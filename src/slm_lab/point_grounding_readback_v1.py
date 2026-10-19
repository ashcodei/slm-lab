"""Independent epoch/export accounting and matched original attention results."""
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
import random

import torch

from .apps_native_metrics_v1 import matched,summarize as native_summary
from .corpus import file_hash
from .diverse_images import atomic_json
from .localized_pointer_evidence_v1 import dense_target,image_layout
from .point_grounding_protocol_v1 import prior_logits,score,summarize


def mapping_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        digest.update(json.dumps([name,list(value.shape),str(value.dtype)],separators=(',',':')).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def recount_attention(observations,rows,photos,pairs,position_prior):
    source={r['id']:r for r in rows};pairing={p['id']:p for p in pairs}
    expected={(r['id'],condition) for r in rows for condition in
        ['proper','blank_image','wrong_image','uniform','center','training_position']}
    expected.update((p['id'],'wrong_query') for p in pairs if p['wrong_query_id'] is not None)
    actual={};by_condition=defaultdict(dict)
    blank_sha=hashlib.sha256(bytes([128,128,128])*(448*448)).hexdigest()
    for record in observations:
        key=(record['id'],record['condition'])
        if key not in expected or key in actual:
            raise ValueError('Every original query/control must occur exactly once')
        row=source[key[0]];condition=key[1];pair=pairing[key[0]]
        query_id=pair['wrong_query_id'] if condition=='wrong_query' else row['id']
        image_id=None if condition=='blank_image' else pair['wrong_image_id'] if condition=='wrong_image' else row['image_id']
        sha=blank_sha if image_id is None else photos[image_id]['model_pixel_sha256_448']
        target,valid=dense_target(row,photos[row['image_id']]['stored_dimensions'])
        if (record['image_id']!=row['image_id'] or record['actual_query_id']!=query_id
                or record['actual_query_input_ids']!=source[query_id]['query_input_ids']
                or record['actual_image_id']!=image_id or record['actual_448px_RGB_sha256']!=sha
                or record['original_attention_grid_histogram']!=row['attention_grid_histogram']
                or record['original_target_valid_cells']!=valid
                or record['scores']!=score(record['logits'],target,valid)):
            raise ValueError('Actual original image/query/control pixels, targets and scores must reconstruct')
        if condition in ['uniform','center','training_position']:
            (left,top,width,height),_=image_layout(photos[row['image_id']]['stored_dimensions'])
            center=int((top+height/2)//32)*14+int((left+width/2)//32)
            if record['logits']!=prior_logits(condition,valid,position_prior,center):
                raise ValueError('Exact original train-only or fixed geometry prior required')
        actual[key]=record;by_condition[condition][row['id']]=record
    if actual.keys()!=expected:
        raise ValueError('Missing actual original attention diagnostic query/control')
    return summarize(actual.values()),by_condition


def bootstrap_photo_gain(proper,control,*,seed=48303,replicates=1000):
    if not control or control.keys()-proper.keys():
        raise ValueError('Same original query subset required for paired gain')
    groups=defaultdict(list)
    for identity,row in control.items():
        if row['image_id']!=proper[identity]['image_id']:
            raise ValueError('Paired gain requires original target photo identity')
        groups[row['image_id']].append(row['scores']['soft_target_cross_entropy']-proper[identity]['scores']['soft_target_cross_entropy'])
    values=[sum(v)/len(v) for _,v in sorted(groups.items())]
    rng=random.Random(seed)
    means=sorted(sum(values[rng.randrange(len(values))] for _ in values)/len(values) for _ in range(replicates))
    return dict(original_queries=len(control),original_photos=len(values),equal_photo_mean_CE_gain=sum(values)/len(values),
        percentile95_interval=[means[int(.025*replicates)],means[min(replicates-1,int(.975*replicates))]],
        bootstrap_replicates=replicates,bootstrap_seed=seed,
        scope='Paired original photo-level development diagnostic, not independent final performance.')


def reconcile(root,reports,output,data):
    plan=json.loads((reports/'plan.json').read_text());options=plan['training_options']
    complete=json.loads((reports/'training-complete.json').read_text())
    if complete['steps']!=1137 or file_hash(output/'resume.pt')!=complete['resumable_checkpoint_sha256']:
        raise ValueError('Actual complete original attention checkpoint required')
    state=torch.load(output/'resume.pt',map_location='cpu',weights_only=True,mmap=True)
    native=torch.load(output/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    extra=torch.load(output/'grounding-head.pt',map_location='cpu',weights_only=True)
    base_path=root/'checkpoints/extended-curriculum-v1/joint/model.pt'
    base=torch.load(base_path,map_location='cpu',weights_only=True,mmap=True)
    if (state['steps']!=1137 or state['planned_steps']!=1137 or state['world_size']!=4
            or state['training_kind']!='original_human_pointer_vision_head_v1'
            or native['training_kind']!='original_human_pointer_vision_head_v1'
            or state['config']!=native['config'] or native['config']!=base['config']
            or state['training_options']!=options or state['sources']['run_plan_sha256']!=file_hash(reports/'plan.json')
            or native['source_checkpoint_sha256']!=file_hash(base_path) or native['steps']!=1137
            or native['tokenizer_sha256']!=plan['tokenizer_sha256']
            or native['data_sha256']!=file_hash(data/'admission.json')
            or extra['data_admission_sha256']!=file_hash(data/'admission.json')
            or extra['native_checkpoint_sha256']!=file_hash(output/'model.pt')
            or extra['run_plan_sha256']!=file_hash(reports/'plan.json') or extra['steps']!=1137):
        raise ValueError('Exact completed own model, source, step and head/native bindings required')
    expected={**{'base.'+n:t for n,t in native['model'].items()},
        'query_projection.weight':extra['query_projection']['weight'],'patch_projection.weight':extra['patch_projection']['weight']}
    if expected.keys()!=state['model'].keys() or len(expected)!=334:
        raise ValueError('Every actual native base and additional head tensor required')
    for name,tensor in expected.items():
        if not bool(torch.isfinite(tensor).all()) or not torch.equal(tensor,state['model'][name]):
            raise ValueError('Actual finite complete export must equal the final resumable state')
    frozen=hashlib.sha256();changed_vision=0
    for name,tensor in native['model'].items():
        if not name.startswith('vision.'):
            if not torch.equal(tensor,base['model'][name]):
                raise ValueError('Every original language/critic tensor must remain exactly unchanged')
        else:
            changed_vision+=not torch.equal(tensor,base['model'][name])
    for name,tensor in sorted(native['model'].items()):
        if not name.startswith('vision.'):
            frozen.update(json.dumps([name,list(tensor.shape),str(tensor.dtype)],separators=(',',':')).encode())
            frozen.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    if not changed_vision or frozen.hexdigest()!=plan['frozen_language_critic_sha256']:
        raise ValueError('Original frozen language digest and genuinely changed own vision required')
    groups=state['optimizer']['param_groups'];ids=[i for g in groups for i in g['params']]
    names=state['trainable_parameter_names']
    if len(ids)!=114 or len(names)!=114 or len(set(ids))!=114 or state['optimizer']['state'].keys()!=set(ids):
        raise ValueError('All114 actual vision/head Adam states required')
    for identity,name in zip(ids,names):
        entry=state['optimizer']['state'][identity];parameter=state['model'][name]
        if (float(entry['step'])!=1137 or any(entry[k].shape!=parameter.shape or not bool(torch.isfinite(entry[k]).all())
                for k in ['exp_avg','exp_avg_sq']) or bool((entry['exp_avg_sq']<0).any())):
            raise ValueError('Actual complete finite114Adam steps and moments required')
    if len(state['worker_rng_states'])!=4:
        raise ValueError('Every actual original worker RNG must be retained')
    for rng in state['worker_rng_states']:
        random.Random().setstate(rng['python']);torch.Generator(device='cpu').set_state(rng['torch'])
        import numpy as np
        n=rng['numpy'];np.random.RandomState(0).set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
        if not isinstance(rng['cuda'],torch.Tensor) or rng['cuda'].dtype!=torch.uint8 or not rng['cuda'].numel():
            raise ValueError('Every original actual CUDA RNG payload must be retained')
    rows=[json.loads(line) for line in (data/'train.jsonl').read_text().splitlines()]
    order=list(range(len(rows)));random.Random(options['seed']).shuffle(order)
    metrics=[json.loads(line) for line in (reports/'training-metrics.jsonl').read_text().splitlines()]
    if metrics!=state['metrics'] or len(metrics)!=1137 or len(rows)!=36354:
        raise ValueError('Every actual saved/source metric and original training query required')
    exposures=[];samples=0;tokens=0
    for step,metric in enumerate(metrics):
        if metric['step']!=step+1 or len(metric['workers'])!=4 or metric['global_queries']!=32:
            raise ValueError('Actual ordered1137collective updates required')
        point_total=0;token_total=0
        for micro in range(4):
            for rank,worker in enumerate(metric['workers']):
                start=step*32+(micro*4+rank)*2;positions=[order[(start+j)%len(order)] for j in range(2)]
                observed=worker['micros'][micro]
                points=sum(rows[p]['source_point_count'] for p in positions);length=sum(len(rows[p]['query_input_ids']) for p in positions)
                if (worker['rank']!=rank or observed['positions']!=positions
                        or observed['original_attention_samples']!=points or observed['complete_query_tokens']!=length):
                    raise ValueError('Every actual original query position, point sample and whole token draw must reconstruct')
                exposures.extend(positions);point_total+=points;token_total+=length
        if (metric['original_attention_samples']!=point_total or metric['whole_query_tokens']!=token_total
                or not math.isfinite(metric['query_mean_nll'])
                or not math.isclose(metric['query_mean_nll'],sum(w['query_mean_nll'] for w in metric['workers'])/4,abs_tol=1e-12)):
            raise ValueError('Actual whole-query objective weights and sample accounting must reconcile')
        samples+=point_total;tokens+=token_total
    cursor=dict(completed_steps=1137,population=36354,seed=options['seed'],source_exposures=36384,
        order_sha256=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest())
    if (state['sampler_state']!=cursor or exposures[:36354]!=order or exposures[36354:]!=order[:30]):
        raise ValueError('Every original training query exactly once, then only30sealed rounding repeats required')
    proof=dict(completed_at=datetime.now(timezone.utc).isoformat(),actual_steps=1137,original_unique_queries=36354,
        actual_query_exposures=36384,rounding_repeats=30,actual_original_attention_samples=samples,actual_whole_query_tokens=tokens,
        all334_export_and_resumable_tensors_equal_and_finite=True,complete_finite_Adam_states=114,all_four_worker_RNG_payloads_retained=True,
        exact_CPU_collective_recovery_test_passed_but_full_model_CUDA_recovery_not_exercised=True,
        all_original_language_critic_tensors_byte_equal=True,frozen_language_critic_sha256=frozen.hexdigest(),
        changed_vision_tensors=changed_vision,actual_inventory=state['actual_inventory'],
        resumable_checkpoint_sha256=file_hash(output/'resume.pt'),native_export_sha256=file_hash(output/'model.pt'),
        additional_head_sha256=file_hash(output/'grounding-head.pt'),source_plan_sha256=file_hash(reports/'plan.json'),
        high_end_or_allround_capability_established=False,goal_achieved=False)
    atomic_json(reports/'complete-neural-source-readback.json',proof)
    held=[json.loads(line) for line in (data/'validation.jsonl').read_text().splitlines()]
    photos={r['image_id']:r for r in map(json.loads,(data/'photos.jsonl').read_text().splitlines()) if r['split']=='validation'}
    pairs=[json.loads(line) for line in (data/'validation-control-pairs.jsonl').read_text().splitlines()]
    prior=json.loads((data/'training-position-prior.json').read_text());full={};mapped={}
    before_plan=json.loads((reports/'baseline/grounding/plan.json').read_text())
    after_plan=json.loads((reports/'candidate/grounding/plan.json').read_text())
    if {k:v for k,v in before_plan.items() if k not in ['actual_model_state_sha256','created_at']}!={k:v for k,v in after_plan.items() if k not in ['actual_model_state_sha256','created_at']}:
        raise ValueError('Exactly matched original attention inputs, controls, batch, dtype, device and runtime required')
    capacity=json.loads((reports/'capacity.json').read_text())
    if (before_plan['actual_model_state_sha256']!=capacity['initial_state_sha256']
            or after_plan['actual_model_state_sha256']!=mapping_digest(state['model'])):
        raise ValueError('Actual assessed complete initial/final own-model tensor bytes must match')
    for phase in ['baseline','candidate']:
        path=reports/phase/'grounding/observations.jsonl'
        observed=[json.loads(line) for line in path.read_text().splitlines()]
        full[phase],mapped[phase]=recount_attention(observed,held,photos,pairs,prior)
        report=json.loads((reports/phase/'grounding/complete.json').read_text())
        if report['observations_sha256']!=file_hash(path) or report['conditions']!=full[phase]['conditions']:
            raise ValueError('Every actual retained attention condition and original score must reconcile')
    gains={condition:bootstrap_photo_gain(mapped['candidate']['proper'],mapped['candidate'][condition])
        for condition in ['blank_image','wrong_image','uniform','center','training_position','wrong_query']}
    gains['before_to_after_proper']=bootstrap_photo_gain(mapped['candidate']['proper'],mapped['baseline']['proper'])
    native_scores={}
    for name in ['ocr.json','objects.json','vision.json']:
        before=json.loads((reports/'baseline'/name).read_text());after=json.loads((reports/'candidate'/name).read_text());matched(before,after,name)
        native_scores[name]=dict(baseline=native_summary(before,name),candidate=native_summary(after,name))
    atomic_json(reports/'matched-attention-and-native-comparison.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        attention=full,paired_original_photo_gain_intervals=gains,matched_native_image_assessments=native_scores,
        every_raw_original_query_image_pixel_target_prior_and_score_reconstructed=True,
        attention_is_approximate_mouse_grid_not_object_box_OCR_or_general_visual_reasoning_accuracy=True,
        original_validation_not_certified_independent_final_test=True,automatic_promotion=False,goal_achieved=False))
