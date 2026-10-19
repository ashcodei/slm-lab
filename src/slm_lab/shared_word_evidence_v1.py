"""Full source exposure, retained Adam and actual held reading delivery audit."""
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path

import torch

from .codenet_stage_v1 import EpochOrder
from .code_residual_checkpoint_v1 import _validate_adam
from .corpus import file_hash
from .diverse_images import atomic_json
from .shared_word_inference_v1 import options
from .shared_word_stage_v1 import PREFIXES


@lru_cache(maxsize=16)
def _order(size,seed): return EpochOrder(size,seed)


def positions(data, setting, step):
    width=setting['microbatch']*setting['accumulation'];per_epoch=math.ceil(len(data)/width)
    epoch,within=divmod(step,per_epoch)
    return [_order(len(data),setting['seed']).at(epoch*len(data)+p)
            for p in range(within*width,min((within+1)*width,len(data)))]


def counts(data, selected):
    rows=[data.rows[p] for p in selected]
    return dict(words=len(rows),UTF8_byte_targets=sum(len(r['response'].encode('utf-8')) for r in rows),
        complete_context_tokens=sum(r['native']['length'] for r in rows),
        response_EOS_targets=sum(r['native']['length']-r['native']['target_start'] for r in rows),
        by_family=dict(Counter(r['byte_family'] for r in rows)))


def budget(data, setting):
    actual=counts(data,range(len(data)));epochs=setting['epochs']
    if setting['steps']!=epochs*math.ceil(len(data)/(setting['microbatch']*setting['accumulation'])):
        raise ValueError('Exact full-source unrepeated-final-batch word budget required')
    return dict(source_population=len(data),source_exposures=len(data)*epochs,unique_original_sources=len(data),
        complete_passes=epochs,repeated_rounding_exposures=0,UTF8_byte_targets=actual['UTF8_byte_targets']*epochs,
        complete_context_tokens=actual['complete_context_tokens']*epochs,response_EOS_targets=actual['response_EOS_targets']*epochs,
        by_family={k:v*epochs for k,v in actual['by_family'].items()})


def _edits(a,b):
    # Independent row-score reconstruction; no runtime scorer imported.
    row=list(range(len(b)+1))
    for i,x in enumerate(a):
        next_row=[i+1]
        for j,y in enumerate(b):next_row.append(min(row[j+1]+1,next_row[-1]+1,row[j]+(x!=y)))
        row=next_row
    return row[-1]


def readback_held(data, tokenizer, path, generation_budget, *, ctc_frames=140):
    path=Path(path);report=json.loads(path.read_text());rows=report['results']
    if (len(rows)!=len(data) or report['generation_budget']!=generation_budget
            or rows!=[json.loads(line) for line in path.with_suffix('.jsonl').open()]
            or sorted(r['source_position'] for r in rows)!=list(range(len(data)))):
        raise ValueError('Every original held source and flushed raw delivery must reconcile exactly')
    ids={row['id']:row for row in data.rows};wrong=[]
    for p,record in enumerate(rows):
        source=data.rows[p];donor=ids[record['wrong_image_id']];wrong.append(record['wrong_image_id'])
        offset=source['native']['offset'];start=source['native']['target_start'];size=source['native']['length']
        sequence=data.tokens[offset:offset+size].tolist()
        if (record['source_position']!=p or record['id']!=source['id'] or record['expected']!=source['response']
                or record['group_id']!=source['group_id'] or record['family']!=source['byte_family']
                or record['source_image_sha256']!=source['stored_image_sha256']
                or record['native_prompt_ids']!=sequence[:start]
                or donor['byte_family']!=source['byte_family'] or donor['group_id']==source['group_id']
                or donor['response']==source['response'] or donor['source_RGB_pixels_sha256']==source['source_RGB_pixels_sha256']
                or set(record['conditions'])!={'proper','blank','wrong'}):
            raise ValueError('Every actual held source and within-family mismatched-photo control must reconstruct')
        for condition,item in record['conditions'].items():
            ctc=item['CTC'];native=item['native'];path_ids=ctc['path'];raw=[];previous=None
            if len(path_ids)!=ctc_frames or any(type(v) is not int or not 0<=v<=256 for v in path_ids):raise ValueError('Every original full byte path required')
            for value in path_ids:
                if value!=256 and value!=previous:raw.append(value)
                previous=value
            try: generated=bytes(raw).decode('utf-8');valid=True
            except UnicodeDecodeError:generated=bytes(raw).decode('utf-8',errors='replace');valid=False
            if (ctc['generated_UTF8_bytes']!=raw or ctc['generated']!=generated or ctc['valid_UTF8']!=valid
                    or ctc['byte_edits']!=_edits(list(source['response'].encode('utf-8')),raw)
                    or ctc['reference_bytes']!=len(source['response'].encode('utf-8'))
                    or not math.isfinite(ctc['byte_normalized_CTC'])):
                raise ValueError('Every complete CTC byte collapse, invalid-UTF8 delivery and byte edit score must reconstruct')
            generated_ids=native['generated_ids']
            if (not generated_ids or len(generated_ids)>generation_budget or 2 in generated_ids[:-1]
                    or native['stop']!=('EOS' if generated_ids[-1]==2 else 'length')
                    or (native['stop']=='length' and len(generated_ids)!=generation_budget)
                    or native['generated']!=tokenizer.decode(generated_ids,skip_special_tokens=True)
                    or native['target_ids']!=sequence[start:]
                    or len(native['per_target_NLL'])!=size-start
                    or any(not math.isfinite(v) or v<0 for v in native['per_target_NLL'])):
                raise ValueError('Every native prompt/whole-responseEOS/greedy token delivery and target likelihood record must reconstruct')
            for head in (ctc,native):
                expected=source['response'];actual=head['generated']
                scores=dict(exact=expected==actual,character_edits=_edits(expected,actual),reference_characters=len(expected),
                    word_edits=_edits(expected.split(),actual.split()),reference_words=len(expected.split()))
                if any(head[k]!=v for k,v in scores.items()):raise ValueError('Every literal complete native/CTC exact and edit score must reconstruct')
    if len(set(wrong))!=len(data):raise ValueError('Every held image must be used exactly once as a within-family wrong control')
    # Score fields above have been independently reconstructed before using
    # the runtime's shared aggregation routine. Neural logits are not replayed.
    from .shared_word_evaluation_v1 import summarize
    if (report['overall']!=summarize(rows) or report['by_family']!={f:summarize([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})}):
        raise ValueError('Every all-source and per-family held aggregate must reconstruct')
    return dict(overall=report['overall'],by_family=report['by_family'],all_raw_paths_tokens_literal_scores_and_original_source_roles_reconstructed=True,
                numerical_neural_logits_and_CTC_likelihoods_not_replayed=True,report_sha256=file_hash(path))


def reconcile(root, reports, output, data, held, tokenizer):
    root,reports,output=map(Path,(root,reports,output));plan=json.loads((reports/'plan.json').read_text())
    initial=torch.load(output/'initial/model.pt',map_location='cpu',weights_only=True,mmap=True)
    initial_parts=initial['aligned_model']['extensions'];training={};reading={};native={}
    if budget(data,plan['training_options'])!=plan['source_exposure_budget']:raise ValueError('Every full original training-word exposure must reconstruct')
    for arm in plan['arms']:
        receipt=json.loads((reports/(arm+'-training-complete.json')).read_text());folder=output/arm
        if file_hash(folder/'model.pt')!=receipt['checkpoint_sha256'] or file_hash(folder/'resume.pt')!=receipt['resume_sha256']:
            raise ValueError('Actual complete word-worker model and optimizer bytes required')
        saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=True,mmap=True)
        resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True,mmap=True);core=resume['core'];parts=saved['aligned_model']['extensions']
        names=[n for n in parts if n.startswith(PREFIXES)];setting=options(plan,arm)
        if (core['steps']!=setting['steps'] or core['planned_steps']!=setting['steps'] or resume['options']!=setting
                or core['fixed_original_state_sha256']!=plan['original_state_sha256']
                or core['optimized_names']!=names or len(core['metrics'])!=setting['steps']
                or saved['arm']!=arm or saved['actual_shared_word_updates']!=setting['steps'] or saved['run_plan_sha256']!=file_hash(reports/'plan.json')
                or set(parts)!=set(initial_parts) or any(not torch.equal(t,core['model']['extensions'][n]) for n,t in parts.items())
                or any(not torch.equal(t,initial_parts[n]) for n,t in parts.items() if not n.startswith(PREFIXES))):
            raise ValueError('Every full trained export/resume and unchanged language/reward/critic tensor must reconcile')
        _validate_adam(core['optimizer'],[parts[n] for n in names],setting['steps'])
        total=Counter();seen=Counter()
        for step,record in enumerate(core['metrics']):
            selected=positions(data,setting,step);wanted=counts(data,selected);seen.update(selected)
            if (record['source_positions']!=selected or record['step']!=step+1 or record['learning_rate']!=setting['learning_rate']
                    or any(record[k]!=v for k,v in wanted.items())
                    or any(not math.isfinite(record[k]) for k in ('word_mean_byte_normalized_CTC','token_mean_native_CE','gradient_norm_before_clip'))
                    or record['summed_objective']!=setting['ctc_weight']*record['word_mean_byte_normalized_CTC']+setting['native_CE_weight']*record['token_mean_native_CE']):
                raise ValueError('Every source draw/family/complete target normalizer and joint objective must reconstruct')
            total.update({k:v for k,v in wanted.items() if k!='by_family'})
        if len(seen)!=len(data) or set(seen.values())!={setting['epochs']}:
            raise ValueError('Each original source exactly once per declared epoch; no rounded repeats')
        if any(total[k]!=plan['source_exposure_budget'][k] for k in ('UTF8_byte_targets','complete_context_tokens','response_EOS_targets')):
            raise ValueError('All original full-word token exposures must match')
        cold=json.loads((reports/arm/'trained-shared-reload.json').read_text())
        if not cold['actual_trained_image_word_text_and_all_original_routes_exact'] or not receipt['all_fixed_weights_metadata_text_only_original_routes_and_code_cache_exact']:
            raise ValueError('Actual full trained cold reload and complete fixed-language/original retention required')
        training[arm]=dict(actual_updates=core['steps'],actual_Adam_states=len(core['optimizer']['state']),
            source_exposure_budget=plan['source_exposure_budget'],all_trained_tensors_equal_complete_resume=True,
            every_fixed_language_reward_critic_extension_equal_initial=True,checkpoint_sha256=receipt['checkpoint_sha256'])
    for arm in ['initial',*plan['arms']]:
        reading[arm]=readback_held(held,tokenizer,reports/arm/'all-held-words.json',plan['evaluation']['generation_budget'])
        native[arm]={name:json.loads((reports/arm/(name+'.json')).read_text())['overall'] for name in ('ocr','objects')}
        vision=json.loads((reports/arm/'vision.json').read_text())
        native[arm]['captions']={k:vision[k] for k in ('images','proper_nll','blank_nll_gain','mismatch_nll_gain','caption_retrieval')}
    report=dict(complete_original_word_training=training,all_original_held_word_and_image_controls=reading,
        matched_native_vision=native,language_weights_and_text_only_computation_unchanged=True,
        original_generic_image_and_final_policy_quality_must_be_assessed_from_results=True,
        old_own_code_language_anchor_not_later_mixed_alignment_policy=True,
        word_crops_not_full_photo_detection=True,genuine_human_RLHF_not_established_by_word_training=True,
        candidate_promoted=False,goal_achieved=False)
    atomic_json(reports/'independent-comparison.json',report);return report
