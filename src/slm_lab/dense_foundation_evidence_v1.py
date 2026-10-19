"""Complete dense foundation training/export/Adam/source/native reconciliation."""
from collections import Counter
import json
import math
from pathlib import Path

import torch

from .codenet_stage_v1 import EpochOrder
from .code_residual_checkpoint_v1 import _validate_adam
from .corpus import file_hash
from .dense_foundation_data_v1 import NAMES,REFERENCE_ROUTES
from .dense_foundation_held_v1 import readback as held_readback
from .dense_foundation_inference_v1 import options,sources,FORMAT as EXPORT_FORMAT
from .dense_foundation_stage_v1 import FORMAT as STAGE_FORMAT
from .own_dense_checkpoint_v1 import FORMAT as CORE_FORMAT
from .diverse_images import atomic_json


def positions(data,setting,order,step):
    width=setting['accumulation'];per_epoch=math.ceil(len(data)/width);epoch,within=divmod(step,per_epoch)
    return [order.at(epoch*len(data)+p) for p in range(within*width,min((within+1)*width,len(data)))]


def validate_envelope(plan,arm,initial,exported,state,data,plan_sha256):
    setting=options(plan,arm);core=state['core'];saved=exported['model'];expected=sources(plan,plan_sha256)
    cursor=dict(completed_updates=setting['steps'],source_population=len(data),complete_passes=setting['epochs'],
        next_pass_sources=0,actual_draws=len(data)*setting['epochs'],repeated_rounding_sources=0,
        seed=setting['seed'],world_size=1)
    group=core['optimizer']['param_groups'][0]
    if (state.get('format')!=STAGE_FORMAT or core.get('format')!=CORE_FORMAT or core.get('world_size')!=1
            or state.get('source_populations')!=data.populations or core.get('cursor')!=cursor
            or exported.get('format')!=EXPORT_FORMAT or exported.get('arm')!=arm
            or exported.get('training_options')!=setting or state.get('options')!=setting
            or exported.get('actual_dense_foundation_updates')!=setting['steps']
            or saved.get('sources')!=expected or core['model'].get('sources')!=expected
            or saved.get('config')!=initial['config'] or saved.get('settings')!=initial['settings']
            or saved.get('actual_counts')!=plan['actual_counts'] or core['model'].get('actual_counts')!=plan['actual_counts']
            or group['lr']!=setting['learning_rate'] or tuple(group['betas'])!=tuple(setting['betas'])
            or group['weight_decay']!=setting['weight_decay'] or group.get('foreach') is not False
            or any(bundle.get('candidate_promoted') is not False or bundle.get('goal_achieved') is not False
                for bundle in (exported,state,core))
            or any(saved[part].keys()!=initial[part].keys() or core['model'][part].keys()!=initial[part].keys()
                for part in ('network','heads'))):
        raise ValueError('Exact full source/options/optimizer/cursor and original complete model envelope required')


def reconcile(root,reports,output,data,held):
    root,reports,output=map(Path,(root,reports,output));plan=json.loads((reports/'plan.json').read_text())
    if data.budget()!=plan['source_exposure_budget']:raise ValueError('Every complete original source budget must reconstruct')
    initial=torch.load(output/'initial/model.pt',map_location='cpu',weights_only=True,mmap=True)['model']
    language=[n for n in initial['network'] if not n.startswith(('vision.','value_head.')) and n!='lm_head.weight']
    training={};capability={};teacher={}
    for arm in plan['arms']:
        receipt=json.loads((reports/(arm+'-training-complete.json')).read_text());folder=output/arm
        if file_hash(folder/'model.pt')!=receipt['checkpoint_sha256'] or file_hash(folder/'resume.pt')!=receipt['resume_sha256']:
            raise ValueError('Exact actually complete dense model and full Adam source snapshots required')
        exported=torch.load(folder/'model.pt',map_location='cpu',weights_only=True,mmap=True)
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True,mmap=True);core=state['core'];saved=exported['model'];setting=options(plan,arm)
        validate_envelope(plan,arm,initial,exported,state,data,file_hash(reports/'plan.json'))
        if (core['completed']!=setting['steps'] or core['planned']!=setting['steps'] or len(core['metrics'])!=setting['steps']
                or state['options']!=setting or core['optimized_names']!=language or exported['actual_dense_foundation_updates']!=setting['steps']
                or exported['source_exposure_budget']!=plan['source_exposure_budget'] or exported['run_plan_sha256']!=file_hash(reports/'plan.json')
                or state['reference_digest']!=(plan['own_reference_state_sha256'] if setting['reference_KL_alpha'] else None)
                or any(not torch.equal(t,core['model']['network'][n]) for n,t in saved['network'].items())
                or any(not torch.equal(t,core['model']['heads'][n]) for n,t in saved['heads'].items())
                or any(not torch.equal(t,initial['network'][n]) for n,t in saved['network'].items() if n not in language and n!='lm_head.weight')
                or any(not torch.equal(t,initial['heads'][n]) for n,t in saved['heads'].items())):
            raise ValueError('Every actual trained dense tensor, full source/options and unchanged vision/critic/head value must reconcile')
        _validate_adam(core['optimizer'],[saved['network'][n] for n in language],setting['steps'])
        if core['metrics']!=[json.loads(line) for line in (reports/(arm+'-metrics.jsonl')).open()]:
            raise ValueError('Every actual flushed training delivery must equal the complete saved optimizer history')
        seen=Counter();totals={n:Counter() for n in NAMES};order=EpochOrder(len(data),setting['seed'])
        for step,record in enumerate(core['metrics']):
            selected=positions(data,setting,order,step);wanted=[data.information(p) for p in selected];seen.update(selected)
            denominator=math.fsum(setting['coefficients'][r['family']]*r['response_EOS_targets'] for r in wanted)
            if (record['step']!=step+1 or record['source_positions']!=selected or record['original_source_records']!=wanted
                    or record['learning_rate']!=setting['learning_rate'] or record['weighted_target_normalizer']!=denominator
                    or not math.isfinite(record['gradient_norm_before_clip']) or record['gradient_norm_before_clip']<0
                    or record['reference_queried']!=bool(setting['reference_KL_alpha'])
                    or set(record['streams'])!={r['family'] for r in wanted}):
                raise ValueError('Every actual original full-source draw and weighted response/next-token normalizer must reconstruct')
            for name,item in record['streams'].items():
                chosen=[r for r in wanted if r['family']==name];count=sum(r['response_EOS_targets'] for r in chosen)
                if (item['examples']!=len(chosen) or item['complete_context_tokens']!=sum(r['complete_context_tokens'] for r in chosen)
                        or item['response_EOS_targets']!=count or item['coefficient']!=setting['coefficients'][name]
                        or item['reference_route']!=REFERENCE_ROUTES[name]
                        or not math.isfinite(item['CE_sum']) or not math.isfinite(item['reference_KL_sum'])
                        or item['token_mean_CE']!=item['CE_sum']/count or item['token_mean_reference_KL']!=item['reference_KL_sum']/count
                        or (setting['reference_KL_alpha']==0 and item['reference_KL_sum']!=0)):
                    raise ValueError('Every actual original source family and CE/KL term must reconstruct')
                totals[name].update(dict(examples=item['examples'],complete_context_tokens=item['complete_context_tokens'],response_EOS_targets=count))
            objective=math.fsum(r['coefficient']*(r['CE_sum']+setting['reference_KL_alpha']*r['reference_KL_sum']) for r in record['streams'].values())/denominator
            if record['summed_objective']!=objective:raise ValueError('Every full-vocabulary weighted original objective must reconstruct')
        if len(seen)!=len(data) or set(seen.values())!={setting['epochs']}:
            raise ValueError('All original examples/windows once per declared pass and no rounded repeats required')
        for name,budget in plan['source_exposure_budget'].items():
            if dict(totals[name])!={k:budget['original_train_examples' if k=='examples' else k]*setting['epochs'] for k in ('examples','complete_context_tokens','response_EOS_targets')}:
                raise ValueError('Every full original source/token exposure total must match')
        reload=json.loads((reports/arm/'trained-shared-reload.json').read_text())
        if not reload['actual_full_trained_dense_native_cold_reload_exact'] or not receipt['all_fixed_vision_critic_heads_and_own_reference_values_metadata_exact']:
            raise ValueError('Actual complete trained cold reload and all fixed/reference retention required')
        training[arm]=dict(actual_updates=core['completed'],actual_Adam_states=len(core['optimizer']['state']),
            every_complete_original_source_once_per_epoch=True,source_exposure_budget=plan['source_exposure_budget'],
            every_export_tensor_equal_complete_resume=True,all_vision_critic_auxiliary_heads_equal_initial=True,
            checkpoint_sha256=receipt['checkpoint_sha256'])
    from .code_quality import execute
    questions={q['id']:q for q in map(json.loads,(root/'data/public-code-assessment-v1/questions.jsonl').open())}
    for arm in ['initial',*plan['arms']]:
        destination=reports/arm;teacher[arm]=held_readback(held,destination/'held-foundation.json')
        domain=json.loads((destination/'knowledge.json').read_text());math_result=json.loads((destination/'math-complete.json').read_text())
        public=[r for i in range(4) for r in json.loads((destination/f'public-{i}.json').read_text())['results']]
        if len(domain['predictions'])!=559 or len(public)!=175 or {r['id'] for r in public}!=set(questions):raise ValueError('Every original native question and supported public problem exactly once required')
        if math_result['evaluated_examples']!=1319 or not math_result['all_original_IDs_prompts_generated_tokens_numeric_answers_and_scores_reconciled']:
            raise ValueError('All original full native math questions/raw tokens and exact numeric scores required')
        passing=[]
        for row in public:
            if row['execution'].get('functional_passed'):
                q=questions[row['id']];actual=execute(dict(language='Python',code=row['code'],function=q['function'],cases=q['cases'],stress=[]))
                if not actual.get('functional_passed'):raise ValueError('Each delivered public passing program must reexecute all original cases')
                passing.append(row['id'])
        local={}
        for mode in ('completion','instruction'):
            rows=json.loads((destination/('code-'+mode+'.json')).read_text())['results']
            if len(rows)!=24:raise ValueError('Every original local Python/JS task required')
            local[mode]=dict(syntax=sum(r['candidate'].get('syntax_valid',False) for r in rows),functional=sum(r['candidate'].get('functional_passed',False) for r in rows),efficiency=sum(r['efficiency_stress_passed'] for r in rows))
        cpp=json.loads((destination/'C++.json').read_text());vision=json.loads((destination/'vision.json').read_text())
        capability[arm]=dict(knowledge=domain['overall'],public_code=dict(tasks=175,syntax=sum(r['execution'].get('syntax_valid',False) for r in public),
            functional=len(passing),independently_reexecuted_passing_ids=passing),local_code=local,
            CPP={k:v for k,v in cpp.items() if k not in ('results','evaluation_provenance')},
            math={k:v for k,v in math_result.items() if k not in ('results','evaluation_provenance','shard_reports_sha256')},
            OCR=json.loads((destination/'ocr.json').read_text())['overall'],objects=json.loads((destination/'objects.json').read_text())['overall'],
            captions={k:vision[k] for k in ('images','proper_nll','blank_nll_gain','mismatch_nll_gain','caption_retrieval')})
    report=dict(complete_original_dense_training=training,complete_held_teacher_metrics=teacher,matched_native_capability=capability,
        separate_own_reference_is_training_only_not_in_exported_model=True,no_external_teacher_weights=True,
        full_language_foundation_not_final_multimodal_or_genuine_human_RLHF=True,
        all_code_languages_training_not_a_correctness_or_efficiency_proof=True,
        source_and_public_development_exposure_qualifications_retained=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(reports/'independent-comparison.json',report);return report
