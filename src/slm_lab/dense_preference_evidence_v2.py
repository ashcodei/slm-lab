"""Complete-source policy accounting and final matched native evidence."""
from collections import Counter
import json
import math
from pathlib import Path

import torch

from .codenet_stage_v1 import EpochOrder
from .code_residual_checkpoint_v1 import _validate_adam,_validate_rng
from .corpus import file_hash
from .cuda_rng_readback_v1 import review_cuda_rng_layout
from .dense_human_preference_stage_v1 import validate_metric
from .dense_human_preference_evaluation_v1 import readback as preference_readback
from .dense_preference_inference_v1 import options,sources
from .diverse_images import atomic_json
from .shared_word_evidence_v1 import readback_held


def positions(data,setting,order,step):
    width=setting['accumulation'];per=math.ceil(len(data)/width);epoch,within=divmod(step,per)
    return [order.at(epoch*len(data)+p) for p in range(within*width,min((within+1)*width,len(data)))]


def budget(data,setting):
    epochs=setting['epochs'];width=setting['accumulation'];weight=setting['DPO_weight']
    contexts={f:sum(r['native'][f]['length'] for r in data.rows) for f in ('chosen','rejected')}
    targets={f:sum(r['native'][f]['length']-r['native'][f]['target_start'] for r in data.rows) for f in ('chosen','rejected')}
    if setting['steps']!=epochs*math.ceil(len(data)/width):raise ValueError('Complete unrepeated source epoch required')
    return dict(original_pairs=len(data),epochs=epochs,pair_exposures=len(data)*epochs,rounded_pair_repeats=0,
        original_chosen_targets=targets['chosen'],original_rejected_targets=targets['rejected'],
        actual_policy_context_tokens=epochs*(contexts['chosen']+weight*contexts['rejected']),
        actual_reference_context_tokens=epochs*weight*sum(contexts.values()),
        actual_policy_chosen_targets=epochs*targets['chosen'],actual_policy_rejected_targets=epochs*weight*targets['rejected'],
        actual_reference_chosen_targets=epochs*weight*targets['chosen'],actual_reference_rejected_targets=epochs*weight*targets['rejected'],
        actual_last_update_pairs=len(data)%width or width)


def reconcile(root,reports,output,data,held,words,tokenizer):
    root,reports,output=map(Path,(root,reports,output));plan_path=reports/'plan.json';plan=json.loads(plan_path.read_text());sha=file_hash(plan_path)
    initial=torch.load(output/'initial/model.pt',map_location='cpu',weights_only=True,mmap=True)['policy']
    names=[n for n in initial['parts'] if not n.startswith('critic.')];training={};preference={};reading={};native={}
    for arm in plan['arms']:
        setting=options(plan,arm);folder=output/arm;receipt=json.loads((reports/(arm+'-training-complete.json')).read_text())
        if file_hash(folder/'model.pt')!=receipt['checkpoint_sha256'] or file_hash(folder/'resume.pt')!=receipt['resume_sha256']:
            raise ValueError('Exact complete policy and optimizer bytes required')
        model=torch.load(folder/'model.pt',map_location='cpu',weights_only=True,mmap=True)
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True,mmap=True);core=state['core'];saved=model['policy'];steps=setting['steps']
        if (state['options']!=setting or state['source_population']!=len(data) or core['completed']!=steps or core['planned']!=steps
                or core['optimized_names']!=names or core['groups']!=['actor'] or core['world_size']!=1
                or saved['sources']!=sources(plan,sha) or core['model']['sources']!=sources(plan,sha)
                or saved['frozen_dense_sha256']!=plan['frozen_dense_sha256'] or state['frozen_dense_sha256']!=plan['frozen_dense_sha256']
                or core['model']['frozen_dense_sha256']!=plan['frozen_dense_sha256'] or model['actual_human_preference_updates']!=steps
                or model['training_options']!=setting or model['source_exposure_budget']!=budget(data,setting)
                or model['source_exposure_budget']!=plan['source_exposure_budget'][arm]
                or saved['actual_counts']!=plan['actual_counts'] or saved['parts'].keys()!=initial['parts'].keys()
                or any(not torch.equal(t,core['model']['parts'][n]) for n,t in saved['parts'].items())
                or not torch.equal(saved['parts']['critic.weight'],initial['parts']['critic.weight'])):
            raise ValueError('Every actual complete policy tensor, frozen base/critic and objective/source provenance must match')
        _validate_adam(core['optimizer'],[saved['parts'][n] for n in names],steps)
        # CPU reviewer never initializes CUDA. Actual CUDA state restoration
        # is checked by each owned capacity worker; here check its raw shape.
        _validate_rng({**core['rng'],'cuda':None},'cpu');cuda=core['rng']['cuda']
        cuda_layout=review_cuda_rng_layout(cuda)
        if core['metrics']!=[json.loads(line) for line in (reports/(arm+'-metrics.jsonl')).open()]:
            raise ValueError('Every original flushed update must equal the complete saved history')
        order=EpochOrder(len(data),setting['seed']);seen=Counter();counts=Counter()
        for step,record in enumerate(core['metrics']):
            chosen=positions(data,setting,order,step);validate_metric(data,chosen,setting,record,step);seen.update(chosen)
            counts.update({k:record[k] for k in ('actual_policy_context_tokens','actual_reference_context_tokens','actual_policy_chosen_targets',
                'actual_policy_rejected_targets','actual_reference_chosen_targets','actual_reference_rejected_targets')})
        if len(seen)!=len(data) or set(seen.values())!={setting['epochs']}:
            raise ValueError('Every original human pair once per declared pass and no rounded repeats required')
        expected=budget(data,setting)
        if any(counts[k]!=expected[k] for k in counts):raise ValueError('Actual CE/DPO branch-specific token exposures must reconcile')
        cursor=dict(completed_updates=steps,source_population=len(data),complete_passes=setting['epochs'],next_pass_sources=0,
            actual_pair_draws=len(data)*setting['epochs'],repeated_rounding_pairs=0,seed=setting['seed'],world_size=1)
        if core['cursor']!=cursor or not receipt['actual_full_trained_policy_native_cold_reload_exact']:
            raise ValueError('Complete exact source cursor and actual trained cold reload required')
        training[arm]=dict(actual_updates=steps,actual_actor_Adam_states=len(core['optimizer']['state']),actual_exposures=expected,retained_CUDA_RNG_layout=cuda_layout,
            every_original_pair_once_per_epoch=True,all_exported_policy_values_equal_complete_resume=True,
            same_frozen_dense_base_and_untouched_separate_critic=True,checkpoint_sha256=receipt['checkpoint_sha256'])
    from .code_quality import execute
    questions={q['id']:q for q in map(json.loads,(root/'data/public-code-assessment-v1/questions.jsonl').open())}
    for arm in ['initial',*plan['arms']]:
        dest=reports/arm;preference[arm]=preference_readback(held,dest/'held-preferences.json')
        reading[arm]=readback_held(words,tokenizer,dest/'held-words.json',64)
        knowledge=json.loads((dest/'knowledge.json').read_text());maths=json.loads((dest/'math-complete.json').read_text())
        public=[r for i in range(4) for r in json.loads((dest/f'public-{i}.json').read_text())['results']]
        if (len(knowledge['predictions'])!=559 or len(public)!=175 or {r['id'] for r in public}!=set(questions)
                or maths['evaluated_examples']!=1319 or not maths['all_original_IDs_prompts_generated_tokens_numeric_answers_and_scores_reconciled']):
            raise ValueError('All original knowledge/public-code/native-math questions required')
        passing=[]
        for row in public:
            if row['execution'].get('functional_passed'):
                q=questions[row['id']];result=execute(dict(language='Python',code=row['code'],function=q['function'],cases=q['cases'],stress=[]))
                if not result.get('functional_passed'):raise ValueError('Every public passing program must independently execute all original cases')
                passing.append(row['id'])
        local={}
        for mode in ('completion','instruction'):
            rows=json.loads((dest/('code-'+mode+'.json')).read_text())['results']
            if len(rows)!=24:raise ValueError('Every original Python/JavaScript functional/efficiency task required')
            local[mode]=dict(syntax=sum(r['candidate'].get('syntax_valid',False) for r in rows),
                functional=sum(r['candidate'].get('functional_passed',False) for r in rows),efficiency=sum(r['efficiency_stress_passed'] for r in rows))
        cpp=json.loads((dest/'C++.json').read_text());vision=json.loads((dest/'vision.json').read_text())
        native[arm]=dict(knowledge=knowledge['overall'],public_code=dict(tasks=175,syntax=sum(r['execution'].get('syntax_valid',False) for r in public),
            functional=len(passing),independently_reexecuted_pass_ids=passing),local_code=local,
            CPP={k:v for k,v in cpp.items() if k not in ('results','evaluation_provenance')},
            math={k:v for k,v in maths.items() if k not in ('results','evaluation_provenance','shard_reports_sha256')},
            OCR=json.loads((dest/'ocr.json').read_text())['overall'],objects=json.loads((dest/'objects.json').read_text())['overall'],
            captions={k:vision[k] for k in ('images','proper_nll','blank_nll_gain','mismatch_nll_gain','caption_retrieval')})
    result=dict(complete_human_policy_training=training,complete_held_preferences=preference,complete_held_word_image_controls=reading,
        matched_initial_all_arm_native_capability=native,matched_chosen_CE_control_and_DPO=True,
        all_reference_weights_share_the_counted_frozen_base=True,critic_untrained_and_no_PPO=True,
        published_human_ranked_other_model_responses_not_fresh_own_output_judgments=True,
        reused_source_and_development_selection_qualifications_retained=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(reports/'independent-comparison.json',result);return result
