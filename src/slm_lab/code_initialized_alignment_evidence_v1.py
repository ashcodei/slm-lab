"""Complete mixed-source budgets and actual trained/native endpoint readback."""
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path

import torch

from .code_residual_checkpoint_v1 import _validate_adam
from .codenet_stage_v1 import EpochOrder
from .corpus import file_hash
from .diverse_images import atomic_json


@lru_cache(maxsize=16)
def _order(population,seed):
    return EpochOrder(population,seed)


def positions(streams, options, step):
    return {name:[_order(len(streams[name]),options['seed']+i*1009).at(step*width+j) for j in range(width)]
            for i,(name,width) in enumerate([('human',4),('generation',1)])}


def source_counts(streams, name, selected):
    if name=='human':
        rows=[streams[name].rows[p]['native']['response'] for p in selected]
        return sum(r['length'] for r in rows),sum(r['length']-r['target_start'] for r in rows)
    rows=[streams[name].pack.index[streams[name].pack.positions[p]] for p in selected]
    return sum(int(r['length']) for r in rows),sum(int(r['length'])-int(r['target_start']) for r in rows)


def budget(streams, options):
    results={}
    for index,(name,width) in enumerate([('human',4),('generation',1)]):
        order=EpochOrder(len(streams[name]),options['seed']+index*1009)
        draws=[order.at(i) for i in range(options['steps']*width)];counts=Counter(draws)
        if len(counts)!=len(streams[name]):raise ValueError('Every original human conversation and verified program must appear')
        context,targets=source_counts(streams,name,draws)
        results[name]=dict(source_population=len(streams[name]),source_exposures=len(draws),unique_original_positions=len(counts),
            complete_passes=len(draws)//len(streams[name]),repeated_exposures=len(draws)-len(counts),
            repeated_source_positions={str(p):n for p,n in counts.items() if n>1},complete_context_tokens=context,response_EOS_targets=targets)
    return results


def reconcile(root,reports,output,streams):
    root,reports,output=map(Path,(root,reports,output));plan=json.loads((reports/'plan.json').read_text())
    initial=torch.load(output/'initial/model.pt',map_location='cpu',weights_only=True,mmap=True)
    initial_parts=initial['aligned_model']['extensions'];training={};capability={}
    expected_budget=budget(streams,plan['training_options'])
    if expected_budget!=plan['source_exposure_budget']:raise ValueError('Every complete mixed-source exposure budget must reconstruct')
    for arm in plan['arms']:
        receipt=json.loads((reports/(arm+'-training-complete.json')).read_text())
        checkpoint=output/arm/'model.pt';resume=output/arm/'resume.pt'
        if file_hash(checkpoint)!=receipt['checkpoint_sha256'] or file_hash(resume)!=receipt['resume_sha256']:
            raise ValueError('Exact completed source worker export and optimizer snapshot bytes required')
        exported=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        state=torch.load(resume,map_location='cpu',weights_only=True,mmap=True);core=state['core'];parts=exported['aligned_model']['extensions']
        names=[n for n in initial_parts if n.startswith(('language_linears.','language_norms.'))]
        options={**plan['training_options'],'reference_KL_alpha':plan['arms'][arm]['reference_KL_alpha']}
        if (exported['run_plan_sha256']!=file_hash(reports/'plan.json') or exported['arm']!=arm
                or core['steps']!=options['steps'] or len(core['metrics'])!=options['steps'] or core['optimized_names']!=names
                or state['options']!=options or set(parts)!=set(initial_parts)
                or any(not torch.equal(t,core['model']['extensions'][n]) for n,t in parts.items())
                or any(not torch.equal(t,initial_parts[n]) for n,t in parts.items() if n not in names)):
            raise ValueError('Every completed mixed-trained tensor and unchanged unused extension must reconcile')
        _validate_adam(core['optimizer'],[parts[n] for n in names],options['steps'])
        actual_totals={name:dict(context=0,targets=0) for name in streams}
        for step,metric in enumerate(core['metrics']):
            selected=positions(streams,options,step)
            if (metric['step']!=step+1 or metric['source_positions']!=selected or metric['learning_rate']!=options['learning_rate']
                    or metric['reference_KL_alpha']!=options['reference_KL_alpha']):
                raise ValueError('Every original mixed source draw and fixed objective profile must reconstruct')
            for name,chosen in selected.items():
                context,targets=source_counts(streams,name,chosen);record=metric['streams'][name]
                if (record['complete_context_tokens']!=context or record['response_EOS_targets']!=targets
                        or any(not math.isfinite(record[k]) for k in ('token_mean_CE','token_mean_reference_KL'))):
                    raise ValueError('Every source-complete target/context normalizer must reconstruct')
                actual_totals[name]['context']+=context;actual_totals[name]['targets']+=targets
        if any(actual_totals[name]!=dict(context=value['complete_context_tokens'],targets=value['response_EOS_targets']) for name,value in expected_budget.items()):
            raise ValueError('All original per-stream full source token exposures must reconcile')
        cold=json.loads((reports/arm/'trained-shared-reload.json').read_text())
        if not cold['full_trained_aligned_and_original_3routes_exact'] or not receipt['all_fixed_bytes_versions_and_original_native_routes_exact']:
            raise ValueError('Actual full trained reload and preserved original routes required')
        training[arm]=dict(actual_updates=core['steps'],actual_Adam_states=len(core['optimizer']['state']),source_exposure_budget=expected_budget,
            every_export_tensor_equal_complete_resume=True,every_unused_extension_equal_initial=True,
            held_human=json.loads((reports/arm/'human-validation.json').read_text())['overall'],checkpoint_sha256=receipt['checkpoint_sha256'])
    from .code_quality import execute
    questions={q['id']:q for q in map(json.loads,(root/'data/public-code-assessment-v1/questions.jsonl').open())}
    for arm in ['initial',*plan['arms']]:
        destination=reports/arm;knowledge=json.loads((destination/'knowledge.json').read_text())
        if len(knowledge['predictions'])!=559:raise ValueError('Every original native knowledge question required')
        public=[r for i in range(4) for r in json.loads((destination/f'public-{i}.json').read_text())['results']]
        if len(public)!=175 or {r['id'] for r in public}!=set(questions):raise ValueError('Every original supported public code problem required exactly once')
        passing=[]
        for row in public:
            if row['execution'].get('functional_passed'):
                source=questions[row['id']];actual=execute(dict(language='Python',code=row['code'],function=source['function'],cases=source['cases'],stress=[]))
                if not actual.get('functional_passed'):raise ValueError('Every claimed public pass must independently reexecute its complete original cases')
                passing.append(row['id'])
        local={}
        for mode in ('completion','instruction'):
            rows=json.loads((destination/('code-'+mode+'.json')).read_text())['results']
            if len(rows)!=24:raise ValueError('Every original local Python/JavaScript task required')
            local[mode]=dict(syntax=sum(r['candidate'].get('syntax_valid',False) for r in rows),functional=sum(r['candidate'].get('functional_passed',False) for r in rows),efficiency=sum(r['efficiency_stress_passed'] for r in rows))
        cpp=json.loads((destination/'C++.json').read_text());math_result=json.loads((destination/'math-complete.json').read_text())
        if math_result['evaluated_examples']!=1319 or not math_result['all_original_IDs_prompts_generated_tokens_numeric_answers_and_scores_reconciled']:
            raise ValueError('Complete original math prompts/raw tokens/answers and numeric scores required')
        capability[arm]=dict(knowledge=knowledge['overall'],local_code=local,public_code=dict(tasks=175,syntax=sum(r['execution'].get('syntax_valid',False) for r in public),functional=len(passing),independently_reexecuted_passing_ids=passing),
            CPP={k:v for k,v in cpp.items() if k not in ('results','evaluation_provenance')},
            math={k:v for k,v in math_result.items() if k not in ('results','evaluation_provenance','shard_reports_sha256')},
            OCR=json.loads((destination/'ocr.json').read_text())['overall'],objects=json.loads((destination/'objects.json').read_text())['overall'],
            vision={k:v for k,v in json.loads((destination/'vision.json').read_text()).items() if k in ('images','proper_nll','blank_nll_gain','mismatch_nll_gain','caption_retrieval')})
    report=dict(completed_full_human_and_verified_program_training=training,matched_initial_and_both_CE_KL_arms=capability,
        exact_reference_KL_uses_only_already_counted_own_routes=True,no_external_teacher=True,
        SFT_and_reference_regularization_not_genuine_ranked_RLHF=True,formal_algorithmic_complexity_not_proven=True,
        public_development_and_semantic_source_overlap_qualifications_retained=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(reports/'independent-comparison.json',report);return report
