"""Read back every diagnostic and record concrete spec failures, without training."""
from collections import Counter,defaultdict
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from record_public_code_instruction_screen_v1 import read,sha,atomic,live
from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted
from slm_lab.ordinary_python_function_cases_v1 import execute

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/initial-local-code-prefix-diagnosis-CPU-v1'
OUT=ROOT/'reports/local-code-prefix-diagnosis-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU readback required')
    diagnostic=read(SOURCE/'complete.json')
    with (SOURCE/'all24-program-diagnoses.jsonl').open() as stream:rows=list(map(json.loads,stream))
    original=read(ROOT/'reports/public-pretraining-interim-CPU-v2/initial/code-completion.json')
    if (not diagnostic.get('complete') or len(rows)!=24 or diagnostic['neural_forwards']!=0
            or not diagnostic['optimization_forbidden'] or diagnostic['source_checkpoint_sha256']!=original['checkpoint_sha256']):
        raise ValueError('Complete original24programs and explicit post-hoc optimization prohibition required')
    totals=Counter();by_language=defaultdict(Counter)
    for position,(row,base) in enumerate(zip(rows,original['results'])):
        if row['position']!=position or row['language']!=base['language'] or row['task']!=base['task'] or not row['optimization_forbidden']:
            raise ValueError('Every original identity/order and held diagnostic role must remain')
        code=base['code'];seen=set()
        for name,result in row['variants'].items():
            if (not code.startswith(result['exact_code'])
                    or hashlib.sha256(result['exact_code'].encode()).hexdigest()!=result['exact_code_sha256']
                    or len(result['exact_code'])!=result['original_program_prefix_characters']):
                raise ValueError('Every derived program must remain an exact original contiguous prefix')
            if name=='original_native_program' and result['exact_code']!=code:raise ValueError('The actual original generated program must be unchanged')
            totals[name+':native_syntax']+=bool(result['native_finite_cases'].get('syntax_valid'))
            totals[name+':native_functional']+=bool(result['native_finite_cases'].get('functional_passed'))
            by_language[row['language']][name+':syntax']+=bool(result['native_finite_cases'].get('syntax_valid'))
            by_language[row['language']][name+':functional']+=bool(result['native_finite_cases'].get('functional_passed'))
            if result['ordinary_Python'] is not None:
                totals[name+':ordinary_Python_functional']+=result['ordinary_Python']['functional_passed']
            if result['exact_code_sha256'] not in seen:
                totals['actual_distinct_native_program_runs']+=1
                totals['actual_distinct_ordinary_Python_runs']+=result['ordinary_Python'] is not None
                seen.add(result['exact_code_sha256'])
        totals['original_programs']+=1
    if dict(totals)!=diagnostic['actual_counts']:raise ValueError('Every actual diagnostic aggregate must reconstruct')
    refs=diagnostic['all24_original_reference_controls']
    if (len(refs)!=24 or not all(r['native'].get('functional_passed') for r in refs)
            or sum('ordinary_Python' in r for r in refs)!=12
            or not all(r['ordinary_Python']['functional_passed'] for r in refs if 'ordinary_Python' in r)):
        raise ValueError('Every24native and12ordinary reference controls must actually pass')
    files=dict(diagnostic['source_files_sha256'])
    for p in [Path(__file__),SOURCE/'complete.json',SOURCE/'all24-program-diagnoses.jsonl']:files[str(p)]=sha(p)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every original/derived artifact must remain exact')
    examples=[];lookup={r['task']:r for r in rows if r['language']=='Python'}
    for task,args,expected in [('sum_even',[[2,4]],6),('count_runs',[[1,1,2]],2),
            ('prefix_sums',[[1,2,3]],[1,3,6]),('longest_unique',['abcabcbb'],3)]:
        program=lookup[task]['variants']['first_target_function_prefix']['exact_code']
        actual=execute(program,task,[dict(args=args,expected=expected)])
        if not actual['all_cases_observed'] or actual['functional_passed']:raise ValueError('The proposed public-spec counterexample was not actually observed; do not fabricate a failure')
        examples.append(dict(task=task,explicit_public_spec_args=args,expected=expected,
            actual_returned_value=actual['raw_observations']['results'][0]['value'],
            unchanged_exact_generated_function_prefix=program,actual_OS_confined_observation=actual,
            posthoc_only_optimization_forbidden=True))
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten original successors must remain admitted')
    paths=[ROOT/'reports/public-pretraining-recovery-v1/plan.json',ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json',
        ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json',ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json',
        ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json',ROOT/QUEUE,
        ROOT/'reports/full-public-code-shard-qualification-CPU-v1/plan.json',*[ROOT/item['queue'] for item in handoff['new_controllers']]]
    plans=[read(p) for p in paths];maps=[p['source_files_sha256'] for p in plans]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in m for m in maps):raise ValueError('Never edit a live source-bound index/document')
    files[str(previous)]=sha(previous)
    for p in paths:files[str(p)]=sha(p)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every fixed source must stay exact')
    owners=[live(p['owner']) for p in plans[:7]]
    if any(r['state'] in {'T','t'} for r in owners[:3]):raise ValueError('No neural worker may be stopped')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    metrics=[json.loads(x) for x in prefix.split(b'\n') if x]
    if [r['step'] for r in metrics]!=list(range(1,len(metrics)+1)):raise ValueError('Every complete actual main update required')
    current=dict(owner=plans[0]['owner'],complete_actual_updates=len(metrics),planned_updates=4447,
        response_EOS_targets=sum(int(r['response_EOS_targets']) for r in metrics))
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: full100000row public shard source qualification active; actual CPU cache control excludes repeated casting on the tested path',
        current_goal_turn_classification='progress: all24older initial code programs receive syntax-only prefix/ordinary-runtime diagnosis; four actual wrong calculations survive cropping; no model gain',
        source_checkpoint_sha256=diagnostic['source_checkpoint_sha256'],endpoint='initial_before_current_public_pretraining',
        source_generated_programs=24,actual_distinct_native_variant_program_runs=49,actual_distinct_ordinary_Python_variant_runs=29,
        all24_native_and12_ordinary_reference_controls_passed=True,actual_syntax_selector_controls_passed=2,
        actual_native_original=dict(syntax_valid=12,functional_correct=0,total=24),
        actual_posthoc_first_function_prefix=dict(syntax_valid=20,functional_correct=0,total=24),
        actual_posthoc_longest_checked_syntax_prefix=dict(syntax_valid=20,functional_correct=0,total=24),
        actual_by_language={k:dict(v) for k,v in by_language.items()},
        actual_ordinary_Python_original_and_both_prefix_policies=dict(original=0,first_function_prefix=0,longest_checked_syntax_prefix=0,total=12),
        actual_four_public_spec_counterexamples=examples,source_and_generated_originals_unchanged=True,
        prefix_selection_uses_syntax_only_not_functional_outcomes=True,
        interpretation='These two checked syntax-only stopping changes are insufficient on this older endpoint; the concrete failures establish calculation errors in clean generated functions.',
        all_stopping_strategies_or_latest_checkpoint_abilities_not_determined=True,
        diagnosis_is_not_native_pass_at1_or_model_improvement=True,stress_efficiency_not_assessed=True,
        every_diagnostic_source_and_derived_record_forbidden_for_optimization=True,
        neural_forwards=0,new_production_optimizer_updates_from_this_work=0,original_native_scores_unmodified=True,
        actual_original_main=current,actual_original_live_owners=owners,
        actual_current_new_code_data_progress=read(ROOT/'reports/full-public-code-shard-qualification-CPU-v1/progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        latest_complete_local_functional_correctness='0/48_on_older_checkpoint',newer_matched_quality_pending=True,
        candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    lines=[f'Older initial-checkpoint code diagnosis, {at}. Every original24completed CPU BF16 completion program is retained. '
        'At most64syntax-only cuts select up to3exact contiguous-prefix variants per program. No character is inserted or replaced and no functional-case outcome chooses a variant. '
        'All original finite cases remain; empirical stress efficiency is outside this diagnosis.\n',
        '| Program view | Syntax valid | Functional correct |','|---|---:|---:|',
        '| Original native program |12/24|0/24|','| First target-function prefix |20/24|0/24|',
        '| Longest checked syntax prefix |20/24|0/24|','',
        'There are49distinct native and29ordinary-Python variant runs. Every24native reference and12ordinary Python references pass. '
        'All12generated Python programs remain0/12functional under ordinary stdlib execution for the original and both prefix policies. '
        'That rules out these specific stopping/cropping policies and the constrained Python namespace as sufficient fixes for the observed zero score. '
        'It does not exhaust every possible stopping policy or establish the abilities of the newer model.\n',
        '| Public specification example | Expected | Actual clean generated function |','|---|---|---|']
    for item in examples:lines.append('| '+item['task']+'('+repr(item['explicit_public_spec_args'])[1:-1]+') | '+repr(item['expected'])+' | '+repr(item['actual_returned_value'])+' |')
    lines += ['', 'These are four fresh actual confined evaluations of already generated function prefixes. They establish calculation errors beyond trailing text. '
        'The diagnosis supports prioritizing broader task-conditioned, execution-screened code learning over EOS-only changes. '
        'The public100,000row candidate audit and full-source repair/preference/reasoning experiments continue; none of these held diagnostic programs, reference answers or outcomes may be used for optimization.\n',
        'Prefix scores are post-hoc diagnostic outcomes, not native model improvements or benchmarks. The original code report/journals, native stopping and all scores remain unchanged. '
        'The observed endpoint is the initial model before the current billion-target run. The fixed newer interim assessment remains incomplete.\n',
        f'The main is live at{len(metrics):,}/4,447updates and{current["response_EOS_targets"]:,}targets. '
        'All original source/human/RL/nine-endpoint scopes remain. Latest full completed local correctness remains0/48 on an older checkpoint. '
        'No frontier parity or current model-quality gain is established. All ten final requirements remain incomplete.','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    if len(review['requirements'])!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Post-hoc failure diagnosis establishes none of the requested final capabilities')
    review['current_actual_runtime']['main']=current
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_local_code_prefix_diagnosis=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every bound byte must stay exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),latest_actual_main_runtime=current,
        latest_actual_local_code_prefix_diagnosis=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Older initial-code failure diagnosis complete, **{at}**: all24completion programs receive syntax-only contiguous-prefix checks. '
        '**Syntax12/24→20/24, functional correctness0/24→0/24**; these are post-hoc crop outcomes, **not model gains**. '
        'All24native/12ordinary reference controls pass, while generated Python stays0/12under ordinary execution. '
        'Four actual public-spec examples show wrong calculations even in clean function prefixes. '
        f'The original main continues at**{len(metrics)}/4,447**; newer checkpoint quality remains pending. '
        '**Latest full completed correctness remains0/48 on an older checkpoint; all ten final requirements remain incomplete.** '
        '[Cases, scope and interpretation](reports/local-code-prefix-diagnosis-v1-review/metrics.md), '
        '[full review](reports/local-code-prefix-diagnosis-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=True,actual_main_updates=len(metrics),actual_prefix_functional_correct=0,
        actual_public_spec_counterexamples=4,original_benchmark_unchanged=True,goal_achieved=False)))


if __name__=='__main__':main()
