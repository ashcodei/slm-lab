"""Record complete candidate-data findings while retaining every final goal."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-code-instruction-screen-v1-review'
READBACK=ROOT/'reports/public-code-instruction-readback-CPU-v1/complete.json'
QUAL=ROOT/'reports/public-code-instruction-qualification-CPU-v2/complete.json'
VETO=ROOT/'reports/public-code-instruction-precision-counterexample-CPU-v1/complete.json'


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    pending=path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    pending.replace(path)


def live(owner):
    fields=(Path('/proc')/str(owner['pid'])/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=owner['ticks'] or fields[0] in {'Z','X'}:raise ValueError('The exact original live birth is required')
    return dict(owner=owner,state=fields[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh CPU-only review required')
    qualified=read(QUAL);replay=read(READBACK);veto=read(VETO)
    if (not all(r.get('complete') for r in [qualified,replay,veto])
            or any(r['training_admitted'] for r in [qualified,replay,veto])
            or replay['original_entries_and_full_native_masks_rebuilt']!=1024
            or replay['complete_protected_input_stems_rechecked']!=31349
            or replay['actual_counts']!=qualified['actual_counts']
            or replay['fresh_complete_candidate_program_runs']!=qualified['actual_counts']['screen_pass_candidate']
            or not veto['independent_semantic_veto_for_this_source']):
        raise ValueError('Complete actual source execution/readback and mathematical veto evidence required')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten original successors must remain admitted')
    plan_paths=[ROOT/'reports/public-pretraining-recovery-v1/plan.json',
        ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json',
        ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json',
        ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json',
        ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json',ROOT/QUEUE,
        *[ROOT/item['queue'] for item in handoff['new_controllers']]]
    plans=[read(p) for p in plan_paths]
    maps=[p['source_files_sha256'] for p in plans]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in m for m in maps):raise ValueError('An active immutable index or document must never be edited')
    files={**qualified['source_files_sha256'],**replay['source_files_sha256'],**veto['source_files_sha256']}
    for p in [Path(__file__),previous,READBACK,QUAL,VETO,*plan_paths]:files[str(p)]=sha(p)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every exact source/artifact must remain unchanged')
    owners=[live(plans[i]['owner']) for i in [0,1,3,4,5]]
    assessment=live(plans[2]['owner'])
    if any(r['state'] in {'T','t'} for r in owners[:2]+[assessment]):raise ValueError('No neural worker may be stopped')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=[json.loads(line) for line in prefix.split(b'\n') if line]
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Complete actual contiguous update prefix required')
    current=dict(owner=plans[0]['owner'],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(r['response_EOS_targets']) for r in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    at=datetime.now(timezone.utc).isoformat();counts=qualified['actual_counts'];OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: bounded polling continuation with measured CPU allocation; no accuracy gain',
        current_goal_turn_classification='progress: pinned synthetic coding candidate collection, whole-test execution/readback, exact legal mathematical counterexample; data remains unadmitted',
        actual_counts=counts,actual_all_source_rows_rebuilt=1024,actual_protected_input_stems_rechecked=31349,
        actual_fresh_surviving_candidate_program_runs=replay['fresh_complete_candidate_program_runs'],
        screen_pass_response_EOS_targets=qualified['screen_pass_response_EOS_targets'],
        synthetic_response_and_test_generator='Qwen2.5-Coder-32B-Instruct',
        student_weights_remain_own_scratch_origin=True,any_external_model_weights_or_live_teacher_API_used=False,
        actual_independent_semantic_veto=str(VETO),known_wrong_screen_pass_source=veto['source_id'],
        actual_execution_screen_tests_passed=5,
        original_disk_reserve_failure_preserved_and_missing8rows_completed=True,
        teacher_programs_and_tests_not_independent_semantic_oracles=True,
        first1024generic_examples_only_not_whole_dataset_or_any_language_coverage=True,
        all_new_candidates_remain_unadmitted=True,new_neural_updates_from_these_candidates=0,
        actual_original_live_owners=owners,actual_original_CPU_assessment=assessment,actual_original_main=current,
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        latest_complete_local_functional_correctness='0/48',new_model_quality_gain_established=False,
        candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    text=(f'Public coding candidate screen complete, {at}. All 1,024 fixed original examples reconstruct against the pinned public shard, '
        f'the own tokenizer and all 31,349 declared protected input stems. {counts["all_published_tests_pass"]} programs pass every published test; '
        f'{counts["published_test_mismatch"]} return observed mismatches; {counts["runtime_resource_or_confinement_abstention"]} execution/resource abstentions '
        f'and {counts["unsupported_whole_example_abstention"]} unsupported whole test sets remain excluded from that count. '
        f'Fresh readback re-executes all {replay["fresh_complete_candidate_program_runs"]} passing candidates. '
        'These are synthetic Qwen-produced answers/tests. No external weights or live teacher API were used, and the student remains scratch-origin.\n\n'
        'One program passes all its published tests yet fails an exact, legal closest-pair input: points (-1000000000,0), (0,1), (1000000000,1). '
        'The first and last neighboring squared distances differ by one, but their float square roots compare equal. An explicit whole-source semantic veto is retained. '
        'All new candidates remain unadmitted. These are data-quality results, not a measured neural capability gain.\n\n'
        'The first producer stopped at its original 40 GiB disk reserve during the main atomic checkpoint save. Its 1,016 complete observations and failure remain unchanged. '
        'After the checkpoint save restored free space, a new version rebuilt that complete prefix and finished the missing eight rows; no neural worker was restarted.\n\n'
        f'The main continues at {len(rows):,}/4,447 updates and {current["response_EOS_targets"]:,} response/EOS targets. '
        'Latest complete local functional correctness remains 0/48; the newer matched assessment is pending. '
        'All original training, human feedback, RL, and nine endpoint scopes remain. All ten final requirements remain incomplete.\n')
    (OUT/'metrics.md').write_text(text)
    if len(review['requirements'])!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Candidate-data research establishes none of the requested final capabilities')
    review['current_actual_runtime']['main']=current
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_public_code_instruction_screen=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every bound artifact must remain unchanged before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),
        latest_actual_main_runtime=current,latest_actual_public_code_instruction_screen=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Public synthetic code screen, **{at}**: all1,024 original rows and native framing rebuild; all31,349 declared protected stems are checked. '
        f'**{counts["all_published_tests_pass"]}** programs pass every published test and fresh candidate re-execution. '
        '**One of those still fails an independent exact-integer legal counterexample**, so its whole source is explicitly vetoed and **all new data remains unadmitted**. '
        'Answers/tests are Qwen-generated public data; our student remains scratch-origin and no external weights or live teacher API were used. '
        f'The main continues at **{len(rows)}/4,447**. **Latest complete model correctness remains0/48; all ten final requirements remain incomplete.** '
        '[Findings and limits](reports/public-code-instruction-screen-v1-review/metrics.md), [full review](reports/public-code-instruction-screen-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=True,review=str(OUT/'review.json'),actual_main_updates=len(rows),training_admitted=False,goal_achieved=False)))


if __name__=='__main__':main()
