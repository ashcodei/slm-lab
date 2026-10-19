"""Record the complete-source code successor and preserve every final requirement."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/broad-anchored-code-queued-v3-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():
        raise ValueError('Fresh CPU-only complete-scope review required')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    previous_path=ROOT/'reports/expanded-execution-and-phase3-v1/review.json'
    old=json.loads(previous_path.read_text())
    queue_path=ROOT/'reports/broad-anchored-code-v3-queue.json'; queue=json.loads(queue_path.read_text())
    cohort_path=ROOT/'reports/broad-anchored-code-v3-held-APPS-cohort.json'; cohort=json.loads(cohort_path.read_text())
    proof_path=ROOT/'reports/broad-anchored-execution-four-CPU-v1/complete.json'; proof=json.loads(proof_path.read_text())
    shape_path=ROOT/'reports/broad-code-capacity-CPU-v1/complete.json'; shape=json.loads(shape_path.read_text())
    if (len(cohort['selected_original_test_ids'])!=64 or not proof['own_reference_preserves_actual_trained_starting_actor']
            or not proof['all_four_next_update_weights_Adam_actions_objectives_and_private_CPU_RNGs_exact']
            or shape['proof']['actual_forced_action_tokens']!=3059 or shape['proof']['optimizer_commits']!=0):
        raise ValueError('Actual fixed held sources, trained own-reference recovery and full-horizon CPU backward required')
    def identity(owner,hidden_childless=False):
        path=Path('/proc')/str(owner['pid']); fields=(path/'stat').read_text().split(') ',1)[1].split()
        if fields[19]!=owner['ticks'] or fields[0] in ['Z','X']: raise ValueError('Exact current live PID/start ticks required')
        children=(path/'task'/str(owner['pid'])/'children').read_text().strip()
        env=(path/'environ').read_bytes().decode().split('\0')
        visible=next((x.split('=',1)[1] for x in env if x.startswith('CUDA_VISIBLE_DEVICES=')),None)
        if hidden_childless and (visible!='' or children): raise ValueError('Queued controller must be CPU-hidden and childless now')
        return dict(pid=owner['pid'],ticks=fields[19],state=fields[0],children=children,
            CUDA_VISIBLE_DEVICES=visible,argv=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode())
    waiting=identity(queue['owner'],True)
    if (ROOT/'reports/broad-anchored-code-v3/plan.json').exists():
        raise ValueError('New code run has already acquired ownership; inspect current plan instead of recording waiting')
    main_plan_path=ROOT/'reports/phase-joint-v3/plan.json'; main_plan=json.loads(main_plan_path.read_text())
    main_owner=identity(main_plan['owner']); human=identity(queue['predecessor'],True)
    workers=[identity(p) for p in json.loads((ROOT/'reports/phase-joint-v3/children.json').read_text()) if p['subject'].startswith('train-rank-')]
    if len(workers)!=4: raise ValueError('All four actual broad-training workers required')
    rows=[]
    for line in (ROOT/'reports/phase-joint-v3/training-metrics.jsonl').read_bytes().splitlines():
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: break
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)): raise ValueError('Contiguous committed production prefix required')
    OUT.mkdir(); at=datetime.now(timezone.utc).isoformat(); targets=int(sum(r['response_EOS_targets'] for r in rows))
    atomic_json(OUT/'actual-main-journal-prefix.json',rows)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True)
    runtime=dict(at=at,actual_main_owner=main_owner,actual_four_main_workers=workers,
        actual_main_updates=len(rows),planned_main_updates=35284,actual_main_targets=targets,planned_main_targets=467581587,
        GPU_observation=gpu,human_waiting_owner=human,new_code_waiting_owner=waiting,
        full_new_human_or_code_production_updates=0,new_full_profile_GPU_admission_pending=True,goal_achieved=False)
    atomic_json(OUT/'runtime.json',runtime)
    requirements=old['requirements']
    requirements['own_architecture']['evidence'] += (' A frozen snapshot of our own starting actor now shares its exact backbone '
        'while copying only its private adapters/norms/critic. This preserves the complete human-DPO starting policy as the code KL reference. '
        'Real small-network trained-reference text/image cold-load, four-CPU recovery, wrong-reference rejection and coordinated failure controls pass.')
    requirements['under_1b_parameters']['evidence'] += (' The queued code experiment exports the existing 999,706,442-value student only. '
        'Its additional frozen own-reference adapter/critic snapshot is training-only; the full-size resident inventory will be measured during fresh GPU admission. '
        'Small-network inventory and student-only cold reload have been checked.')
    requirements['serious_full_model_training']['evidence']=(f'The original full-size broad policy remains active at {len(rows):,}/35,284 '
        f'updates and {targets:,}/467,581,587 targets on all four local L40s in this snapshot. Human-v4 waits for full completion. '
        'The new broad-code-v3 controller is live, CPU-hidden and childless, waiting for all complete human-v4 train epochs, '
        'native assessments and restoration. It schedules all 3,721 sources once in each matched arm, every complete reference target, '
        'all four GPUs per arm, compact complete Adam/private-RNG states plus durable external full trajectory journals, and '
        'two retained atomic checkpoint generations. Fresh full-size four-GPU admission still gates its first production update.')
    requirements['gpu_runtime']['evidence']=('All four exact broad-training workers are verified live. The exact human and new code '
        'waiting owners are GPU-hidden and childless. No predecessor is preempted. New code admission must exercise actual full source '
        'extremes, a 3,059-action cached/full/backward control on all four L40s, exact next-update weights/Adam/all private CPU-CUDA RNG, '
        'and rejection of a bad private CUDA RNG before mutation. No new full-size code GPU capacity or production has occurred yet.')
    requirements['human_feedback']['evidence'] += (' The code successor starts only at the declared complete own human-DPO checkpoint '
        'and then uses an exact frozen snapshot of that policy for KL. It adds zero human labels. All 4,642 held public human pairs '
        'are assessed at initial/control/RL endpoints to observe alignment drift; fresh human judgments of own outputs remain missing.')
    requirements['broad_independent_benchmarks']['evidence'] += (' The code successor additionally preselects 64 original APPS '
        'publisher-test problems by difficulty/interface before candidate generations. Each uses the unchanged native prompt, '
        '1,024-token greedy budget and every original case under the retained local runtime. No test solutions are extracted and '
        'optimization is forbidden. Missing/duplicated/mismatched shard tests pass. It remains a bounded diagnostic and is not an official '
        'APPS score or semantic-independence certificate; full independent final testing is still required.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (' The complete new code experiment is now concretely queued '
        'behind the original work, rather than being only a source/loss prototype. No additional held correctness or efficiency gain '
        'has been measured. Latest scores remain 1/24 instruction and 0/24 completion in both completed old arms.')
    report=dict(at=at,previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        requirements=requirements,actual_runtime=runtime,new_code_queue=queue,
        actual_CPU_trained_reference_recovery=proof,actual_CPU_longest_horizon_control=shape,
        held_APPS_cohort=cohort,latest_native_scores_unchanged=old['native_development_scores_unchanged'],
        new_full_size_code_training_updates=0,new_full_profile_GPU_capacity_passed=False,
        source_audit_large_integer_fix='Version2 parses irrelevant metadata integers as strings; original raw tests and global integer limits remain unchanged. Three regression/source-closure tests pass.',
        new_queue_binds_reachable_Python_and_active_data_owner_inputs=True,
        unstarted_v2_controller_stop_record="reports/broad-anchored-code-v2-preparation-stop-for-runtime-binding-v1.json",
        v1_code_driver_was_never_launched=True,legacy_or_live_sources_not_edited=True,
        all_ten_final_requirements_incomplete=True,no_remaining_scope_removed=True,
        new_held_capability_gain_proven=False,candidate_promoted=False,goal_achieved=False)
    files=[Path(__file__),previous_path,queue_path,cohort_path,proof_path,shape_path,main_plan_path,
        OUT/'runtime.json',OUT/'actual-main-journal-prefix.json',ROOT/'src/slm_lab/local_source_closure_v2.py',
        ROOT/'reports/local-source-closure-v2-tests.log',ROOT/'reports/broad-anchored-code-v3-driver-tests.log',
        ROOT/'reports/broad-anchored-policy-v1-tests.log',ROOT/'reports/frozen-own-actor-reference-v1-tests.log',
        ROOT/'reports/held-apps-native-shards-v1-tests.log',ROOT/'reports/active-execution-input-bindings-v1-tests.log',
        ROOT/'reports/broad-anchored-code-v2-preparation-stop-for-runtime-binding-v1.json']
    report['source_files_sha256']={str(p):file_hash(p) for p in files}
    atomic_json(OUT/'review.json',report)
    audit_path=ROOT/'reports/completion-audit.json'; docs=[ROOT/'README.md',ROOT/'RESEARCH.md',audit_path]
    if any(str(p) in queue['source_files_sha256'] for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),root_documents_source_bound_not_modified=True))); return
    audit=json.loads(audit_path.read_text()); backup=ROOT/'reports/completion-audit-before-broad-anchored-code-queue-v3.json'
    with backup.open('xb') as f: f.write(audit_path.read_bytes())
    audit.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),live_joint_training=runtime,
        queued_broad_anchored_code=dict(owner=waiting,predecessor=human,original_sources=3721,steps_each_arm=3721,
            arms=queue['arms'],held_APPS_problems=64,held_human_pairs=4642,new_GPU_capacity_passed=False,production_updates=0),
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(audit_path,audit)
    paragraph=(f'Complete broad-code successor queued, **{at}**: an exact live CPU-hidden, childless controller now waits '
        'behind the complete foundation and human-v4 epochs/assessments. It will compare **3,721-source supervised code training** '
        'against **execution RL plus the same complete reference supervision**, using all four L40s per arm. The KL reference is '
        'a frozen snapshot of our own complete starting human-DPO actor sharing the same backbone. The exported student remains '
        '**999,706,442 parameters**; the private own-reference snapshot is training-only. **Fresh full-size GPU admission remains pending; '
        'new full-size code updates are zero.**\n\n'
        'The trained-own-reference four-CPU source-epoch/recovery/failure controls pass, as does a **3,059-action / 3,837-context** '
        'cached/full/backward stress on a **1,110,722-parameter small CPU model**, with no stress optimizer commit. These checks '
        'do not prove full-model capacity or quality. The queued experiment preserves both native coding frames, broad knowledge/math/images, '
        'all **4,642 held public human rankings**, and adds **64 fixed original APPS publisher-test problems**, with complete case sets '
        'and a matched 1,024-token budget. This is a bounded local-runtime diagnostic, not an official benchmark score. '
        'Reachable Python imports and active data-owner inputs are bound; a valid huge-integer audit edge case is covered by regression tests.\n\n'
        f'The original four-GPU broad training continues at **{len(rows):,}/35,284 updates** and **{targets:,}/467,581,587 targets** '
        'in this snapshot. Latest completed coding scores remain **1/24 instruction tasks and 0/24 completion** in both old CE/RL arms. '
        '**All ten final requirements remain incomplete; no candidate is promoted.** See the [current full review](reports/broad-anchored-code-queued-v3-review/review.json), '
        '[new code queue](reports/broad-anchored-code-v3-queue.json), and [live broad training](reports/phase-joint-v3/progress.json). '
        'Earlier entries below are historical snapshots.\n\n')
    for path in docs[:2]:
        header,rest=path.read_text().split('\n\n',1); path.write_text(header+'\n\n'+paragraph+rest)
    print(json.dumps(dict(review=str(OUT/'review.json'),main_updates=len(rows),main_targets=targets,
        new_code_waiting_owner=waiting['pid'],new_full_size_code_updates=0,all_ten_requirements_incomplete=True)))


if __name__=='__main__': main()
