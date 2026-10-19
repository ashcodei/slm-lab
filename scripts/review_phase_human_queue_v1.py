"""Full goal review with actual queued human training and unresolved live wait."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT/'reports/phase-human-queued-v1-review.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU review must hide every GPU')
    if DESTINATION.exists():
        raise ValueError('Preserve every previous goal review')
    import assess_joint_java_comparison_v1 as resources
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    def read(name):
        return json.loads((ROOT/name).read_text())
    run = 'reports/verified-views-execution-v1/'
    current = read(run+'plan.json')
    phase = read('reports/phase-joint-v1-queue.json')
    human = read('reports/phase-human-preference-v1-queue.json')
    for actor in [current['owner'],phase['owner'],human['owner']]:
        if not resources.same(actor):
            raise ValueError('Every stated current and queued process must be live now')
    live = [p for p in read(run+'children.json') if resources.same(p) and p['subject'].startswith('RL-rank-')]
    if len(live) != 4:
        raise ValueError('All four current distributed workers must be live')
    if (Path('/proc')/str(human['owner']['pid'])/'task'/str(human['owner']['pid'])/'children').read_text().strip():
        raise ValueError('Waiting human queue must remain childless')
    frozen = read('reports/execution-step128-readback-v1/complete.json')
    host = read('reports/execution-group-host-four-CPU-v6/complete.json')
    FileBindings(frozen['source_files_sha256']).check(full=True)
    for proof in host['proofs']:
        FileBindings(proof['source_files_sha256']).check(full=True)
    if not host['exact_baseline_scalar_trajectories_weights_and_Adam']:
        raise ValueError('Actual host-transport baseline comparison must pass')
    progress = {a:read(run+a+'-progress.json') for a in ['verified_solution','execution_hybrid']}
    previous = read('reports/bounded-probes-and-human-parallel-v1-review.json')
    requirements = previous['requirements']
    additions = {
        'own_architecture':'A new phase-trained-base policy loader rejects incomplete/older foundation cursors, keeps the actual shared own reference and unused critic, and passes actual small trained-foundation/DPO text/image cold-load and tamper tests. This is not a new final architecture or quality result.',
        'public_data':'The upcoming preference comparison enumerates all85,786original admitted public human-ranked training pairs exactly once per arm:5,362global width16updates, with a final10pair batch split3/3/2/2. Every original cohort position is enumerated by an actual source-loader test. No new external teacher or raw corpus was added this turn.',
        'human_feedback':'A complete matched chosen-CE/human-DPO controller is now genuinely queued behind the exact complete broader foundation. Each arm uses all four GPUs in sequence. New full configured four-L40 next-update/private-RNG recovery and longest chosen/rejected whole-response/target backward controls must pass before either production arm. Both arms retain the same counted own foundation as reference; all4,642held human-ranked other-model response pairs are scheduled for sharded evaluation at initial/control/DPO endpoints. No full-size human production update has occurred yet, and no fresh own-output human judgment exists.',
        'serious_full_model_training':'All four actual128-update saved states now reconcile against their original journals, all561Adam clocks/moments and all private CPU RNGs; the561actor tensors differ between objectives while frozen base and unused critic remain exact. This verifies retained progress, not128neural-backward replay or held improvement. The actual current trial has an unresolved live communication/runtime wait: control160andhybrid138at the latest observation. No process was restarted or bound source changed.',
        'high_end_llm_capability':'No new native post-RL result exists. Latest completed coding correctness remains0/24in both512-token frames after warm-up. Human scheduling, source enumerations, state integrity and transport mechanics do not establish any high-end capability.',
        'broad_independent_benchmarks':'Held preference sharding now reproduces serial neural scores, source/target fingerprints and length-bias controls, and rejects duplicated/mismatched policy or altered score shards. Full4,642pair assessment plus unchanged24case512-token code in both frames and broad native development comparisons are scheduled for three human-policy endpoints. These do not replace the still-required complete independent broad/code/multimodal and held word final evaluation.',
        'gpu_runtime':'Three live PID/start-tick/resource snapshots show no child code-test worker, a busy GPU2with zero reported memory activity and idle peer GPUs while counters stop. This suggests a runtime/communication wait but proves neither a terminal failure nor its exact cause. Python-stack sampling is denied by host tracing permissions, including the sudo attempt; no tracing setting was changed. A new CPU/Gloo-gradient fallback reproduces actual four-CPU original trajectories, weights and Adam exactly; it is not activated and still needs actual full-horizon GPU capacity. Current, broader and human owners are all confirmed live; both queued owners remain GPU-hidden and childless.'}
    for key, addition in additions.items():
        requirements[key]['current_evidence_and_limits'] += ' '+addition
    names = ['reports/execution-step128-readback-v1/complete.json',
        'reports/execution-group-host-four-CPU-v6/complete.json',
        'reports/execution-RL-runtime-wait-v1.json',
        'reports/phase-human-policy-v1-tests-v2.log',
        'reports/held-human-shards-v1-tests.log',
        'reports/phase-human-preference-v1-driver-tests-v3.log',
        'reports/execution-recovery-semantics-v1-tests.log',
        'src/slm_lab/phase_human_policy_v1.py','src/slm_lab/held_human_shards_v1.py',
        'scripts/train_phase_human_preference_v1.py',str(Path(__file__).relative_to(ROOT))]
    at = datetime.now(timezone.utc).isoformat()
    review = dict(at=at, objective_preserved=True, previous_goal_turn_classification='progress',
        current_goal_turn_classification='progress',
        current_goal_turn_progress=[
            'New complete-foundation human policy loading, actual small trained DPO cold-load/tamper tests and held preference sharding implemented and verified.',
            'All85,786original human cohort source draws and width16tails verified; matched full-epoch four-GPU human comparison is queued behind the actual broader owner.',
            'Actual128-step all-four policy/Adam/private CPU-RNG state reconciled; every actor tensor differs between the actual objectives.',
            'Runtime wait investigated using exact live process/thread/child/GPU observations; no terminal failure inferred or restart performed.',
            'New host gradient transport reproduces all-four original CPU trajectories/weights/Adam; actual GPU use remains pending.'],
        requirements=requirements, latest_completed_native_outcomes=previous['latest_completed_native_outcomes'],
        actual_current_RL_progress=progress, observed_runtime_wait_requires_followup=True,
        verified_terminal_failure=False, no_current_trial_process_signaled_or_restarted=True,
        broader_curriculum=dict(owner=phase['owner'],planned_targets=467581587,planned_updates=35284,production_updates=0),
        queued_human_comparison=dict(owner=human['owner'],actual_childless_GPU_hidden_waiter=True,
            original_train_pairs=85786,original_held_pairs=4642,arms=['chosen_CE','human_DPO'],
            planned_updates_per_arm=5362,global_original_pairs_per_update=16,final_batch_original_pairs=10,
            actual_full_profile_GPU_capacity_pending=True,production_updates=0,
            public_human_ranked_other_model_responses_not_fresh_own_output_judgments=True),
        actual_last_common_checkpoint_updates_per_arm=128, fallback_CPU_baseline_exact_GPU_not_tested=True,
        all_ten_final_requirements_incomplete=True,no_remaining_scope_removed=True,
        candidate_promoted=False,held_coding_gain_proven=False,goal_achieved=False,
        source_files_sha256={str(ROOT/p):file_hash(ROOT/p) for p in names})
    for name in ['README.md','RESEARCH.md','reports/completion-audit.json']:
        if any(str(ROOT/name) in plan['source_files_sha256'] for plan in [current,phase,human]):
            raise ValueError('Never edit bound current or queued source files')
    atomic_json(DESTINATION,review)
    path = ROOT/'reports/completion-audit.json'
    backup = ROOT/'reports/completion-audit-before-phase-human-v1.json'
    if backup.exists():
        raise ValueError('Preserve previous full-scope audit')
    shutil.copyfile(path,backup)
    audit = json.loads(path.read_text())
    audit.update(updated_at=at,latest_full_scope_review=str(DESTINATION.relative_to(ROOT)),
        requirements={k:v['current_evidence_and_limits'] for k,v in requirements.items()},
        queued_human_preference_comparison=review['queued_human_comparison'],
        current_posttraining=dict(actual_control_updates=progress['verified_solution']['actual_updates'],
            actual_execution_updates=progress['execution_hybrid']['actual_updates'],warmup_updates=88,
            phase='live_runtime_wait_under_investigation',last_common_verified_checkpoint=128,
            verified_terminal_failure=False,held_coding_gain_proven=False),
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(path,audit)
    message = f'''Complete human-preference training queued, **2026-10-17**: a new matched **chosen-CE / human-DPO** comparison is now waiting behind the complete broader foundation, with **85,786original human-ranked training pairs per arm**, **5,362updates each**, and **all four L40s used by each arm in sequence**. The native loader test enumerates every original pair once; the last global batch contains10pairs split3/3/2/2. New complete-foundation policy loading rejects partial/older base cursors, and actual small trained-foundation/DPO text/image cold-load tests pass. Fresh full configured four-L40 capacity/recovery and longest whole-response backward controls still gate production. **No full-size human-preference production update has occurred yet.**

All **4,642held public human-ranking pairs** are scheduled across all4GPUs at initial/control/DPO endpoints. Sharded neural results match serial scores and preserve source/mask fingerprints, ties and length-prior controls; missing/changed shards reject. These are human preferences over other models' public responses, not new human judgments of our generated answers. Unchanged512-token code in both frames and broad native comparisons remain scheduled; full final broad/code/image/word evaluations remain required.

The preceding code experiment has an **unresolved live runtime wait**: current observed counts are **{progress['verified_solution']['actual_updates']}/173control** and **{progress['execution_hybrid']['actual_updates']}/173execution RL**. Exact owner and all4worker PID/start ticks remain live; no child execution worker is running. Repeated samples show GPU2busy with no reported memory activity while peer GPUs wait, suggesting communication/runtime trouble without proving the exact cause or a terminal failure. Python stack reads are denied by host tracing permissions; no tracing setting was changed and no current process was restarted. The last common **128-update** four-worker state is independently reconciled, including561Adam clocks per replica and all private CPU RNG states. Both objectives'561actor tensors differ, with identical frozen base and unused critic. This is retained-state integrity, not neural-backward replay or quality.

A separate **CPU/Gloo gradient transport** fallback reproduces the original four-CPU scalar trajectories, weights and Adam exactly. It remains unactivated and requires actual full-horizon GPU admission. The broader **467,581,587-target** foundation/instruction/vision curriculum remains queued. Latest completed native code correctness remains **0/24** after warm-up; post-RL quality is pending. **All ten final-model requirements remain incomplete; no candidate is promoted.** See the [full-scope review](reports/phase-human-queued-v1-review.json), [human queue](reports/phase-human-preference-v1-queue.json), [verified128state](reports/execution-step128-readback-v1/complete.json), [runtime observations](reports/execution-RL-runtime-wait-v1.json) and [CPU transport comparison](reports/execution-group-host-four-CPU-v6/complete.json). Earlier entries are historical.

'''
    for name in ['README.md','RESEARCH.md']:
        path = ROOT/name
        first,_,rest = path.read_text().partition('\n\n')
        path.write_text(first+'\n\n'+message+rest)
    print(json.dumps(dict(review=str(DESTINATION.relative_to(ROOT)),
        actual_control_updates=progress['verified_solution']['actual_updates'],
        actual_execution_updates=progress['execution_hybrid']['actual_updates'],
        human_planned_updates_per_arm=5362,all_ten_final_requirements_incomplete=True)),flush=True)


if __name__ == '__main__':
    main()
