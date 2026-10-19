"""Full goal review after verified numerical diagnosis and resumed production."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT/'reports/single-update-recovery-v3-review.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Review must hide every GPU')
    if DESTINATION.exists():
        raise ValueError('Preserve all earlier goal reviews')
    import assess_joint_java_comparison_v1 as resources
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    def read(name):
        return json.loads((ROOT/name).read_text())
    folder = ROOT/'reports/execution-single-update-recovery-v3'
    plan = read('reports/execution-single-update-recovery-v3/plan.json')
    phase = read('reports/phase-joint-v2-queue.json')
    human = read('reports/phase-human-preference-v2-queue.json')
    for actor in [plan['owner'],phase['owner'],human['owner']]:
        if not resources.same(actor):
            raise ValueError('Every stated current and queued owner must be live now')
    live = [p for p in json.loads((folder/'children.json').read_text()) if resources.same(p)]
    if sum(p['subject'].startswith('train-rank-') for p in live) != 4:
        raise ValueError('Every actual new production worker required')
    for queue in [phase,human]:
        pid = queue['owner']['pid']
        if (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().strip():
            raise ValueError('Both waiting queues must remain childless')
    capacity = json.loads((folder/'capacity.json').read_text())
    if not capacity['passed']:
        raise ValueError('Fresh actual configured four-GPU admission must pass')
    calibration = read('reports/code-probability-rounding-v1/admission.json')
    FileBindings(calibration['source_files_sha256']).check(full=True)
    retained = read('reports/execution-diagnostic138-readback-v1/complete.json')
    FileBindings(retained['source_files_sha256']).check(full=True)
    progress = {a:json.loads((folder/(a+'-progress.json')).read_text()) for a in ['verified_solution','execution_hybrid']}
    if min(p['actual_updates'] for p in progress.values()) < 139:
        raise ValueError('Both real production arms must pass the original failed step')
    predecessor_log = ROOT/'reports/verified-views-execution-v1/RL-launcher.log'
    text = predecessor_log.read_text()
    if 'Cached and full actual behavior differ before update' not in text or 'SeqNum=139, OpType=ALLREDUCE' not in text:
        raise ValueError('Preserve both the originating local error and secondary collective timeout')
    previous = read('reports/phase-human-queued-v1-review.json')
    requirements = previous['requirements']
    additions = {
        'own_architecture':'The same own35-layer counted policy now has a declared fresh single-update score-function objective. Its gradient matches the original clipped-ratio first derivative at ratio1 and an independent analytic score-function/KL gradient. It introduces no additional neural capacity. The original archived experiment remains unchanged.',
        'public_data':'No new raw data was collected this turn. Original173function reward-v2 treatment stays unchanged in the resumed comparison; the546previously admitted added machine probes remain future work. The complete467,581,587target/28family pretraining/instruction/vision pass and85,786human pairs per preference arm are genuinely requeued with all prior repetition/overlap/source limits retained.',
        'under_1b_parameters':'Every actual new four-L40 capacity worker counts999,706,442unique policy values, including all own heads and the separate unused critic. The disposable FP64 oracle is a precision copy of the same own weights used only for diagnosis and is discarded. It is not retained in deployment or used as a model teacher. No final qualified unified candidate is promoted.',
        'text_and_images':'The current code continuation freezes all own base/vision/word/head values while training the shared actor, so text-conditioned image answers can still change. There is no new visual quality result. Full new phase training and native/final generic vision, OCR, grounding and held-word evaluations remain required.',
        'human_feedback':'The corrected current method is execution-feedback RL, not human-preference RLHF. The rebuilt human-v2 owner is live and childless behind phase-v2, with0human production updates. It schedules matched chosen-CE/DPO arms, all four GPUs per arm,85,786original human-ranked training pairs each and4,642held pairs for initial/control/DPO. Rankings concern other-model public responses; fresh own-output human judgments remain absent.',
        'serious_full_model_training':'The actual original trial ended with control160/RL138and no final native result. All original data/checkpoints/journals are preserved. From the durable128-state, ten actual disposable updates per arm reconstruct138with exactly reproduced original actions, cached log probabilities, rewards and semantic update records; all four138states/561Adam clocks/private CPU RNGs reconcile. The new objective is adopted explicitly after138. Fresh actual four-L40 admission performs two logical/three physical new updates per replica, exact next actions/rewards/weights/Adam/all private CPU/CUDA RNG recovery and bad-CUDA-state rejection, then longest1438prompt/fixed512horizon actual backward without commit. Those controls pass before new production. They are not the full broad source epoch.',
        'high_end_llm_capability':'Still not achieved. No new post-RL native coding/knowledge/math/image result exists. The latest completed warm-up code result remains0/24functional and0/24efficient in both512token frames. Numerical equivalence and repaired runtime are not capability gains or high-end parity.',
        'broad_independent_benchmarks':'The resumed comparison schedules unchanged24case native code in both512token frames sharded across all four GPUs, plus the same broad development controls after173. A fresh single-update CPU fault test coordinates an injected one-replica error before reduction/commit. These are mechanics/development evidence; all full559knowledge,24codebothframes,12C++,175publiccode,1319math,170OCR,164objects,128retrieval,3429held-word and genuinely independent final evaluations remain required.',
        'gpu_runtime':'The original rank3cached/full probability guard failed first; its rank2peer then timed out in NCCL all-reduce139. After authoritative terminal cleanup, a transport-only disposable attempt was explicitly stopped before production because it did not address that originating guard. The four-GPU diagnostic reproduces the failure: maxFP32cached/full action-logp difference0.0001556873versus FP64same-weight equations1.0436e-13, with exact original cached replay. Eight own trajectories/610actions and raw logits/vectors are read back. The admitted runtime bound is0.000623(logp), about0.06232percentrelative action probability, with future violations rejected. It is empirical, not a future accuracy/error guarantee. The new stage coordinates local errors before reduction and uses CPU/Gloo gradient SUM. Fresh actual full-size capacity passes on all four original local L40s; production is genuinely live beyond139. No external GPUs or teacher weights were acquired.',
        'any_code_correctness_and_efficiency':'Still unachieved. The original finite published test weakness and the sole public-full-pass program rejected by added probes remain preserved. New numerical calibration does not alter publisher tests or mark wrong programs correct. Complete173source comparison, posttraining native correctness/efficiency and broad independent multilingual/general-code tests are still pending.'}
    for key,addition in additions.items():
        requirements[key]['current_evidence_and_limits'] += ' '+addition
    names = ['reports/execution-single-update-recovery-v3/capacity.json',
        'reports/execution-probability-diagnostic-v7/complete.json',
        'reports/execution-diagnostic138-readback-v1/complete.json',
        'reports/code-probability-rounding-v1/admission.json',
        'reports/single-update-four-CPU-v8/complete.json',
        'reports/single-update-failure-four-CPU-v8/complete.json',
        'reports/single-update-code-policy-v8-tests.log',
        'reports/execution-single-update-recovery-v3-driver-tests.log',
        'reports/phase-joint-driver-v2-tests.log','reports/phase-human-preference-v2-driver-tests.log',
        'reports/verified-views-execution-v1/failure.json','reports/verified-views-execution-v1/restored.json',
        'reports/execution-host-recovery-v2/research-stop-before-production.json',
        'reports/execution-host-recovery-v2/restored.json',str(Path(__file__).relative_to(ROOT))]
    at = datetime.now(timezone.utc).isoformat()
    review = dict(at=at,objective_preserved=True,previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        current_goal_turn_progress=[
            'Original originating probability-guard failure identified from authoritative terminal trace; records and unused branch states preserved.',
            'Actual four-GPU prefix reconstruction and own-weight FP64 reference reproduce and explain the rejected FP32difference.',
            'Raw vectors/logits read back; fail-closed calibrated guard and analytic single-update gradient implemented and tested.',
            'Both138states reconcile; fresh full four-L40 recovery/invalid-RNG/full512horizon admission passes before production resumes beyond139.',
            'Full467.6Mtarget phase-v2 and complete85,786pair-per-arm human-v2 controllers are genuinely requeued and live.'],
        requirements=requirements,latest_completed_native_outcomes=previous['latest_completed_native_outcomes'],
        actual_current_production_progress=progress,explicit_legacy_objective_prefix=138,
        actual_full_profile_GPU_admission_passed=True,production_uses_original_published_reward_v2=True,
        numerical_calibration={k:calibration[k] for k in ['observed_own_trajectories','observed_own_action_tokens',
            'original_guard_rejections_reproduced','maximum_observed_FP32_cache_full_logp_difference',
            'maximum_observed_FP64_cache_full_logp_difference','admitted_absolute_action_logp_bound',
            'relative_action_probability_error_bound']},
        queued_broad_training=dict(owner=phase['owner'],planned_targets=467581587,planned_updates=35284,production_updates=0),
        queued_human_comparison=dict(owner=human['owner'],train_pairs_each_arm=85786,held_pairs=4642,updates_each_arm=5362,
            arms=['chosen_CE','human_DPO'],production_updates=0,actual_future_GPU_admission_pending=True),
        prior_failed_queues_terminal_not_silently_restarted=True,all_ten_final_requirements_incomplete=True,
        no_remaining_scope_removed=True,held_coding_gain_proven=False,candidate_promoted=False,goal_achieved=False,
        source_files_sha256={str(ROOT/p):file_hash(ROOT/p) for p in names})
    for name in ['README.md','RESEARCH.md','reports/completion-audit.json']:
        if any(str(ROOT/name) in p['source_files_sha256'] for p in [plan,phase,human]):
            raise ValueError('Never edit a current or queued bound document')
    atomic_json(DESTINATION,review)
    path = ROOT/'reports/completion-audit.json'
    backup = ROOT/'reports/completion-audit-before-single-update-recovery-v3.json'
    if backup.exists():
        raise ValueError('Preserve previous full-scope audit')
    shutil.copyfile(path,backup)
    audit = json.loads(path.read_text())
    audit.update(updated_at=at,latest_full_scope_review=str(DESTINATION.relative_to(ROOT)),
        requirements={k:v['current_evidence_and_limits'] for k,v in requirements.items()},
        current_posttraining=dict(phase='production_fresh_single_update_execution_control_comparison',
            legacy_updates=138,actual_control_updates=progress['verified_solution']['actual_updates'],
            actual_execution_updates=progress['execution_hybrid']['actual_updates'],actual_full_GPU_capacity_passed=True,
            held_coding_gain_proven=False),queued_full_source_phase_training=review['queued_broad_training'],
        queued_human_preference_comparison=review['queued_human_comparison'],
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(path,audit)
    message = f'''Training resumed after verified numerical diagnosis, **2026-10-17**: both actual production arms passed the original failing update139and now stand at **{progress['verified_solution']['actual_updates']}/173control** and **{progress['execution_hybrid']['actual_updates']}/173execution RL** on all4localL40s. The original failure began with rank3rejecting cached/full FP32probabilities; rank2then timed out in NCCLall-reduce139. The failed trial and both stopped queue versions remain archived. A disposable transport-only admission was explicitly stopped before production when the originating guard was identified.

The real failure is reproduced using the same own student: maxFP32action-logp difference **0.0001556873**, versus **1.0436e-13** between cached/full FP64equation recomputations of the same weights. Original cached behavior replays exactly. Eight actual own trajectories/610actions and raw logits/vectors were read back; independent FP64normalization of the original FP32logits uses the original numerical tolerance rather than demanding CPU/GPUbit equality. A calibrated **0.000623absolute logp** runtime bound(about0.06232percentrelative action probability) now rejects larger deviations. It is empirical, not a future accuracy guarantee.

The new method uses one fresh score-function update per sampled group plus the same own-base KL and original complete-target fallback. Its gradient matches the original clipped-ratio gradient at ratio1and an independent analytic gradient; it avoids assigning an importance ratio to FP32rounding. Local errors are coordinated before reduction. Actual four-CPU migration/recovery/fault tests pass. From the durable128-state, ten real diagnostic updates reconstruct138with exact original behavior/reward/semantic-record replay; all4saved138states and561Adam clocks reconcile. Objective migration is explicit after138. **Fresh actual four-L40 next-update/weights/Adam/all-private-RNG recovery, bad-CUDA-state rejection and longest1438prompt/fixed512horizon backward pass on all4cards before new production.** These controls are not held quality gains.

The **467,581,587target/35,284update** foundation/instruction/vision pass and matched **85,786human-pair-per-arm/5,362update** preference comparison are requeued as phase-v2andhuman-v2behind the corrected run. Both exact waiting owners are live, GPU-hidden and childless. Human training will also evaluate all4,642held public rankings; no human production update has occurred yet. All original source/repetition limits, broad image/OCR/word work and full independent final evaluations remain in scope. Latest completed native coding correctness remains **0/24** after warm-up; post-RL native results are pending. **All ten final-model requirements remain incomplete; no candidate is promoted.** See [full review](reports/single-update-recovery-v3-review.json), [live RL](reports/execution-single-update-recovery-v3/execution_hybrid-progress.json), [actual GPU admission](reports/execution-single-update-recovery-v3/capacity.json), [numerical readback](reports/code-probability-rounding-v1/admission.json), [broader queue](reports/phase-joint-v2-queue.json) and [human queue](reports/phase-human-preference-v2-queue.json). Earlier entries are historical.

'''
    for name in ['README.md','RESEARCH.md']:
        path = ROOT/name
        first,_,rest = path.read_text().partition('\n\n')
        path.write_text(first+'\n\n'+message+rest)
    print(json.dumps(dict(review=str(DESTINATION.relative_to(ROOT)),
        control=progress['verified_solution']['actual_updates'],execution=progress['execution_hybrid']['actual_updates'],
        all_ten_final_requirements_incomplete=True)),flush=True)


if __name__ == '__main__':
    main()
