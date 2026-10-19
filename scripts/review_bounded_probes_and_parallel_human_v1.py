"""Full objective review after train-only counterexamples and parallel DPO proof."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT/'reports/bounded-probes-and-human-parallel-v1-review.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Review must hide all GPUs')
    if DESTINATION.exists():
        raise ValueError('Preserve every previous full-scope review')
    import assess_joint_java_comparison_v1 as resources
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    def read(name):
        return json.loads((ROOT/name).read_text())
    run = 'reports/verified-views-execution-v1/'
    plan = read(run+'plan.json')
    queue = read('reports/phase-joint-v1-queue.json')
    if not resources.same(plan['owner']) or not resources.same(queue['owner']):
        raise ValueError('Confirm exact live training and queued successor identities')
    workers = [row for row in read(run+'children.json') if resources.same(row)]
    if sum(row['subject'].startswith('RL-rank-') for row in workers) != 4:
        raise ValueError('All four actual current RL/control workers required')
    for name in ['README.md', 'RESEARCH.md', 'reports/completion-audit.json']:
        if any(str(ROOT/name) in p['source_files_sha256'] for p in [plan, queue]):
            raise ValueError('Never edit current or queued source-bound documentation')
    prepared = read('reports/bounded-code-probes-v1/prepared.json')
    admission = read('reports/bounded-code-probes-v1/admission.json')
    reader = read('reports/bounded-code-probes-v1/independent-reader.json')
    replay = read('reports/execution-prefix96-augmented-v1/complete.json')
    shared = read('reports/execution-prefix-reuse-full-CPU-v5/complete.json')
    distribution = read('reports/parallel-human-preference-four-CPU-v2/complete.json')
    cache_distribution = read('reports/execution-group-prefix-four-CPU-v5/complete.json')
    if (not admission['bounded_reference_probe_scoring_admitted']
            or not reader['all_original_reference_outputs_reproduced']
            or not reader['all_added_probe_no_ops_failed']
            or not replay['all_applied_original_rewards_reproduced']
            or not distribution['all_four_CPU_workers_passed']
            or not cache_distribution['all_four_CPU_workers_passed']):
        raise ValueError('All separately executed controls must have completed')
    for proof in [admission, replay, shared, *distribution['proofs'], *cache_distribution['proofs']]:
        FileBindings(proof['source_files_sha256']).check(full=True)
    prior = read('reports/phase-joint-queued-v1-review.json')
    requirements = prior['requirements']
    additions = {
        'own_architecture': 'New one-group immutable prompt-cache reuse preserves exact sampled trajectories/RNG on the full counted warm-up policy. This changes inference computation, not network capacity; no frontier capability is shown.',
        'public_data': 'New bounded machine-generated training observations extend existing public functions: 546 independently replayed reference/no-op observations on 88 of 173 original functions. The other 85 retain explicit original reward treatment. No new raw function, external teacher response or human judgment was collected.',
        'under_1b_parameters': 'The full cache-reuse measurement uses the actual 999,706,442-parameter policy without additional neural values. The new parallel human preference kernel shares its original counted frozen own base; four-worker proof uses a clearly labelled small random network.',
        'text_and_images': 'No new final image/word/grounding quality result is available. Existing failed/mixed native results remain controlling. Later human preference training must also be checked against broad and visual regressions.',
        'human_feedback': 'A complete-pair four-replica DPO/chosen-CE kernel now uses global pair and chosen-token denominators and gradient SUM, including empty tails. Four tests compare unequal masks/empty replicas against serial gradients and reject held/machine labels. Actual four CPU workers execute two logical/eight unique original human-pair updates, three physical optimizer calls with exact all-worker Adam and private Python/NumPy/Torch RNG recovery. These are small-network controls, not full-size production RLHF. The 85,786 public human-ranked training pairs contain other-model responses; they are not fresh human judgments of our outputs.',
        'serious_full_model_training': 'The current actual 173-update-per-arm experiment continues unchanged. No bound current or queued source/test/log/plan was edited. New reward admission, 768-response replay and sampler/DPO controls add no production updates. The complete broader 467,581,587-target phase-separated curriculum remains queued with its own actual four-L40 capacity gate.',
        'high_end_llm_capability': 'Still not achieved. Latest completed full512-token code correctness remains 0/24 after warm-up in both frames, with no post-RL native result yet. Stronger public training rewards and sampling speed do not establish held quality or high-end parity.',
        'broad_independent_benchmarks': 'All 768 actual own responses in the matched first96 training problems are independently reexecuted under the actual applied reward and separately under the augmented diagnostic. Applied rewards reproduce. This is train-only source/reference agreement, not independent final evaluation. All original complete broad/code/multimodal final benchmark requirements remain.',
        'gpu_runtime': 'All four actual RL/control workers and the childless GPU-hidden successor are live. One full-size, longest1438-token-prompt CPU trial with horizon4 reduces prefills2to1 and time25.088to13.411seconds (1.871ratio), with exact actions/probabilities/rewards/final RNG. This is one short CPU trial including reward execution, not a measured full-horizon GPU speedup. New sampler/reward and human preference production still require full-profile actual GPU admission.',
        'any_code_correctness_and_efficiency': 'The first96 matched training problems give control12/384positive original rewards,0full,mean0.01125; RL19/384positive,1full,mean0.0208687. Added probes reject that sole full-public-test program; augmented full passes are0/384for both. The program violates documented string/bytes exclusions. Augmented positives20/384control,26/384RL are finite reference checks, not arbitrary-code correctness/efficiency. Original live rewards remain unchanged and are not relabelled.'}
    for key, value in additions.items():
        requirements[key]['current_evidence_and_limits'] += ' '+value
    progress = {arm: read(run+arm+'-progress.json') for arm in ['verified_solution', 'execution_hybrid']}
    files = ['reports/bounded-code-probes-v1/prepared.json',
        'reports/bounded-code-probes-v1/admission.json', 'reports/bounded-code-probes-v1/independent-reader.json',
        'reports/execution-prefix96-augmented-v1/complete.json',
        'reports/execution-prefix-reuse-full-CPU-v5/complete.json',
        'reports/parallel-human-preference-four-CPU-v2/complete.json',
        'reports/execution-group-prefix-four-CPU-v5/complete.json',
        'reports/parallel-human-preference-v2-tests-v2.log',
        'reports/first-full-published-pass-doc-edge-diagnostic-v1.json',
        'reports/first-full-published-pass-augmented-diagnostic-v1.json',
        'src/slm_lab/parallel_human_preference_v2.py', str(Path(__file__).relative_to(ROOT))]
    at = datetime.now(timezone.utc).isoformat()
    review = dict(at=at, objective_preserved=True, current_goal_turn_classification='progress',
        current_goal_turn_progress=[
            'Actual current production RL/control continues; no trial is replaced or relabelled.',
            'A real published-test-full-pass own program fails documented edges; independent 546-observation augmentation is admitted for future reward use.',
            'All768actual own-response applied rewards replay exactly; stronger diagnostic rejects the sole published-full-pass program.',
            'Exact scalar prompt reuse passes four-worker recovery and full-count longest-prompt CPU timing.',
            'Complete original human DPO/chosen-CE global-denominator gradients and empty tails pass tests; four actual CPU replicas recover next weights/Adam/all private CPU RNGs.'],
        requirements=requirements, latest_completed_native_outcomes=prior['current_native_outcomes'],
        actual_current_RL_progress=progress, actual_matched_prefix96_reward_diagnostics=replay['summaries'],
        added_training_probes=dict(functions=prepared['functions_with_qualified_extra_probes'],
            observations=prepared['new_machine_probe_observations'], future_scoring_admitted=True,
            applied_to_current_run=False, reference_agreement_not_formal_domain_or_general_correctness=True),
        sampler_CPU_measurement={k:shared[k] for k in ['actual_unique_parameters',
            'complete_longest_original_prefix_tokens','maximum_new_tokens','baseline_seconds',
            'shared_seconds','short_CPU_wall_time_ratio','full_horizon_GPU_throughput_not_measured']},
        parallel_human_preference=dict(small_network_four_CPU_workers_passed=True, logical_unique_human_pairs=8,
            full_size_GPU_capacity_pending=True, production_updates=0),
        queued_successor=dict(owner=queue['owner'], planned_updates=35284, planned_targets=467581587,
            production_updates=0, actual_GPU_capacity_pending=True,
            all_selected_family_records_once=True, all_original_sources_once=False,
            repeated_references_and_cross_family_overlap_explicit=True),
        all_ten_final_requirements_incomplete=True, no_remaining_scope_removed=True,
        held_coding_gain_proven=False, full_size_human_preference_production_started=False,
        source_files_sha256={str(ROOT/p):file_hash(ROOT/p) for p in files},
        candidate_promoted=False, goal_achieved=False)
    atomic_json(DESTINATION, review)
    audit_path = ROOT/'reports/completion-audit.json'
    snapshot = ROOT/'reports/completion-audit-before-bounded-probes-v1.json'
    if snapshot.exists():
        raise ValueError('Preserve previous complete audit')
    shutil.copyfile(audit_path, snapshot)
    audit = json.loads(audit_path.read_text())
    audit.update(updated_at=at, latest_full_scope_review=str(DESTINATION.relative_to(ROOT)),
        requirements={k:v['current_evidence_and_limits'] for k,v in requirements.items()},
        current_posttraining=dict(phase='controlled_execution_RL_and_supervised_comparison',
            actual_execution_updates=progress['execution_hybrid']['actual_updates'],
            actual_control_updates=progress['verified_solution']['actual_updates'], warmup_updates=88,
            model_unique_parameters=999706442, production_execution_RL_started=True,
            held_coding_gain_proven=False),
        queued_full_source_phase_training=review['queued_successor'],
        latest_future_reward_and_human_preference_work=review['added_training_probes'],
        all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(audit_path, audit)
    message = f'''Training reward coverage and faster post-training, **2026-10-17**: the actual four-L40 controlled run continues at **{progress['execution_hybrid']['actual_updates']}/173 execution updates** and **{progress['verified_solution']['actual_updates']}/173 supervised-control updates**. The latest completed native evaluation remains **0/24 coding correctness** after warm-up in both full512-token frames; post-RL native assessments are pending. No model is promoted.

I reproduced a real weakness in public training tests: the first fully rewarded generated `is_sequence` program passes the original examples but mishandles documented strings/bytes and other reference observations. **546 added bounded machine observations across88functions** now have independent full reference/no-op replay and future scoring admission. The other85functions keep their declared original reward treatment. These checks measure finite reference agreement, not formal input-domain validity or general correctness. Current live rewards and original publisher records remain unchanged.

All **768 own responses** from the matched first96problems were reexecuted. Original rewards give control **12/384 positive,0full** and RL **19/384 positive,1full**. The new observations reject that sole full-public-test program; augmented full passes are **0/384 both arms**. Augmented partial positives are20/384control and26/384RL. These are training diagnostics, not held capability gains or human preferences.

New single-group prompt-cache reuse preserves exact actions, behavior probabilities, rewards and final RNG. Four CPU workers pass distributed update/recovery controls. On the **actual999,706,442-parameter** policy and longest1438-token training prompt, one short horizon4CPUtrial takes **13.41svs25.09s**, with prefills1vs2. **GPU/full-horizon speedup remains unmeasured.** The future four-replica human DPO/chosen-CE kernel also passes unequal-mask/empty-tail serial-gradient tests and actual four-CPU next-update/Adam/all-private-CPU-RNG recovery on eight original human-ranked pairs. These small-network controls are not full-size production RLHF. Public human rankings concern other-model responses, not new human judgments of our model.

The queued **467,581,587-target,35,284-update** foundation/instruction/vision curriculum is preserved, with fresh actual four-L40 capacity still required. This is one pass over selected family records with explicit exercise references/overlap, not all original raw sources. Broad code, factual knowledge, math, ordinary images, OCR, human preference learning and independent final evaluations remain in scope. **All ten final-model requirements remain incomplete.** See the [full-scope review](reports/bounded-probes-and-human-parallel-v1-review.json), [live training](reports/verified-views-execution-v1/execution_hybrid-progress.json), [reward replay](reports/execution-prefix96-augmented-v1/complete.json), [exact full-size cache measurement](reports/execution-prefix-reuse-full-CPU-v5/complete.json) and [human preference controls](reports/parallel-human-preference-four-CPU-v2/complete.json). Earlier entries are historical.

'''
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT/name
        first, _, rest = path.read_text().partition('\n\n')
        path.write_text(first+'\n\n'+message+rest)
    print(json.dumps(dict(review=str(DESTINATION.relative_to(ROOT)),
        actual_execution_updates=progress['execution_hybrid']['actual_updates'],
        added_probes=546, all_ten_final_requirements_incomplete=True)), flush=True)


if __name__ == '__main__':
    main()
