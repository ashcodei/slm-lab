"""Record closed experiments and immutable observations without changing workers."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import read, sha, atomic
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/visual-dispatch-three-repair-arms-and-casts-v1-review'
PREVIOUS = ROOT / 'reports/visual-connected-controller-and-aligned-continuation-v1-review/review.json'


def main():
    if OUT.exists():
        raise ValueError('Preserve all historical results, failures and reviews')
    checks = [('visual-native-assessment-binding-v1-tests.log', '6 passed'),
        ('visual-aligned-GPU-driver-v1-tests.log', '10 passed'),
        ('frozen-linear-cast-cache-v2-tests.log', '16 passed')]
    for name, expected in checks:
        if expected not in (ROOT / 'reports' / name).read_text():
            raise ValueError('Every named dispatch or actual own-small cache check must pass')
    arms_path = ROOT / 'reports/all-three-GPU-repair-training-readback-CPU-v1/complete.json'
    arms = read(arms_path)
    if (not arms['complete'] or not arms['passed'] or arms['actual_total_separate_arm_updates'] != 918
            or not arms['all15actual_production_owner_births_terminal'] or arms['new_accuracy_result']):
        raise ValueError('All three real complete training arms need their independent state/source readback')
    for name, arm in arms['actual_profiles'].items():
        if (name not in ['plain_repeat', 'semantic_repair', 'weighted_semantic_repair']
                or arm['actual_complete_updates'] != 306 or arm['actual_source_presentations'] != 4851
                or arm['actual_original_response_EOS_targets'] != 2287110
                or arm['actual_Adam_leaves_at_clock306'] != 561
                or arm['actual_actor_tensors_changed_from_common_initializer'] != 561
                or not arm['four_distinct_private_RNGs_and_manifest_verified']
                or not arm['actual_full_native_cold_proof_retained']):
            raise ValueError('Do not reduce any original arm, target budget, private state or cold proof')
    real_path = ROOT / 'reports/visual-real-assessment-binding-CPU-v1.json'
    real = read(real_path)
    if (not real['passed'] or not real['actual_real_numerical_worker_and_reconciler_function_objects_unchanged']
            or not real['actual_existing_GPU_report_root_unchanged'] or real['actual_neural_forwards'] != 0):
        raise ValueError('Real evaluator numerical functions and original output bindings must be unchanged')
    scheduler_path = ROOT / 'reports/visual-launch-scheduler-capability-CPU-v1.json'
    scheduler = read(scheduler_path)
    if scheduler['return_from_IDLE_to_OTHER_allowed'] or scheduler['actual_own_transition_error']['errno'] != 1:
        raise ValueError('Reassess actual scheduler constraints instead of assuming an idle child can escape')
    prep_path = ROOT / 'reports/joint-visual-aligned-continuation-preparation-v1/plan.json'
    prep = read(prep_path)
    prep_live = not terminal(prep['owner'])
    if not prep_live or prep['GPU_leases'] != 0 or prep['models_constructed'] != 0:
        raise ValueError('This review expects the exact source-only preparation still live')
    code_path = ROOT / 'reports/full-public-code-shard-readback-CPU-v1/complete.json'
    code = read(code_path)
    pixel_path = ROOT / 'reports/full-multimodal-pixel-validation-CPU-v1/complete.json'
    pixels = read(pixel_path)
    if (not code['integrity_readback_passed'] or code['actual_original_positions'] != 100000
            or code['all_original_candidate_programs_freshly_executed'] != 24271
            or code['actual_fresh_execution_counts'] != dict(all_published_tests_pass=24248,
                runtime_resource_or_confinement_abstention=13, published_test_mismatch=10)
            or code['training_admitted'] or not pixels['passed']
            or pixels['actual_counts']['physical_paths_observed'] != 195121):
        raise ValueError('Full-source checks and all preserved code failures must be represented exactly')
    proofs = sorted((ROOT / 'reports/frozen-linear-cast-cache-v2-pytest').rglob('actual*.json'))
    if len(proofs) != 2:
        raise ValueError('Both actual own-small dispatch-count and next-Adam proof files are required')
    cast = next(read(p) for p in proofs if p.name == 'actual35layer-cast-observation.json')
    adam = next(read(p) for p in proofs if p.name == 'actual35layer-two-Adam-updates.json')
    if (cast['actual_small_research_parameters'] != 370482 or cast['actual_layers'] != 35
            or cast['actual_uncached_frozen_FP32_to_BF16_conversions'] != 738
            or cast['actual_warmup_conversions'] != 246
            or cast['actual_warmed_frozen_FP32_to_BF16_conversions'] != 0
            or not cast['actual_logits_bit_exact'] or cast['measured_GPU_speedup']
            or adam['actual_Adam_leaves'] != 561 or adam['actual_updates'] != 2
            or not adam['all_gradients_weights_Adam_and_RNG_bit_exact']):
        raise ValueError('Do not turn small CPU cast-count evidence into a full-size GPU speed or quality claim')
    at = datetime.now(timezone.utc).isoformat()
    study = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
    children = read(study / 'children.json')
    active = [p for p in children if not terminal(p)]
    source_owners = [dict(pid=106808, ticks='54228332'), dict(pid=108345, ticks='54257301'),
        dict(pid=109189, ticks='54271186')]
    snapshot = dict(at=at, actual_original_GPU_parent_live=not terminal(dict(pid=101066, ticks='54137054')),
        actual_live_original_GPU_children=active, all_original_training_arms_complete=True,
        old_global_progress_is_a_last_training_record_not_current_phase=True,
        closed_current_study_endpoints=[str(p.parent.relative_to(study)) for p in study.glob('*/complete.json')],
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True),
        exact_preparation_owner=prep['owner'], preparation_live=True,
        preparation_actual_scheduler=os.sched_getscheduler(prep['owner']['pid']),
        exact_original_predecessors=prep['exact_predecessors'],
        original_predecessor_births_live={n: not terminal(p) for n, p in prep['exact_predecessors'].items()},
        full_visual_source_readers=[dict(owner=p, live=not terminal(p)) for p in source_owners],
        independent_pixel_readback_progress=read(ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1/progress.json'),
        actual_free_bytes=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
        future_full_visual_launch_estimated_free_bytes=round(54.5 * 1024**3),
        no_existing_neural_worker_signalled_reprioritized_restarted_or_changed=True,
        no_competing_GPU_context_lease_or_job=True,
        no_mutable_queue_progress_journal_or_child_registry_used_as_a_source_binding=True)
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    updates = dict(
        public_data=('The original pixel producer completes all195121physical paths and all recorded model-input protocols, '
            'covering302627views/91585words. Independent fresh pixel, full4733step rank-batch readback and separate source-only '
            'admission are still running. The public code reader reconstructs all100000positions and freshly executes all24271 '
            'weak synthetic candidates:24248published-test agreement,10mismatches,13abstentions. Every outcome and original role '
            'is preserved. This public precomputed Qwen2.5-Coder32B candidate is still unadmitted weak SFT data, not independent '
            'semantic truth, human preference, own on-policy data or an RL reward oracle. No external model weights/API are loaded.'),
        serious_full_model_training=('All3actual four-L40coding arms finish306updates/4851presentations/2287110positive response-EOS '
            'targets each:918separate-arm updates. Independent closed-state readback reconstructs every original source/rank '
            'delivery and verifies all561Adam leaves at306, all561actor tensors changed, four distinct private RNGs and original '
            '4096text/448photo/real-word cold evidence per arm. Online training NLL falls within each arm over3epochs. '
            'Those training metrics do not establish held-out correctness. Full matched native evaluations are active; '
            'all17public families, original human/RL scopes and9final endpoints remain.'),
        text_and_images=('The complete owned visual GPUdriver now connects the tested capacity/recovery/training/controller '
            'to final standalone export and all7full candidate suites. Sixbinding and10launch/paired-scope checks pass. A real '
            'canonical evaluator import confirms worker/reconciler function identities/bytecode stay unchanged and allbindings '
            'restore without constructing a model or running a forward. The exact source-only preparation is live and cannot '
            'spawn GPUs. Actual4GPU984Mcapacity, all4733updates/302627views, final standalone cold/export and quality remain '
            'pending; all original failed numerical gates remain unwaived.'),
        gpu_runtime=('The original exact GPUowner is evaluating its completed three-arm study; no current worker was changed. '
            'A real own-only scheduler probe receivesEPERM when returning fromSCHED_IDLE toSCHED_OTHER atnice19/RLIMIT_NICE0. '
            'The source-only idle preparation therefore finishes separately; actual future GPUlaunch requires a new normal '
            'nice0process, complete original aligned/source predecessors and original shared lease. A new derived-frozen-cast '
            'cache passes16own-small controls: a370482parameter35layer CPUgraph reduces738repeated frozen conversions over3forwards '
            'to0after246warmup casts, with exact logits/greedy actions/all561gradients/twoAdam states/RNG and unchanged learned '
            'inventory. It adds649216derived runtime bytes for this small graph,0learned values. Full-size CUDA equivalence, '
            'actual GPU speed and future production integration are unproved; it is not injected into current workers.'),
        human_feedback=('The visual continuation initializer remains the fixed completed aligned machine-DPO native student '
            'after the intact canonical pipeline. Preparation checks exact pipeline/finalizer/source births and retains85786human '
            'pairs,16961QA/3088AI2Dtimes3/600MBPP/4642held human pairs/3429held words times3 and all9original final endpoints. '
            'New visualcandidate evaluation uses all7suites and the exact already completed canonical baseline. No human/RL '
            'milestone or model promotion is inferred from source checks, training fit or runtime tests.'))
    for name, value in requirements.items():
        value['previous_evidence'] = str(PREVIOUS)
        if name in updates:
            value['evidence'] = updates[name]
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('Every requested final requirement is still incomplete')
    names = ['scripts/train_joint_visual_aligned_continuation_GPU_v1.py',
        'scripts/prepare_joint_visual_aligned_continuation_CPU_v1.py',
        'scripts/compare_visual_native_endpoints_v1.py', 'scripts/check_real_visual_assessment_binding_CPU_v1.py',
        'scripts/read_all_GPU_repair_training_arms_CPU_v1.py',
        'src/slm_lab/visual_native_assessment_binding_v1.py', 'src/slm_lab/frozen_linear_cast_cache_v1.py',
        'tests/test_visual_native_assessment_binding_v1.py', 'tests/test_visual_aligned_GPU_driver_v1.py',
        'tests/test_frozen_linear_cast_cache_v1.py', 'tests/test_frozen_linear_cast_cache_v2.py',
        'scripts/run_frozen_linear_cast_cache_tests_v1.py', 'scripts/run_frozen_linear_cast_cache_tests_v2.py',
        'reports/frozen-linear-cast-cache-v1-tests.log', 'reports/frozen-linear-cast-cache-v1-venv-tests.log',
        'reports/full-function-repair-comparison-GPU-v2/all-arms-training-complete.json',
        'reports/completed-public-epoch-and-visual-integration-v1-review/matched-metrics.md']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, arms_path, real_path, scheduler_path, prep_path,
        code_path, pixel_path, *proofs, *[ROOT / 'reports' / n for n, _ in checks], *[ROOT / n for n in names]]}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    classification = ('progress:all3four-L40coding arms complete with independent full306update source/Adam readback; '
        'full evaluation active; owned full visual GPUdispatch and separate live source-only preparation implemented; '
        '32new dispatch/paired-scope/own-small cast checks pass; no new held-out quality claim')
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        all_three_actual_GPU_training_arms_complete=True, actual_updates_per_arm=306,
        actual_total_separate_arm_updates=918, actual_presentations_per_arm=4851,
        actual_response_EOS_targets_per_arm=2287110, actual_Adam_leaves_at306_per_arm=561,
        actual_changed_actor_tensors_per_arm=561, all_three_independent_state_and_source_readbacks_passed=True,
        actual_new_tests=32, actual_new_dispatch_binding_and_paired_scope_tests=16,
        actual_own_small_frozen_cast_tests=16, real_evaluator_function_identity_and_bytecode_unchanged=True,
        actual_source_only_visual_preparation_live=True, actual_source_only_models_GPUleases_and_updates=0,
        actual_scheduler_return_to_normal_rejected_errno1=True,
        original_full_pixel_producer_complete_paths=195121, full_code_fresh_candidates=24271,
        fresh_code_test_agreements=24248, fresh_code_mismatches=10, fresh_code_abstentions=13,
        weak_synthetic_code_candidate_still_not_admitted=True,
        small_CPU_cast_probe_research_parameters=370482, small_CPU_cast_probe_layers=35,
        small_CPU_uncached_conversions_over3forwards=738, small_CPU_warmup_conversions=246,
        small_CPU_warmed_conversions_over3forwards=0,
        small_CPU_logits_561gradients_twoAdam_updates_RNG_exact=True,
        cache_not_injected_into_existing_workers=True, measured_full_size_GPU_speedup=False,
        full_visual_GPUcapacity_training_and_all_original_final_quality_pending=True,
        new_accuracy_result=False, candidate_promoted=False, all_ten_final_requirements_incomplete=True,
        goal_achieved=False))
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json']:
        files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, previous_full_scope_review=str(PREVIOUS),
        current_goal_turn_classification=classification, source_files_sha256=files, goal_achieved=False))
    rows = []
    for name, arm in arms['actual_profiles'].items():
        values = [m['original_token_mean_NLL'] for m in arm['online_source_epoch_metrics']]
        rows.append('| ' + name + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    metrics = ('All three four-L40 coding arms complete **306 updates, 4,851 source presentations and 2,287,110 '
        'response/EOS targets each**. Independent readback verifies every original delivery, all561Adam leaves at306, '
        'all561actor tensors changed, four distinct private RNGs and the original native cold proof in each arm. '
        '**918 is the sum of three separate experiments, not one model\'s update count.**\n\n'
        'Online training response-token NLL, with762370gold targets per source epoch in each arm:\n\n'
        '| Arm | Epoch1 | Epoch2 | Epoch3 |\n|---|---:|---:|---:|\n' + '\n'.join(rows) + '\n\n'
        '**This is training fit, not held-out correctness.** Prompt contexts differ across arms, so this table cannot '
        'select a winning approach. Full matched native assessments are running. The last independently verified public-epoch '
        'result remains1/48functional,0/48efficient and0/24ordinaryPython.\n\n'
        'The owned visual training/export/evaluation driver is implemented. **16new dispatch/binding/paired-scope checks pass**; '
        'the real frozen canonical worker/reconciler functions remain unchanged. Its separate live source-only preparation '
        'creates no model, GPUcontext or lease and cannot automatically launch GPUs. A real own-process probe confirmed that '
        'an idle nice19process cannot return to normal scheduling here; actual future launch uses a new normal nice0process. '
        'The fixed initializer remains the completed aligned native after all original canonical stages. Full4GPUcapacity, '
        'all4733visual updates/302627views/4669243targets and full final quality remain pending.\n\n'
        'The original pixel producer passes all**195121physical image paths**. Independent pixel and complete-rank source '
        'readbacks still precede separate data admission. Full100000row code reconstruction and24271fresh candidate runs '
        'finish with24248published-test agreements,10mismatches and13abstentions. All are preserved. The public precomputed '
        'Qwen-generated code candidate remains **unadmitted weak synthetic SFT**, never independent correctness or anRLoracle.\n\n'
        '**16own-small cache checks pass.** A370482parameter35layer CPUgraph makes738frozen FP32→BF16conversions over3forwards '
        'without caching; after246warmup conversions,3further forwards make0. Logits, native greedy tokens, all561adapter '
        'gradients, twoAdam updates andRNG are exact. Its649216additional derived bytes add0learned values and are absent '
        'from state/export. Source mutations/replacements reject before forward. The original missing-pytest launch and '
        'bound-method object-identity test failure are preserved; the latter is corrected by checking the same actual bound '
        'module/function, with all numerical checks unchanged. **No full-size CUDA equivalence or GPU speedup is established; '
        'the cache is not installed in existing workers.**\n\n'
        'All17public families,85786human pairs, complete human/RLstages and9final endpoints remain. '
        '**All ten requested final requirements remain incomplete.**\n')
    (OUT / 'metrics.md').write_text(metrics)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, goal_achieved=False,
        latest_full_scope_review=str(OUT / 'review.json'),
        latest_actual_visual_dispatch_three_repair_arms_and_casts=str(OUT / 'complete.json'),
        current_goal_turn_classification=classification)
    atomic(audit_path, audit)
    entry = (f'All three coding arms trained; visual dispatch and cast experiment, **{at}**: '
        '**3×306four-L40updates complete**, with independent full source/Adam/state readback. Training NLL falls in all3arms; '
        '**full held-out evaluation is active and no new correctness gain is claimed**. The full visual driver is implemented; '
        'its live idle preparation cannot launch GPUs.32new dispatch/scope/own-small cast tests pass. '
        'A35layer small CPUprobe removes repeated frozen-weight casts with exact logits/gradients/Adam/RNG; full CUDA speed '
        'and equivalence remain unmeasured. The full image producer and100krow code readback finish; code mismatches remain '
        'preserved and the weak synthetic candidate unadmitted. **All ten final requirements remain incomplete.** '
        '[Evidence](reports/visual-dispatch-three-repair-arms-and-casts-v1-review/metrics.md), '
        '[full review](reports/visual-dispatch-three-repair-arms-and-casts-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        path.write_text(entry + path.read_text())
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual_separate_arm_GPUupdates=918,
        new_tests=32, full_size_GPU_speedup_measured=False, new_accuracy_result=False, goal_achieved=False)))


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('Run the source-only review separately with CUDA hidden and nice19')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    main()
