"""Record actual independent coding outcomes and the unlaunched cached runtime."""
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import read, sha, atomic
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/first-repair-quality-and-cached-visual-runtime-v1-review'
PREVIOUS = ROOT / 'reports/visual-dispatch-three-repair-arms-and-casts-v1-review/review.json'


def main():
    if OUT.exists():
        raise ValueError('Preserve every previous completed or failed result')
    checks = [('device-native-code-action-readback-v1-tests.log', 13),
        ('joint-visual-cast-equivalence-v1-tests.log', 5),
        ('joint-visual-cached-controller-v1-tests.log', 10),
        ('visual-aligned-GPU-driver-v2-tests.log', 10)]
    for name, count in checks:
        if str(count) + ' passed' not in (ROOT / 'reports' / name).read_text():
            raise ValueError('All exact device-protocol, cross-objective, connected-controller and launch controls must pass')
    reader_dir = ROOT / 'reports/first-GPU-repair-native-program-readback-CPU-v1'
    reader = read(reader_dir / 'complete.json')
    if (not reader['passed'] or not reader['actual24fresh_reference_controls_passed']
            or reader['actual_native_verdict_differences'] or reader['actual_source_verdict_differences']
            or not reader['all96native_and48ordinary_Python_programs_freshly_executed']
            or not reader['all30source_validation_programs_freshly_executed']):
        raise ValueError('Every complete original program, action, ordinary Python case and source-validation outcome must be freshly checked')
    totals = reader['actual_fixed_endpoint_totals']
    before, after = totals['baseline'], totals['plain_repeat']
    for value, wanted in [(before, dict(syntax=14, functional=1, fresh_empirical_efficiency=0,
            ordinary_Python_functional=0, genuine_EOS_stops=10, source_syntax=3, source_all_original_examples_pass=0)),
        (after, dict(syntax=21, functional=1, fresh_empirical_efficiency=1,
            ordinary_Python_functional=1, genuine_EOS_stops=17, source_syntax=10, source_all_original_examples_pass=1))]:
        if any(value[k] != v for k, v in wanted.items()) or value['native_code_programs'] != 48 or value['ordinary_Python_programs'] != 24:
            raise ValueError('Reconcile any changed actual independent results before recording a quality claim')
    if (before['passing_tasks'] != [dict(language='JavaScript', task='lower_bound', frame='instruction')]
            or after['passing_tasks'] != [dict(language='Python', task='sum_even', frame='completion')]):
        raise ValueError('The gained Python case and lost JavaScript case must both remain explicit')
    prefix_proofs = sorted((ROOT / 'reports/joint-visual-cast-equivalence-v1-pytest').rglob('actual35layer-visual-cast-equivalence.json'))
    combined_proofs = sorted((ROOT / 'reports/joint-visual-cached-controller-v1-pytest').rglob('actual-cached-controller.json'))
    four_prefix = [p for p in (ROOT / 'reports/joint-visual-cast-equivalence-v1-pytest').rglob('0.json')]
    four_combined = [p for p in (ROOT / 'reports/joint-visual-cached-controller-v1-pytest').rglob('0.json')]
    if not all(len(p) == 1 for p in [prefix_proofs, combined_proofs, four_prefix, four_combined]):
        raise ValueError('Require both actual complete four-worker runs and the actual35layer/single-controller proofs')
    own = read(prefix_proofs[0])
    if (own['actual_research_parameters'] != 412482 or own['actual_native_parameters'] != 374338
            or own['actual_complete_prefix_presentations'] != 6 or own['actual_separate_physical_capacity_updates'] != 6
            or own['cache_inventory']['cached_original_parameter_tensors'] != 246
            or own['cache_inventory']['patched_detach_only_wrappers'] != 492
            or not own['all_three_objectives_and_Adam_input_gradients_and_weights_Adam_metrics_private_RNG_exact']
            or own['full_size_GPU_equivalence']):
        raise ValueError('Actual small35layer mechanics must not be relabelled full-size GPUevidence')
    proofs = [*prefix_proofs, *combined_proofs]
    for folder in [four_prefix[0].parent, four_combined[0].parent]:
        proofs.extend(folder / (str(r) + '.json') for r in range(4))
    for rank, path in enumerate([four_combined[0].parent / (str(r) + '.json') for r in range(4)]):
        value = read(path)
        original = value['original_complete_capacity']
        if (not value['passed'] or original['rank'] != rank or original['actual_actor_parameters'] != 70338
                or original['actual_native_parameters'] != 68098 or value['actual_total_capacity_physical_optimizer_calls'] != 17
                or original['actual_complete_capacity_entries'] != 10 or original['actual_capacity_source_presentations'] != 13
                or original['actual_final_sparse_group_clocks'] != dict(vision=10, pointing=1, word=2)
                or not original['independent_native_cold']['actual_text_photo_logits_and_word_readout_bit_exact']
                or original['full_size_GPU_capacity_passed']):
            raise ValueError('All4actual own-small replicas must preserve every original objective/recovery/native-cold scope')
    old_driver = ROOT / 'scripts/train_joint_visual_aligned_continuation_GPU_v1.py'
    new_driver = ROOT / 'scripts/train_joint_visual_aligned_continuation_GPU_v2.py'
    functions = lambda p: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef)}
    old, new = functions(old_driver), functions(new_driver)
    same = {k: old[k] == new[k] for k in ['stage_sources', 'initializer', 'gpu_setup', 'load_candidate', 'assessment']}
    if not all(same.values()):
        raise ValueError('The fixed initializer, devices, original native candidate loader and full evaluator must stay unchanged')
    prep_path = ROOT / 'reports/joint-visual-aligned-continuation-preparation-v1/plan.json'
    prep = read(prep_path)
    if terminal(prep['owner']) or any(sha(p) != h for p, h in prep['source_files_sha256'].items()):
        raise ValueError('The exact original source-only preparation and all its frozen source bindings must remain intact')
    if (ROOT / 'reports/joint-visual-aligned-continuation-GPU-v2').exists():
        raise ValueError('This report expects the new GPUdriver still unlaunched')
    at = datetime.now(timezone.utc).isoformat()
    study = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
    snapshot = dict(at=at, actual_live_GPUchildren=[p for p in read(study / 'children.json') if not terminal(p)],
        exact_original_GPUparent_live=not terminal(dict(pid=101066, ticks='54137054')),
        complete_original_endpoints=[p.parent.name for p in study.glob('*/complete.json')],
        exact_original_source_only_preparation_live=True, preparation_owner=prep['owner'],
        exact_original_predecessors=prep['exact_predecessors'],
        actual_original_predecessor_births_live={k: not terminal(v) for k, v in prep['exact_predecessors'].items()},
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True), actual_free_bytes=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
        full_visual_pixel_readback_progress=read(ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1/progress.json'),
        no_existing_neural_process_signalled_reprioritized_restarted_or_changed=True,
        no_new_GPU_context_lease_or_job=True, no_mutable_queue_journal_progress_or_child_registry_source_binding=True)
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['any_code_correctness_and_efficiency']['evidence'] = (
        'The predeclared first coding arm now has fresh independent checks of all96baseline/candidate native programs, all48ordinary '
        'Python deliveries, all30source-validation programs and24fresh reference controls, with zero verdict differences. '
        'Syntax14/48to21/48; functional1/48to1/48; empirical efficiency0/48to1/48; ordinaryPython0/24to1/24; genuineEOS10/48to17/48. '
        'Python completion sum_even is gained; the previous JavaScript instruction lower_bound pass is lost. Total correctness is flat. '
        'Public source-validation examples pass0/15to1/15, with answers visible in prompts and no certified all-history independence. '
        'The remaining two matched arms are still evaluating; first-arm evidence does not select a winner, promote a candidate or '
        'establish arbitrary-code or high-end capability.')
    requirements['text_and_images']['evidence'] = (
        'The cached visual integration passes5cross-objective and10connected-controller controls. An actual412482parameter research/'
        '374338native35layer CPUgraph verifies complete native CE, full image/text matching and FP32 CTC/word-native objectives with '
        'exact Adam-input gradients/selected weights/Adam/metrics/private RNG, then restores the durable untrained boundary. '
        'Four actual70338research/68098native CPUreplicas also run the6extra prefix updates plus the intact original11capacity calls: '
        '17physical calls, all9families/10small entries/13presentations, sparse clocksvision10/pointing1/word2, private recovery and '
        'independent native cold remain exact. These are mechanics results. Full4GPU984Mcast equivalence, original10actual '
        'capacity cases/510presentations and4733production updates/302627views/4669243targets remain pending.')
    requirements['gpu_runtime']['evidence'] = (
        'A new unlaunched owned visual GPUv2driver consumes the intact original source-only preparation and extends immutable runtime '
        'bindings. Ten new launch/paired-scope checks pass; AST comparison confirms fixed initializer/device/native-loader/evaluator '
        'helpers unchanged. Before training, all4actual GPUs must pass a192presentation native/photo/word uncached-versus-cached '
        'gradient/Adam/RNG prefix, return to the committed initializer, then pass every original full-size capacity/recovery/cold gate. '
        'All6extra and11original physical calls are reported separately. Cache values add no learned parameters and remain outside '
        'state/export; source changes reject. FullsizeCUDAequivalence and GPU speed are unmeasured; no current worker is patched. '
        'Future launch still requires original aligned/source predecessors terminal, a new normalnice0process, original exclusive '
        'lease and full disk/memory budgets. The existing GPUstudy continues evaluating its remaining matched arms.')
    requirements['broad_independent_benchmarks']['evidence'] = (
        'The first predeclared coding arm has complete original local broad producer reports: knowledge2/56to1/56, math0/16to0/16, '
        'OCR0/20to0/20 andCER9.879167to11.15625(worse), objects0/15to0/15, photosI2T1/32to1/32andT2I2/32to2/32. Broad neural '
        'scores have not received a new independent forward readback. The separate program reader certifies only unchanged actions/'
        'programs and fresh execution, not broad knowledge gains. All original16961QA/3088AI2Dtimes3/600MBPP/4642human/'
        '3429wordtimes3populations and9final endpoints remain intact and incomplete for the requested final model.')
    for value in requirements.values():
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('All ten full requested requirements remain incomplete')
    names = ['src/slm_lab/device_native_code_action_readback_v1.py', 'scripts/read_first_GPU_repair_endpoint_programs_CPU_v1.py',
        'tests/test_device_native_code_action_readback_v1.py', 'scripts/run_device_native_code_action_readback_tests_v1.py',
        'src/slm_lab/joint_visual_cast_equivalence_v1.py', 'src/slm_lab/joint_visual_cached_controller_v1.py',
        'tests/test_joint_visual_cast_equivalence_v1.py', 'tests/test_joint_visual_cached_controller_v1.py',
        'scripts/run_joint_visual_cast_equivalence_tests_v1.py', 'scripts/run_joint_visual_cached_controller_tests_v1.py',
        'tests/test_visual_aligned_GPU_driver_v2.py', 'scripts/run_visual_aligned_GPU_driver_tests_v2.py',
        'src/slm_lab/frozen_linear_cast_cache_v1.py']
    paths = [Path(__file__), PREVIOUS, prep_path, old_driver, new_driver, *proofs,
        *[ROOT / n for n in names], *[ROOT / 'reports' / n for n, _ in checks],
        *[p for p in reader_dir.iterdir() if p.is_file() and p.name not in ['progress.json']],
        *[study / p / 'complete.json' for p in ['baseline', 'plain_repeat']]]
    files = {str(p): sha(p) for p in paths}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    entrypoint = dict(implementation_ready=True, driver=str(new_driver), source_files_sha256={str(new_driver): sha(new_driver)},
        original_source_only_preparation=str(prep_path.parent), full_source_and_aligned_pipeline_still_pending=True,
        GPUjob_started=False, explicit_later_fresh_normal_nice0_launch_required=True,
        actual_four_GPU_prefix_equivalence_and_original_capacity_required=True,
        all4733steps302627views4669243targets_and_full_final_candidate_suites_preserved=True, goal_achieved=False)
    atomic(OUT / 'next-visual-runtime.json', entrypoint)
    classification = ('progress:first coding arm independently improves syntax14to21/48 and ordinaryPython0to1/24, '
        'but total correctness stays1/48 with gained/lost tasks;38new protocol/cross-objective/cached-controller/launch '
        'checks pass and full owned cached GPUv2runtime is implemented but unlaunched')
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        actual_fixed_baseline_and_first_arm_totals=totals,
        all96native48ordinary_Python30source_programs_and24references_freshly_executed=True,
        actual_native_or_source_verdict_differences=0, first_arm_total_code_correctness_improved=False,
        first_arm_gained_Python_case_and_lost_JavaScript_case_both_preserved=True,
        actual_new_controls=38, actual_device_action_protocol_controls=13,
        actual_cross_objective_cache_controls=5, actual_connected_cached_controller_controls=10,
        actual_GPUv2launch_and_paired_scope_controls=10, actual35layer_research_parameters=412482,
        actual35layer_native_parameters=374338, actual_small_four_replica_connected_capacity=True,
        actual_separate_cache_prefix_capacity_calls=6, actual_original_controller_capacity_calls=11,
        actual_small_total_physical_capacity_calls=17, fixed_loader_device_and_assessment_AST_unchanged=same,
        original_live_preparation_and_all_frozen_bindings_unchanged=True,
        cached_visual_GPUv2implemented=True, new_GPUjob_started=False, actual_full_size_GPUcache_equivalence=False,
        measured_GPU_speedup=False, candidate_promoted=False, broad_quality_improvement_not_established=True,
        all_ten_final_requirements_incomplete=True, goal_achieved=False))
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'next-visual-runtime.json']:
        files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, previous_full_scope_review=str(PREVIOUS),
        current_goal_turn_classification=classification, source_files_sha256=files, goal_achieved=False))
    metrics = ('The fixed first arm now has independent complete checks of **96native programs,48ordinary Python programs, '
        '30source-validation programs and24fresh reference controls**. All original prompt IDs, actual generated tokens, EOS '
        'boundaries, raw text and delivered program bytes reconstruct; zero native/source verdict differences. No new neural '
        'generation or program repair was used.\n\n'
        '| Local metric | Baseline | First coding arm |\n|---|---:|---:|\n'
        '| Syntax-valid code |14/48|21/48|\n| Functional code |1/48|1/48|\n'
        '| Empirical efficiency pass |0/48|1/48|\n| Ordinary Python functional |0/24|1/24|\n'
        '| Genuine EOS stops |10/48|17/48|\n| Public source-example pass |0/15|1/15|\n\n'
        '**Overall code correctness is flat.** Python completion`sum_even` is gained; JavaScript instruction`lower_bound` '
        'is lost. Source-validation answers are visible in prompts and historical independence is not certified. The other '
        'two coding arms are still evaluating; this first arm is not a selected winner and is not promoted.\n\n'
        'Complete broad producer reports show knowledge**2/56→1/56**, math0/16→0/16, OCR0/20→0/20with '
        '**CER9.879167→11.15625(worse)**, objects0/15→0/15 and unchanged photo retrieval1/32I2Tand2/32T2I. '
        'These broad scores have not received a new independent neural forward readback. **There is no broad-quality gain claim.**\n\n'
        '**38new checks pass:**13actual device-action protocol/tampering controls,5cross-objective cache checks, '
        '10connected cached-controller checks and10new GPUdriver launch/paired-scope checks. A412482parameter research/'
        '374338native35layer CPUgraph retains exact native/image-matching/FP32word objectives, actual Adam-input gradients, '
        'weights, moments, metrics and private RNG. Four actual70338research/68098native CPUreplicas complete the6extra '
        'uncached/cached prefix calls plus the intact11original capacity calls. All9families,10small entries/13presentations, '
        'sparse clocksvision10/pointing1/word2, original recovery controls and independent native cold remain. '
        '**Small mechanics are not full-size CUDAcapacity or speed evidence.**\n\n'
        'The owned **visual GPUv2driver is implemented and unlaunched**. It uses the original source-only preparation without '
        'changing that live process or its frozen files, then extends runtime bindings. It requires all4actual full-size '
        'native/photo/word gradient/Adam/RNG equivalence prefixes (192presentations), exact initializer restoration, and '
        'the complete original10capacity cases/510presentations before training. All6extra and11original physical calls '
        'are recorded separately. All4733production updates/302627views/4669243targets, final complete standalone export/cold '
        'and all7full candidate suites remain. No extra learned values or cached export state; no current worker is patched. '
        '**Actual full-size CUDAequivalence and GPU speedup remain unmeasured.**\n\n'
        'All17public families,85786human pairs, full human/RLstages and9final endpoints remain. '
        '**All ten requested final requirements remain incomplete.**\n')
    (OUT / 'metrics.md').write_text(metrics)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, goal_achieved=False,
        latest_full_scope_review=str(OUT / 'review.json'),
        latest_actual_first_repair_quality_and_cached_runtime=str(OUT / 'complete.json'),
        next_visual_GPU_driver=str(new_driver), current_goal_turn_classification=classification)
    atomic(audit_path, audit)
    entry = (f'First coding-arm quality and cached visual runtime, **{at}**: independent execution confirms '
        '**syntax14/48→21/48, ordinaryPython0/24→1/24, empirical efficiency0/48→1/48; total functional stays1/48**. '
        'Python`sum_even` is gained and JavaScript`lower_bound` lost. Broad producer knowledge and OCR regress; no broad gain '
        'or winner is claimed.38new protocol/cache/controller/launch checks pass. The full cached visual GPUv2driver is '
        'implemented but unlaunched, with all4actual full-size gradient/recovery/cold gates still required. '
        '**All ten final requirements remain incomplete.** '
        '[Evidence](reports/first-repair-quality-and-cached-visual-runtime-v1-review/metrics.md), '
        '[full review](reports/first-repair-quality-and-cached-visual-runtime-v1-review/review.json), '
        '[next visual runner](scripts/train_joint_visual_aligned_continuation_GPU_v2.py). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        path.write_text(entry + path.read_text())
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual_new_controls=38,
        ordinary_Python_before=0, ordinary_Python_after=1, total_code_before=1, total_code_after=1,
        new_GPUjob_started=False, goal_achieved=False)))


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('CUDA-hidden nice19 source-only review required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    main()
