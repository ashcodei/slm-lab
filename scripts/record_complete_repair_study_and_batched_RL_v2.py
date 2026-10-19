"""Record the closed complete coding study and new batched RL mechanics."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from record_visual_stage_and_first_repair_arm_v1 import read, sha, atomic
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/complete-repair-study-and-batched-RL-v2-review'
PREVIOUS = ROOT / 'reports/full-human-token-suffix-curriculum-v1-review/review.json'
STUDY = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
HUMAN = ROOT / 'reports/phase-human-preference-v8'


def main():
    if OUT.exists():
        raise ValueError('Preserve every previous complete or failed result')
    final_path = ROOT / 'reports/remaining-GPU-repair-native-program-readback-CPU-v1/complete.json'
    first_path = ROOT / 'reports/first-GPU-repair-native-program-readback-CPU-v1/complete.json'
    final, first = map(read, [final_path, first_path])
    if (not final['passed'] or not first['passed'] or not terminal(dict(pid=101066, ticks='54137054'))
            or not read(STUDY / 'complete.json')['all_actual_GPU_capacity_and_three_matched_arms_complete']
            or not read(STUDY / 'restored.json')['all_registered_point_actors_terminal']
            or any(not terminal(p) for p in read(STUDY / 'children.json'))
            or not final['all192native_and96ordinary_Python_and60source_programs_freshly_executed_across_both_closed_readbacks']
            or final['actual_native_verdict_differences'] or final['actual_source_verdict_differences']):
        raise ValueError('The entire original study and both complete independent program readbacks must finish unchanged')
    totals = final['actual_combined_all_four_endpoint_totals']
    expected = dict(baseline=(14, 0, 0, 10), plain_repeat=(21, 1, 1, 17),
        semantic_repair=(30, 1, 1, 24), weighted_semantic_repair=(25, 1, 1, 17))
    profiles, broad, paired = [], {}, {}
    baseline_passes = {(v['language'], v['task'], v['frame']) for v in totals['baseline']['passing_tasks']}
    for name, expected_values in expected.items():
        value = totals[name]
        if (value['native_code_programs'] != 48 or value['ordinary_Python_programs'] != 24 or value['functional'] != 1
                or (value['syntax'], value['ordinary_Python_functional'], value['fresh_empirical_efficiency'], value['genuine_EOS_stops']) != expected_values):
            raise ValueError('All original48native/24ordinary cases and exact observed gains/losses must remain explicit')
        passes = {(v['language'], v['task'], v['frame']) for v in value['passing_tasks']}
        paired[name] = dict(gained=[list(v) for v in sorted(passes - baseline_passes)],
            lost=[list(v) for v in sorted(baseline_passes - passes)], original_population=48,
            total_functional_gain=value['functional'] - totals['baseline']['functional'])
        for frame in ['completion', 'instruction']:
            report = read(STUDY / name / ('code-' + frame + '.json'))
            profiles.append({k: report[k] for k in ['generation_budget', 'decoding', 'suite_sha256', 'tokenizer_sha256', 'inference_device_type', 'inference_precision']})
        reports = {suite: read(STUDY / name / (suite + '.json')) for suite in ['knowledge', 'math', 'ocr', 'objects', 'vision']}
        if (reports['knowledge']['overall']['questions'] != 56 or reports['math']['evaluated_examples'] != 16
                or reports['ocr']['overall']['examples'] != 20 or reports['objects']['overall']['examples'] != 15 or reports['vision']['images'] != 32):
            raise ValueError('Every original complete local broad population must remain present')
        broad[name] = dict(knowledge_generated=round(56 * reports['knowledge']['overall']['generated_accuracy']),
            knowledge_likelihood=round(56 * reports['knowledge']['overall']['likelihood_accuracy']),
            math_correct=reports['math']['correct'], OCR_exact=round(20 * reports['ocr']['overall']['exact_match_accuracy']),
            OCR_character_error_rate=reports['ocr']['overall']['character_error_rate'],
            objects_exact=round(15 * reports['objects']['overall']['correct_image_exact_accuracy']),
            image_to_text=round(32 * reports['vision']['caption_retrieval']['image_to_text_accuracy']),
            text_to_image=round(32 * reports['vision']['caption_retrieval']['text_to_image_accuracy']),
            caption_NLL=reports['vision']['proper_nll'], independent_neural_forward_readback=False)
    if any(p != profiles[0] for p in profiles):
        raise ValueError('All four actual endpoints must retain the identical original native runtime profile')
    tests = ROOT / 'reports/batched-native-RL-v2-tests.log'
    if '15 passed' not in tests.read_text():
        raise ValueError('All original sampler/kernel controls plus nonfinite guards and exact-gradient recovery must pass')
    gap_path = ROOT / 'reports/batched-rollout-v1-nonfinite-rescore-gap.json'
    gap = read(gap_path)
    if not gap['actual_v1_gap_observed'] or gap['new_production_updates'] != 0:
        raise ValueError('The original own-small nonfinite comparison blind spot must stay recorded')
    proofs = sorted((ROOT / 'reports/batched-native-RL-v2-pytest').rglob('actual*.json'))
    if len(proofs) != 5:
        raise ValueError('Both actual35layer sampler traces and all3actual strict-verifier/update proofs are required')
    sample = next(read(p) for p in proofs if p.name == 'actual35layer-four-private-attempts.json')
    variable = next(read(p) for p in proofs if p.name == 'actual35layer-variable-stops.json')
    if (sample['actual_actor_parameters'] != 370482 or sample['actual_group_size'] != 4
            or sample['actual_generated_actions'] != 48 or sample['actual_prefill_batch_sizes'] != [4]
            or len(sample['actual_decode_batch_sizes']) != 11 or sample['nonfinite_full_forward_behavior_observed']
            or not sample['behavior_rescore_passed'] or sample['cached_full_max_absolute_difference'] > .001
            or variable['actual_group_size'] != 8 or variable['actual_generated_actions'] != 161
            or min(variable['actual_decode_batch_sizes']) != 5):
        raise ValueError('Actual private samples must match the full retained neural observations')
    for proof in [read(p) for p in proofs if p.name == 'actual-batched-public-RL-mechanics.json']:
        if (proof['actual_small_student_values'] != 267334 or not proof['deliberately_initialized_fixture']
                or not proof['matched_control_and_RL_samples_and_CE_identical']
                or not proof['original_fixed_base_and_critic_preserved'] or not proof['actual_global_RNG_unchanged']):
            raise ValueError('Strict public-reward mechanics must stay explicitly small and deliberately initialized')
    human_plan_path = HUMAN / 'plan.json'
    human = read(human_plan_path)
    if (terminal(human['owner']) or human['human_train_pairs'] != 85786 or human['training_options']['steps'] != 5362
            or not read(HUMAN / 'capacity.json')['passed'] or (HUMAN / 'failure.json').exists()):
        raise ValueError('The original whole human-pair run must be live after both actual full capacity gates pass')
    human_proofs = []
    for arm in ['chosen_CE', 'human_DPO']:
        for rank in range(4):
            path = HUMAN / ('capacity-' + arm + '-' + str(rank) + '.json')
            proof = read(path)
            if (proof['physical_GPU'] != rank or proof['actual_counts']['actual_unique'] != 999706442
                    or proof['actual_Adam_states'] != 561 or proof['actual_logical_updates'] != 2
                    or proof['physical_optimizer_calls'] != 3
                    or not proof['exact_next_global_update_weights_Adam_private_RNG']
                    or not proof['invalid_private_CUDA_RNG_rejected_before_mutation']
                    or [r['global_original_positions'] for r in proof['complete_source_extremes']] != human['capacity_original_pair_batches']):
                raise ValueError('All8actual full-size human capacity/recovery/extreme-source reports must pass')
            human_proofs.append(path)
    at = datetime.now(timezone.utc).isoformat()
    children = [p for p in read(HUMAN / 'children.json') if not terminal(p)]
    if not any(p['subject'].startswith('train-chosen_CE-') for p in children):
        raise ValueError('Reassess actual phase: this review expects the first whole human arm training workers live')
    snapshot = dict(at=at, original_coding_study_owner_terminal=True, all_original_coding_children_terminal=True,
        original_human_owner=human['owner'], original_human_training_children=children,
        original_human_progress={arm: read(HUMAN / (arm + '-progress.json')) for arm in ['chosen_CE', 'human_DPO']
            if (HUMAN / (arm + '-progress.json')).exists()},
        actual_GPU_rows=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'], text=True),
        source_suffix_producer_progress=read(ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1/progress.json'),
        source_suffix_producer_live=not terminal(dict(pid=146084, ticks='55111340')),
        source_suffix_fresh_reader_live=not terminal(dict(pid=146633, ticks='55128888')),
        visual_pixel_fresh_reader_progress=read(ROOT / 'reports/full-multimodal-pixel-readback-CPU-v1/progress.json'),
        full_cached_visual_preparation_live=not terminal(dict(pid=136092, ticks='54738417')),
        actual_free_bytes=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
        no_existing_neural_worker_signalled_reprioritized_restarted_or_patched=True,
        no_competing_GPU_context_lease_or_job_added=True, no_mutable_progress_registry_journal_or_human_queue_bound=True)
    requirements = deepcopy(read(PREVIOUS)['requirements'])
    requirements['any_code_correctness_and_efficiency']['evidence'] = (
        'The entire original3arm coding study is complete/restored, all actual producers terminal. Both independent readbacks '
        'freshly execute all192native/96ordinaryPython/60source-validation programs plus fresh reference controls with0verdict '
        'differences, reconstructing all actual prompt/action/raw/program bytes. Baseline/repeat/repair/weighted syntax14/21/30/25 '
        'of48; functional1/1/1/1of48; empirical efficiency0/1/1/1; ordinaryPython0/1/1/1of24; genuineEOS10/17/24/17. All3arms '
        'gain Python completion sum_even and lose JavaScript instruction lower_bound. No arm improves total correctness; none '
        'is promoted. Source-example pass0/1/0/1of15has visible public answers and no all-history independence certificate. '
        'This does not establish broad coding, arbitrary code or frontier capability.')
    requirements['broad_independent_benchmarks']['evidence'] = (
        'All four complete local broad producer populations remain56knowledge/16math/20OCR/15object/32photo examples. Generated '
        'knowledge2/1/2/2of56; likelihood4/4/3/3; math0each; OCR0each withCER9.879167/11.15625/13.370833/11.8; objects0each; '
        'retrievalI2T1/32andT2I2/32each. These are producer neural scores, without a new independent broad forward readback. '
        'The fresh program reader proves code execution/actions, not broad neural gains. All original16961QA/3088AI2Dtimes3/'
        '600MBPP/4642human/3429wordtimes3and9final endpoints remain incomplete for the requested final model.')
    requirements['human_feedback']['evidence'] = (
        'The original canonical whole human-preference stage has resumed after the complete restored coding study. Both '
        'chosen_CE and human_DPO pass all4actual L40capacity reports:999706442research values,561Adam states,2logical/3physical '
        'capacity updates, exact next-global-update weights/Adam/privateRNG, invalid CUDA RNG rejection and all4original '
        'extreme-source batches. The chosen-response production training workers are now live. Each arm preserves85786public '
        'human-ranked pairs/5362updates; these public response texts include model outputs ranked by humans, not fresh human '
        'judgments on own outputs. Full production completion and held preference/general/code quality remain pending. '
        'New batched-verifier mechanics are machine feedback, not human RLHF.')
    requirements['gpu_runtime']['evidence'] = (
        'All4original L40s now belong to the intact canonical human stage; no current worker was patched/restarted or new '
        'competing lease/context created. A new batched native sampler and strict reasoning-gradient kernel pass15own-small '
        'CPUchecks. A370482parameter35layer graph preserves full-vocabulary private-seed serial actions/logps/RNG with batched '
        'KVhistories;48actions use12batched neural calls versus48serial, without a measured CUDA speedup. Eight attempts '
        'produce161actual actions with live batch8to5and genuine variableEOSstops. Full singleton rescore differs byatmost '
        '4.7684e-7in these examples, within unchanged.001guard. The originalv1NaNcomparison gap is deliberately reproduced '
        'and preserved; v2rejects nonfinite inference/training rescoring with complete JSON-safe observations and clears '
        'partial gradients. No full-size CUDAcapacity, distributed durability or RLquality is inferred.')
    requirements['serious_full_model_training']['evidence'] = (
        'The complete4447update public epoch and3x306actual repair updates remain source/weight/Adam verified. All coding '
        'matched assessments and independent program checks now finish. The original whole human stage passes both4GPU '
        'capacity gates and launches production chosen_CE training, retaining all85786pairs in each chosen/DPOarm. New batched '
        'RLtesting uses an explicitly initialized267334parameter fixture and2CPUmechanics Adam updates with exact next-action/'
        'verifier/probability/unclipped-gradient/weight/Adam/privateRNG replay; these are not production model updates. '
        'Full source suffix qualification/fresh controls, actual production RL and complete4733stepvisual training remain pending.')
    for value in requirements.values():
        value['previous_evidence'] = str(PREVIOUS)
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('All ten requested final requirements remain incomplete')
    names = ['src/slm_lab/batched_own_native_rollouts_v1.py', 'src/slm_lab/batched_own_native_rollouts_v2.py',
        'src/slm_lab/batched_reasoning_group_gradients_v1.py', 'src/slm_lab/batched_reasoning_group_gradients_v2.py',
        'tests/test_batched_own_native_rollouts_v1.py', 'tests/test_batched_own_native_rollouts_v2.py',
        'tests/test_batched_reasoning_group_gradients_v1.py', 'tests/test_batched_reasoning_group_gradients_v2.py',
        'scripts/run_batched_own_native_rollout_tests_v1.py', 'scripts/run_batched_reasoning_group_gradient_tests_v1.py',
        'scripts/run_batched_native_RL_v2_tests.py', 'scripts/probe_batched_rollout_nonfinite_rescore_CPU_v1.py',
        'reports/batched-own-native-rollouts-v1-tests.log', 'reports/batched-reasoning-group-gradients-v1-tests.log',
        'reports/full-public-token-suffix-metadata-readback-CPU-v1/complete.json',
        'scripts/train_phase_human_preference_v8_assessment_recovery_gate_v1.py']
    files = {str(p): sha(p) for p in [Path(__file__), PREVIOUS, final_path, first_path, tests, gap_path,
        human_plan_path, HUMAN / 'capacity.json', *human_proofs, *proofs, *[ROOT / n for n in names],
        STUDY / 'complete.json', STUDY / 'restored.json', *[STUDY / n / 'complete.json' for n in expected]]}
    OUT.mkdir()
    atomic(OUT / 'runtime-snapshot.json', snapshot)
    atomic(OUT / 'full-matched-comparison.json', dict(actual_native_profile=profiles[0],
        all_four_fresh_code_and_source_endpoint_totals=totals, paired_gains_and_losses=paired,
        complete_broad_producer_results=broad, no_arm_improves_total_code_correctness=True,
        no_candidate_promoted=True, all_original_endpoint_populations_and_raw_failures_preserved=True,
        independent_broad_neural_forward_readback_pending=True, goal_achieved=False))
    classification = ('progress:the complete3arm coding study and all192native/96ordinary/60source fresh executions finish; '
        'syntax/termination improve but correctness stays1/48allarms; original human stage passes8actual capacity gates and '
        'starts chosen-response production workers;15new batched native/RLmechanics controls pass, including finite-probability '
        'rejection and exact next-gradient/Adam/RNG replay; no new broad or RLquality gain')
    atomic(OUT / 'complete.json', dict(complete=True, at=at, source_files_sha256=files,
        full_original_GPUcoding_study_complete_and_restored=True,
        all192native96ordinaryPython60source_programs_freshly_executed=True,
        actual_code_verdict_differences=0, actual_combined_code_totals=totals,
        all_three_arms_functional1of48_and_no_promotion=True,
        original_human_stage_all8actual_full_size_capacity_reports_passed=True,
        actual_human_pairs_per_arm=85786, actual_planned_updates_per_arm=5362,
        original_human_production_training_workers_live=True,
        actual_new_batched_sampler_and_gradient_controls=15,
        actual_own_small_sampler_parameters=370482, actual_own_small_gradient_fixture_parameters=267334,
        original_v1_nonfinite_comparison_gap_preserved=True,
        v2nonfinite_inference_and_gradient_rescores_rejected_with_complete_observations=True,
        exact_next_small_RL_actions_unclipped_gradients_weights_Adam_and_private_RNG=True,
        full_size_GPUbatched_sampling_capacity_throughput_distributed_recovery_and_quality_pending=True,
        no_competing_GPUjob_or_existing_worker_change=True,
        new_production_RL_updates=0, all_ten_final_requirements_incomplete=True, goal_achieved=False))
    for p in [OUT / 'complete.json', OUT / 'runtime-snapshot.json', OUT / 'full-matched-comparison.json']:
        files[str(p)] = sha(p)
    atomic(OUT / 'review.json', dict(at=at, requirements=requirements, previous_full_scope_review=str(PREVIOUS),
        previous_goal_turn_classification='progress:all14884human-code suffix views independently reconstructed and full control/readback chain launched',
        current_goal_turn_classification=classification, source_files_sha256=files, goal_achieved=False))
    rows = ['| ' + name + ' | ' + ' | '.join(str(totals[name][k]) for k in
        ['syntax', 'functional', 'fresh_empirical_efficiency', 'ordinary_Python_functional', 'genuine_EOS_stops']) + ' |' for name in expected]
    metrics = ('**The entire three-arm coding experiment is complete and restored.** Independent checks cover every '
        '**192native program,96ordinary Python program and60source-validation program**, with zero verdict differences. '
        'All prompts, actual actions, EOS stops, raw text and program bytes reconstruct under the same512action greedyBF16profile.\n\n'
        '| Endpoint | Syntax/48 | Correct/48 | Efficient/48 | Ordinary Python/24 | EOS/48 |\n'
        '|---|---:|---:|---:|---:|---:|\n' + '\n'.join(rows) + '\n\n'
        '**No arm improves total coding correctness.** Each trained arm gains Python completion`sum_even` and loses '
        'JavaScript instruction`lower_bound`. Higher syntax and more EOS stops do not establish algorithmic understanding. '
        'All variants remain unpromoted. Source-example passes0/1/0/1of15have public answers in their prompts and are not '
        'certified independent historical benchmarks.\n\n'
        'Complete broad producer results show knowledge2/1/2/2of56, math0each, OCR0each withCER '
        '9.879167/11.15625/13.370833/11.8, objects0each and unchanged photo retrieval1/32I2Tand2/32T2I. '
        '**These broad scores have no new independent neural forward readback and establish no broad-quality gain.**\n\n'
        '**The original canonical human-preference run has resumed on all4L40s.** All8actual chosen-CE/DPOcapacity '
        'reports pass at999706442research values/561Adam leaves, with full extreme-source batches, exact next-update '
        'weights/Adam/privateRNG and invalid CUDA-RNG rejection. Chosen-response production workers are live. Each arm '
        'retains all85786public human-ranked pairs/5362updates. The response texts include model outputs ranked by humans; '
        'these are not fresh human judgments on this student. Full training and held-out quality remain pending.\n\n'
        '**15new batched sampling/RLchecks pass.** A370482parameter35layer CPUgraph generates4private-seed attempts '
        'with the same actions/logps/generator states as independent serial sampling:48actual actions use12batched '
        'native neural calls versus48serial. An8attempt case preserves161actions while finished histories shrink the '
        'batch8→5. Cached/full selected logps differ byatmost4.7684e-7in these traces; the.001guard is unchanged. '
        '**No full-size CUDA speedup is measured.**\n\n'
        'A deliberately initialized267334parameter fixture connects actual strict public train rewards to the batched '
        'gradient kernel. Identical sampled groups/CEanchors give different control/RLgradients when rewards vary; '
        'all-zero groups keep exact control gradients and unconditional CE. Exact next-update recovery now also hashes '
        'every unclipped actor gradient, alongside all actions/verifiers/probabilities/weights/Adam/privateRNG. '
        '**These are small mechanics, not production student performance or full-size training.**\n\n'
        'Fault injection found a realv1comparison gap:NaNfull-forward logps could report a zero maximum difference. '
        'The original source and reproduction are preserved. **V2rejects nonfinite inference and training rescoring**, '
        'keeps every generated attempt and explicit JSON-safe nonfinite observations, and clears partial gradients '
        'before any optimizer commit. No production worker used this prototype or was patched.\n\n'
        'The full human-code suffix qualification/fresh execution chain and visual pixel/readback chain continue. '
        'The cached visual GPUv2runner remains unlaunched. All17public families,85786human pairs, complete RL/visual '
        'budgets and9final endpoints remain. **All ten requested final requirements remain incomplete.**\n')
    (OUT / 'metrics.md').write_text(metrics)
    audit_path = ROOT / 'reports/completion-audit.json'
    audit = read(audit_path)
    audit.update(at=at, updated_at=at, requirements=requirements, goal_achieved=False,
        latest_full_scope_review=str(OUT / 'review.json'), latest_actual_complete_repair_and_batched_RL=str(OUT / 'complete.json'),
        next_batched_native_sampler=str(ROOT / 'src/slm_lab/batched_own_native_rollouts_v2.py'),
        next_batched_reasoning_gradient_kernel=str(ROOT / 'src/slm_lab/batched_reasoning_group_gradients_v2.py'),
        current_goal_turn_classification=classification)
    atomic(audit_path, audit)
    entry = (f'Completed coding study and batched RLmechanics, **{at}**: all3coding arms and all192native/96ordinaryPython/'
        '60source fresh executions complete. Syntax14/21/30/25of48improves; **functional correctness stays1/48at every endpoint** '
        'and none is promoted. The original human-preference stage passes all8actual L40capacity reports and launches '
        'chosen-response production workers for the full85786pairs/arm.15new own-small batched-sampler/RLchecks pass, '
        'including preserved variableEOS/privateRNG, actual strict rewards, exact next-gradient/Adam recovery and explicit '
        'NaNrejection. Full-size CUDAcapacity/speed/RLquality remain unproved. **All ten final requirements remain incomplete.** '
        '[Evidence](reports/complete-repair-study-and-batched-RL-v2-review/metrics.md), '
        '[full review](reports/complete-repair-study-and-batched-RL-v2-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        path.write_text(entry + path.read_text())
    print(json.dumps(dict(review=str(OUT / 'review.json'), complete_code_study=True,
        all_arms_functional=1, actual_new_controls=15, human_production_workers_live=True, goal_achieved=False)))


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('CUDA-hidden nice19 source-only review required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    main()
