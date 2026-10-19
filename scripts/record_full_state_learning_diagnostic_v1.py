"""Preserve full scope and report the actual state-learning diagnostic honestly."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import observe, terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-state-learning-diagnostic-v1-review'
NAMES = (
    'code-first-continuation-v2', 'phase-human-preference-v6',
    'broad-anchored-code-v5', 'code-first-final-standalone-CPU-v2',
    'full-held-QA-pipeline-v3', 'full-AI2D-pipeline-v3',
    'full-original-MBPP-pipeline-v3', 'AI2D-fixed-exposure-strata-v3',
)


def read(path):
    return json.loads(Path(path).read_text())


def live(identity):
    fields = (Path('/proc') / str(identity['pid']) / 'stat').read_text().split(') ', 1)[1].split()
    if fields[19] != identity['ticks'] or fields[0] in {'Z', 'X'}:
        raise ValueError('The same original live process birth is required for this snapshot')
    return dict(pid=identity['pid'], ticks=fields[19], state=fields[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CPU-hidden full-scope review is required')
    master_path = ROOT / 'reports/completion-audit.json'
    master = read(master_path)
    previous_path = ROOT / master['latest_full_scope_review']
    prior = read(previous_path)
    requirements = deepcopy(prior['requirements'])
    proof_path = ROOT / 'reports/full-state-backward-CPU-v1/complete.json'
    reader_path = ROOT / 'reports/full-state-backward-CPU-v1-readback/complete.json'
    diagnostic_path = ROOT / 'reports/state-update-direction-CPU-v1/complete.json'
    figure_path = ROOT / 'reports/fixed-code-metrics-v1/complete.json'
    proof, reader, diagnostic, figure = map(read, [proof_path, reader_path, diagnostic_path, figure_path])
    proof_plan = read(proof_path.parent / 'plan.json')
    diagnostic_plan = read(diagnostic_path.parent / 'plan.json')
    if not terminal(proof_plan['owner']) or not terminal(diagnostic_plan['owner']):
        raise ValueError('Both actual full-size CPU experiments must finish before this final review')
    if (not proof['complete'] or not reader['complete'] or not diagnostic['complete']
            or proof['actual_logical_updates'] != 2 or proof['actual_active_language_tensors'] != 317
            or reader['actual_changed_language_tensors'] != 317
            or reader['actual_fixed_vision_word_critic_auxiliary_tensors'] != 144
            or reader['actual_unique_native_values'] != 984399690
            or not proof['actual_in_memory_next_update_weights_Adam_and_private_CPU_RNG_exact']
            or not proof['every_exported_parameter_and_whole4095_logits_and_real_word_CTC_cold_load_exact']
            or not diagnostic['exact_original_short_update_and_long_gradient_and_warm_final_parameters_reproduced']
            or diagnostic['before_source_NLL'] != proof['longest_source_pre_update_NLL']):
        raise ValueError('Complete actual full-model rehearsal, independent tensor reader and matched diagnostic required')
    observations = diagnostic['actual_observations']
    if ([row['kind'] for row in observations] != ['warm_Adam_full_step',
            'scaled_actual_warm_parameter_displacement', 'scaled_actual_warm_parameter_displacement',
            'fresh_Adam_same_pre_update_weights_and_gradient']
            or [row['scale'] for row in observations] != [1., .25, .0625, 1.]
            or observations[0]['source_NLL'] != proof['longest_source_post_update_NLL']):
        raise ValueError('Every predeclared actual diagnostic control must remain present')
    test_path = ROOT / 'reports/state-backward-rehearsal-v1-final-tests.log'
    if '1 passed' not in test_path.read_text():
        raise ValueError('The actual small-network recovery and fixed-head control must pass')
    main_path = ROOT / 'reports/phase-joint-v3/plan.json'
    main_plan = read(main_path)
    progress = read(main_path.parent / 'progress.json')
    main_owner = live(main_plan['owner'])
    workers = [row for row in read(main_path.parent / 'children.json')
               if row['subject'].startswith('train-rank-')]
    if len(workers) != 4:
        raise ValueError('All four original main workers are required for this live snapshot')
    worker_observations = [live(row) for row in workers]
    queues = {}
    protected = dict(main_plan['source_files_sha256'])
    for name in NAMES:
        path = ROOT / 'reports' / (name + '-queue.json')
        queue = read(path)
        queues[name] = dict(owner=observe(queue['owner']), queue_sha256=file_hash(path))
        protected.update(queue['source_files_sha256'])
        if (ROOT / 'reports' / name).exists() and name != 'AI2D-fixed-exposure-strata-v3':
            raise ValueError('This review expects the replacement production controllers still waiting')
    protected.update(proof_plan['source_files_sha256'])
    protected.update(diagnostic_plan['source_files_sha256'])
    original_jobs = {}
    for name in (
        'CodeContests-full-Python-references-v1', 'CodeContests-public-execution-traces-candidate-v1',
        'CodeContests-reference-outputs-v1-readback', 'public-execution-targets-v1-readback-admission',
        'public-state-traces-full-v1', 'public-state-traces-full-v1-readback-admission',
    ):
        folder = ROOT / 'reports' / name
        plan = read(folder / 'plan.json')
        done = read(folder / 'complete.json')
        if not terminal(plan['owner']) or (folder / 'failure.json').exists():
            raise ValueError('Every original public-source job must remain terminal and complete')
        protected.update(plan['source_files_sha256'])
        original_jobs[name] = dict(owner=plan['owner'], terminal=True, complete=True,
                                   complete_sha256=file_hash(folder / 'complete.json'))
    metric_paths = sorted((main_path.parent).glob('held-packed-*.json'),
                          key=lambda p: 0 if p.stem.endswith('initial') else int(p.stem.rsplit('-', 1)[1]))
    latest_path = metric_paths[-1]
    latest = read(latest_path)
    foundation = read(main_path.parent / 'held-packed-12853.json')
    if any(latest[key] != foundation[key] for key in (
            'selected_original_positions', 'complete_causal_targets', 'original_manifest_sha256')):
        raise ValueError('The latest code-retention comparison requires the same fixed cohort')
    at = datetime.now(timezone.utc).isoformat()
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used',
                                   '--format=csv,noheader'], text=True)
    gpu_processes = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader',
    ], text=True)
    if {int(line.split(',', 1)[0]) for line in gpu_processes.splitlines() if line.strip()} != {
            row['pid'] for row in workers}:
        raise ValueError('Only the four original main workers may own GPU compute in this snapshot')
    runtime = dict(at=at, actual_main_updates=progress['actual_updates'], planned_main_updates=35284,
                   original_main_owner=main_owner, original_four_workers=worker_observations,
                   actual_GPU_observation=gpu, waiting_controllers=queues,
                   actual_GPU_process_observation=gpu_processes,
                   original_public_source_jobs=original_jobs, new_code_first_production_updates=0)
    finding = dict(
        actual_full_student_values=984399690, partial_own_main_source_step=28672,
        actual_experimental_state_updates=2, actual_physical_optimizer_calls=3,
        actual_changed_language_tensors=317, actual_fixed_other_tensors=144,
        whole_state_context_positions=[354, 4095], whole_state_response_EOS_targets=[69, 3609],
        in_memory_next_update_weights_Adam_and_private_CPU_RNG_exact=True,
        durable_Adam_recovery_not_certified=True,
        all_parameter_and_whole4095_logits_and_real_word_CTC_cold_load_exact=True,
        independent_reader_checks_stored_parameters_and_source_masks_only=True,
        same_source_NLL_before=proof['longest_source_pre_update_NLL'],
        same_source_NLL_after=proof['longest_source_post_update_NLL'],
        actual_direction_controls=observations,
        warm_step_non_descent_for_measured_combined_gradient=diagnostic[
            'warm_step_is_non_descent_for_measured_current_combined_gradient'],
        source_loss_not_generated_or_held_quality=True,
        full_peak_LR_microbatch1_test_not_production_batch16_warmup_ablation=True,
        production_recipe_unchanged=True, production_state_updates=0,
        graph_completed_metric_points=figure['actual_completed_metric_points'],
        graph_last_metric_step=figure['last_main_token_loss_update'],
        latest_fixed_retention_step=int(latest_path.stem.rsplit('-', 1)[1]),
        latest_fixed_retention_NLL=latest['token_mean_NLL'],
        foundation_fixed_retention_NLL=foundation['token_mean_NLL'],
        latest_native_functional_result_is_older_checkpoints_0_of48_each=True,
        measured_held_capability_gain=False, goal_achieved=False,
    )
    finding['observations_support_overshoot_in_this_rehearsal'] = (
        observations[0]['clipped_combined_gradient_dot_displacement'] < 0
        and observations[0]['source_NLL'] > diagnostic['before_source_NLL']
        and observations[1]['source_NLL'] < diagnostic['before_source_NLL']
    )
    finding['interpretation_limits'] = (
        'A negative combined-gradient dot displacement and lower source NLL at a smaller '
        'actual displacement support overshoot in this particular two-example rehearsal. '
        'Scaled weights are not claimed bit-identical to changing Adam learning rate. '
        'Fresh Adam resets both moments. No held quality gain or production recipe change follows.'
        if finding['observations_support_overshoot_in_this_rehearsal'] else
        'The complete observations are retained without selecting a better production recipe from this source example.'
    )
    notes = {
        'own_architecture': ' The execution-state auxiliary task now passes an actual984MCPU backward/recovery/cold-export rehearsal on an isolated partial own step28672 copy, with0new parameters. It changes all317language tensors while144vision/word/critic/auxiliary tensors remain exact. This is training-mechanics evidence, not an architectural quality gain or a new production initializer.',
        'public_data': ' All1310separately admitted public state examples remain in the7558update queued source schedule. The rehearsal actually uses two preselected source positions949and675, of354and4095context positions and69and3609response/EOS targets. It is not a completed1310source training epoch, and no private/generated case enters the state target.',
        'under_1b_parameters': ' The actual retained rehearsal standalone has984399690unique finite values across461full-shape tensors and one stored tied vocabulary; an independent static reader verifies these counts and every changed/fixed tensor. Final production standalone still waits for complete learning and its own fold/cold checks.',
        'text_and_images': ' All144nonlanguage tensors, including vision and word paths, remain byte exact in the actual text-state rehearsal. Its whole4095token outputs and real-word CTC cold-load exactly, and CTC is unchanged from its source. This verifies preservation and serialization, not visual or reading quality improvement. Full final photo/diagram/word assessments remain required.',
        'human_feedback': ' Human chosenCE/DPO and code executionRL retain the applied state-inclusive predecessor gates, all85786human training pairs per arm,4642held pairs and3721code sources per arm. No new human/DPO/RL production updates were made by the CPU experiment. The new public-state pack remains CE-only.',
        'serious_full_model_training': f' Current original four-L40 main is at{progress["actual_updates"]}/35284updates in this snapshot; all eight applied code-first/state/human/code/final/assessment controllers still wait. The CPU rehearsal performs2logical/3physical optimizer calls, exact in-memory next-update weights/Adam/privateCPU-RNG recovery, corrupt-source/RNG rejection before mutation, and actual full-model cold export. No durableAdam snapshot is claimed and no production recipe was changed.',
        'high_end_llm_capability': f' The actual same-source rehearsal NLL rises from{proof["longest_source_pre_update_NLL"]}to{proof["longest_source_post_update_NLL"]}. The matched control reproduces every original final language tensor before evaluating two predeclared scaled displacements and a freshAdam control. This is not an improvement claim, and two peakLR single-example steps do not establish behavior of the queued batch16warmup recipe. High-end parity remains unsupported.',
        'broad_independent_benchmarks': ' All eight fixed endpoints and complete16961QA/3088AI2Dtimes3conditions/600originalMBPP evaluations remain queued unchanged. A standalone PNG/SVG/CSV figure records17completed fixed packed-code checks through30720and the two older matched native-code endpoints. Token retention, source-training NLL, syntax and functional correctness remain explicitly separate; no new all-domain or final-model score exists.',
        'gpu_runtime': ' The same original main owner and four workers remain live, all eight replacement controllers remain CUDA-hidden and childless, and all original public-source jobs are terminal complete. The isolated rehearsal and direction controls use2CPUthreads and initialize noCUDA. Fresh four-L40 full-size capacity/recovery gates still precede production code-first, human and code learning.',
        'any_code_correctness_and_efficiency': f' Latest fixed packed-code NLL at{latest_path.stem.rsplit("-",1)[1]}is{latest["token_mean_NLL"]}, versus{foundation["token_mean_NLL"]}atfoundation12853on exactly128windows/524160targets. Retention has worsened during the instruction/vision phase; the applied complete code-first continuation includes raw-code and education replay to address this. Latest matched generated-code results remain the older12853/14336checkpoints: syntax20/48to36/48, functional0/48andempiricalefficiency0/48atboth. No new correctness or efficiency gain is claimed.',
    }
    for key, note in notes.items():
        requirements[key]['evidence'] += note
    requirements['serious_full_model_training']['evidence'] += (
        ' Full direction observations: ' + json.dumps(observations, separators=(',', ':'))
        + '. ' + finding['interpretation_limits']
    )
    if len(requirements) != 10 or any(row['status'] != 'incomplete_for_requested_final_model'
                                      for row in requirements.values()):
        raise ValueError('All ten original final requirements must remain explicitly incomplete')
    OUT.mkdir()
    atomic_json(OUT / 'findings.json', finding)
    evidence = [Path(__file__), previous_path, proof_path, reader_path, diagnostic_path, figure_path,
                test_path, latest_path, OUT / 'findings.json']
    atomic_json(OUT / 'review.json', dict(
        at=at, requirements=requirements, previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        actual_runtime=runtime, current_findings=finding,
        source_files_sha256={str(path): file_hash(path) for path in evidence},
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True, no_original_scope_removed=True,
        candidate_promoted=False, goal_achieved=False,
    ))
    docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', master_path]
    if any(str(path) in protected for path in docs):
        raise ValueError('Review is complete; protected root documents must remain unchanged')
    # Dated documentation digests record mutable historical snapshots. Actual
    # source_files_sha256 bindings, including completed experiments, are fixed.
    bound = subprocess.run([
        'rg', '-l', '--glob', '*.json', '--glob', '*.jsonl',
        r'"(/home/jovyan/slm-lab/)?(README\.md|RESEARCH\.md|reports/completion-audit\.json)"\s*:',
        str(ROOT / 'reports'),
    ], capture_output=True, text=True)
    if bound.returncode not in (0, 1):
        raise ValueError('Historical source-binding search failed')
    def source_bound(node):
        if isinstance(node, dict):
            if 'source_files_sha256' in node:
                for name in node['source_files_sha256']:
                    if (ROOT / name).resolve() in docs:
                        return True
            return any(source_bound(value) for value in node.values())
        return isinstance(node, list) and any(source_bound(value) for value in node)
    for name in bound.stdout.splitlines():
        path = Path(name)
        nodes = (json.loads(line) for line in path.read_text().splitlines()) if path.suffix == '.jsonl' else [read(path)]
        if any(source_bound(node) for node in nodes):
            raise ValueError('A root document is an actual fixed scientific source and must stay unchanged')
    (OUT / 'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at, requirements={key: row['evidence'] for key, row in requirements.items()},
                  latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)),
                  latest_full_size_state_backward=str(proof_path.relative_to(ROOT)),
                  latest_full_size_state_backward_readback=str(reader_path.relative_to(ROOT)),
                  latest_state_update_direction_diagnostic=str(diagnostic_path.relative_to(ROOT)),
                  latest_fixed_code_metric_figure='reports/fixed-code-metrics-v1/code-metrics.png',
                  latest_full_public_state_preparation='reports/public-state-traces-full-v1/complete.json',
                  latest_full_public_state_admission='reports/public-state-traces-full-v1-readback-admission/complete.json',
                  latest_actual_main_runtime=runtime,
                  previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
                  all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (
        f'Full-size state-learning diagnostic, **{at}**: the original four-L40 main continues at '
        f'**{progress["actual_updates"]}/35,284 updates**. An isolated own **984M** partial checkpoint '
        'passes two actual state-source updates, exact in-memory next-update recovery and full-context '
        'cold export. **317 language tensors change; 144 other tensors stay exact.** '
        'An independent static reader checks every stored tensor and both source masks. '
        '**The long-source NLL rises from0.3749to0.4105**; matched direction, scaled-displacement and '
        'fresh-Adam observations are retained in the diagnostic. This two-example peak-learning-rate '
        'test is not an evaluation of the queued batch16warmup recipe. Production recipes and all '
        'eight applied waiting controllers are unchanged.\n\n'
        'The latest matched native coding result remains **0/48 functional passes** at both older '
        'checkpoints; no new quality gain or frontier parity is established. All complete-source, '
        'human/DPO, execution-RL, standalone and full final assessment requirements remain. '
        '**All ten final requirements remain incomplete.** '
        '[Metrics figure](reports/fixed-code-metrics-v1/code-metrics.png), '
        '[actual controls](reports/state-update-direction-CPU-v1/complete.json), '
        '[full review](reports/full-state-learning-diagnostic-v1-review/review.json). '
        'Earlier entries are historical snapshots.\n\n'
    )
    for path in docs[:2]:
        first, rest = path.read_text().split('\n', 1)
        path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), main_updates=progress['actual_updates'],
                          diagnostic_complete=True, measured_held_capability_gain=False,
                          all_ten_final_requirements_incomplete=True, goal_achieved=False)), flush=True)


if __name__ == '__main__':
    main()
