"""Fresh complete remaining-arm checks after the original full study closes.

These are the predeclared baseline and first arm, never a selected winner. All
of their actual producer births must be terminal and full endpoint receipts
present. Other profiles, queues, neural workers and GPUleases stay untouched.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/full-function-repair-comparison-GPU-v2'
DEST = ROOT / 'checkpoints/full-function-repair-comparison-GPU-v2'
OUT = ROOT / 'reports/remaining-GPU-repair-native-program-readback-CPU-v1'
WAIT = ROOT / 'reports/remaining-GPU-repair-native-program-readback-CPU-v1-wait'
FIRST = ROOT / 'reports/first-GPU-repair-native-program-readback-CPU-v1'
PROFILES = ('semantic_repair', 'weighted_semantic_repair')
PARENT = dict(pid=101066, ticks='54137054')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lines(path):
    with Path(path).open('rb') as stream:
        result = []
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('Closed complete LF action/program records are required')
            result.append(json.loads(line))
    return result


def main():
    if (OUT.exists() or WAIT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden one-thread source-only readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    # Pure source/control waiting. Do not construct a model, initialize CUDA,
    # acquire a GPUlease or import torch while the exact producer is live.
    import time
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from record_visual_stage_and_first_repair_arm_v1 import atomic
    first = read(FIRST / 'complete.json')
    if not first['passed'] or first['actual_native_verdict_differences'] or first['actual_source_verdict_differences']:
        raise ValueError('The full independently verified baseline/first-arm evidence is required unchanged')
    original = read(SOURCE / 'plan.json')
    if {k: original['owner'][k] for k in PARENT} != PARENT:
        raise ValueError('The exact actual original study birth is required')
    WAIT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    bound = {str(p): sha(p) for p in [Path(__file__), SOURCE / 'plan.json', FIRST / 'complete.json',
        FIRST / 'all96-fresh-native-and48-ordinary-programs.jsonl', FIRST / 'all30-fresh-source-validation-programs.jsonl']}
    atomic(WAIT / 'plan.json', dict(owner=owner, exact_original_producer=PARENT, source_files_sha256=bound,
        original_remaining_profiles=list(PROFILES), no_model_GPUlease_or_neural_generation=True, automatic_retry=False))
    expires = time.monotonic() + 7 * 86400
    try:
        while not terminal(PARENT):
            if time.monotonic() > expires or any(sha(p) != h for p, h in bound.items()):
                raise ValueError('Original source-only waiting bindings or declared deadline changed')
            atomic(WAIT / 'progress.json', dict(owner=owner, exact_original_producer_live=True,
                phase='waiting for original complete four-endpoint study and restoration', neural_forwards=0, GPUleases=0))
            time.sleep(15)
        if ((SOURCE / 'failure.json').exists() or not read(SOURCE / 'complete.json')['all_actual_GPU_capacity_and_three_matched_arms_complete']
                or not read(SOURCE / 'restored.json')['all_registered_point_actors_terminal']
                or any(not terminal(x) for x in read(SOURCE / 'children.json'))):
            raise ValueError('All original producer work must complete and restore before final readback')
        if any(sha(p) != h for p, h in bound.items()):
            raise ValueError('All original immutable sources and first complete readback must remain unchanged')
    except BaseException as exc:
        atomic(WAIT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False))
        raise
    from tokenizers import Tokenizer
    from slm_lab.bounded_disk_reserve_wait_v1 import wait_for_reserve
    from slm_lab.code_quality import execute, efficiency
    from slm_lab.code_tasks import TASKS, request
    from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
    from slm_lab.device_native_code_action_readback_v1 import local, source
    from slm_lab.diverse_images import atomic_json
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.native_code_shards_v1 import reconcile, positions
    from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute
    from slm_lab.public_function_validation_observation_v1 import observe
    tests = ROOT / 'reports/device-native-code-action-readback-v1-tests.log'
    if '13 passed' not in tests.read_text():
        raise ValueError('The exact real device-observer protocol and rejection controls must pass first')
    plan = read(SOURCE / 'plan.json')
    if {k: plan['owner'][k] for k in PARENT} != PARENT:
        raise ValueError('The exact original actual four-L40study is required')
    actual = [x for x in read(SOURCE / 'children.json') if any(x['subject'].startswith('assess-' + p + '-') for p in PROFILES)]
    expected = {'assess-' + p + '-' + s + '-' + str(r) for p in PROFILES for s in ['source', 'code', 'broad'] for r in range(4)}
    if len(actual) != 24 or {x['subject'] for x in actual} != expected or any(not terminal(x) for x in actual):
        raise ValueError('Every actual producer of these two complete endpoints must be terminal, without touching other arms')
    if any(not read(SOURCE / p / 'complete.json')['complete'] for p in PROFILES):
        raise ValueError('Both complete full local endpoints must exist; never infer completion from logs')
    training = ROOT / 'reports/all-three-GPU-repair-training-readback-CPU-v1/complete.json'
    if not read(training)['passed']:
        raise ValueError('Original complete training-source/weights/Adam readback is required')
    cohort_path = ROOT / 'reports/source-function-validation-runtime-CPU-v1/all15-fixed-native-validation-cases.json'
    cohort = read(cohort_path)
    if len(cohort) != 15 or [c['position'] for c in cohort] != list(range(15)):
        raise ValueError('All15source-validation functions must remain present')
    inputs = read(SOURCE / 'native-assessment-inputs.json')
    paths = [Path(__file__), tests, SOURCE / 'plan.json', SOURCE / 'native-assessment-inputs.json', training, cohort_path,
        SOURCE / 'complete.json', SOURCE / 'restored.json', FIRST / 'complete.json',
        FIRST / 'all96-fresh-native-and48-ordinary-programs.jsonl', FIRST / 'all30-fresh-source-validation-programs.jsonl',
        DEST / 'tokenizer.json', *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_device_native_code_action_readback_v1.py'])]
    for profile in PROFILES:
        paths.extend(p for p in (SOURCE / profile).rglob('*') if p.is_file())
        checkpoint = DEST / (profile + '-actor.pt')
        if sha(checkpoint) != inputs[profile]['checkpoint_sha256']:
            raise ValueError('The complete actual exported actor must match every original report')
        paths.append(checkpoint)
    files = {str(p.resolve()): sha(p) for p in paths}
    states = {p: (Path(p).stat().st_dev, Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    def guard():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any(
                (Path(p).stat().st_dev, Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != s
                for p, s in states.items()):
            raise ValueError('All closed endpoints/source code and own idle scheduling must stay unchanged')
    def reserve():
        wait_for_reserve(ROOT, minimum_bytes=40 * 1024**3, guard=guard)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic_json(OUT / 'closed-producer-births.json', dict(actual_original_parent=PARENT,
        actual24closed_endpoint_producer_births=actual, every_closed_endpoint_producer_terminal=True,
        other_profile_producers_and_current_neural_jobs_unchanged=True))
    files[str(OUT / 'closed-producer-births.json')] = sha(OUT / 'closed-producer-births.json')
    states.update({p: (Path(p).stat().st_dev, Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files})
    atomic_json(OUT / 'plan.json', dict(owner=owner, source_files_sha256=files, fixed_original_profiles=list(PROFILES),
        complete_native_programs=96, additional_ordinary_Python_programs=48, complete_source_validation_programs=30,
        every_failure_and_verdict_difference_preserved=True, no_new_generation_or_program_repair=True,
        source_role='evaluation', optimization_forbidden=True, models_constructed=0, GPUleases=0, goal_achieved=False))
    try:
        token = Tokenizer.from_file(str(DEST / 'tokenizer.json'))
        jobs = []
        for profile in PROFILES:
            for frame in ['completion', 'instruction']:
                merged = reconcile([SOURCE / profile / f'code-{frame}-shard-{r}.json' for r in range(4)], OUT / f'{profile}-{frame}-original.json')
                if merged['task_count'] != 24 or merged['checkpoint_sha256'] != inputs[profile]['checkpoint_sha256']:
                    raise ValueError('Every original local task, source profile and actual checkpoint must match')
                original = read(SOURCE / profile / f'code-{frame}.json')
                if any(merged[k] != original[k] for k in ['results', 'generation_budget', 'decoding', 'suite_sha256',
                        'tokenizer_sha256', 'inference_device_type', 'inference_precision']):
                    raise ValueError('Fresh whole-shard reconstruction must equal the complete original endpoint')
                for rank in range(4):
                    events = lines(SOURCE / profile / f'code-{frame}-actions-{rank}.jsonl')
                    rows = read(SOURCE / profile / f'code-{frame}-shard-{rank}.raw.json')['results']
                    if len(rows) != len(events) or len(rows) != 2 * len(positions(rank, 4)):
                        raise ValueError('Every original per-frame native delivery must have exactly one original action event')
                    for i, (row, event) in enumerate(zip(rows, events)):
                        jobs.append(dict(profile=profile, frame=frame, case=row,
                            action_proof=local(row, frame, event, token, case_position=i)))
        if len(jobs) != 96:
            raise ValueError('All96native code programs must reconstruct before fresh execution')
        reserve()
        controls = {(language, task[0]): execute(request(task, language, code, True))
            for task in TASKS for language, code in [('Python', task[4]), ('JavaScript', task[5])]}
        controls_passed = all(v.get('functional_passed') and v.get('efficiency_completed') for v in controls.values())
        atomic_json(OUT / 'all24-fresh-reference-controls.json', dict(results=[dict(language=k[0], task=k[1], execution=v) for k, v in controls.items()]))
        rows, differences = [], []
        with (OUT / 'all96-fresh-native-and48-ordinary-programs.jsonl').open('x') as journal:
            for i, job in enumerate(jobs):
                if i % 8 == 0:
                    reserve()
                case = job['case']
                task, code = reconstruct(case, job['frame'])
                payload = request(task, case['language'], code)
                actual = execute(payload)
                changed = any(bool(actual.get(k)) != bool(case['candidate'].get(k)) for k in ['syntax_valid', 'functional_passed'])
                if changed:
                    differences.append(dict(profile=job['profile'], frame=job['frame'], task=task[0],
                        language=case['language'], producer=case['candidate'], fresh=actual))
                reference = controls[(case['language'], task[0])]
                efficient, measurement = efficiency(actual, reference, task[3]) if reference.get('efficiency_completed') else (False, {})
                ordinary = ordinary_execute(code, task[0], payload['cases']) if case['language'] == 'Python' else None
                row = dict(profile=job['profile'], frame=job['frame'], language=case['language'], task=task[0],
                    actual_original_program=code, action_proof=job['action_proof'], fresh_native_execution=actual,
                    fresh_ordinary_Python=ordinary, fresh_empirical_efficiency=efficient,
                    fresh_efficiency_measurement=measurement, original_empirical_efficiency=case['efficiency_stress_passed'],
                    actual_original_case_count=len(payload['cases']), original_verdict_difference=changed,
                    source_role='evaluation', optimization_forbidden=True)
                rows.append(row)
                journal.write(json.dumps(row, ensure_ascii=True) + '\n')
                journal.flush()
                atomic_json(OUT / 'progress.json', dict(owner=owner, phase='fresh whole native programs',
                    complete_programs=len(rows), expected_programs=96, actual_verdict_differences=len(differences),
                    neural_forwards=0))
            os.fsync(journal.fileno())
        sources, source_differences = [], []
        with (OUT / 'all30-fresh-source-validation-programs.jsonl').open('x') as journal:
            for profile in PROFILES:
                for rank in range(4):
                    raw = read(SOURCE / profile / f'source-validation-shard-{rank}.json')
                    events = lines(SOURCE / profile / f'source-validation-actions-{rank}.jsonl')
                    programs = lines(SOURCE / profile / f'source-validation-programs-{rank}.jsonl')
                    if programs != raw['results'] or len(programs) != len(events):
                        raise ValueError('Every closed source action/program journal must match its complete original shard')
                    for i, (row, event) in enumerate(zip(programs, events)):
                        reserve()
                        case = cohort[row['position']]
                        action = source(row, case, event, token, case_position=i)
                        actual = observe(case['original_source_row'], row['program'])
                        changed = any(actual[k] != row['public_execution'][k] for k in ['syntax_valid', 'all_original_examples_passed', 'passed_examples'])
                        if changed:
                            source_differences.append(dict(profile=profile, position=row['position'],
                                producer=row['public_execution'], fresh=actual))
                        result = dict(profile=profile, position=row['position'], action_proof=action,
                            actual_original_program=row['program'], fresh_execution=actual, verdict_difference=changed,
                            source_role='validation', optimization_forbidden=True)
                        sources.append(result)
                        journal.write(json.dumps(result, ensure_ascii=True) + '\n')
                        journal.flush()
            os.fsync(journal.fileno())
        if len(sources) != 30:
            raise ValueError('All30original source validation programs must be reexecuted')
        totals = {}
        for profile in PROFILES:
            group = [r for r in rows if r['profile'] == profile]
            source_group = [r for r in sources if r['profile'] == profile]
            totals[profile] = dict(native_code_programs=len(group), syntax=sum(bool(r['fresh_native_execution'].get('syntax_valid')) for r in group),
                functional=sum(bool(r['fresh_native_execution'].get('functional_passed')) for r in group),
                fresh_empirical_efficiency=sum(bool(r['fresh_empirical_efficiency']) for r in group),
                ordinary_Python_programs=sum(r['language'] == 'Python' for r in group),
                ordinary_Python_functional=sum(bool(r['fresh_ordinary_Python']['functional_passed']) for r in group if r['language'] == 'Python'),
                genuine_EOS_stops=sum(r['action_proof']['actual_stopped_on_EOS'] for r in group),
                passing_tasks=[dict(language=r['language'], task=r['task'], frame=r['frame']) for r in group if r['fresh_native_execution'].get('functional_passed')],
                source_validation_programs=len(source_group), source_syntax=sum(r['fresh_execution']['syntax_valid'] for r in source_group),
                source_all_original_examples_pass=sum(r['fresh_execution']['all_original_examples_passed'] for r in source_group))
        reserve()
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError('All closed original source/action/code/runtime bytes must remain unchanged')
        passed = controls_passed and not differences and not source_differences
        atomic_json(OUT / 'complete.json', dict(complete=True, passed=passed, at=datetime.now(timezone.utc).isoformat(),
            owner=owner, source_files_sha256=files, actual_fixed_endpoint_totals=totals,
            actual_combined_all_four_endpoint_totals=dict(first['actual_fixed_endpoint_totals'], **totals),
            all192native_and96ordinary_Python_and60source_programs_freshly_executed_across_both_closed_readbacks=True,
            original_baseline_and_first_arm_closed_readback_reused_without_rerunning_or_selecting=True,
            actual24fresh_reference_controls_passed=controls_passed,
            all96actual_prompts_actions_raw_text_and_programs_reconstructed=True,
            all96native_and48ordinary_Python_programs_freshly_executed=True,
            all30source_validation_programs_freshly_executed=True,
            actual_native_verdict_differences=differences, actual_source_verdict_differences=source_differences,
            all_original_failures_and_fresh_differences_preserved=True,
            source_validation_original_public_answers_visible_in_prompt=True,
            empirical_stress_timings_not_asymptotic_proof=True, no_action_probabilities_invented=True,
            every_original_matched_arm_and_failure_preserved=True,
            complete_three_arm_code_program_readback_finished=True,
            independent_broad_neural_score_readbacks_still_pending=True,
            source_role='evaluation', optimization_forbidden=True, neural_forwards=0, GPUleases=0,
            new_production_optimizer_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, passed=passed, actual_fixed_endpoint_totals=totals)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
