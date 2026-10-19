"""Freshly execute all144 unchanged native SFT/DPO development programs.

Reconcile all original shards, reconstruct code from each retained decoded
response and preserve every fresh verdict. No model is loaded or trained;
the producer did not retain original action IDs/EOS journals for these cases.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

from slm_lab.code_quality import execute, efficiency
from slm_lab.code_tasks import TASKS, request
from slm_lab.decoded_native_code_reexecution_v1 import reconstruct
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.native_code_shards_v1 import reconcile
from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/phase-human-preference-v8'
RESERVE = 40 * 1024**3
MAX_OUTPUT = 64 * 1024**2
ARMS = ['initial', 'chosen_CE', 'human_DPO']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main(output):
    output = Path(output).absolute()
    if (output.exists() or output.resolve() != output or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(output.parent).free < RESERVE + MAX_OUTPUT):
        raise ValueError('A fresh funded one-thread CUDA-hidden CPU assessment readback is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    plan = read(SOURCE / 'plan.json'); done = read(SOURCE / 'complete.json')
    if (not terminal(plan['owner']) or (SOURCE / 'failure.json').exists()
            or done.get('all_held_human_pairs_and_native_development_assessments_complete') is not True
            or done.get('actual_source_epoch_updates_per_arm') != 5362):
        raise ValueError('The exact full original SFT/DPO producer and all development assessments must finish successfully')
    files = {str(p): sha(p) for p in [SOURCE / 'plan.json', SOURCE / 'complete.json', *python_sources(ROOT, [Path(__file__)])]}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    output.mkdir()

    def write(name, value):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        if (sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) + len(raw) > MAX_OUTPUT
                or shutil.disk_usage(output).free - len(raw) < RESERVE):
            raise ValueError('Preserve every complete program/verdict and the40GiBactual output-filesystem reserve')
        pending = output / (name + '.pending')
        with pending.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        pending.replace(output / name)

    write('plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_original_parent=plan['owner'], actual_original_programs=144, secondary_ordinary_Python_programs=72,
        original_frames=['completion', 'instruction'], original_languages=['Python', 'JavaScript'], original_generation_budget=512,
        no_new_generation_or_program_repair=True, original_action_IDs_and_EOS_journals_unavailable=True,
        source_role='evaluation', optimization_forbidden=True, production_updates=0, goal_achieved=False))
    try:
        jobs = []; profiles = []
        for arm in ARMS:
            (output / arm).mkdir()
            for frame in ['completion', 'instruction']:
                report = reconcile([SOURCE / arm / ('code-' + frame + '-shard-' + str(rank) + '.json') for rank in range(4)],
                    output / arm / ('all-original-code-' + frame + '.json'))
                if report['task_count'] != 24 or report['tokenizer_sha256'] != plan['tokenizer_sha256']:
                    raise ValueError('Every complete original development task/frame and own vocabulary must remain')
                files.update(report['source_files_sha256'])
                profiles.append({k: report[k] for k in ['generation_budget', 'decoding', 'suite_sha256', 'tokenizer_sha256', 'inference_device_type', 'inference_precision']})
                jobs.extend(dict(arm=arm, frame=frame, case=case) for case in report['results'])
        if len(jobs) != 144 or any(p != profiles[0] for p in profiles):
            raise ValueError('All144original programs must share the matched complete generation/runtime protocol')
        write('closed-source-bindings.json', dict(source_files_sha256=files, actual_matched_generation_profile=profiles[0],
            original_programs=144, all_original_shards_and_decoded_deliveries_retained=True))

        def guard():
            if (os.sched_getscheduler(0) != os.SCHED_IDLE or shutil.disk_usage(output).free < RESERVE
                    or any(sha(p) != h for p, h in files.items())):
                raise ValueError('Every closed source/program/runtime byte and output reserve must remain exact')

        guard(); controls = {}
        for task in TASKS:
            for language, code in [('Python', task[4]), ('JavaScript', task[5])]:
                controls[(language, task[0])] = execute(request(task, language, code, True))
        write('all24-fresh-reference-controls.json', dict(results=[dict(language=k[0], task=k[1], execution=v) for k, v in controls.items()]))
        controls_passed = all(v.get('functional_passed') and v.get('efficiency_completed') for v in controls.values())
        rows = []; differences = []
        with (output / 'all144-original-programs-fresh-executions.jsonl').open('x') as stream:
            for index, job in enumerate(jobs):
                if index % 8 == 0:
                    guard()
                case = job['case']; task, code = reconstruct(case, job['frame'])
                payload = request(task, case['language'], code); actual = execute(payload)
                difference = any(bool(actual.get(k)) != bool(case['candidate'].get(k)) for k in ['syntax_valid', 'functional_passed'])
                if difference:
                    differences.append(dict(arm=job['arm'], frame=job['frame'], language=case['language'], task=case['task'], producer=case['candidate'], fresh=actual))
                reference = controls[(case['language'], task[0])]
                efficient, measurement = efficiency(actual, reference, task[3]) if reference.get('efficiency_completed') else (False, {})
                ordinary = ordinary_execute(code, task[0], payload['cases']) if case['language'] == 'Python' else None
                row = dict(arm=job['arm'], frame=job['frame'], language=case['language'], task=task[0], actual_unchanged_code=code,
                    actual_code_sha256=hashlib.sha256(code.encode()).hexdigest(), actual_original_finite_cases=len(payload['cases']),
                    fresh_native_execution=actual, fresh_ordinary_Python_execution=ordinary,
                    fresh_empirical_efficiency_pass=bool(efficient), fresh_efficiency_measurement=measurement,
                    original_empirical_efficiency_pass=case['efficiency_stress_passed'], actual_native_verdict_difference=difference,
                    source_role='evaluation', optimization_forbidden=True, no_program_repair_or_new_neural_generation=True)
                rows.append(row); stream.write(json.dumps(row) + '\n'); stream.flush()
                write('progress.json', dict(completed=len(rows), planned=144, actual_native_verdict_differences=len(differences), new_neural_forwards=0))
            os.fsync(stream.fileno())
        totals = {}
        for arm in ARMS:
            group = [r for r in rows if r['arm'] == arm]
            totals[arm] = dict(tasks=len(group), syntax=sum(bool(r['fresh_native_execution'].get('syntax_valid')) for r in group),
                functional=sum(bool(r['fresh_native_execution'].get('functional_passed')) for r in group),
                empirical_efficiency=sum(r['fresh_empirical_efficiency_pass'] for r in group),
                ordinary_Python_tasks=sum(r['language'] == 'Python' for r in group),
                ordinary_Python_functional=sum(bool(r['fresh_ordinary_Python_execution']['functional_passed']) for r in group if r['language'] == 'Python'))
        if any(r['tasks'] != 48 or r['ordinary_Python_tasks'] != 24 for r in totals.values()):
            raise ValueError('Every original full endpoint and Python population must remain')
        guard()
        write('complete.json', dict(complete=True, passed=controls_passed and not differences, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_endpoint_totals=totals, all24_fresh_reference_controls_passed=controls_passed,
            all144_unchanged_native_programs_and72ordinary_Python_deliveries_reexecuted=True,
            actual_native_verdict_differences=differences, actual_fresh_program_observations_sha256=sha(output / 'all144-original-programs-fresh-executions.jsonl'),
            original_action_IDs_EOS_stops_weights_and_neural_scores_not_independently_verified=True,
            empirical_stress_timings_not_asymptotic_or_arbitrary_program_efficiency_proof=True,
            source_role='evaluation', optimization_forbidden=True, neural_forwards=0, production_updates=0, goal_achieved=False))
        print(json.dumps(dict(complete=True, passed=controls_passed and not differences, actual_endpoint_totals=totals)), flush=True)
    except BaseException as exc:
        write('failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/human-v8-native-program-readback-CPU-v1')
    main(parser.parse_args().output)
