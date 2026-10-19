"""Freshly execute all144 native code deliveries and12 training-source programs.

Reads each complete immutable endpoint as soon as it exists, then waits on the
exact original producer PID/start ticks for the next. Does not rerun neural
generation or training. Native code reports retain raw delivered text, rather
than actual BPE action IDs; this reader cannot reconstruct missing token IDs.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

import torch

from slm_lab.code_quality import execute, efficiency, instruction_code
from slm_lab.code_tasks import TASKS, request
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.own_execution_contrast_CPU_v1 import ExecutionContrastPairs

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/own-execution-contrast-full-CPU-v1'
CHECKPOINTS = ROOT / 'checkpoints/own-execution-contrast-full-CPU-v1'
OUT = ROOT / 'reports/own-execution-contrast-full-CPU-v2-outcome-readback'
ENDPOINTS = ('initial', 'anchor_CE', 'anchor_contrast')


def live(identity):
    path = Path('/proc') / str(identity['pid']) / 'stat'
    try: fields = path.read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return False
    return fields[19] == identity['ticks'] and fields[0] not in ['Z', 'X']


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden all-output readback required')
    torch.set_num_threads(1); OUT.mkdir(); expires = time.time() + 24 * 3600
    owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    source_plan_path = RUN / 'plan.json'; source_plan = json.loads(source_plan_path.read_text())
    producer = source_plan['owner']; data = ExecutionContrastPairs(ROOT / 'data/own-execution-negative-pairs-CPU-v1')
    corpus = ExpandedExecutionCorpus(ROOT); tasks = {t[0]: t for t in TASKS}
    files = dict(source_plan['source_files_sha256'])
    files.update({str(p): file_hash(p) for p in [source_plan_path, *python_sources(ROOT, [Path(__file__)])]})
    guard = FileBindings(files)
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, exact_producer=producer,
        endpoints=list(ENDPOINTS), code_cases_each_endpoint=48, original_training_sources_each_endpoint=4,
        actual_candidate_execution_workers=2, source_files_sha256=files, expires_at_unix=expires,
        neural_generation_or_updates_replayed=False, raw_code_text_available_actual_code_action_IDs_not_retained=True,
        automatic_retry=False, candidate_promoted=False, goal_achieved=False))
    summaries = {}; evidence = [OUT / 'plan.json']
    try:
        for endpoint in ENDPOINTS:
            folder = RUN / endpoint; ready = folder / 'complete.json'
            while not ready.is_file():
                if not live(producer) or (RUN / 'failure.json').exists(): raise ValueError('Exact code producer ended without a complete next endpoint')
                if time.time() > expires: raise ValueError('Declared output-reader lifetime exceeded; no automatic retry')
                guard.check(full=False); time.sleep(10)
            done = json.loads(ready.read_text()); checkpoint = CHECKPOINTS / (endpoint + '-actor.pt')
            checkpoint_sha = file_hash(checkpoint); selected = {}; jobs = []; inputs = [ready, checkpoint]
            if done.get('native_cold_text_and_synthetic_image_probe_bitexact') is not True: raise ValueError('Original actual native cold reload receipt required')
            for frame in ['completion', 'instruction']:
                path = folder / ('code-' + frame + '.json'); report = json.loads(path.read_text()); inputs.append(path)
                if (report['task_count'] != 24 or len(report['results']) != 24 or report['generation_budget'] != 512
                        or report['prompt_mode'] != frame or report['checkpoint_sha256'] != checkpoint_sha
                        or report['suite_sha256'] != file_hash(ROOT / 'src/slm_lab/code_tasks.py')
                        or report['tokenizer_sha256'] != corpus.provenance['tokenizer_sha256']
                        or report['inference_device_type'] != 'cpu' or report['inference_precision'] != 'fp32'
                        or [(r['language'], r['task']) for r in report['results']] != [(lang, t[0]) for lang in ['Python', 'JavaScript'] for t in TASKS]):
                    raise ValueError('Every original native code case, source checkpoint, tokenizer, frame and budget required')
                for name, digest in report['evaluation_provenance']['files_sha256'].items():
                    if file_hash(ROOT / name) != digest: raise ValueError('Original native code evaluator source changed')
                for row in report['results']:
                    task = tasks[row['task']]; name, args, spec, scaling, _, _ = task
                    prefix = (f'def {name}({args}):\n    """{spec}"""\n' if row['language'] == 'Python'
                        else f'function {name}({args}) {{\n  // {spec}\n')
                    derived = instruction_code(row['raw_response']) if frame == 'instruction' else prefix + row['raw_response'].split('```')[0]
                    if row['language'] == 'Python': derived = re.split(r'\n(?=File:)', derived)[0]
                    if derived != row['code'] or row['specification'] != spec or row['intended_scaling'] != scaling:
                        raise ValueError('Actually delivered raw code text must reconstruct without repair or replacement')
                    jobs.append((frame, row))
                selected[frame] = report

            def replay(job):
                frame, row = job; task = tasks[row['task']]
                actual = execute(request(task, row['language'], row['code']))
                for key in ['syntax_valid', 'functional_passed', 'passed_cases', 'total_cases']:
                    if actual.get(key) != row['candidate'].get(key): raise ValueError('Fresh unchanged original candidate execution differs: ' + str((endpoint, frame, row['task'], key)))
                passed = False; measure = {}; reference = None
                if actual.get('functional_passed'):
                    reference = execute(request(task, row['language'], task[4] if row['language'] == 'Python' else task[5], True))
                    if not reference.get('functional_passed') or not reference.get('efficiency_completed'): raise ValueError('Fresh canonical runtime control must pass before efficiency scoring')
                    passed, measure = efficiency(actual, reference, task[3])
                category = ('correct' if actual.get('functional_passed') else 'syntax_invalid' if not actual.get('syntax_valid')
                    else 'wrong_answer' if actual.get('error') == 'Hidden case mismatch' else 'runtime_or_supported_subset_error')
                return dict(endpoint=endpoint, frame=frame, language=row['language'], task=row['task'],
                    raw_delivered_response=row['raw_response'], actual_code=row['code'], fresh_execution=actual,
                    fresh_reference_execution=reference, efficiency_stress_passed=passed, efficiency_measurement=measure,
                    empirical_timing_verdict_agrees_with_producer=passed == row['efficiency_stress_passed'], category=category)
            with ThreadPoolExecutor(max_workers=2) as pool: replayed = list(pool.map(replay, jobs))
            scores = {}
            for frame, report in selected.items():
                rows = [r for r in replayed if r['frame'] == frame]
                item = dict(tasks=24, syntax_passes=sum(bool(r['fresh_execution'].get('syntax_valid')) for r in rows),
                    correct=sum(bool(r['fresh_execution'].get('functional_passed')) for r in rows),
                    empirical_efficiency_stress_passes=sum(r['efficiency_stress_passed'] for r in rows),
                    empirical_timing_disagreements=sum(not r['empirical_timing_verdict_agrees_with_producer'] for r in rows),
                    failures=dict(Counter(r['category'] for r in rows)))
                if (item['syntax_passes'] / 24 != report['syntax_pass_rate'] or item['correct'] / 24 != report['functional_pass_at_1']):
                    raise ValueError('Every independently executed functional/syntax aggregate must match actual raw programs')
                scores[frame] = item
            sources_path = folder / 'actual-unassisted-training-source-evaluation.json'
            sources = json.loads(sources_path.read_text()); inputs.append(sources_path); source_replayed = []
            if [r['source_id'] for r in sources] != list(data.by_source): raise ValueError('Every fixed original source problem required')
            for row in sources:
                position = corpus.by_id[row['source_id']]; source = corpus.row(position); seq, _ = corpus.sequence(position)
                actual = row['actual_rollout']; actions = actual['actual_generated_action_ids']; stopped = actual['stopped_on_generated_EOS']
                if (actual['original_native_prefix_ids'] != seq[:source['prefix_positions']] or actual['supplied_public_reference_prefix_ids']
                        or actual['horizon'] != 512 or actual['sampler'] != 'greedy' or actual['seed'] != 970417
                        or not 1 <= len(actions) <= 512 or 2 in actions[:-1] or stopped != (actions[-1] == 2)
                        or (not stopped and len(actions) != 512) or actual['complete_candidate_response'] != corpus.tokenizer.decode(actions)
                        or actual['inference_device_type'] != 'cpu' or actual['inference_precision'] != 'FP32'
                        or row['public_training_source_not_held'] is not True or row['model_updates'] != 0):
                    raise ValueError('Every actual source prompt/action/EOS/budget must reconstruct without hints')
                cached, forwarded = actual['actual_cached_logps'], actual['actual_full_forward_logps']
                if (len(cached) != len(actions) or len(forwarded) != len(actions)
                        or any(not math.isfinite(p) or p > 0 for p in cached + forwarded)
                        or max(abs(a - b) for a, b in zip(cached, forwarded)) > .0001):
                    raise ValueError('Every retained actual full/cached probability observation must respect its numerical bound')
                fresh = corpus.score(position, actual['complete_candidate_response'], stopped_on_eos=stopped)
                if any(fresh[k] != row['actual_original_execution'][k] for k in ['reward', 'status']): raise ValueError('Fresh source-program execution must reproduce original finite-case credit')
                source_replayed.append(dict(source_id=row['source_id'], actual_code=actual['complete_candidate_response'], fresh_original_execution=fresh))
            summary = dict(native_code=scores, original_train_sources=4,
                public_train_finite_case_passes=sum(r['fresh_original_execution']['reward'] == 1. for r in source_replayed))
            if (summary['public_train_finite_case_passes'] != done['public_train_unassisted_full_finite_case_passes']
                    or done['public_train_sources'] != 4): raise ValueError('Every completed source-fit aggregate must reconstruct')
            target = OUT / endpoint; target.mkdir(); outputs = []
            for filename, rows in [('fresh-native-code-executions.jsonl', replayed), ('fresh-training-source-executions.jsonl', source_replayed)]:
                path = target / filename; path.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows)); outputs.append(path)
            atomic_json(target / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), endpoint=endpoint,
                independent_outcomes=summary, actual_raw_code_deliveries_reexecuted=48, actual_training_source_deliveries_reexecuted=4,
                actual_native_code_action_IDs_not_retained_by_producer_and_not_reconstructed=True,
                empirical_efficiency_remeasured_under_current_host_load_not_asymptotic_proof=True,
                source_files_sha256={str(p): file_hash(p) for p in [*inputs, *outputs]},
                neural_forward_or_training_updates=0, candidate_promoted=False, goal_achieved=False))
            summaries[endpoint] = summary; evidence.append(target / 'complete.json')
            print(json.dumps(dict(endpoint=endpoint, independent_outcomes=summary)), flush=True)
        guard.check(full=True)
        atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True, independent_outcomes=summaries,
            all144_native_development_code_and12_actual_train_source_programs_freshly_executed=True,
            original_frames_languages_functions_and512_action_budgets_preserved=True,
            neural_generation_or_training_not_replayed=True, no_independent_final_benchmark_or_human_RLHF_claim=True,
            source_files_sha256={str(p): file_hash(p) for p in evidence}, candidate_promoted=False, goal_achieved=False))
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__': main()
