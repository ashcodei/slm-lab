"""Reexecute completed native code and independently recalculate bounded QA metrics.

This reader does not run neural generation, alter a benchmark, or train a model.
Retrieval embeddings were not saved by the producer, so retrieval accuracy is
reported as a producer observation rather than an independently replayed result.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/completed-execution-outcomes-readback-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This independent output reader must hide all GPUs')
    if OUT.exists():
        raise ValueError('Preserve complete or partial previous evidence')
    import torch
    from slm_lab.code_quality import execute, efficiency
    from slm_lab.code_tasks import TASKS, request
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.knowledge_evaluation import letter
    from slm_lab.math_corpus import final_answer
    from slm_lab.ocr_evaluation import normalized, transcription_scores, summarize as ocr_summary
    from slm_lab.object_evaluation import summarize as object_summary
    torch.set_num_threads(1)
    OUT.mkdir()
    bindings = {}

    def read(path):
        path = Path(path)
        bindings[str(path)] = file_hash(path)
        return json.loads(path.read_text())

    def close(a, b):
        if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError(f'Recomputed aggregate differs: {a!r} {b!r}')

    current = ROOT / 'reports/completed-execution-assessment-v4'
    plan = read(current / 'plan.json')
    complete = read(current / 'training-complete.json')
    if complete['evaluation_only_training_updates'] != 0:
        raise ValueError('Completed-policy evaluator must have performed zero updates')
    baseline = ROOT / 'reports/verified-views-execution-v1/warmup'
    old_reader = read(ROOT / 'reports/verified-views-warmup88-readback-v1/complete.json')
    for path, digest in old_reader['source_files_sha256'].items():
        if Path(path).parent == baseline and file_hash(path) != digest:
            raise ValueError('Previously independently replayed warm-up output changed')
    folders = {arm: current / arm for arm in ['verified_solution', 'execution_hybrid']}
    folders['warmup_CPU_FP32'] = ROOT / 'reports/warmup88-code-CPU-FP32-v1'
    tasks = {task[0]: task for task in TASKS}
    expected_cases = {(lang, task[0]) for lang in ['Python', 'JavaScript'] for task in TASKS}
    jobs, scores, reports = [], {}, {}
    for arm, folder in folders.items():
        scores[arm] = {}; reports[arm] = {}
        for frame in ['completion', 'instruction']:
            d = read(folder / f'code-{frame}.json')
            old = read(baseline / f'code-{frame}.json')
            if (d['task_count'] != 24 or d['generation_budget'] != 512
                    or d['prompt_mode'] != frame
                    or {(r['language'], r['task']) for r in d['results']} != expected_cases
                    or len(d['results']) != 24):
                raise ValueError('Every unchanged task and generation budget is required')
            for key in ['generation_budget', 'prompt_mode', 'suite_sha256', 'tokenizer_sha256']:
                if d[key] != old[key]:
                    raise ValueError('Matched task/tokenizer/frame/budget required')
            if arm != 'warmup_CPU_FP32':
                checkpoint = ROOT / 'checkpoints/execution-single-update-recovery-v3' / (arm + '-actor.pt')
                bindings[str(checkpoint)] = file_hash(checkpoint)
                if d['checkpoint_sha256'] != bindings[str(checkpoint)]:
                    raise ValueError('Actual completed native actor checkpoint required')
                for key in ['inference_device_type', 'inference_precision', 'decoding']:
                    if d[key] != old[key]:
                        raise ValueError('Native training comparison must keep inference configuration')
            else:
                if d['inference_device_type'] != 'cpu' or d['inference_precision'] != 'fp32':
                    raise ValueError('Explicit CPU/FP32 comparison required')
            for path, digest in d['evaluation_provenance']['files_sha256'].items():
                if file_hash(ROOT / path) != digest:
                    raise ValueError('Bound evaluator implementation changed')
                bindings[str(ROOT / path)] = digest
            reports[arm][frame] = d
            jobs.extend((arm, frame, row) for row in d['results'])
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(),
        code_outputs_to_reexecute=len(jobs), workers=2, neural_generation_replayed=False,
        training_updates=0, source_files_sha256={**bindings, str(Path(__file__)): file_hash(__file__)},
        candidate_promoted=False, goal_achieved=False))

    def replay(job):
        arm, frame, row = job
        task = tasks[row['task']]
        actual = execute(request(task, row['language'], row['code']))
        for key in ['syntax_valid', 'functional_passed', 'passed_cases', 'total_cases']:
            if actual.get(key) != row['candidate'].get(key):
                raise ValueError('Independent candidate execution differs: ' + str((arm, frame, row['task'], key)))
        passed, measure, reference = False, {}, None
        if actual.get('functional_passed'):
            code = task[4] if row['language'] == 'Python' else task[5]
            reference = execute(request(task, row['language'], code, True))
            if not reference.get('functional_passed') or not reference.get('efficiency_completed'):
                raise ValueError('Fresh canonical reference must pass before any efficiency claim')
            passed, measure = efficiency(actual, reference, task[3])
        category = ('correct' if actual.get('functional_passed') else
                    'syntax_invalid' if not actual.get('syntax_valid') else
                    'wrong_answer' if actual.get('error') == 'Hidden case mismatch' else
                    'runtime_or_supported_subset_error')
        return dict(arm=arm, frame=frame, language=row['language'], task=row['task'],
            category=category, actual_execution=actual, fresh_reference_execution=reference,
            efficiency_stress_passed=passed, efficiency_measurement=measure,
            timing_verdict_agrees_with_producer=passed == row['efficiency_stress_passed'])

    with ThreadPoolExecutor(max_workers=2) as pool:
        replayed = list(pool.map(replay, jobs))
    for arm in folders:
        for frame in ['completion', 'instruction']:
            rows = [r for r in replayed if r['arm'] == arm and r['frame'] == frame]
            d = reports[arm][frame]
            syntax = sum(bool(r['actual_execution'].get('syntax_valid')) for r in rows)
            correct = sum(bool(r['actual_execution'].get('functional_passed')) for r in rows)
            efficient = sum(r['efficiency_stress_passed'] for r in rows)
            close(syntax / 24, d['syntax_pass_rate']); close(correct / 24, d['functional_pass_at_1'])
            scores[arm][frame] = dict(tasks=24, syntax=syntax, correct=correct, efficient=efficient,
                failures=dict(Counter(r['category'] for r in rows)),
                timing_verdict_disagreements=sum(not r['timing_verdict_agrees_with_producer'] for r in rows))

    broad = {}
    data = {name: [json.loads(line) for line in Path(path).read_text().splitlines()]
            for name, path in plan['pulse_sources'].items()}
    for name, path in plan['pulse_sources'].items():
        bindings[path] = file_hash(path)
    math_path = ROOT / 'data/math-reasoning-v2/test.jsonl'
    math_rows = [json.loads(line) for line in math_path.read_text().splitlines()]
    math_rows.sort(key=lambda row: hashlib.sha256(row['question'].encode()).digest())
    bindings[str(math_path)] = file_hash(math_path)
    for arm in ['verified_solution', 'execution_hybrid']:
        folder = folders[arm]; broad[arm] = {}
        for suite in ['knowledge', 'math', 'ocr', 'objects', 'vision']:
            d = read(folder / (suite + '.json')); old = read(baseline / (suite + '.json'))
            for key in ['tokenizer_sha256', 'inference_device_type', 'inference_precision']:
                if d[key] != old[key]:
                    raise ValueError('Broad assessment configuration changed')
            for key in ['data_sha256', 'questions_sha256', 'generation_budget', 'max_new_tokens', 'image_size']:
                if key in d and d[key] != old[key]:
                    raise ValueError('Broad data or inference budget changed')
            ckpt = ROOT / 'checkpoints/execution-single-update-recovery-v3' / (arm + '-actor.pt')
            if d['checkpoint_sha256'] != bindings[str(ckpt)]:
                raise ValueError('Broad report must use same actual completed policy')
            rows = d.get('results', d.get('predictions', d.get('samples')))
            old_rows = old.get('results', old.get('predictions', old.get('samples')))
            if [r['id'] for r in rows] != [r['id'] for r in old_rows]:
                raise ValueError('Actual matched broad cases required')
            if suite == 'knowledge':
                originals = {r['id']: r for r in data[suite]}
                for row in rows:
                    original = originals[row['id']]
                    if row['answer_index'] != original['answer_index']:
                        raise ValueError('Original gold answer differs')
                    prediction = letter(row['generated_response'], len(original['options']))
                    scores_ = row['candidate_scores']
                    if len(scores_) != len(original['options']) or not all(math.isfinite(s) for s in scores_):
                        raise ValueError('Every finite option likelihood required')
                    choice = max(range(len(scores_)), key=lambda i: scores_[i])
                    if (prediction != row['generated_index'] or choice != row['likelihood_index']
                            or (prediction == row['answer_index']) != row['generated_correct']
                            or (choice == row['answer_index']) != row['likelihood_correct']):
                        raise ValueError('Generated/likelihood QA verdict differs')
                n = len(rows); generated = sum(r['generated_correct'] for r in rows)
                likelihood = sum(r['likelihood_correct'] for r in rows)
                close(generated / n, d['overall']['generated_accuracy'])
                close(likelihood / n, d['overall']['likelihood_accuracy'])
                chance = sum(1 / len(r['candidate_scores']) for r in rows) / n
                close(chance, d['overall']['chance_accuracy'])
                broad[arm][suite] = dict(questions=n, generated_correct=generated,
                    likelihood_correct=likelihood, mean_random_choice_accuracy=chance,
                    generated_format_valid=sum(r['generated_index'] is not None for r in rows),
                    gold_letter_positions=dict(Counter(r['answer_index'] for r in rows)),
                    likelihood_letter_positions=dict(Counter(r['likelihood_index'] for r in rows)))
            elif suite == 'math':
                if [r['id'] for r in rows] != [r['id'] for r in math_rows[:16]]:
                    raise ValueError('Original fixed math prefix required')
                for row, original in zip(rows, math_rows):
                    actual = final_answer(row['raw_response'])
                    gold = Fraction(original['answer_numerator'], original['answer_denominator'])
                    if (row['correct'] != (actual == gold)
                            or row['format_compliant'] != (actual is not None)
                            or (actual is not None and (actual.numerator != row['answer_numerator']
                                                      or actual.denominator != row['answer_denominator']))):
                        raise ValueError('Exact rational answer readback differs')
                broad[arm][suite] = dict(problems=len(rows), correct=sum(r['correct'] for r in rows))
                if broad[arm][suite]['correct'] != d['correct']:
                    raise ValueError('Math aggregate differs')
            elif suite in ['ocr', 'objects']:
                originals = {r['id']: r for r in data[suite]}
                for row in rows:
                    if row['expected'] != originals[row['id']]['response']:
                        raise ValueError('Original visual target differs')
                    if suite == 'ocr':
                        actual = transcription_scores(row['expected'], row['generated'])
                        if any(row[k] != value for k, value in actual.items()):
                            raise ValueError('Exact OCR edit/word counts differ')
                    else:
                        for view in ['proper', 'blank', 'wrong']:
                            actual = normalized(row[view + '_generated']).casefold() == normalized(row['expected']).casefold()
                            if actual != row[view + '_exact']:
                                raise ValueError('Visual intervention answer verdict differs')
                actual = ocr_summary(rows) if suite == 'ocr' else object_summary(rows)
                for key, value in actual.items():
                    close(value, d['overall'][key])
                broad[arm][suite] = actual
            else:
                n = sum(r['target_tokens'] for r in rows)
                actual = dict(proper_nll=sum(r['target_tokens'] * r['proper_nll'] for r in rows) / n,
                    blank_nll_gain=sum(r['target_tokens'] * (r['blank_nll'] - r['proper_nll']) for r in rows) / n,
                    mismatch_nll_gain=sum(r['target_tokens'] * (r['mismatched_nll'] - r['proper_nll']) for r in rows) / n)
                for key, value in actual.items():
                    close(value, d[key])
                broad[arm][suite] = dict(**actual, producer_retrieval=d['caption_retrieval'],
                    embeddings_not_saved_retrieval_accuracy_not_independently_replayed=True)
    for path in [Path(__file__), ROOT / 'reports/execution-final173-readback-v1/complete.json',
                 ROOT / 'reports/execution-final173-reward-replay-v1/complete.json']:
        bindings[str(path)] = file_hash(path)
    for path, digest in bindings.items():
        if file_hash(path) != digest:
            raise ValueError('Evidence changed during reader')
    result = dict(at=datetime.now(timezone.utc).isoformat(), native_code_outputs_reexecuted=96,
        CPU_FP32_comparison_outputs_reexecuted=48, previously_reexecuted_warmup_outputs_unchanged=True,
        code=scores, broad=broad, all_code_reexecutions=replayed,
        neural_generation_not_replayed=True, CPU_FP32_comparison_changes_both_device_and_precision=True,
        one_local_instruction_task_recovered_both_arms=True, RL_coding_advantage_over_CE_proven=False,
        native_completion_correctness_improved=False, broad_capability_improvement_proven=False,
        final_independent_benchmarks_and_human_own_output_judgments_still_required=True,
        candidate_promoted=False, goal_achieved=False, source_files_sha256=bindings)
    atomic_json(OUT / 'complete.json', result)
    print(json.dumps(dict(code=scores, broad=broad)), flush=True)


if __name__ == '__main__':
    main()
