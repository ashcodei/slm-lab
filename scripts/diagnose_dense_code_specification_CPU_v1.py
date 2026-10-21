"""Score all 48 protected code references under true/conflicting specifications.

Uses only the first completed standalone full dense model, on CPU. Actual GPU
generation journals are checked, not regenerated. Nothing is admitted to SFT,
pretraining, DPO or RL. The four live GPU training workers remain untouched.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT/'checkpoints/durable-distributed-code-v1/rate3e6/model.pt'
ASSESSMENT = ROOT/'reports/durable-distributed-code-v1/rate3e6/assessment-attempt-2'
OUT = ROOT/'reports/dense-code-specification-CPU-v1'
EXPECTED_MODEL = 'f039c0d5acea149a36be326f8d1127bbba56632789b9a149f7240bcbd3f61a5e'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    path = OUT/name
    if shutil.disk_usage(OUT).free < 8*1024**3 + 32*1024**2:
        raise ValueError('Retain the shared persistent writer reserve')
    if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) > 32*1024**2:
        raise ValueError('The complete diagnostic exceeds its declared report budget')
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh explicitly CPU-only diagnostic is required')
    os.nice(10); os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    import torch
    from tokenizers import Tokenizer
    from slm_lab.code_tasks import TASKS
    from slm_lab.code_specification_likelihood_v1 import comparison, score
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.native_code_action_readback_v1 import validate
    from slm_lab.native_code_shards_v1 import positions
    from slm_lab.native_greedy_observer_v1 import signature
    from slm_lab.own_native_query_v1 import inspect_package, load_package
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    OUT.mkdir(); start = time.perf_counter()
    files = {str(p): sha(p) for p in python_sources(ROOT, [Path(__file__).resolve()])}
    files[str(ROOT/'tests/test_code_specification_likelihood_v1.py')] = sha(ROOT/'tests/test_code_specification_likelihood_v1.py')
    files[str(MODEL)] = EXPECTED_MODEL
    tokenizer_path = MODEL.parent/'tokenizer.json'
    expected_tokenizer = '0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f'
    files[str(tokenizer_path)] = expected_tokenizer
    write('plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner_pid=os.getpid(),
        owner_birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
        all_original_cases=48, conditions_per_case=2, complete_correct_response_EOS_per_condition=True,
        no_optimizer_no_training_no_new_generation=True, CPU_threads=1, nice=os.nice(0),
        CPU_affinity=sorted(os.sched_getaffinity(0)), source_files_sha256=files,
        no_CUDA_context_allowed=True, candidate_promoted=False, goal_achieved=False,
        limitations=['conflicting specifications have different lengths and retain name hints',
            'one reference per task; no exhaustive semantic equivalence or causal proof',
            'CPU FP32 scores do not establish exact GPU BF16 probability parity',
            'gold-prefix token predictions are not free-running code correctness',
            'local protected cases do not replace full final benchmarks']))
    guard = FileBindings(files); guard.check(full=True)
    package = inspect_package(MODEL, expected_sha256=EXPECTED_MODEL,
                              expected_tokenizer_sha256=expected_tokenizer)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError('No tokenizer clipping or padding is allowed')
    model, metadata = load_package(package)
    if torch.cuda.is_initialized():
        raise ValueError('This diagnostic must never initialize CUDA')
    before = signature(model); rng = torch.get_rng_state().clone()
    write('loaded.json', dict(package=package, metadata=metadata, seconds=time.perf_counter()-start))
    rows = []
    for frame in ['completion', 'instruction']:
        for rank in range(4):
            raw_path = ASSESSMENT/f'code-{frame}-shard-{rank}.raw.json'
            journal = ASSESSMENT/f'code-{frame}-rank-{rank}-actual-actions.jsonl'
            report = json.loads(raw_path.read_text())
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            files.update({str(p): sha(p) for p in [raw_path, journal]})
            expected = [(lang, TASKS[i]) for lang in ['Python', 'JavaScript'] for i in positions(rank, 4)]
            if (report['checkpoint_sha256'] != EXPECTED_MODEL or report['task_count'] != 6
                    or len(report['results']) != 6 or len(events) != 6):
                raise ValueError('Every original full-model delivery and shard must match')
            for local, (case, event, (language, task)) in enumerate(zip(report['results'], events, expected)):
                if case['language'] != language or case['task'] != task[0]:
                    raise ValueError('Every original language/task position must match')
                proof = validate(case, frame, event, tokenizer, expected_case_position=local)
                pair = comparison(task, language, frame)
                targets = tokenizer.encode(pair['response'], add_special_tokens=False).ids + [2]
                result = dict(pair=pair, response_EOS_ids=targets, original_delivery=proof, conditions={})
                for condition in ['original', 'conflicting']:
                    ids = [1] + tokenizer.encode(pair[condition+'_prompt'], add_special_tokens=False).ids
                    if condition == 'original' and ids != event['original_input_token_ids']:
                        raise ValueError('The unmodified native prompt must be exact')
                    measured = score(model, ids, targets)
                    result['conditions'][condition] = dict(complete_prompt_ids=ids, **measured)
                    if signature(model) != before or not torch.equal(torch.get_rng_state(), rng) or torch.cuda.is_initialized():
                        raise ValueError('All model metadata and caller RNG must remain unchanged; CUDA forbidden')
                    write('progress.json', dict(complete_cases=len(rows), current_case=[frame, language, task[0]],
                        complete_condition=condition, seconds=time.perf_counter()-start))
                true = result['conditions']['original']; other = result['conditions']['conflicting']
                result['correct_specification_NLL_advantage'] = other['mean_token_NLL'] - true['mean_token_NLL']
                actions = event['actual_generated_token_ids']
                divergence = next((i for i, (a, b) in enumerate(zip(actions, targets)) if a != b), None)
                result['original_GPU_gold_prefix_divergence'] = divergence
                if divergence is not None:
                    result['CPU_gold_rank_at_original_GPU_divergence'] = true['target_best_tied_rank'][divergence]
                write(f'{frame}-{language}-{task[0]}.json', result); rows.append(result)
                guard.check()
    if len(rows) != 48 or len({(r['pair']['frame'], r['pair']['language'], r['pair']['task']) for r in rows}) != 48:
        raise ValueError('Every original task/language/frame must close once')
    summary = {}
    for condition in ['original', 'conflicting']:
        parts = [r['conditions'][condition] for r in rows]
        total = sum(p['response_EOS_targets'] for p in parts)
        summary[condition] = dict(response_EOS_targets=total,
            token_NLL=sum(p['summed_token_NLL'] for p in parts)/total,
            gold_prefix_top1_correct=sum(p['gold_prefix_top1_correct'] for p in parts),
            gold_prefix_top1_accuracy=sum(p['gold_prefix_top1_correct'] for p in parts)/total,
            mean_EOS_NLL=sum(p['EOS_NLL'] for p in parts)/48)
    summary['cases_with_lower_reference_NLL_under_correct_specification'] = sum(r['correct_specification_NLL_advantage'] > 0 for r in rows)
    summary['mean_case_NLL_advantage_for_correct_specification'] = sum(r['correct_specification_NLL_advantage'] for r in rows)/48
    guard = FileBindings(files); guard.check(full=True)
    write('complete.json', dict(complete=True, cases=48, neural_forward_calls=96,
        actual_registered_parameters=metadata['actual_registered_parameters'], checkpoint_sha256=EXPECTED_MODEL,
        summary=summary, seconds=time.perf_counter()-start, source_files_sha256=files,
        no_optimization_no_admission_no_new_native_generation=True,
        candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        if OUT.exists():
            write('failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), goal_achieved=False))
        raise
