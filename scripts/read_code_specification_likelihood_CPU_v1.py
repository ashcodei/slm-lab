"""Reconstruct the complete protected specification diagnostic and arithmetic.

No model, optimizer, GPU or neural replay. This checks source/target identities
and recorded arithmetic, not the numerical truth of producer neural scores.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/dense-code-specification-CPU-v1'
ASSESSMENT = ROOT/'reports/durable-distributed-code-v1/rate3e6/assessment-attempt-2'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('Explicit CPU-only non-neural reader required')
    os.nice(10); os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    from tokenizers import Tokenizer
    from slm_lab.code_tasks import TASKS
    from slm_lab.native_code_action_readback_v1 import prompt_for, validate
    from slm_lab.native_code_shards_v1 import positions
    read = lambda path: json.loads(Path(path).read_text())
    complete = read(OUT/'complete.json'); plan = read(OUT/'plan.json')
    if (not complete['complete'] or complete['cases'] != 48 or complete['neural_forward_calls'] != 96
            or complete['actual_registered_parameters'] != 984399690):
        raise ValueError('The complete actual full-size 48-case diagnostic is required')
    files = dict(complete['source_files_sha256'])
    files.update({str(p): sha(p) for p in [Path(__file__).resolve(), OUT/'complete.json', OUT/'plan.json']})
    for name, digest in files.items():
        if sha(name) != digest:
            raise ValueError('An original bound diagnostic byte differs: '+name)
    if any(files[k] != v for k, v in plan['source_files_sha256'].items()):
        raise ValueError('Every original plan binding must survive completion')
    checkpoint = Path(next(k for k in files if k.endswith('/rate3e6/model.pt')))
    tokenizer = Tokenizer.from_file(str(checkpoint.parent/'tokenizer.json'))
    rows = []
    for frame in ['completion', 'instruction']:
        for rank in range(4):
            raw = read(ASSESSMENT/f'code-{frame}-shard-{rank}.raw.json')
            events = [json.loads(line) for line in (ASSESSMENT/f'code-{frame}-rank-{rank}-actual-actions.jsonl').read_text().splitlines()]
            expected = [(lang, TASKS[i]) for lang in ['Python', 'JavaScript'] for i in positions(rank, 4)]
            if len(raw['results']) != 6 or len(events) != 6:
                raise ValueError('Every complete original native shard is required')
            for local, ((language, task), case, event) in enumerate(zip(expected, raw['results'], events)):
                path = OUT/f'{frame}-{language}-{task[0]}.json'; files[str(path)] = sha(path)
                row = read(path); pair = row['pair']; true = row['conditions']['original']
                proof = validate(case, frame, event, tokenizer, expected_case_position=local)
                if row['original_delivery'] != proof or (pair['task'], pair['language'], pair['frame']) != (task[0], language, frame):
                    raise ValueError('Every exact original native delivery and protected case must reconstruct')
                peers = [t for t in TASKS if t[1] == task[1]]
                other = peers[(peers.index(task)+1) % len(peers)]
                prompt = prompt_for(case, frame); reference = task[4 if language == 'Python' else 5]
                response = reference if frame == 'instruction' else reference.split('\n' if language == 'Python' else '{', 1)[1]
                if (pair['original_prompt'] != prompt or pair['conflicting_prompt'] != prompt.replace(task[2], other[2])
                        or pair['conflicting_specification_task'] != other[0] or pair['response'] != response
                        or pair['original_reference'] != reference or pair['evaluation_only_optimization_forbidden'] is not True):
                    raise ValueError('One entire same-signature specification intervention and unchanged full reference required')
                target = tokenizer.encode(response, add_special_tokens=False).ids + [2]
                if target != row['response_EOS_ids']:
                    raise ValueError('Every original complete code response/EOS token must reconstruct')
                n = len(target)
                for condition in ['original', 'conflicting']:
                    part = row['conditions'][condition]
                    ids = [1] + tokenizer.encode(pair[condition+'_prompt'], add_special_tokens=False).ids
                    if (part['complete_prompt_ids'] != ids or part['prompt_tokens'] != len(ids)
                            or part['response_EOS_targets'] != n or part['inference_device'] != 'cpu'
                            or part['inference_precision'] != 'fp32'
                            or any(len(part[k]) != n for k in ['token_NLL', 'target_best_tied_rank', 'target_tied_values', 'teacher_prefix_top1_ids'])):
                        raise ValueError('Every complete native context and recorded response score must align')
                    losses = part['token_NLL']; predictions = part['teacher_prefix_top1_ids']
                    if any(not math.isfinite(x) or x < 0 for x in losses):
                        raise ValueError('All original token losses must be finite and nonnegative')
                    wrong = [i for i, (a, b) in enumerate(zip(predictions, target)) if a != b]
                    if (not math.isclose(sum(losses), part['summed_token_NLL'], abs_tol=1e-8)
                            or not math.isclose(sum(losses)/n, part['mean_token_NLL'], abs_tol=2e-6)
                            or part['EOS_NLL'] != losses[-1] or part['gold_prefix_top1_correct'] != n-len(wrong)
                            or part['first_gold_prefix_top1_divergence'] != (wrong[0] if wrong else None)):
                        raise ValueError('Every original complete-target arithmetic summary must reconstruct')
                actions = event['actual_generated_token_ids']
                divergence = next((i for i, (a, b) in enumerate(zip(actions, target)) if a != b), None)
                if row['original_GPU_gold_prefix_divergence'] != divergence:
                    raise ValueError('Exact original GPU/reference token divergence must reconstruct')
                if divergence is not None and row['CPU_gold_rank_at_original_GPU_divergence'] != true['target_best_tied_rank'][divergence]:
                    raise ValueError('Recorded reference rank must address the exact original divergence')
                advantage = row['conditions']['conflicting']['mean_token_NLL'] - true['mean_token_NLL']
                if row['correct_specification_NLL_advantage'] != advantage:
                    raise ValueError('The signed paired reference NLL difference must reconstruct')
                rows.append(row)
    if len(rows) != 48 or 'torch' in sys.modules:
        raise ValueError('All 48 original cases must close without importing Torch or a model')
    summary = {}
    for condition in ['original', 'conflicting']:
        parts = [r['conditions'][condition] for r in rows]; n = sum(p['response_EOS_targets'] for p in parts)
        actual = dict(response_EOS_targets=n, token_NLL=sum(p['summed_token_NLL'] for p in parts)/n,
            gold_prefix_top1_correct=sum(p['gold_prefix_top1_correct'] for p in parts),
            gold_prefix_top1_accuracy=sum(p['gold_prefix_top1_correct'] for p in parts)/n,
            mean_EOS_NLL=sum(p['EOS_NLL'] for p in parts)/48)
        if actual != complete['summary'][condition]:
            raise ValueError('Every original aggregate must agree with all complete cases')
        summary[condition] = actual
    preferred = sum(r['correct_specification_NLL_advantage'] > 0 for r in rows)
    if preferred != complete['summary']['cases_with_lower_reference_NLL_under_correct_specification']:
        raise ValueError('All paired preference signs must reconstruct')
    groups = {}
    for group in ['all', 'completion', 'instruction', 'Python', 'JavaScript']:
        selected = [r for r in rows if group == 'all' or group in [r['pair']['frame'], r['pair']['language']]]
        parts = [r['conditions']['original'] for r in selected]
        groups[group] = dict(cases=len(parts), gold_prefix_EOS_top1=sum(p['teacher_prefix_top1_ids'][-1] == 2 for p in parts),
            full_reference_gold_prefix_top1=sum(p['gold_prefix_top1_correct'] == p['response_EOS_targets'] for p in parts),
            mean_EOS_NLL=statistics.mean(p['EOS_NLL'] for p in parts),
            cases_correct_specification_preferred=sum(r['correct_specification_NLL_advantage'] > 0 for r in selected))
    output = OUT/'source-arithmetic-readback.json'
    result = dict(complete=True, original_cases=48, original_neural_calls_not_repeated=96,
        every_original_prompt_reference_token_mask_and_recorded_arithmetic_rebuilt=True,
        neural_scores_not_numerically_replayed=True, no_training_or_new_generation=True,
        summary=summary, groups=groups, source_files_sha256=files,
        specification_length_and_retained_name_hints_limit_interpretation=True,
        exact_reference_divergence_not_proof_of_semantic_failure=True,
        CPU_scores_not_exact_GPU_BF16_policy_probabilities=True,
        candidate_promoted=False, goal_achieved=False)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(complete=True, summary=summary, groups=groups), indent=2))


if __name__ == '__main__':
    main()
