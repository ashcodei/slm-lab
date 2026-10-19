"""Reconstruct all held-human targets and aggregate retained per-token scores.

This independently checks source/mask/score accounting. It does not replay the
GPU numerical forward or turn teacher metrics into generation correctness.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.whole_human_data_v2 import WholeHumanData

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/whole-human-SFT-v1'


def review(data, report):
    if (report.get('source_manifest_sha256') != file_hash(data.root/'manifest.json')
            or report.get('whole_source_responses_and_EOS') is not True
            or report.get('response_or_prompt_truncation') is not False
            or len(report.get('results', [])) != len(data)):
        raise ValueError('Every original held human source and full response token record required')
    rows = report['results']
    for position, (source, row) in enumerate(zip(data.rows, rows)):
        ids, labels = data.sequence(position, 'response')
        targets = [v for v in labels[1:] if v != -100]
        nll = row['per_target_nll']; argmax = row['teacher_argmax_ids']
        if (row['source_position'] != position or row['id'] != source['id'] or row['group_id'] != source['group_id']
                or row['language'] != source['language'] or row['context_tokens'] != len(ids)
                or row['target_ids'] != targets or row['response_EOS_targets'] != len(targets)
                or len(nll) != len(targets) or len(argmax) != len(targets) or targets[-1] != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in nll)
                or any(type(v) is not int or not 0 <= v < 32768 for v in argmax)
                or row['token_nll'] != math.fsum(nll)/len(targets)
                or row['teacher_correct'] != sum(a == b for a, b in zip(argmax, targets))):
            raise ValueError('Every complete original target/EOS, source identity, prediction and retained score must reconstruct')
    def summary(items):
        targets = sum(len(r['target_ids']) for r in items)
        return dict(examples=len(items), response_EOS_targets=targets,
            token_mean_nll=math.fsum(v for r in items for v in r['per_target_nll'])/targets,
            example_mean_nll=math.fsum(r['token_nll'] for r in items)/len(items),
            teacher_argmax_accuracy=sum(r['teacher_correct'] for r in items)/targets)
    by_language = {language:summary([r for r in rows if r['language'] == language]) for language in sorted({r['language'] for r in rows})}
    if report['overall'] != summary(rows) or report['by_language'] != by_language:
        raise ValueError('All complete target and language aggregates must recompute from original per-token evidence')
    return summary(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('Preserve previous independent target readbacks')
    data = WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'dialogue', 'valid', 'evaluation')
    plan = json.loads((REPORTS/'plan.json').read_text()); evidence = {}
    for arm in ['initial', *plan['arms']]:
        path = REPORTS/arm/'human-validation.json'
        evidence[arm] = dict(reconstructed=review(data, json.loads(path.read_text())), retained_report_sha256=file_hash(path))
    import torch
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    initial = torch.load(ROOT/'checkpoints/whole-human-SFT-v1/initial/model.pt', map_location='cpu', weights_only=True, mmap=True)
    if tensor_digest(initial['aligned_model']['extensions'].items()) != plan['initial_parts_digest']:
        raise ValueError('Actual initial checkpoint must retain every original seeded extension value')
    baseline = evidence['initial']['reconstructed']
    for arm, item in evidence.items():
        measured = item['reconstructed']
        item['response_token_NLL_relative_change'] = measured['token_mean_nll']/baseline['token_mean_nll']-1
        item['teacher_argmax_accuracy_percentage_point_change'] = 100*(measured['teacher_argmax_accuracy']-baseline['teacher_argmax_accuracy'])
    atomic_json(args.output, dict(observed_at=datetime.now(timezone.utc).isoformat(), run_plan_sha256=file_hash(REPORTS/'plan.json'),
        actual_initial_parts_digest_matches_plan=True, all_four_complete_held_source_teacher_reports=evidence,
        every_original_source_BOS_prompt_response_EOS_target_and_all_aggregates_reconstructed=True,
        numerical_GPU_forwards_not_independently_replayed=True, free_generation_or_factual_accuracy_not_established=True,
        model_updates_by_this_review=0, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    main()
