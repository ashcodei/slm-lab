"""Correct one transcribed pool total in a new review, preserving the original."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-hidden arithmetic correction required')
    old = ROOT / 'reports/contrastive-fit-and-full-audits-v1-review/review.json'
    output = old.with_name('review-corrected.json')
    if output.exists(): raise ValueError('Preserve earlier corrections')
    plan_path = ROOT / 'reports/AI2D-full-text-exposure-audit-v1/plan.json'; plan = json.loads(plan_path.read_text())
    total = sum(plan['older_possible_source_tokens'].values())
    value = json.loads(old.read_text()); field = value['requirements']['broad_independent_benchmarks']['evidence']
    if field.count('2638877519additional positions') != 1 or total != 2639877519: raise ValueError('Only the known transcribed total may be corrected')
    value['requirements']['broad_independent_benchmarks']['evidence'] = field.replace('2638877519additional positions', str(total) + 'additional positions')
    value['arithmetic_correction'] = dict(at=datetime.now(timezone.utc).isoformat(), original_review_sha256=file_hash(old),
        original_text_total=2638877519, actual_sum_of_three_original_pools=total, training_or_evaluation_changed=False)
    value['source_files_sha256'].update({str(p): file_hash(p) for p in [old, plan_path, Path(__file__)]})
    atomic_json(output, value)
    master_path = ROOT / 'reports/completion-audit.json'; master = json.loads(master_path.read_text())
    if master['latest_full_scope_review'] != str(old.relative_to(ROOT)): raise ValueError('Exact original latest review required')
    master['latest_full_scope_review'] = str(output.relative_to(ROOT)); master['requirements'] = {k: v['evidence'] for k, v in value['requirements'].items()}
    atomic_json(master_path, master)
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name; text = path.read_text()
        path.write_text(text.replace(str(old.relative_to(ROOT)), str(output.relative_to(ROOT))))
    print(json.dumps(dict(corrected_review=str(output), original_preserved=str(old), actual_older_tokens=total)), flush=True)


if __name__ == '__main__': main()
