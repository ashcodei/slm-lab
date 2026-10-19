"""Measure every admitted pair before choosing a more local learning objective."""
from collections import Counter
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path

import numpy as np

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_contrast_data_v1 import ExecutionContrastData

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/execution-preference-edit-census-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CPU-only whole-pair census is required')
    data = ExecutionContrastData(ROOT, purpose='audit')
    if len(data) != 1620:
        raise ValueError('All1620original admitted preferences must remain')
    rows = []; histogram = Counter(); ratios = []; lengths = []
    for i, row in enumerate(data.rows):
        a, b = row['chosen'].splitlines(keepends=True), row['rejected'].splitlines(keepends=True)
        matching = sum(block.size for block in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_matching_blocks())
        similarity = 2 * matching / (len(a) + len(b))
        bucket = next(label for threshold, label in [(.9,'at_least_90percent'),(.75,'75_to90percent'),
                      (.5,'50_to75percent'),(.25,'25_to50percent'),(0.,'below25percent')] if similarity >= threshold)
        histogram[bucket] += 1; ratios.append(similarity)
        chosen = sum(v != -100 for v in row['chosen_labels']); rejected = sum(v != -100 for v in row['rejected_labels'])
        lengths.append(chosen/rejected)
        rows.append(dict(original_admitted_position=i, id=row['id'], chosen_lines=len(a), rejected_lines=len(b),
            exact_matching_lines=matching, complete_original_line_similarity=similarity,
            chosen_response_EOS_targets=chosen, rejected_response_EOS_targets=rejected))
    data.guard.check(full=True); OUT.mkdir()
    with (OUT/'all1620-original-pair-census.jsonl').open('x') as stream:
        for row in rows: stream.write(json.dumps(row)+'\n')
    bindings = {str(p):file_hash(p) for p in [Path(__file__), data.folder/'manifest.json',data.folder/'admission.json',
                data.folder/'pairs.jsonl',OUT/'all1620-original-pair-census.jsonl']}
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        actual_original_pairs=len(rows), exact_line_similarity_histogram=dict(histogram),
        exact_line_similarity_quantiles=dict(zip(['minimum','p25','median','p75','maximum'],[float(v) for v in np.quantile(ratios,[0,.25,.5,.75,1])])),
        chosen_over_rejected_target_length_quantiles=dict(zip(['minimum','p25','median','p75','maximum'],[float(v) for v in np.quantile(lengths,[0,.25,.5,.75,1])])),
        every_admitted_pair_retained_no_training_or_held_scoring=True,
        line_similarity_is_not_semantic_equivalence_or_a_model_quality_score=True,
        source_files_sha256=bindings, models_constructed=0, model_updates=0, goal_achieved=False))
    print(json.dumps(dict(report=str(OUT/'complete.json'), histogram=dict(histogram), median_line_similarity=float(np.median(ratios)))),flush=True)


if __name__=='__main__':main()
