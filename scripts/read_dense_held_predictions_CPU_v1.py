"""Rebuild every reserved input/mask and recorded loss for a dense profile.

Reads original sealed token arrays and producer scores, without Torch, a model
or neural replay. Historical same-native baseline is distinguished from the
new study's final matched baseline, which remains separately required.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'reports/durable-distributed-code-v1'
DATA = ROOT/'data/public-reasoned-code-SFT-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['rate3e6', 'rate3e5', 'rate1e4', 'rate1e5'], required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('Explicit model-free CPU source/arithmetic readback required')
    os.nice(10); os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    import numpy as np
    output = RUN/(args.profile+'-held-source-arithmetic-readback-v2.json')
    if output.exists():
        raise ValueError('Preserve an existing complete readback')
    read = lambda path: json.loads(Path(path).read_text())
    manifest = read(DATA/'manifest.json'); files = {str(DATA/'manifest.json'): sha(DATA/'manifest.json'),
                                                  str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    for name, digest in manifest['files_sha256'].items():
        path = DATA/name
        if sha(path) != digest:
            raise ValueError('An original sealed source byte changed: '+name)
        files[str(path)] = digest
    index = np.load(DATA/'index.npy', allow_pickle=False, mmap_mode='r')
    tokens = np.memmap(DATA/'input-ids.bin', dtype='<u2', mode='r')
    held = np.flatnonzero(index['role'] == manifest['roles']['validation'])
    rows = {}
    with gzip.open(DATA/'original-rows.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            if row['split'] == 'validation':
                role = row['original_fixed_role']
                if role['source_role'] != 'reserved' or not role['optimization_forbidden']:
                    raise ValueError('Every source role must remain reserved and forbidden to optimization')
                rows[row['token_pack_index']] = row
    if len(held) != 425 or set(rows) != set(held.tolist()):
        raise ValueError('All425unchanged reserved source records are required')
    paths = {'retained_same_native_baseline': ROOT/'reports/recovered-dense-code-v1/initial/assessment/held-code-NLL.json',
             args.profile: RUN/args.profile/'assessment-attempt-2/held-code-NLL.json'}
    results = {}
    for label, path in paths.items():
        report = read(path); files[str(path)] = sha(path)
        if (not report['complete'] or report['reserved_views'] != 425 or len(report['records']) != 425
                or report['response_EOS_targets'] != 367945 or report['reserved_optimization_updates'] != 0):
            raise ValueError('Every original reserved source and complete prediction record is required')
        groups = {}; all_CE = 0.; all_EOS = 0.; total = 0
        for local, (position, record) in enumerate(zip(held, report['records'])):
            position = int(position); entry = index[position]; start = int(entry['target_start'])
            ids = tokens[int(entry['offset']):int(entry['offset'])+int(entry['length'])].astype(np.int64).tolist()
            labels = [-100]*start + ids[start:]; n = len(ids)-start
            if (record['reserved_local_position'] != local or record['original_pack_position'] != position
                    or record['complete_input_token_ids'] != ids or record['original_response_EOS_labels'] != labels
                    or record['response_EOS_targets'] != n or ids[0] != 1 or ids[-1] != 2
                    or any(not math.isfinite(record[k]) or record[k] < 0 for k in ['CE_sum', 'EOS_CE_sum'])):
                raise ValueError('Every original whole input, response/EOS mask and recorded loss must reconstruct')
            group = groups.setdefault(rows[position]['view'], dict(views=0, targets=0, CE=0.))
            group['views'] += 1; group['targets'] += n; group['CE'] += record['CE_sum']
            total += n; all_CE += record['CE_sum']; all_EOS += record['EOS_CE_sum']
        if (total != 367945 or not math.isclose(all_CE/total, report['token_mean_NLL'], abs_tol=1e-12)
                or not math.isclose(all_EOS/425, report['EOS_token_mean_NLL'], abs_tol=1e-12)):
            raise ValueError('Every original producer aggregate must reconstruct from all425records')
        results[label] = dict(views=425, response_EOS_targets=total, all_token_NLL=all_CE/total, EOS_token_NLL=all_EOS/425,
            groups={key: dict(views=g['views'], targets=g['targets'], token_NLL=g['CE']/g['targets']) for key, g in groups.items()})
    old, new = results['retained_same_native_baseline'], results[args.profile]
    reductions = dict(all_relative_reduction=1-new['all_token_NLL']/old['all_token_NLL'],
        **{key+'_relative_reduction': 1-new['groups'][key]['token_NLL']/old['groups'][key]['token_NLL'] for key in old['groups']})
    if 'torch' in sys.modules:
        raise ValueError('No model framework may be imported by this source/arithmetic reader')
    result = dict(complete=True, at=datetime.now(timezone.utc).isoformat(), profile=args.profile,
        all_original425inputs_and_target_masks_rebuilt=True, results=results, relative_reductions=reductions,
        source_files_sha256=files, same_study_final_matched_baseline_still_required=True,
        no_neural_scores_replayed_or_new_generation=True, lower_prediction_loss_not_code_correctness=True,
        candidate_promoted=False, goal_achieved=False)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(results=results, relative_reductions=reductions), indent=2))


if __name__ == '__main__':
    main()
