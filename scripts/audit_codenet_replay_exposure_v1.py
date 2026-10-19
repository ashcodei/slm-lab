"""Enumerate the proposed runner's actual replay draws, without starting training."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_replay_v1 import INDEX_DTYPE
from slm_lab.codenet_stage_v1 import EpochOrder, rehearsal_selection
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Use CPU-only index enumeration')
    output = ROOT / 'reports/codenet-replay-exposure-v1'
    if output.exists():
        raise ValueError('Preserve earlier replay exposure evidence')
    folder = ROOT / 'data/codenet-replay-v1'
    manifest_path = folder / 'manifest.json'
    audit_path = ROOT / 'reports/codenet-replay-v1-independent-audit/results.json'
    code_audit_path = ROOT / 'reports/codenet-stage-v1-source-exposure-audit.json'
    manifest, audit, code = (json.loads(p.read_text()) for p in [manifest_path, audit_path, code_audit_path])
    if (file_hash(manifest_path) != audit['replay_manifest_sha256']
        or audit['original_source_rows_verified'] != manifest['source_rows']
        or audit['complete_fit_rows_verified'] != manifest['complete_fit_rows']
        or audit['source_rows_dropped_or_duplicated'] or audit['encoder_truncated_sequences']):
        raise ValueError('Completed independent full replay reconstruction required')
    paths = [Path(__file__), ROOT / 'src/slm_lab/codenet_runner_v1.py', ROOT / 'src/slm_lab/codenet_stage_v1.py',
        ROOT / 'src/slm_lab/codenet_replay_v1.py', ROOT / 'src/slm_lab/codenet_checkpoint_v1.py',
        manifest_path, audit_path, code_audit_path, ROOT / 'data/codenet-durable-pack-v1/manifest.json']
    bound = {str(p): file_hash(p) for p in paths}
    bound.update({str(folder / name): digest for name,digest in manifest['artifact_files_sha256'].items()})
    guard = FileBindings(bound)
    if code['source_manifest_sha256'] != file_hash(ROOT / 'data/codenet-durable-pack-v1/manifest.json'):
        raise ValueError('Earlier complete-code exposure plan differs')
    index = np.load(folder / 'index.npy', mmap_mode='r', allow_pickle=False)
    if index.dtype != INDEX_DTYPE or len(index) != manifest['complete_fit_rows']:
        raise ValueError('Complete replay source index differs')
    names = manifest['family_names']
    positions = {name: np.flatnonzero(index['family'] == i) for i,name in enumerate(names)}
    for name in names:
        if len(positions[name]) != manifest['family_counts'][name]:
            raise ValueError('Original indexed family population differs')
    seed = code['seed']
    steps = code['complete_pass_layout']['optimizer_steps']
    orders = {name: EpochOrder(len(positions[name]), seed + 1000 + i) for i,name in enumerate(names)}
    counts = np.zeros(len(index), dtype='<u4')
    draws, family_updates = [], Counter()
    for step in range(steps):
        name, occurrence = rehearsal_selection(step,names)
        if occurrence != family_updates[name]:
            raise ValueError('Actual source-family cursor differs from runner formula')
        family_updates[name] += 1
        for rank in range(4):
            local = orders[name].rank_groups(occurrence,rank,4,1,1)[0][0]
            position = int(positions[name][local])
            counts[position] += 1
            draws.append(position)
    actual = np.asarray(draws,dtype='<u4')
    if len(actual) != steps * 4 or int(counts.sum()) != len(actual):
        raise ValueError('Replay draw accounting differs')
    families = {}
    for name in names:
        local = counts[positions[name]]
        families[name] = {'available_complete_fit_rows':len(local),'optimizer_updates':family_updates[name],
            'actual_example_exposures':int(local.sum()),'distinct_examples_reached':int(np.count_nonzero(local)),
            'examples_not_reached':int(np.count_nonzero(local==0)),
            'exposure_count_distribution':{str(k):int(v) for k,v in Counter(map(int,local)).items()},
            'actual_complete_context_token_exposures':int((local.astype(np.uint64)*index['context_tokens'][positions[name]]).sum()),
            'actual_complete_response_target_exposures':int((local.astype(np.uint64)*index['target_tokens'][positions[name]]).sum())}
    output.mkdir()
    np.save(output/'draw-source-index-u32.npy',actual,allow_pickle=False)
    np.save(output/'source-row-exposure-counts-u32.npy',counts,allow_pickle=False)
    guard.check(full=True)
    summary = {'recorded_at':datetime.now(timezone.utc).isoformat(),'hypothetical_unlaunched_layout':True,
        'source_files_sha256':bound,'seed':seed,'optimizer_steps':steps,'world_size':4,
        'code_batch_size':1,'code_gradient_accumulation':16,'code_requested_passes':3,
        'replay_batch_size':1,'replay_accumulation':1,'replay_objective_coefficient':.5,
        'family_selection':'one family per optimizer update, round-robin with independent absolute exposure cursors',
        'families':families,'available_complete_fit_replay_rows':len(index),
        'actual_replay_example_exposures':len(actual),'distinct_replay_examples_reached':int(np.count_nonzero(counts)),
        'unused_complete_fit_replay_rows':int(np.count_nonzero(counts==0)),
        'whole_original_over_window_rows_preserved_but_not_in_fit_sampler':manifest['over_window_review_rows'],
        'actual_code_response_target_exposures':code['actual_complete_response_targets'],
        'actual_replay_response_target_exposures':sum(v['actual_complete_response_target_exposures'] for v in families.values()),
        'all_fit_replay_examples_reached':bool(np.all(counts>0)),
        'artifact_files_sha256':{name:file_hash(output/name) for name in ['draw-source-index-u32.npy','source-row-exposure-counts-u32.npy']},
        'training_started':False,'dataset_admitted_to_training':False,'goal_achieved':False,
        'scope':'Actual index-based counts for one proposed unlaunched schedule, not an adopted retention strategy or learned capability result. Separate stream token means give coefficient0.5 regardless of raw token-volume ratio. Full index validation does not imply this schedule consumes every replay row. Final replay/schedule must match the intended own checkpoint.'}
    atomic_json(output/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['source_files_sha256','families','artifact_files_sha256']}),flush=True)


if __name__=='__main__':
    main()
