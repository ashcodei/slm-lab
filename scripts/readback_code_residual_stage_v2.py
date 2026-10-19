"""Reconcile the actual source interface checks and retained four-CPU artifacts."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_stage_v2 import FULL_REPLAY_FAMILIES
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/code-residual-stage-v2'


def read(path):
    return json.loads(path.read_text())


def main():
    output = REPORTS / 'independent-complete-readback.json'
    if output.exists():
        raise ValueError('Preserve earlier independent readback')
    source_path = ROOT / 'reports/code-residual-stage-v2-source-check/results.json'
    source = read(source_path)
    guard = FileBindings(source['source_files_sha256'])
    checkpoint = torch.load(REPORTS / 'complete.pt', weights_only=True, mmap=True)
    partial = torch.load(REPORTS / 'resume.pt', weights_only=True, mmap=True)
    inference = torch.load(REPORTS / 'inference.pt', weights_only=True, mmap=True)
    assert checkpoint['steps'] == checkpoint['planned_steps'] == 19
    assert partial['steps'] == 7 and partial['planned_steps'] == 19
    assert [m['step'] for m in checkpoint['metrics']] == list(range(1, 20))
    assert checkpoint['metrics'][:7] == partial['metrics']
    assert state_digest(checkpoint['residuals']) == checkpoint['residual_state_sha256']
    assert fingerprint(checkpoint['optimizer']) == checkpoint['optimizer_state_sha256']
    assert fingerprint(checkpoint['worker_rng_states']) == checkpoint['worker_rng_states_sha256']
    assert len(checkpoint['worker_rng_states']) == len({fingerprint(r) for r in checkpoint['worker_rng_states']}) == 4
    for index, tensor in enumerate(checkpoint['residuals'].values()):
        adam = checkpoint['optimizer']['state'][index]
        assert set(adam) == {'step', 'exp_avg', 'exp_avg_sq'} and adam['step'].item() == 19
        assert adam['exp_avg'].shape == adam['exp_avg_sq'].shape == tensor.shape
        assert all(bool(torch.isfinite(t).all()) for t in [tensor, adam['exp_avg'], adam['exp_avg_sq']])
    assert inference['training_checkpoint_sha256'] == file_hash(REPORTS / 'complete.pt')
    assert inference['base_checkpoint_sha256'] == file_hash(REPORTS / 'base.pt')
    assert inference['run_plan_sha256'] == file_hash(REPORTS / 'plan.json')
    assert inference['tokenizer_sha256'] == file_hash(REPORTS / 'tokenizer.json')
    assert inference['data_sha256'] == file_hash(REPORTS / 'data-manifest.json')
    assert fingerprint(inference['residuals']) == fingerprint(checkpoint['residuals'])
    assert not {'optimizer', 'worker_rng_states', 'rng'} & inference.keys()
    progress = checkpoint['sampler_state']
    assert set(progress['replay_families']) == set(FULL_REPLAY_FAMILIES)
    assert progress['code_examples'] == progress['replay_examples'] == 76
    assert set(m['replay_family'] for m in checkpoint['metrics']) == set(FULL_REPLAY_FAMILIES)
    workers = [read(REPORTS / f'rank-{rank}.json') for rank in range(4)]
    assert all(w['exact_residual_Adam_and_all_rank_RNG_continuation'] for w in workers)
    pack = CompleteCodePack(ROOT / 'data/codenet-durable-pack-v2', 'train', purpose='training')
    expected_code_positions = {0, int(np.argmax(pack.index[pack.positions]['length'])),
        int(np.argmax(pack.index[pack.positions]['length'] - pack.index[pack.positions]['target_start']))}
    expected_code_positions.update(group[0] for rank in range(4)
        for group in EpochOrder(len(pack), 71).rank_groups(0, rank, 4, 1, 16))
    assert {row['local_position'] for row in source['selected_actual_code_batches']} == expected_code_positions
    for row in source['selected_actual_code_batches']:
        ids, labels = pack.sequence(row['local_position'])
        assert row['global_position'] == int(pack.positions[row['local_position']])
        assert row['context_tokens'] == len(ids) and row['target_tokens'] == sum(y != -100 for y in labels)
        assert row['complete_input_sha256'] == hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()
    families = {}
    for folder in ['codenet-replay-v1', 'codenet-fact-replay-v1', 'codenet-dialogue-replay-v1', 'codenet-image-replay-v1']:
        path = ROOT / 'data' / folder
        manifest = read(path / 'manifest.json')
        index = np.load(path / 'index.npy', mmap_mode='r', allow_pickle=False)
        for i, name in enumerate(manifest['family_names']):
            rows = index[index['family'] == i]
            assert len(rows) == manifest['family_counts'][name]
            assert name not in families
            families[name] = rows
    assert tuple(families) == FULL_REPLAY_FAMILIES and sum(len(rows) for rows in families.values()) == 348133
    raw_replay_batches = 0
    for record in source['selected_actual_replay_batches']:
        rows = families[record['family']]
        assert record['complete_fit_population'] == len(rows)
        selected = {0, int(np.argmax(rows['context_tokens'])), int(np.argmax(rows['target_tokens']))}
        assert {row['local_position'] for row in record['checks']} == selected
        for row in record['checks']:
            original = rows[row['local_position']]
            assert row['context_tokens'] == int(original['context_tokens'])
            assert row['targets'] == int(original['target_tokens'])
            assert row['complete_input_sha256'] == original['input_sha256'].tobytes().hex()
            raw_replay_batches += 1
    guard.check(full=True)
    files = dict(source['source_files_sha256'])
    files[str(Path(__file__))] = file_hash(Path(__file__))
    files[str(source_path)] = file_hash(source_path)
    for path in REPORTS.iterdir():
        # This inspector's redirected log changes after the manifest is written.
        # Bind stable artifacts and the separately frozen focused-test log.
        if path.is_file() and path.suffix != '.log':
            files[str(path)] = file_hash(path)
    files[str(ROOT / 'src/slm_lab/code_residual_export_v2.py')] = file_hash(ROOT / 'src/slm_lab/code_residual_export_v2.py')
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), source_and_artifact_files_sha256=files,
        actual_prepared_code_population=len(pack), actual_prepared_replay_population=348133,
        actual_replay_families=list(families), selected_real_code_batches_rederived=len(expected_code_positions),
        selected_real_replay_batches_index_fields_rederived=raw_replay_batches,
        prospective_first_update_seed_used_for_source_check=71,
        actual_maximum_context_in_code_checks=max(r['context_tokens'] for r in source['selected_actual_code_batches']),
        actual_maximum_response_targets_in_code_checks=max(r['target_tokens'] for r in source['selected_actual_code_batches']),
        CPU_world_size=4, CPU_model_counts=checkpoint['model_counts'], CPU_completed_kernel_steps=19,
        all19_saved_cursors_complete=True, complete_training_and_separate_inference_residuals_identical=True,
        complete_Adam_and_all_rank_RNG_preserved=True,
        real_source_encoder_pixels_and_masks_checked_in_preceding_report=True,
        GPU_training_updates_this_interface_verification=0,
        complete_checkpoint_bytes=(REPORTS / 'complete.pt').stat().st_size,
        inference_checkpoint_bytes=(REPORTS / 'inference.pt').stat().st_size,
        production_launcher_and_final_base_match=False, all_later_queued_sources_covered=False,
        candidate_promoted=False, goal_achieved=False,
        scope='Full retained CPU fixture checkpoint/export and real-source index/report readback. '
            'The preceding source check actually batched selected first/extreme real inputs and448px pixels; '
            'this reader rederives their indexed identities/counts, not every image render. '
            'Four-rank neural continuation uses an8688-parameter own replica and fixture programs. '
            'Not actual857M/all-source training, learned capability or readiness for final production launch.')
    atomic_json(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'source_and_artifact_files_sha256'}), flush=True)


if __name__ == '__main__':
    main()
