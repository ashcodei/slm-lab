"""Read the full saved inventory and reconcile all eight actual worker proofs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import _validate_adam
from slm_lab.code_residual_collective_checkpoint_v2 import FORMAT, fingerprint, _metadata
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_replay_v1 import FullReplayEncoder
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exclusive_inference_resources_v1 import identity
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/code-residual-collective-cuda-v2'


def read(path):
    return json.loads(path.read_text())


def main():
    output = REPORTS / 'independent-complete-readback.json'
    if output.exists():
        raise ValueError('Preserve earlier readback')
    plan_path = REPORTS / 'plan.json'
    plan = read(plan_path)
    guard = FileBindings(plan['source_files_sha256'])
    result = read(REPORTS / 'results.json')
    receipt = read(REPORTS / 'snapshot.json')
    snapshot = Path(receipt['checkpoint'])
    assert file_hash(snapshot) == receipt['checkpoint_sha256']
    saved = torch.load(snapshot, map_location='cpu', weights_only=True, mmap=True)
    assert saved['format'] == FORMAT and saved['world_size'] == 4 and saved['steps'] == 1 and saved['planned_steps'] == 2
    assert snapshot.stat().st_size == receipt['checkpoint_bytes'] < plan['maximum_checkpoint_bytes']
    assert saved['run_plan_sha256'] == file_hash(plan_path)
    base_path = ROOT / 'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
    base, metadata = load_checkpoint(base_path, 'cpu', training_state=False)
    model = RoutedCodeResidual(base, width=2048)
    assert model.parameter_counts() == saved['model_counts'] == result['workers'][0]['model_counts']
    assert state_digest(model.base.state_dict()) == saved['frozen_backbone_state_sha256']
    assert saved['residual_parameter_names'] == [name for name, _ in model.residuals.named_parameters()]
    expected = model.residuals.state_dict()
    assert set(saved['residuals']) == set(expected) and len(expected) == 48
    for name, tensor in saved['residuals'].items():
        assert tensor.shape == expected[name].shape and tensor.dtype == expected[name].dtype
        assert bool(torch.isfinite(tensor).all())
    _validate_adam(saved['optimizer'], model.adapter_parameters(), 1)
    assert state_digest(saved['residuals']) == saved['residual_state_sha256']
    assert fingerprint(saved['optimizer']) == saved['optimizer_state_sha256']
    assert fingerprint(saved['worker_rng_states']) == saved['worker_rng_states_sha256']
    assert len(saved['worker_rng_states']) == len({fingerprint(r) for r in saved['worker_rng_states']}) == 4
    for rng in saved['worker_rng_states']:
        random.Random().setstate(rng['python'])
        n = rng['numpy']
        np.random.RandomState().set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
        torch.Generator().set_state(rng['torch'])
        assert rng['cuda'].dtype == torch.uint8 and rng['cuda'].ndim == 1 and rng['cuda'].numel() > 0
    meta = _metadata(model, {key: saved[key] for key in ['base_checkpoint_sha256', 'tokenizer_sha256', 'run_plan_sha256']},
        1, 2, dict(updates=1, complete_code_programme_exposures=8, replay_exposures=4))
    assert fingerprint(dict(meta=meta, base=saved['frozen_backbone_state_sha256'],
        residuals=saved['residuals'], optimizer=saved['optimizer'])) == saved['shared_state_fingerprint']
    pack = CompleteCodePack(ROOT / 'data/codenet-durable-pack-v2', 'train', purpose='training')
    code_targets = 0
    for row in plan['original_code_rows']:
        ids, labels = pack.sequence(row['local_position'])
        assert int(pack.positions[row['local_position']]) == row['global_position']
        assert hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest() == row['full_input_ids_sha256']
        assert len(ids) == row['context_tokens'] and sum(y != -100 for y in labels) == row['target_tokens']
        code_targets += row['target_tokens']
    tokenizer = Tokenizer.from_file(str(pack.folder / 'tokenizer.json'))
    assert metadata['tokenizer_sha256'] == file_hash(pack.folder / 'tokenizer.json') == saved['tokenizer_sha256']
    encoder = FullReplayEncoder(tokenizer, model.config, ROOT / 'data/extended-curriculum-v1', 448)
    replay_targets = 0
    for row in plan['original_replay_rows']:
        ids, labels = encoder.sequence(row, 'response')
        replay_targets += sum(y != -100 for y in labels)
        if row.get('image'):
            rgb = np.rint((encoder.pixels(row).permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            assert hashlib.sha256(rgb.tobytes()).hexdigest() == row['model_pixel_sha256_448']
    workers = []
    for rank in range(4):
        before = read(REPORTS / f'reference-rank-{rank}.json')
        after = read(REPORTS / f'resume-rank-{rank}.json')
        assert after == result['workers'][rank] and before['rank'] == after['rank'] == rank
        assert before['pid'] != after['pid'] and identity(before['pid']) is None and identity(after['pid']) is None
        for key in ['complete_state_sha256', 'shared_state_sha256', 'next_draws', 'second_update', 'original_frozen_base_sha256']:
            assert before[key] == after[key]
        metrics = after['second_update']['metric']['objectives']
        assert metrics['complete_code']['global_target_tokens'] == code_targets * 4 == 1528
        assert metrics['original_replay']['global_target_tokens'] == replay_targets * 2 == 96
        assert after['second_update']['replay_has_image'] == (rank % 2 == 0)
        assert after['fresh_process_exact_restart_verified'] is True
        workers.append(dict(rank=rank, reference_pid=before['pid'], fresh_pid=after['pid'],
            complete_state_sha256=after['complete_state_sha256'],
            maximum_GPU_allocated_bytes=max(before['peak_GPU_allocated_bytes'], after['peak_GPU_allocated_bytes']),
            maximum_GPU_reserved_bytes=max(before['peak_GPU_reserved_bytes'], after['peak_GPU_reserved_bytes'])))
    assert len({w['complete_state_sha256'] for w in workers}) == 4
    raw_bytes = sum(t.numel() * t.element_size() for t in saved['residuals'].values()) * 3
    assert raw_bytes == 1_811_939_328
    gib = 1024**3
    accounting = dict(complete_measured_checkpoint_bytes=snapshot.stat().st_size, raw_weights_and_two_Adam_moments_bytes=raw_bytes,
        measured_step_RNG_and_metadata_overhead_bytes=snapshot.stat().st_size - raw_bytes,
        prospective_maximum_complete_checkpoint_bytes=2*gib, prospective_snapshot_slots=3,
        prospective_maximum_adapter_inference_export_bytes=3*gib//4,
        prospective_reports_tokenizer_and_metadata_bytes=gib//2, prospective_remaining_reserve_bytes=16*gib,
        prospective_initial_additional_free_bytes=23*gib+gib//4,
        prospective_portable_base_copy_cap_bytes=3*gib,
        prospective_portable_initial_additional_free_bytes=26*gib+gib//4,
        implemented_full_source_storage_contract=False,
        limitation='Prospective output accounting for latest/previous/atomic-pending complete adapter states, '
            'split inference export, reports/tokenizer and16GiB remaining. Original retained own base/data already occupy disk. '
            'Neither split export nor full-source driver/lifecycle/final-base/readiness is implemented here. '
            'Existing25GiB main and older full-model runner guards remain unchanged.')
    guard.check(full=True)
    files = dict(plan['source_files_sha256'])
    files[str(Path(__file__))] = file_hash(Path(__file__))
    for path in REPORTS.glob('*.json'):
        files[str(path)] = file_hash(path)
    atomic_json(output, dict(completed_at=datetime.now(timezone.utc).isoformat(),
        all48_saved_residuals_and_full_Adam_inventory_read=True, all_four_saved_worker_RNG_inventories_read=True,
        exact_own_base_counts_and_original_training_sources_rederived=True,
        all_eight_terminal_worker_report_comparisons_rederived=True, workers=workers,
        file_bindings_sha256=files, snapshot_sha256=receipt['checkpoint_sha256'], storage_accounting=accounting,
        checkpoint_is_ephemeral_fixture_not_durable_training_progress=True,
        scope='Readback of the full step1 checkpoint and source/worker proofs. Final step2 full tensors/RNG were '
            'hashed and compared by each actual GPU worker, not saved or rerun by this CPU inspector. '
            'No functional coding, knowledge, multimodal learning, human alignment or final-base/full-source gain assessed.',
        goal_achieved=False, candidate_promoted=False))
    print(json.dumps(dict(workers=workers, storage_accounting=accounting)), flush=True)


if __name__ == '__main__':
    main()
