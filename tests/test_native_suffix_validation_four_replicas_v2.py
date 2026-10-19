"""Run the complete native-worker/recovery/fault trial through the V2 stage."""
import hashlib
import json
from pathlib import Path

from torch.multiprocessing import spawn

from slm_lab.native_suffix_validation_stage_v2 import NativeSuffixValidationStageV2, PROTOCOL
import test_parallel_native_suffix_stage_v1 as original


def worker(rank, folder):
    # This changes only the fixture factory in this new spawned test process.
    # The original source file, original live tests, and canonical GPU jobs
    # retain their own unchanged imports and process state.
    original.ParallelNativeSuffixStage = NativeSuffixValidationStageV2
    factory = original.setup
    actual = []
    def checked_factory(*args, **kwargs):
        stage = factory(*args, **kwargs)
        assert type(stage) is NativeSuffixValidationStageV2
        assert stage.binding()['recorded_source_validation'] == PROTOCOL
        actual.append(dict(profile=stage.profile, stage_class=type(stage).__name__,
            validation_protocol=stage.binding()['recorded_source_validation'], world=stage.world,
            actual_small_parameters=stage.model.counts()['actual_unique']))
        return stage
    original.setup = checked_factory
    original.worker(rank, folder)
    assert {r['profile'] for r in actual} == {'anchor_CE', 'anchor_execution'} and len(actual) == 2
    root = Path(folder)
    files = {}
    for profile in ['anchor_CE', 'anchor_execution']:
        path = root / (profile + '-rank-' + str(rank) + '.json')
        with path.open('rb') as stream: files[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (root / ('actual-v2-rank-' + str(rank) + '.json')).write_text(json.dumps(dict(
        rank=rank, actual_complete_matched_trial=actual, original_trial_evidence_sha256=files,
        all_actual_updates_source_mean_empty_tail_recovery_and_peer_failure_checks_completed=True,
        full_size_GPUcapacity_or_model_quality=False, production_updates=0), indent=2, allow_nan=False))


def test_actual_four_V2_replicas_complete_native_source_mean_recovery_empty_tail_and_peer_faults(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    identities = [json.loads((tmp_path / ('actual-v2-rank-' + str(r) + '.json')).read_text()) for r in range(4)]
    assert all(len(i['actual_complete_matched_trial']) == 2 for i in identities)
    for profile in ['anchor_CE', 'anchor_execution']:
        proofs = [json.loads((tmp_path / (profile + '-rank-' + str(r) + '.json')).read_text()) for r in range(4)]
        assert len({p['actual_final_shared_checkpoint_sha256'] for p in proofs}) == 1
        assert all(p['actual_serial_full_source_mean_matches'] and p['actual_CE_only_source_retained'] for p in proofs)
