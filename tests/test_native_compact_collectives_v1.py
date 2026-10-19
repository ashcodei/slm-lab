"""Four real Gloo transports of closed observations, with peer fault checks."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import pickle

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.native_reference_records_v1 import NativeReferenceRecords, canonical
from slm_lab.native_suffix_compact_stage_v3 import NativeSuffixCompactStageV3
from test_native_reference_records_v1 import fixture


def worker(rank, directory):
    torch.set_num_threads(1)
    folder = Path(directory)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'),
        rank=rank, world_size=4, timeout=timedelta(minutes=10))
    try:
        row, reference = fixture()
        stage = NativeSuffixCompactStageV3.__new__(NativeSuffixCompactStageV3)
        stage.rank, stage.world, stage.control_group = rank, 4, None
        stage.failure_observations = []
        stage.reference_records = NativeReferenceRecords(reference)
        original = row['actual_rank_observations'][rank]
        wire_bytes = len(pickle.dumps(stage.reference_records.pack_observation(original), protocol=2))
        expanded = stage.gather_observations(original)
        assert canonical(expanded) == canonical(row['actual_rank_observations'])
        assert all(o['kernel']['own_reference_binding'] is stage.reference_records.reference for o in expanded)
        wrong = copy.deepcopy(original)
        if rank == 2: wrong['kernel']['own_reference_binding']['external_pretrained_teacher'] = 0
        with pytest.raises(ValueError, match='before compact collective'):
            stage.gather_observations(wrong)
        before = stage.failure_observations[-1]
        expected_bad = copy.deepcopy(row['actual_rank_observations'][2])
        expected_bad['kernel']['own_reference_binding']['external_pretrained_teacher'] = 0
        assert before['actual_rank_errors'][2]['complete_original_local_observation'] == expected_bad
        class BadMarker(NativeReferenceRecords):
            def pack_observation(self, value):
                result = super().pack_observation(value)
                if rank == 2: result['kernel']['own_reference_binding']['sha256'] = '0' * 64
                return result
        stage.reference_records = BadMarker(reference)
        with pytest.raises(ValueError, match='after compact collective'):
            stage.gather_observations(original)
        assert all(e is not None for e in stage.failure_observations[-1]['actual_rank_errors'])
        dist.barrier()
        assert not torch.cuda.is_initialized()
        (folder / ('rank-' + str(rank) + '.json')).write_text(json.dumps(dict(
            rank=rank, actual_closed_binding_count=len(reference['sources']), actual_wire_bytes=wire_bytes,
            every_original_observation_byte_equal=True, full_header_shared_after_actual_collective=True,
            peer_pretransport_and_posttransport_faults_coordinated_without_hang=True,
            complete_original_failure_observations_retained=True,
            neural_updates=0, production_capacity_or_quality_gain=False), indent=2))
    finally:
        dist.destroy_process_group()


def test_actual_four_rank_transport_and_coordinated_peer_faults(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    reports = [json.loads((tmp_path / ('rank-' + str(r) + '.json')).read_text()) for r in range(4)]
    assert all(r['every_original_observation_byte_equal'] for r in reports)
    assert len({r['actual_closed_binding_count'] for r in reports}) == 1
