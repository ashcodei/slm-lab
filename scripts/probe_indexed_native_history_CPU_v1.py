"""Physical 40 GiB-reserve storage proof on closed CPU-fixture observations."""
import hashlib
import json
import os
from pathlib import Path
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/indexed-native-history-real-reserve-CPU-v1'
RESERVE = 40 * 1024**3


def main():
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
        os.environ[key] = '1'
    os.setpriority(os.PRIO_PROCESS, 0, 19)
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.sched_setaffinity(0, {22})
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.indexed_native_history_v1 import Writer, Reader, canonical, sha
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.reserved_atomic_metadata_v1 import write_json
    from test_indexed_native_history_v1 import actual_rows, budgets
    from test_native_reference_records_v1 import resident
    files = [Path(__file__).resolve(), ROOT / 'src/slm_lab/indexed_native_history_v1.py',
        ROOT / 'src/slm_lab/native_reference_records_v1.py', ROOT / 'tests/test_indexed_native_history_v1.py',
        ROOT / 'tests/test_native_reference_records_v1.py',
        ROOT / 'reports/full-native-token-suffix-qualification-gzip-recovery-CPU-v3/closed-source-bindings.json']
    bindings = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    def guard():
        if not all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in bindings.items()):
            raise ValueError('All actual source and original source-map bytes must remain unchanged')
    OUT.mkdir(parents=True, exist_ok=False)
    def publish(name, value):
        write_json(OUT / name, value, root=ROOT, minimum_free_bytes=RESERVE,
            maximum_metadata_bytes=2 * 1024**2, guard=guard)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    publish('plan.json', dict(owner=owner, source_files_sha256=bindings, actual_other_writer_reserve_bytes=RESERVE,
        original_closed_CPU_observations_only=True, production_source_map_for_transport_only=True,
        models_constructed=0, neural_updates=0, GPUcontexts=0, goal_achieved=False))
    before_rng = value_digest(capture_rng('cpu'))
    rows, reference = actual_rows()
    out = Writer(OUT / 'history', reference_binding=reference, starting_completed=0,
        budgets=budgets(maximum_records=3, maximum_physical_bytes=32 * 1024**2,
            maximum_logical_bytes=64 * 1024**2, minimum_free_bytes=RESERVE), guard=guard)
    receipts, allocations, memory = [], [], []
    for row in rows:
        ticket = out.reserve_next()
        assert shutil.disk_usage(ROOT).free >= RESERVE
        assert Path(ticket['path']).stat().st_blocks * 512 >= ticket['allocation_bytes']
        allocations.append(ticket)
        receipts.append(out.append(row))
        memory.append(resident((out.__dict__, out.codec.__dict__)))
    readback = Reader(out.root, expected_header_sha256=out.header_sha256,
        maximum_compact_record_bytes=1024**2, maximum_expanded_record_bytes=8 * 1024**2, guard=guard)
    actual = [sha(canonical(r)) for r in readback.records(receipts)]
    expected = [sha(canonical(r)) for r in rows]
    assert actual == expected
    assert value_digest(capture_rng('cpu')) == before_rng
    assert max(memory) - min(memory) < 64 * 1024
    assert sum(o['kernel'] is None for o in rows[-1]['actual_rank_observations']) == 3
    guard()
    publish('complete.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        source_files_sha256=bindings, actual_other_writer_reserve_bytes=RESERVE,
        actual_free_bytes_after=shutil.disk_usage(ROOT).free, actual_physically_funded_tickets=allocations,
        actual_exact_records=3, actual_closed_source_binding_count=len(reference['sources']),
        all_original_reference_action_probability_verdict_and_timing_bytes_reconstructed=True,
        original_empty_rank_tail_preserved=True, all_python_numpy_torch_RNG_unchanged=True,
        actual_writer_control_and_all_immutable_codec_field_resident_bytes=memory,
        resident_measurement_scope='Recursive primitive/container size of writer and codec dictionaries, identity-deduplicated; opaque Path/guard internals and interpreter/globals excluded',
        actual_checkpoint_receipt_resident_bytes=resident(receipts), original_logical_bytes=out.logical_bytes,
        actual_compressed_and_receipt_file_bytes=out.physical_bytes,
        actual_reference_header_sha256=out.header_sha256, actual_receipts=receipts,
        original_small_CPU_observations_only=True, real_source_map_substitution_for_transport_only=True,
        no_neural_forward_or_update_performed=True, production_GPU_capacity_or_training_admitted=False,
        model_quality_gain=False, goal_achieved=False))
    print(json.dumps(dict(complete=True, actual_records=3, actual_reserve=RESERVE,
        writer_plus_codec_resident_bytes=memory, logical_bytes=out.logical_bytes,
        physical_file_bytes=out.physical_bytes, neural_updates=0)), flush=True)


if __name__ == '__main__':
    main()
