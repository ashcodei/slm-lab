"""Read every retained original V1record under the unchanged16MiBpeak bound.

The first V1run preserved all5362synthetic-metric source-shaped files. Its
read(maximum+1) temporarily allocates32MiB for a tiny header. V2must preserve
every original field/byte/hash while bounding that read by actual file size.
"""
import gc
import gzip
import json
from pathlib import Path
import tracemalloc

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.indexed_native_history_v1 import canonical,read_regular as old_read
from slm_lab.joint_indexed_history_v2 import read_regular,decompress
from slm_lab.joint_indexed_checkpoint_v3 import ENVELOPE,scan
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.shared_joint_checkpoint_v2 import validate_adam
from test_joint_task_gradients_v1 import view

ROOT=Path(__file__).resolve().parents[1]


def test_tiny_header_actual_length_read_preserves_bytes_and_avoids_maximum_buffer_allocation(tmp_path):
    path=tmp_path/'tiny-header';path.write_bytes(b'whole original header')
    tracemalloc.start();old=old_read(path,32*1024**2);_,old_peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    tracemalloc.start();actual=read_regular(path,32*1024**2);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert actual==old==path.read_bytes() and peak<1024**2
    with pytest.raises(ValueError):read_regular(path,4)
    (tmp_path/'actual-header-read-memory.json').write_text(json.dumps(dict(actual_bytes=len(actual),old_incremental_peak=old_peak,new_incremental_peak=peak),indent=2))


def test_read_growth_rejects_without_clipping_to_the_previous_file_length(tmp_path,monkeypatch):
    path=tmp_path/'growing';path.write_bytes(b'abc');original=Path.open
    class Growing:
        def __init__(self,stream):self.stream=stream;self.once=False
        def __enter__(self):return self
        def __exit__(self,*args):return self.stream.__exit__(*args)
        def fileno(self):return self.stream.fileno()
        def read(self,n):
            if not self.once:
                self.once=True
                with original(path,'ab') as stream:stream.write(b'def')
            return self.stream.read(n)
    def changed(self,*args,**kwargs):
        stream=original(self,*args,**kwargs)
        return Growing(stream) if self==path else stream
    monkeypatch.setattr(Path,'open',changed)
    with pytest.raises(ValueError,match='changed during'):read_regular(path,100)


@pytest.mark.parametrize('fault',['over_quota','truncated','CRC'])
def test_chunked_gzip_retains_complete_original_bound_and_validates_trailer(fault):
    original=b'z'*70000;raw=gzip.compress(original,mtime=0)
    assert decompress(raw,len(original))==original
    if fault=='over_quota':
        with pytest.raises(ValueError):decompress(raw,len(original)-1)
    elif fault=='truncated':
        with pytest.raises(EOFError):decompress(raw[:-2],len(original))
    else:
        broken=raw[:-8]+bytes([raw[-8]^1])+raw[-7:]
        with pytest.raises(gzip.BadGzipFile):decompress(broken,len(original))


def test_all_retained5362original_source_records_preserve_complete_semantics_under_original_memory_bound(tmp_path):
    admission=ROOT/'reports/complete-joint-objective-bridge-CPU-v1/complete.json'
    if not admission.exists():pytest.skip('Requires the prepared complete original joint source recipe')
    closed=json.loads(admission.read_text());source=Path(closed['entries_path']);source_sha=file_hash(source)
    assert source_sha==closed['entries_file_sha256']
    entries=json.loads(source.read_text());assert len(entries)==5362 and value_digest(entries)==closed['complete_checkpoint_schedule_typed_sha256']
    root=ROOT/'reports/joint-indexed-full-history-v1-pytest/test_every_original5362source_0/history'
    directory=root/'attempt-000001'
    if not (directory/'reference.json.gz').exists():pytest.skip('Requires the retained complete5362record storage probe; see scripts/probe_joint_indexed_full_history_CPU_v1.py')
    header=json.loads(gzip.decompress((directory/'reference.json.gz').read_bytes()))
    receipts=[];header_sha=file_hash(directory/'reference.json.gz')
    for index in range(1,5363):
        receipt=json.loads((directory/('record-'+str(index).zfill(12)+'.receipt.json')).read_text())
        receipts.append(dict(format=ENVELOPE,directory=str(directory),header_sha256=header_sha,record=receipt))
    current=view();original_stat=source.stat()
    def guard():
        actual=source.stat()
        assert (actual.st_ino,actual.st_size,actual.st_mtime_ns)==(original_stat.st_ino,original_stat.st_size,original_stat.st_mtime_ns)
    history=dict(binding=header['reference_binding'],budgets=header['budgets'],maximum_attempts=1,history_root=str(root))
    gc.collect();tracemalloc.start();boundary=scan(current,entries,len(entries),receipts,history,guard)
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    assert peak<16*1024**2
    clocks=boundary['actual_per_parameter_Adam_steps']
    for name,count in clocks.items():
        assert count==(5362 if name.startswith('actor.') else 4733 if name.startswith('vision.') else 1432 if name.startswith('word.') else 1223),name
    optimizer=torch.optim.AdamW([p for _,p in current.named_selected()],lr=1e-4,foreach=False)
    with pytest.raises(ValueError):validate_adam(optimizer.state_dict(),optimizer,current.named_selected(),clocks)
    assert file_hash(source)==source_sha and not torch.cuda.is_initialized()
    evidence=dict(actual_all_original_source_steps=5362,actual_complete_original_checkpoint_schedule_sha256=value_digest(entries),actual_entries_file_sha256=source_sha,
        actual_full_original_context_tokens=sum(o['complete_context_tokens'] for e in entries for o in e['objectives']),
        actual_full_original_response_EOS_targets=sum(o['response_EOS_targets'] for e in entries for o in e['objectives']),
        actual_full_original_CTC_UTF8_bytes=sum(o['word_UTF8_bytes'] for e in entries for o in e['objectives']),
        actual_sparse_group_clocks=dict(actor=5362,vision=4733,word=1432,pointing=1223),
        actual_incremental_tracemalloc_reader_peak_bytes=peak,existing_schedule_receipts_model_and_interpreter_baseline_not_in_incremental_peak=True,
        actual_receipt_envelopes_JSON_bytes=len(canonical(receipts)),actual_original_history_logical_bytes=len(canonical(header['reference_binding']))+sum(e['record']['original_expanded_bytes'] for e in receipts),
        all_original5362failed_run_records_reused_byte_exact=True,original16MiB_memory_bound_unchanged=True,
        all_metric_values_and_repeated_padding_are_explicit_non_neural_controls=True,empty_untrained_Adam_cannot_adopt_fabricated_full_horizon_clocks=True,
        model_forwards=0,optimizer_updates=0,GPU_contexts=0,full_size_capacity_training_or_model_quality=False)
    (tmp_path/'actual-full-original-joint-history.json').write_text(json.dumps(evidence,indent=2))
