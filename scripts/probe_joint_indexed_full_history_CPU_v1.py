"""Full original source-shaped storage, deliberately non-neural metrics.

Repeated padding stresses incremental reader memory only. Its compression
ratio is not a prediction of real training history. No model forward or
optimizer update occurs and the resulting sparse clocks reject empty Adam.
"""
import gc
import json
from pathlib import Path
import tracemalloc

import pytest
import torch

from slm_lab.corpus import file_hash
from slm_lab.indexed_native_history_v1 import canonical
from slm_lab.joint_indexed_history_v1 import Writer
from slm_lab.joint_indexed_checkpoint_v1 import ENVELOPE,scan
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.shared_joint_checkpoint_v2 import validate_adam
from test_joint_task_gradients_v1 import view

ROOT=Path(__file__).resolve().parents[1]


def test_every_original5362source_step_streams_without_resident_metric_history_or_fabricated_Adam(tmp_path):
    admission=ROOT/'reports/complete-joint-objective-bridge-CPU-v1/complete.json'
    closed=json.loads(admission.read_text());source=Path(closed['entries_path']);source_sha=file_hash(source)
    assert source_sha==closed['entries_file_sha256']
    entries=json.loads(source.read_text());assert len(entries)==5362 and value_digest(entries)==closed['complete_checkpoint_schedule_typed_sha256']
    current=view();root=tmp_path/'history';root.mkdir();original_stat=source.stat()
    def guard():
        actual=source.stat()
        assert (actual.st_ino,actual.st_size,actual.st_mtime_ns)==(original_stat.st_ino,original_stat.st_size,original_stat.st_mtime_ns)
    budgets=dict(maximum_records=5362,maximum_compact_record_bytes=128*1024,maximum_expanded_record_bytes=128*1024,
        maximum_logical_bytes=1024**3,maximum_physical_bytes=96*1024**2,minimum_free_bytes=40*1024**3)
    binding=dict(sources={'actual_original_data_sha256':closed['actual_original_data_identity_sha256']},model_metadata={'actual_own_small_counts':current.actor.counts()},
        own_text_reference={'synthetic_storage_only_not_a_training_reference':True},options={'all_metric_values_synthetic':True},world_size=1,optimizer_settings_sha256='a'*64)
    history=dict(binding=binding,budgets=budgets,maximum_attempts=1,history_root=str(root))
    writer=Writer(root/'attempt-000001',reference_binding=binding,starting_completed=0,budgets=budgets,guard=guard)
    receipts=[];padding='synthetic_non_neural_memory_stress:'+'p'*65536
    for entry in entries:
        active={n for objective in entry['objectives'] for n in current.task_names(objective['task'])}
        metric=dict(step=entry['step'],entry_sha256=value_digest(entry),active_parameter_names=[n for n,_ in current.named_selected() if n in active],
            normalized_task_losses=[0. for _ in entry['objectives']],gradient_norm_before_clip=0.,globally_reduced_task_gradients_before_combining=True,
            gradient_interaction=dict(synthetic_values_not_actual_neural_training=True,original_source_step=entry['step'],non_neural_repeated_padding=padding))
        writer.reserve_next();receipt=writer.append(metric);writer.acknowledge_success(entry['step'])
        receipts.append(dict(format=ENVELOPE,directory=str(writer.root),header_sha256=writer.header_sha256,record=receipt))
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
        actual_receipt_envelopes_JSON_bytes=len(canonical(receipts)),actual_original_history_logical_bytes=writer.logical_bytes,
        all_original_metrics_remain_in_individual_immutable_files=True,all_metric_values_and_repeated_padding_are_explicit_non_neural_controls=True,
        empty_untrained_Adam_cannot_adopt_fabricated_full_horizon_clocks=True,model_forwards=0,optimizer_updates=0,GPU_contexts=0,full_size_capacity_training_or_model_quality=False)
    (tmp_path/'actual-full-original-joint-history.json').write_text(json.dumps(evidence,indent=2))
