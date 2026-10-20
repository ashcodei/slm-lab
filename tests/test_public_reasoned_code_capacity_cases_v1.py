"""Controlled physical packets exercise real whole-batch and short-tail geometry.

Their admission envelopes are loader fixtures, not source-corpus evidence.
"""
from collections import Counter
import copy
import gzip
import hashlib
import json

import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.codenet_token_pack_v1 import CodePackWriter
from slm_lab.corpus import file_hash
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.public_reasoned_code_capacity_cases_v1 import choose
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData, REVISION
from test_public_reasoned_code_train_data_v1 import packet


def expanded_fixture(root, count):
    folder, reports, _ = packet(root)
    with gzip.open(folder/'original-rows.jsonl.gz','rt') as stream:
        template = json.loads(next(stream))
    admission = json.loads((folder/'admission.json').read_text())
    complete = json.loads((reports/'complete.json').read_text())
    # Rebuild only this test's controlled temporary packet, before opening any
    # loader. Never mutate an actual admitted source or a loaded mmap package.
    for name in ['input-ids.bin','index.npy','original-rows.jsonl.gz','manifest.json']:
        (folder/name).unlink()
    writer = CodePackWriter(folder, maximum_bytes=1024**2, minimum_free_gib=0)
    counts = Counter()
    for position in range(count+2):
        train = position<count
        role = 'train' if train else 'validation'
        view = 'reasoning' if position%2 else 'code_only'
        ids, start = ([1,4,5,2],2)
        if train and position == count-1:
            ids, start = ([1]+[4]*4094+[2],2048)
        elif train and position == count-2:
            ids, start = ([1]+[5]*4093+[2],1)
        group = hashlib.sha256(('controlled-source-'+str(position)).encode()).hexdigest()
        row = copy.deepcopy(template)
        row.update(id='expanded-fixture:'+str(position), split=role, view=view, group_id=group,
            original_shard_position=position,
            original_fixed_role=dict(source_role='train' if train else 'reserved', optimization_forbidden=not train,
                role_fixed_before_fresh_replay=True, declared_duplicate_component_sha256=group),
            original_native_metadata=dict(complete_context_tokens=len(ids), response_EOS_targets=len(ids)-start, target_start=start))
        writer.add(row,ids,[-100]*start+ids[start:])
        counts[role+'_examples'] += 1
        counts[role+'_context_tokens'] += len(ids)
        counts[role+'_target_tokens'] += len(ids)-start
        counts[role+'_'+view+'_examples'] += 1
    writer.finish(dict(dataset='nvidia/OpenCodeReasoning-2', revision=REVISION, weak_SFT_only=True))
    manifest_sha = file_hash(folder/'manifest.json')
    admission.update(dataset_manifest_sha256=manifest_sha, complete_packed_rows_read_back=count+2,
        positive_train_rows=count, separately_reserved_rows=2, actual_native_counts=dict(counts))
    (folder/'admission.json').write_text(json.dumps(admission))
    complete.update(source_dataset_manifest_sha256=manifest_sha, admission_sha256=file_hash(folder/'admission.json'),
        complete_native_rows=count+2, actual_native_counts=dict(counts))
    (reports/'complete.json').write_text(json.dumps(complete))


@pytest.mark.parametrize('count',[4,5,16,17,19,32])
def test_every_original_source_epoch_extreme_tail_and_rank_assignment(tmp_path,count):
    expanded_fixture(tmp_path,count)
    data = PublicReasonedCodeSFTData(tmp_path)
    before = value_digest(capture_rng('cpu'))
    plan = choose(data,epochs=3,seed=990231)
    assert value_digest(capture_rng('cpu')) == before
    entries = data.complete_schedule(epochs=3,width=16,seed=990231)
    tail = count%16 or 16
    assert Counter(p for entry in entries for p in entry) == {p:3 for p in range(count)}
    assert plan['complete_training_budget'] == data.budget(entries)
    assert plan['actual_complete_training_tail_sources_per_rank'] == [len(entries[-1][r::4]) for r in range(4)]
    assert plan['actual_original_training_has_empty_ranks'] is (tail<4)
    assert plan['all_four_replicas_receive_each_selected_source_in_declared_plan'] is True
    selected = set()
    for batch in plan['selected_intact_original_batches']:
        assert batch['intact_original_positions'] == entries[batch['original_global_step']-1]
        selected.update(batch['intact_original_positions'])
    # Independently derive placement from every actual declared capacity entry.
    for position in selected:
        assert {i%4 for entry in plan['capacity_entries'] for i,p in enumerate(entry) if p == position} == {0,1,2,3}
    for entry,mapping in zip(plan['capacity_entries'],plan['capacity_case_mapping']):
        assert len(entry) == len(set(entry))
        assert mapping['actual_original_positions'] == entry
        assert mapping['actual_rank_source_counts'] == [len(entry[r::4]) for r in range(4)]
        for position in entry:
            ids,labels = data.sequence(position)
            assert ids[-1] == labels[-1] == 2 and len(ids) == data.information(position)['complete_context_tokens']
    reasons = [reason for batch in plan['selected_intact_original_batches'] for reason in batch['reasons']]
    assert any(r.get('maximum') == 'complete_context_tokens' and r['actual_value'] == 4096 for r in reasons)
    assert any(r.get('maximum') == 'response_EOS_targets' and r['actual_value'] == 4094 for r in reasons)
    assert sum(r.get('actual_original_epoch_end',False) for r in reasons) == 3
    assert plan['models_constructed'] == plan['neural_forwards'] == plan['optimizer_updates'] == plan['GPUcontexts'] == 0
    assert plan['source_selection_only'] and not plan['full_size_capacity_passed'] and not plan['training_admitted']
    capacity_data = PublicReasonedCodeSFTData(tmp_path,purpose='capacity')
    assert choose(capacity_data,epochs=3,seed=990231) == plan
    assert not torch.cuda.is_initialized()
