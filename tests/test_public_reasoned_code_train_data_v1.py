"""Small physical loader fixtures, not evidence of full-corpus admission."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from slm_lab.codenet_token_pack_v1 import CodePackWriter
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import new_backbone, OwnDenseAssistant
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData, REVISION


def packet(root, *, owner=None):
    data = root/'data/public-reasoned-code-SFT-v1'
    reports = root/'reports/public-reasoned-code-pack-admission-v1'
    data.mkdir(parents=True)
    reports.mkdir(parents=True)
    for name in ['tokenizer.json', 'model-config.json', 'DATASET-LICENSE.txt', 'publisher-README.md']:
        (data/name).write_text('{}\n')
    writer = CodePackWriter(data, maximum_bytes=1024**2, minimum_free_gib=0)
    definitions = [('train', 'code_only', 0, [1,4,5,2], 2),
        ('train', 'reasoning', 0, [1,4,6,7,2], 2), ('train', 'code_only', 1, [1,8,4,5,2], 3),
        ('validation', 'code_only', 2, [1,9,5,2], 2), ('validation', 'reasoning', 2, [1,9,6,7,2], 2)]
    counts = Counter()
    for i, (split, view, source, ids, start) in enumerate(definitions):
        fixed = dict(source_role='train' if split=='train' else 'reserved', optimization_forbidden=split!='train',
            role_fixed_before_fresh_replay=True, declared_duplicate_component_sha256=str(source+1)*64)
        row = dict(id='fixture:'+str(i), split=split, publisher_split='train', view=view,
            source_revision=REVISION, original_shard_position=source, group_id=str(source+1)*64,
            original_fixed_role=fixed, model_generated_examples=True, human_preferences=False,
            own_model_on_policy=False, use_as_RL_reward_oracle_forbidden=True, admitted_use='weak_supervised_coding_only',
            external_teacher_weights_loaded_by_this_pipeline=False, live_teacher_API_used=False,
            original_native_metadata=dict(complete_context_tokens=len(ids), response_EOS_targets=len(ids)-start, target_start=start))
        writer.add(row, ids, [-100]*start+ids[start:])
        counts[split+'_examples'] += 1
        counts[split+'_context_tokens'] += len(ids)
        counts[split+'_target_tokens'] += len(ids)-start
        counts[split+'_'+view+'_examples'] += 1
    manifest = writer.finish(dict(dataset='nvidia/OpenCodeReasoning-2', revision=REVISION, weak_SFT_only=True))
    ancestor = root/'controlled-source-ancestor.txt'
    ancestor.write_text('Loader-only fixture; does not establish source-corpus execution or admission.\n')
    owner = owner or dict(pid=99999999, birth='0')
    admission = dict(format='slm-public-reasoned-code-weak-SFT-admission-v1', owner=owner,
        admitted_to_training=True, admitted_use='weak_supervised_coding_only', synthetic_teacher_data=True,
        external_teacher_weights_loaded=False, live_teacher_API_used=False, human_preferences=False,
        own_model_on_policy=False, use_as_RL_reward_oracle_forbidden=True, independent_semantic_correctness_certified=False,
        original_fixed_roles_and_every_whole_native_input_and_EOS_mask_rebuilt=True,
        complete_original_source_decisions=20000, complete_source_execution_candidates_read_back=6116,
        declared_duplicate_component_cross_role_intersections=0, dataset_manifest_sha256=file_hash(data/'manifest.json'),
        complete_packed_rows_read_back=5, positive_train_rows=3, separately_reserved_rows=2,
        actual_native_counts=dict(counts), source_files_sha256={str(ancestor):file_hash(ancestor)})
    (data/'admission.json').write_text(json.dumps(admission))
    (reports/'owner-plan.json').write_text(json.dumps(dict(owner=owner, controlled_fixture_only=True)))
    complete = dict(owner=owner, complete=True, data_only_weak_SFT_admitted=True,
        full_native_text_arrays_and_roles_rebuilt_independently=True, original_source_positions=20000,
        original_fixed_role_cross_intersections=0, reserved_optimization_forbidden=True,
        use_as_RL_reward_oracle_forbidden=True, admission_sha256=file_hash(data/'admission.json'),
        source_dataset_manifest_sha256=file_hash(data/'manifest.json'), complete_native_rows=5, actual_native_counts=dict(counts))
    (reports/'complete.json').write_text(json.dumps(complete))
    return data, reports, ancestor


def test_complete_train_views_schedules_and_copy_isolation(tmp_path):
    packet(tmp_path)
    data = PublicReasonedCodeSFTData(tmp_path)
    assert len(data) == 3 and all(data.row(p)['split']=='train' for p in range(len(data)))
    schedule = data.complete_schedule(epochs=2, width=2, seed=901)
    assert [len(entry) for entry in schedule] == [2,1,2,1]
    assert Counter(p for entry in schedule for p in entry) == {0:2,1:2,2:2}
    assert data.budget(schedule)['response_EOS_targets'] == 14
    code = PublicReasonedCodeSFTData(tmp_path, views='code_only')
    reasoning = PublicReasonedCodeSFTData(tmp_path, views='reasoning')
    assert len(code) == 2 and len(reasoning) == 1
    assert data.identity([[0]]) != code.identity([[0]])
    row = data.row(0)
    row['original_fixed_role']['source_role'] = 'reserved'
    assert data.row(0)['original_fixed_role']['source_role'] == 'train'
    with pytest.raises(ValueError):
        data.batch([0,0])
    with pytest.raises(ValueError):
        data.score('anything')
    data.views = 'code_only'
    with pytest.raises(ValueError):
        data.sequence(0)


def test_live_or_missing_admission_and_changed_native_bytes_are_rejected(tmp_path):
    data_path, reports, _ = packet(tmp_path)
    data = PublicReasonedCodeSFTData(tmp_path)
    path = data_path/'input-ids.bin'
    raw = bytearray(path.read_bytes())
    raw[2] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        data.batch([0])
    (reports/'complete.json').unlink()
    with pytest.raises(ValueError):
        PublicReasonedCodeSFTData(tmp_path)
    live_root = tmp_path/'live'
    import os
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    packet(live_root, owner=owner)
    with pytest.raises(ValueError, match='Terminal'):
        PublicReasonedCodeSFTData(live_root)


def test_hot_native_guard_is_compact_but_final_provenance_audit_is_complete(tmp_path):
    _, _, ancestor = packet(tmp_path)
    data = PublicReasonedCodeSFTData(tmp_path)
    assert str(ancestor) in data.provenance_source_files_sha256
    assert str(ancestor) not in data.source_files_sha256
    expected = data.batch([0])['input_ids'].clone()
    ancestor.write_text('deliberate controlled historical-source corruption\n')
    assert torch.equal(data.batch([0])['input_ids'], expected)
    with pytest.raises(ValueError):
        data.audit_provenance()


def test_actual_own_actor_update_uses_all_complete_masked_targets(tmp_path):
    packet(tmp_path)
    data = PublicReasonedCodeSFTData(tmp_path)
    batch = data.batch([0,1,2])
    assert batch['input_ids'].shape == (3,5)
    assert batch['labels'][0,-1].item() == -100 and not batch['attention_mask'][0,-1]
    config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=32, vision_hidden_size=16,
        vision_intermediate_size=32, vision_layers=1, vision_heads=4, image_size=16, patch_size=8, pool_size=2)
    base = OwnDenseAssistant(new_backbone(config, 991001), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4)
    model = OwnDenseActor(base, rank=2)
    params = [p for _,p in model.named_selected(('actor',))]
    optimizer = torch.optim.AdamW(params, lr=.001, weight_decay=0., foreach=False)
    before = [p.detach().clone() for p in params]
    output = model(**batch)
    logits, labels = output['logits'][:,:-1], batch['labels'][:,1:]
    total = labels.ne(-100).sum()
    assert total.item() == 7
    oracle = F.cross_entropy(logits.reshape(-1,32), labels.reshape(-1), ignore_index=-100, reduction='sum')/total
    torch.testing.assert_close(output['loss'], oracle)
    output['loss'].backward(inputs=params)
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
    assert all(p.grad is None for p in base.parameters())
    optimizer.step()
    assert any(not torch.equal(a,b) for a,b in zip(before,params))
    data.audit_provenance()
