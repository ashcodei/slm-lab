import copy
import json

import pytest
import torch

from slm_lab import own_execution_contrast_policy_CPU_v1 as policy
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


def test_bounded_machine_label_student_io_rejects_other_roles_and_excludes_training_cache(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=32, vision_intermediate_size=64,
        vision_layers=1, vision_heads=4, image_size=32, patch_size=16)
    base = OwnDenseAssistant(new_backbone(config, 969617), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1).eval()
    actor = OwnDenseActor(base, rank=2, seed=969717).eval()
    plan = dict(base_inputs=dict(arm='step-2048', checkpoint='mock-only', plan='mock-only',
        checkpoint_sha256='a' * 64, plan_sha256='b' * 64), actual_base_updates=2048,
        pair_manifest_sha256='c' * 64, own_initial_reference_cache_sha256='d' * 64,
        pairs=20, original_train_sources=4, planned_updates_each_arm=40,
        complete_human_alignment_or_source_epoch=False, actor_settings=actor.settings,
        actual_counts=actor.counts(), initial_parts_sha256=tensor_digest(actor.parts.named_parameters()))
    # The test mocks only the outer base IO; this tiny model has not actually
    # performed2048 foundation or40 trial updates.
    monkeypatch.setattr(policy, 'load_base', lambda *a, **k: (copy.deepcopy(base), dict(mock_only=True)))
    p = tmp_path / 'plan.json'; p.write_text(json.dumps(plan)); digest = file_hash(p)
    checkpoint = tmp_path / 'student.pt'; raw = policy.export(actor, plan, digest, 'initial', 0); torch.save(raw, checkpoint)
    loaded, metadata = policy.load(checkpoint, p, 'initial', expected_checkpoint_sha256=file_hash(checkpoint), expected_plan_sha256=digest)
    ids = torch.tensor([[1, 7, 8, 2]]); image = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        for pixels in [None, image]: assert torch.equal(actor(ids, pixels)['logits'], loaded(ids, pixels)['logits'])
    assert metadata['teacher_or_training_examples_not_loaded']
    assert 'reference_cache' not in raw and 'pairs' not in raw
    with pytest.raises(ValueError): policy.export(actor, plan, digest, 'anchor_contrast', 3)
    raw['complete_human_alignment_or_source_epoch'] = True; torch.save(raw, checkpoint)
    with pytest.raises(ValueError): policy.load(checkpoint, p, 'initial', expected_checkpoint_sha256=file_hash(checkpoint), expected_plan_sha256=digest)
