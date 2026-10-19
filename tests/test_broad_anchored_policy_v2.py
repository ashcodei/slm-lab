import copy
import json

import pytest
import torch

from slm_lab import broad_anchored_policy_v2 as policy
from slm_lab import phase_human_policy_v2 as human
from slm_lab.corpus import file_hash
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.parallel_human_preference_v2 import ParallelHumanPreferenceStage
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_phase_human_policy_v2 import fixture


def trained_start(tmp_path):
    model, data, plan, human_plan = fixture(tmp_path)
    setting = human.options(plan, 'human_DPO')
    optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))],
        lr=setting['learning_rate'], betas=tuple(setting['betas']), weight_decay=setting['weight_decay'], foreach=False)
    ParallelHumanPreferenceStage(model, optimizer, data, setting, human.sources(plan, file_hash(human_plan))).update()
    checkpoint = tmp_path/'human.pt'
    torch.save(human.export(model, plan, file_hash(human_plan), 'human_DPO', 1), checkpoint)
    inputs = dict(checkpoint=str(checkpoint), plan=str(human_plan), checkpoint_sha256=file_hash(checkpoint), plan_sha256=file_hash(human_plan))
    plan = dict(starting_policy_inputs=inputs, actual_counts=model.counts(),
        frozen_dense_sha256=tensor_digest(model.base.named_parameters()),
        starting_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),
        planned_steps=1, original_coding_sources=1, APPS_execution_admission_sha256='e'*64,
        documented_probe_admission_sha256='f'*64, small_synthetic_unit_fixture=True)
    plan['own_reference_binding'] = FrozenOwnActorReference(model, sources=policy.reference_sources(plan)).binding()
    run_plan = tmp_path/'code-plan.json'; run_plan.write_text(json.dumps(plan))
    return model, plan, run_plan


def test_own_trained_start_anchor_update_native_text_image_cold_load_and_student_inventory(tmp_path):
    model, plan, run_plan = trained_start(tmp_path)
    fresh, _ = policy.fresh(plan); reference = policy.make_reference(fresh, plan)
    assert tensor_digest(fresh.parts.named_parameters()) == plan['starting_actor_parts_sha256']
    ids = torch.tensor([[1, 4, 7, 2]]); parameters = [p for _, p in model.named_selected(('actor',))]
    optimizer = torch.optim.AdamW(parameters, lr=.001, foreach=False)
    model(ids, labels=ids)['loss'].backward(inputs=parameters); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    path = tmp_path/'code.pt'
    torch.save(policy.export(model, plan, file_hash(run_plan), 'anchor_execution', 1), path)
    loaded, _ = policy.load(path, run_plan, 'anchor_execution',
        expected_checkpoint_sha256=file_hash(path), expected_plan_sha256=file_hash(run_plan))
    pixels = torch.zeros(1, 3, model.config.image_size, model.config.image_size)
    with torch.no_grad():
        assert torch.equal(model(ids)['logits'], loaded(ids)['logits'])
        assert torch.equal(model(ids, pixels)['logits'], loaded(ids, pixels)['logits'])
    assert loaded.counts() == plan['actual_counts']
    assert reference.resident_inventory(fresh)['extra_training_only_own_reference_values'] > 0


def test_partial_wrong_source_critic_and_reference_bindings_reject(tmp_path):
    model, plan, run_plan = trained_start(tmp_path)
    raw = policy.export(model, plan, file_hash(run_plan), 'anchor_CE', 1); path = tmp_path/'bad.pt'
    for mutate in [lambda r:r.update(completed_updates=0),
                   lambda r:r.update(original_coding_sources_consumed=0),
                   lambda r:r['policy']['parts']['critic.weight'].add_(1),
                   lambda r:r['policy'].update(frozen_dense_sha256='0'*64)]:
        bad = copy.deepcopy(raw); mutate(bad); torch.save(bad, path)
        with pytest.raises(ValueError):
            policy.load(path, run_plan, 'anchor_CE', expected_checkpoint_sha256=file_hash(path), expected_plan_sha256=file_hash(run_plan))
    wrong = copy.deepcopy(plan); wrong['own_reference_binding']['frozen_actor_parts_sha256'] = '0'*64
    with pytest.raises(ValueError, match='snapshot'):
        policy.make_reference(model, wrong)
