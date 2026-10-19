import copy
import json

import pytest
import torch

from slm_lab import phase_human_policy_v1 as policy
from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.fast_joint_inference_v2 import export as foundation_export
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.parallel_human_preference_v2 import ParallelHumanPreferenceStage
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_stage_v1 import pairs, options
from test_own_dense_actor_v1 import actor


def fixture(tmp_path):
    torch.set_num_threads(1)
    base = install(actor().base)
    initializer = tmp_path/'initializer.pt'
    initial_sources = {'small_unit_fixture': 'a'*64}
    torch.save(base.export(initial_sources), initializer)
    settings = dict(initializer=str(initializer), initializer_sha256=file_hash(initializer),
        initializer_sources=initial_sources, actual_counts=base.counts(), tokenizer_sha256='b'*64)
    foundation_plan = dict(initial_dense_settings=settings, actual_counts=base.counts(),
        planned_steps=1, all_selected_family_records_once=True, all_original_sources_once=False,
        later_reference_is_own_completed_foundation=True, tokenizer_sha256='b'*64,
        small_synthetic_unit_fixture=True)
    source_plan = tmp_path/'foundation-plan.json'
    source_plan.write_text(json.dumps(foundation_plan))
    optimizer = torch.optim.AdamW(base.parameters(), lr=.001)
    base(torch.tensor([[1, 4, 7, 2]]), labels=torch.tensor([[-100, 4, 7, 2]]))['loss'].backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    checkpoint = tmp_path/'foundation.pt'
    torch.save(foundation_export(base, foundation_plan, file_hash(source_plan), 'trained', 1), checkpoint)
    model = OwnDenseActor(base, rank=2, seed=958993)
    data = pairs()
    setting = options(data)
    setting.update(epochs=1, steps=1, accumulation=len(data))
    setting.pop('DPO_weight')
    plan = dict(foundation_inputs=dict(checkpoint=str(checkpoint), plan=str(source_plan),
        checkpoint_sha256=file_hash(checkpoint), plan_sha256=file_hash(source_plan)),
        actual_dense_counts=base.counts(), frozen_dense_sha256=tensor_digest(base.named_parameters()),
        actor_settings=model.settings, actual_counts=model.counts(),
        initial_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),
        human_train_pairs=len(data), human_admission_sha256='c'*64, human_manifest_sha256='d'*64,
        training_options=setting, small_synthetic_unit_fixture=True)
    run_plan = tmp_path/'policy-plan.json'
    run_plan.write_text(json.dumps(plan))
    return model, data, plan, run_plan


def test_real_trained_fixture_foundation_DPO_update_and_native_cold_load(tmp_path):
    model, data, plan, run_plan = fixture(tmp_path)
    sha = file_hash(run_plan)
    cold, _ = policy.fresh(plan)
    assert tensor_digest(cold.parts.named_parameters()) == tensor_digest(model.parts.named_parameters())
    setting = policy.options(plan, 'human_DPO')
    optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))],
        lr=setting['learning_rate'], betas=tuple(setting['betas']),
        weight_decay=setting['weight_decay'], foreach=False)
    stage = ParallelHumanPreferenceStage(model, optimizer, data, setting, policy.sources(plan, sha))
    stage.update()
    model.eval()
    payload = policy.export(model, plan, sha, 'human_DPO', 1)
    path = tmp_path/'policy.pt'
    torch.save(payload, path)
    loaded, _ = policy.load(path, run_plan, 'human_DPO',
        expected_checkpoint_sha256=file_hash(path), expected_plan_sha256=sha)
    ids = torch.tensor([[1, 4, 7, 2]])
    pixels = torch.zeros(1, 3, model.config.image_size, model.config.image_size)
    with torch.no_grad():
        assert torch.equal(model(ids)['logits'], loaded(ids)['logits'])
        assert torch.equal(model(ids, pixels)['logits'], loaded(ids, pixels)['logits'])
    for mutate in [lambda s:s.update(completed_updates=0),
        lambda s:s['policy']['parts']['critic.weight'].add_(1),
        lambda s:s.update(actual_global_human_pairs=0),
        lambda s:s['policy'].update(frozen_dense_sha256='0'*64)]:
        bad = copy.deepcopy(payload)
        mutate(bad)
        torch.save(bad, path)
        with pytest.raises(ValueError):
            policy.load(path, run_plan, 'human_DPO',
                expected_checkpoint_sha256=file_hash(path), expected_plan_sha256=sha)


def test_partial_or_wrong_phase_foundation_rejected_before_policy_initialization(tmp_path):
    _, _, plan, _ = fixture(tmp_path)
    source = json.loads(open(plan['foundation_inputs']['plan']).read())
    path = plan['foundation_inputs']['checkpoint']
    raw = torch.load(path, weights_only=True)
    for mutate in [lambda s:s.update(arm='initial'), lambda s:s.update(completed_updates=0),
                   lambda s:s.update(whole_source_epochs_completed=0)]:
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(ValueError, match='no partial'):
            policy.validate_foundation(source, bad)
    torch.save(raw, path)
    plan['foundation_inputs']['checkpoint_sha256'] = '0'*64
    with pytest.raises(ValueError):
        policy.fresh(plan)
