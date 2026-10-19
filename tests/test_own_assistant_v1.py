from dataclasses import asdict
import copy
import json

import pytest
import torch
from tokenizers import Tokenizer, models

from slm_lab.corpus import file_hash
from slm_lab.frozen_base_pointer_stage_v1 import FrozenBasePointerStage, make_head, make_optimizer as head_optimizer
from slm_lab.frozen_base_pointer_inference_v1 import export_payload as head_payload
from slm_lab.own_assistant_v1 import OwnAssistant, attach_grounding, load_manifest, MANIFEST_FORMAT
from slm_lab.own_code_training_stage_v1 import OwnCodeTrainingStage, make_optimizer, components
from slm_lab.own_code_training_inference_v1 import export_payload
from test_own_code_training_stage_v1 import fixtures
from test_own_code_delta_v1 import own_models
from test_frozen_base_pointer_stage_v1 import settings
from test_deterministic_vision_resize_v1 import ResizeFixture


def trained_fixture(folder):
    model, streams, options = fixtures()
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
        **{str(i): i for i in range(4, 32)}}, unk_token='<unk>'))
    token = folder / 'tokenizer.json'; tokenizer.save(str(token))
    base = folder / 'base.pt'; target_path = folder / 'own-target.pt'; _, target = own_models()
    for path, native in [(base, model.base), (target_path, target)]:
        torch.save(dict(model=native.state_dict(), config=asdict(native.config), tokenizer_sha256=file_hash(token)), path)
    initial_plan = folder / 'initial-plan.json'
    initial_plan.write_text(json.dumps(dict(own_base=file_hash(base), own_target=file_hash(target_path), fixture_only=True)))
    counts = model.counts()
    initial = dict(format='own-conditional-code-delta-v1', components=components(model), config=asdict(model.config),
        base_checkpoint_sha256=file_hash(base), own_code_source_checkpoint_sha256=file_hash(target_path),
        tokenizer_sha256=file_hash(token), source_plan_sha256=file_hash(initial_plan), reserved_point_head_parameters=256,
        base_parameters=counts['frozen_general'], additional_code_parameters=counts['trainable_code'],
        total_unique_parameters=counts['total_unique'], total_with_reserved_point_head=counts['total_with_reserved_head'],
        candidate_promoted=False, goal_achieved=False)
    initial_path = folder / 'initial.pt'; torch.save(initial, initial_path)
    code_plan = dict(training_options=options, actual_model_counts=counts,
        initial_component_checkpoint_sha256=file_hash(initial_path), base_checkpoint_sha256=file_hash(base),
        initial_plan_sha256=file_hash(initial_plan), own_code_target_sha256=file_hash(target_path),
        data_admissions_sha256={name: str(i) * 64 for i, name in enumerate(streams)}, fixture_only=True)
    code_plan_path = folder / 'code-plan.json'; code_plan_path.write_text(json.dumps(code_plan))
    stage = OwnCodeTrainingStage(model, make_optimizer(model, options), streams, options,
                                 {'run_plan_sha256': file_hash(code_plan_path)}, 'cpu')
    for _ in range(3):
        stage.update()
    code_path = folder / 'code-trained.pt'
    torch.save(export_payload(model, initial, code_plan, file_hash(code_plan_path), stage.completed), code_path)
    head = make_head(model.base, rank=8); opts = settings()
    plan = dict(training_options=opts, head_rank=8, base_checkpoint_sha256=file_hash(base), data_admission_sha256='a' * 64,
                fixture_only=True)
    head_plan = folder / 'head-plan.json'; head_plan.write_text(json.dumps(plan))
    sources = dict(base_checkpoint_sha256=file_hash(base), tokenizer_sha256=file_hash(token),
                   run_plan_sha256=file_hash(head_plan), data_admission_sha256='a' * 64)
    kernel = FrozenBasePointerStage(head, head_optimizer(head, opts), ResizeFixture(), opts, sources, 'cpu')
    for _ in range(kernel.steps):
        kernel.update()
    head_path = folder / 'head-trained.pt'
    torch.save(head_payload(head, opts, sources, kernel.completed, kernel.frozen_base_digest), head_path)
    model.eval(); head.eval()
    actual = dict(shared_original_backbone=counts['frozen_general'], additional_code=counts['trainable_code'],
                  additional_grounding=256, total_unique=counts['total_unique'] + 256)
    manifest = dict(format=MANIFEST_FORMAT, actual_counts=actual, own_code_target_sha256=file_hash(target_path),
                    grounding_data_admission_sha256='a' * 64, candidate_promoted=False, goal_achieved=False)
    for name, path in [('base_checkpoint', base), ('tokenizer', token), ('code_checkpoint', code_path),
                       ('code_initial_plan', initial_plan), ('code_training_plan', code_plan_path),
                       ('grounding_head', head_path), ('grounding_plan', head_plan)]:
        manifest[name] = path.name; manifest[name + '_sha256'] = file_hash(path)
    manifest_path = folder / 'assistant.json'; manifest_path.write_text(json.dumps(manifest))
    args = dict(expected_head_sha256=file_hash(head_path), expected_base_sha256=file_hash(base),
        expected_tokenizer_sha256=file_hash(token), expected_plan_sha256=file_hash(head_plan), expected_data_admission_sha256='a' * 64)
    return model, head, manifest_path, manifest, (head_path, base, token, head_plan), args, (target_path, initial_path)


def test_complete_trained_reference_factory_loads_one_base_and_reproduces_all_three_routes(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    model, head, path, manifest, files, kwargs, unused = trained_fixture(tmp_path)
    # Neither the dense own source policy nor initial code component weights
    # are needed for serving after their provenance hashes are recorded.
    for old in unused:
        old.unlink()
    from slm_lab import own_code_delta_inference_v1 as code_loader
    real_load = code_loader.load_checkpoint; calls = []
    def counted(path, *args, **kwargs):
        calls.append(str(path))
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(code_loader, 'load_checkpoint', counted)
    loaded, metadata = load_manifest(path, expected_manifest_sha256=file_hash(path))
    assert calls == [str(files[1])]
    assert loaded.language.base is loaded.grounding.base
    assert loaded.language.code.vision is loaded.grounding.base.vision
    assert loaded.counts() == manifest['actual_counts']
    assert metadata['full_backbones_loaded_by_attachment'] == 0
    ids = torch.tensor([[1, 4, 7]])
    pixels = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    with torch.no_grad():
        for route in ['general', 'code']:
            assert torch.equal(model(ids, pixels, route=route)['logits'], loaded(ids, pixels, route=route)['logits'])
            assert torch.equal(model.generate(ids, pixels, route=route, max_new_tokens=3), loaded.generate(ids, pixels, route=route, max_new_tokens=3))
        assert torch.equal(head(ids, pixels)['logits'], loaded.locate(ids, pixels)['logits'])
        assert torch.equal(head(ids, pixels)['logits'], loaded(ids, pixels, route='grounding')['logits'])


def test_attachment_preserves_rng_base_versions_existing_code_cache_and_rejects_duplicate_base(tmp_path):
    torch.set_num_threads(1)
    model, old_head, _, _, files, kwargs, _ = trained_fixture(tmp_path)
    ids = torch.tensor([[1, 4, 7]]); pixels = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        cached = model.code.prefill(ids, pixels)['cache']
        expected = model.code.decode(torch.tensor([[8]]), cached)['logits'].clone()
    versions = [(id(p), p._version, p.data_ptr()) for p in model.base.parameters()]
    rng = torch.get_rng_state().clone()
    combined, _ = attach_grounding(model, *files, **kwargs)
    assert torch.equal(rng, torch.get_rng_state())
    assert versions == [(id(p), p._version, p.data_ptr()) for p in model.base.parameters()]
    with torch.no_grad():
        assert torch.equal(expected, combined.language.code.decode(torch.tensor([[8]]), cached)['logits'])
    duplicate = make_head(copy.deepcopy(model.base), rank=8)
    with pytest.raises(ValueError, match='second backbone'):
        OwnAssistant(model, duplicate)
    with pytest.raises(ValueError, match='below1B'):
        OwnAssistant(model, combined.grounding, parameter_limit=combined.counts()['total_unique'])


def test_false_head_provenance_inventory_and_mutated_base_reject_without_live_changes(tmp_path):
    torch.set_num_threads(1)
    model, _, _, _, files, kwargs, _ = trained_fixture(tmp_path)
    original = {n: t.clone() for n, t in model.state_dict().items()}; rng = torch.get_rng_state().clone()
    payload = torch.load(files[0], weights_only=True)
    for mutate in [lambda s: s['sources'].update(base_checkpoint_sha256='0' * 64),
                   lambda s: s['projections']['query_projection.weight'].fill_(float('nan')),
                   lambda s: s['actual_inventory'].update(actual_trainable_parameters=0)]:
        bad = copy.deepcopy(payload); mutate(bad); path = tmp_path / 'bad-head.pt'; torch.save(bad, path)
        with pytest.raises(ValueError):
            attach_grounding(model, path, *files[1:], **{**kwargs, 'expected_head_sha256': file_hash(path)})
        assert torch.equal(rng, torch.get_rng_state())
        assert all(torch.equal(t, model.state_dict()[n]) for n, t in original.items())
    with torch.no_grad():
        next(model.base.parameters()).add_(.01)
    changed = {n: t.clone() for n, t in model.state_dict().items()}
    with pytest.raises(ValueError, match='shared original base'):
        attach_grounding(model, *files, **kwargs)
    assert all(torch.equal(t, model.state_dict()[n]) for n, t in changed.items())
