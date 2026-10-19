from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp

from slm_lab.codenet_checkpoint_v1 import restore_rng, save_collective, validate_resume
from slm_lab.codenet_runner_v1 import (FixedReductionModel, INTEGER_OPTIONS, RUNTIME_FILES, VerifiedReplayEncoder,
    average_gradients, bind_evaluations,
    training_update, validate_plan)
from slm_lab.codenet_stage_v1 import EpochOrder, Objective, freeze_for_code_stage, objective_gradients
from slm_lab.codenet_token_pack_v1 import packed_encoder
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.diverse_images import pixel_hash
from slm_lab.training import TrainingOutputs


class Tokenizer:
    def get_vocab_size(self):
        return 128

    def token_to_id(self, text):
        return {'<pad>': 0, '<bos>': 1, '<eos>': 2}[text]

    def encode(self, text):
        return SimpleNamespace(ids=[ord(c) for c in text])


class Code:
    def __init__(self, folder):
        self.folder = folder

    def __len__(self):
        return 5

    def sequence(self, position):
        response = [11 + position] * (35 if position == 0 else position)
        return [1, 10] + response + [2], [-100, -100] + response + [2]


class Replay:
    families = ['image_reading', 'human_instruction']

    def __init__(self, image):
        self.image = image
        with Image.open(image) as picture:
            self.golden = pixel_hash(picture, 16)

    def size(self, name):
        return 3

    def row(self, name, index):
        row = {'prompt': 'p', 'response': 'a' * (index + 1), 'split': 'train'}
        if name == 'image_reading':
            row.update(image=str(self.image), task='word_ocr', model_pixel_sha256_16=self.golden)
        return row


def options():
    return {'seed': 71, 'passes': 2, 'batch_size': 1, 'gradient_accumulation': 2,
            'replay_batch_size': 1, 'replay_accumulation': 2, 'image_size': 448,
            'checkpoint_every': 1, 'log_every': 1, 'learning_rate': .003,
            'weight_decay': .1, 'gradient_clip': 1., 'replay_coefficient': .5}


def small_config():
    return ModelConfig(vocab_size=128, hidden_size=16, intermediate_size=32, num_layers=1,
        num_heads=4, num_kv_heads=2, max_text_length=64, vision_hidden_size=16,
        vision_intermediate_size=32, vision_layers=1, vision_heads=4, image_size=16, patch_size=4, pool_size=2)


def test_actual_runner_updates_own_decoder_with_complete_code_and_both_replay_modalities(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(43)
    config = small_config()
    raw = TinyMultimodalLM(config)
    raw.gradient_checkpointing = True
    before = copy.deepcopy(raw.state_dict())
    trainable, frozen = freeze_for_code_stage(raw)
    optimizer = torch.optim.AdamW(trainable, lr=.003)
    model = TrainingOutputs(raw, 'sft')
    code = Code(tmp_path)
    image = tmp_path / 'image.png'
    Image.new('RGB', (8, 4), (21, 55, 90)).save(image)
    replay, tokenizer = Replay(image), Tokenizer()
    code_encoder = packed_encoder(code, tokenizer, config)
    replay_encoder = VerifiedReplayEncoder(tokenizer, config, tmp_path, image_size=16)
    seen_code, modalities = [], []
    for step in range(5):
        result = training_update(model, optimizer, code, code_encoder, replay, replay_encoder,
                                 step, options(), 'cpu', nullcontext, rank=0, world=1)
        positions = [group[0] for group in EpochOrder(5, 71).rank_groups(step, 0, 1, 1, 2)]
        expected_targets = sum(sum(v != -100 for v in code.sequence(p)[1]) for p in positions)
        assert result['objectives']['complete_code']['global_target_tokens'] == expected_targets
        assert result['objectives'][result['replay_family']]['coefficient'] == .5
        assert result['gradient_norm_before_clipping'] > 0
        assert all(p.grad is None for p in frozen)
        seen_code.extend(positions)
        modalities.append(result['replay_family'])
    assert sorted(seen_code[:5]) == list(range(5))
    assert sorted(seen_code[5:]) == list(range(5))
    assert modalities == ['image_reading', 'human_instruction', 'image_reading', 'human_instruction', 'image_reading']
    state = raw.state_dict()
    for name in state:
        if name.startswith(('vision.', 'value_head.')):
            torch.testing.assert_close(state[name], before[name], rtol=0, atol=0)
    assert any(not torch.equal(state[name], before[name]) for name, p in raw.named_parameters() if p.requires_grad)
    assert code_encoder.truncated == replay_encoder.truncated == 0


def own_model_worker(rank, init_path, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_path, rank=rank, world_size=4)
    torch.manual_seed(43)
    path = Path(folder)
    config = small_config()
    raw = TinyMultimodalLM(config)
    trainable, frozen = freeze_for_code_stage(raw)
    raw.gradient_checkpointing = True
    model = FixedReductionModel(raw)
    torch.manual_seed(101 + rank)
    frozen_before = {n: p.detach().clone() for n, p in raw.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW(trainable, lr=.003)
    code, replay, tokenizer = Code(path), Replay(path / 'image.png'), Tokenizer()
    code_encoder = packed_encoder(code, tokenizer, config)
    replay_encoder = VerifiedReplayEncoder(tokenizer, config, path, image_size=16)
    opts = {**options(), 'passes': 4, 'gradient_accumulation': 1}

    def update(step):
        result = training_update(model, optimizer, code, code_encoder, replay, replay_encoder,
            step, opts, 'cpu', nullcontext, rank=rank, world=4)
        assert all(p.grad is None for p in frozen)
        for name, p in raw.named_parameters():
            if name in frozen_before:
                torch.testing.assert_close(p.detach(), frozen_before[name], rtol=0, atol=0)
        return result

    for step in range(2):
        update(step)

    def bundle(states):
        return {'training_kind': 'complete_code_rehearsal_v1', 'run_plan_sha256': 'own-decoder-fixture',
            'world_size': 4, 'planned_steps': 5, 'steps': 2, 'worker_rng_states': states,
            'model': raw.state_dict(), 'optimizer': optimizer.state_dict()}

    save_collective(path / 'resume.pt', bundle, 'cpu', 4, rank, 1_000_000, 0)
    for step in range(2, 5):
        update(step)
    expected, expected_optimizer = copy.deepcopy(raw.state_dict()), copy.deepcopy(optimizer.state_dict())
    raw = TinyMultimodalLM(config)
    trainable, frozen = freeze_for_code_stage(raw)
    raw.gradient_checkpointing = True
    model = FixedReductionModel(raw)
    optimizer = torch.optim.AdamW(trainable, lr=.003)
    saved = torch.load(path / 'resume.pt', weights_only=True)
    start = validate_resume(saved, 'own-decoder-fixture', 4, 5)
    raw.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['worker_rng_states'][rank], 'cpu')
    for step in range(start, 5):
        update(step)
    for name, value in raw.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    for key, state in optimizer.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, expected_optimizer['state'][key][name], rtol=0, atol=0)
    (path / ('own-rank-' + str(rank) + '.json')).write_text(json.dumps({
        'worker': rank, 'exact_resumed_model_and_adam_match': True,
        'frozen_vision_and_critic_unchanged': True, 'image_and_text_rehearsal_executed': True}))
    dist.destroy_process_group()


def test_four_workers_use_the_actual_update_path_with_images_and_exact_resume(tmp_path):
    Image.new('RGB', (8, 4), (21, 55, 90)).save(tmp_path / 'image.png')
    mp.spawn(own_model_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=4, join=True)
    for rank in range(4):
        result = json.loads((tmp_path / ('own-rank-' + str(rank) + '.json')).read_text())
        assert result['exact_resumed_model_and_adam_match'] and result['frozen_vision_and_critic_unchanged']
        assert result['image_and_text_rehearsal_executed']


class AnalyticLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.7))

    def forward(self, input_ids, labels):
        return {'loss': (self.weight * input_ids[:, 1:].float() - labels[:, 1:].float()).square().mean()}


class AnalyticRows:
    def batch(self, rows, field, device):
        assert len(rows) == 1
        return {'input_ids': torch.tensor([[0] + [p[0] for p in rows[0]]], device=device),
                'labels': torch.tensor([[-100] + [p[1] for p in rows[0]]], device=device)}


def analytic_worker(rank, init_path, folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + init_path, rank=rank, world_size=4)
    raw = AnalyticLoss()
    model = FixedReductionModel(raw)
    encoder = AnalyticRows()
    result = objective_gradients(model, [
        Objective('code', encoder, [[[(rank + 1, 2)] * (rank + 1)]], 'response', 1.),
        Objective('replay', encoder, [[[(rank + 5, 3)] * (4 - rank)]], 'response', .5)],
        'cpu', nullcontext, world=4)
    average_gradients(model, 'cpu', 4)
    expected_code = sum((r + 1) * 2 * (.7 * (r + 1) - 2) * (r + 1) for r in range(4)) / 10
    expected_replay = sum((4 - r) * 2 * (.7 * (r + 5) - 3) * (r + 5) for r in range(4)) / 10
    assert float(raw.weight.grad) == pytest.approx(expected_code + .5 * expected_replay, rel=1e-6)
    assert result['objectives']['code']['global_target_tokens'] == 10
    assert result['objectives']['replay']['global_target_tokens'] == 10
    (Path(folder) / ('analytic-rank-' + str(rank))).write_text(str(float(raw.weight.grad)))
    dist.destroy_process_group()


def test_fixed_reduction_retains_global_objective_means_with_unequal_worker_targets(tmp_path):
    mp.spawn(analytic_worker, args=(str(tmp_path / 'rendezvous'), str(tmp_path)), nprocs=4, join=True)
    assert all((tmp_path / ('analytic-rank-' + str(rank))).exists() for rank in range(4))


def test_each_used_replay_image_requires_matching_actual_encoder_pixels(tmp_path):
    image = tmp_path / 'image.png'
    Image.new('RGB', (8, 4), (21, 55, 90)).save(image)
    replay = Replay(image)
    row = replay.row('image_reading', 0)
    encoder = VerifiedReplayEncoder(Tokenizer(), small_config(), tmp_path, image_size=16)
    assert encoder.pixels(row).shape == (3, 16, 16)
    Image.new('RGB', (8, 4), (22, 56, 91)).save(image)
    with pytest.raises(ValueError, match='original golden hash'):
        encoder.pixels(row)


def plan_fixture(tmp_path):
    from slm_lab import codenet_runner_v1
    package = Path(codenet_runner_v1.__file__).parent
    plan = {'format': 'slm-complete-code-run-plan-v1', 'world_size': 4, 'training_options': options(),
            'training_ready': True, 'automatic_candidate_promotion': False,
            'gradient_reduction': 'fixed_parameter_order_worker_average_v1',
            'replay_selection': 'full_pass_floor_then_uniform_extra_family_updates_v1',
            'source_files_sha256': {str(package / name): '1' * 64 for name in RUNTIME_FILES},
            'preceding_processes': [{'pid': 1, 'start_ticks': '1'}],
            'storage': {'maximum_checkpoint_bytes': 9_000_000_000,
                        'maximum_final_model_bytes': 3_200_000_000, 'minimum_free_bytes': 26 * 1024**3}}
    for name in ['base_checkpoint', 'tokenizer', 'code_pack', 'replay', 'replay_audit', 'evaluation_binding', 'output']:
        plan[name] = str(tmp_path / name)
    for name in ['base_checkpoint', 'tokenizer', 'code_manifest', 'code_admission', 'replay_manifest',
                 'replay_audit', 'evaluation_binding']:
        plan[name + '_sha256'] = '1' * 64
    return plan


def test_run_plan_rejects_missing_readiness_and_unsafe_storage(tmp_path):
    plan = plan_fixture(tmp_path)
    assert validate_plan(plan, 4) == options()
    for update in [{'training_ready': False}, {'automatic_candidate_promotion': True},
                   {'source_files_sha256': {}}, {'preceding_processes': []},
                   {'storage': {**plan['storage'], 'minimum_free_bytes': 1}},
                   {'storage': {**plan['storage'], 'maximum_final_model_bytes': 2_900_000_000}},
                   {'base_checkpoint': 'relative/model.pt'}]:
        with pytest.raises(ValueError):
            validate_plan({**plan, **update}, 4)
    with pytest.raises(ValueError, match='four-worker'):
        validate_plan(plan, 1)
    for key in INTEGER_OPTIONS:
        with pytest.raises(ValueError):
            validate_plan({**plan, 'training_options': {**options(), key: -1}}, 4)


def test_stage_cannot_bind_training_loss_in_place_of_broad_capability_baselines(tmp_path):
    from slm_lab.corpus import file_hash
    plan = plan_fixture(tmp_path)
    path = Path(plan['evaluation_binding'])
    path.write_text(json.dumps({'format': 'slm-complete-code-evaluation-binding-v1',
        'baseline_checkpoint_sha256': plan['base_checkpoint_sha256'], 'baseline_completed': True,
        'promote_from_training_loss': False, 'guards': {'training_loss': {}}}))
    plan['evaluation_binding_sha256'] = file_hash(path)
    with pytest.raises(ValueError, match='all capability guards'):
        bind_evaluations(plan)
