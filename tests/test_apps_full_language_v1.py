import copy
from contextlib import nullcontext
import json
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from tokenizers import Tokenizer, models

from slm_lab.apps_full_language_v1 import FullLanguageStage, HumanProgramEncoder
from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_stage_v3 import EXPANDED_REPLAY_FAMILIES
from slm_lab.codenet_stage_v1 import freeze_for_code_stage
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig, TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest


class Source:
    def __init__(self, folder): self.folder, self.bad = Path(folder), False
    def __len__(self): return 479
    def example(self, i):
        ids = [1, 7, 9] + [11 + i % 3] * (1 + i % 4) + [2]
        return dict(id=str(i), role='validation' if self.bad else 'train', original_human_solution=True,
            execution_verified=False, human_preferences=False, positive=(ids, [-100]*3+ids[3:]))


class Replay:
    families = EXPANDED_REPLAY_FAMILIES
    def size(self, name): return 5 + self.families.index(name) % 4
    def row(self, name, i): return dict(family=name, index=i)


class ReplayEncoder:
    truncated = 0
    def batch(self, rows, field, device):
        i = rows[0]['index']; ids = torch.tensor([[1, 8, 10] + [13] * (i % 3) + [2]], device=device)
        labels = ids.clone()
        if field is not None: labels[:, :2] = -100
        pixels = torch.ones(1, 3, 16, 16, device=device) * (i + 1) / 10
        return dict(input_ids=ids, labels=labels, pixel_values=pixels)


def worker(rank, folder):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'), rank=rank, world_size=4)
    try:
        torch.manual_seed(731)
        config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=2, num_heads=4,
            num_kv_heads=2, max_text_length=64, vision_hidden_size=8, vision_intermediate_size=16,
            vision_layers=1, vision_heads=2, image_size=16, patch_size=4, pool_size=2)
        original = TinyMultimodalLM(config)
        tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
            **{f'token{i}': i for i in range(4,32)}}, unk_token='<unk>'))
        options = dict(seed=931001, code_accumulation=2, replay_accumulation=1, image_size=16, learning_rate=.003,
            weight_decay=.01, replay_coefficient=.5, gradient_clip=1.)
        sources = dict(base_checkpoint_sha256='a'*64, tokenizer_sha256='b'*64, run_plan_sha256='c'*64)
        def make():
            raw = copy.deepcopy(original); params, _ = freeze_for_code_stage(raw)
            optimizer = torch.optim.AdamW(params, lr=.003, weight_decay=.01, foreach=False)
            source = Source(folder)
            stage = FullLanguageStage(raw, optimizer, source, HumanProgramEncoder(source, tokenizer, config),
                Replay(), ReplayEncoder(), options, sources, 'cpu', nullcontext)
            return stage
        first = make(); before = state_digest(first.raw.state_dict())
        if rank == 3: first.source.bad = True
        with pytest.raises(ValueError, match='Collective preflight'): first.update()
        assert first.completed == 0 and state_digest(first.raw.state_dict()) == before
        assert all(p.grad is None for p in first.raw.parameters()); first.source.bad = False
        initial_rng = torch.get_rng_state().clone()
        for _ in range(first.steps):
            first.update(); torch.rand(rank + 1)
        reference = dict(model=state_digest(first.raw.state_dict()), optimizer=fingerprint(first.optimizer.state_dict()),
            metrics=first.metrics.copy(), sampler=first.sampler_state(first.completed), rng=torch.get_rng_state().clone())
        second = make(); torch.set_rng_state(initial_rng)
        for _ in range(31): second.update(); torch.rand(rank + 1)
        path = Path(folder) / 'resume.pt'
        second.save(path, maximum_bytes=20*1024**2, minimum_free_bytes=0)
        previous_hash = file_hash(path)
        second.update(); torch.rand(rank + 1)
        with pytest.raises(ValueError, match='Collective snapshot failed'):
            second.save(path, maximum_bytes=1, minimum_free_bytes=0)
        assert file_hash(path) == previous_hash
        restored = make()
        state = torch.load(path, map_location='cpu', weights_only=True)
        restored.restore(state)
        assert restored.completed == 31
        for _ in range(31, restored.steps): restored.update(); torch.rand(rank + 1)
        assert state_digest(restored.raw.state_dict()) == reference['model']
        assert fingerprint(restored.optimizer.state_dict()) == reference['optimizer']
        assert restored.metrics == reference['metrics']
        assert restored.sampler_state(restored.completed) == reference['sampler']
        assert torch.equal(torch.get_rng_state(), reference['rng'])
        assert {m['replay_family'] for m in restored.metrics} == set(EXPANDED_REPLAY_FAMILIES)
        assert state_digest({n:t for n,t in restored.raw.state_dict().items() if n.startswith(('vision.','value_head.'))}) == restored.frozen_digest
        bad = dict(state); bad['source_population'] = 1; bad['sampler_state'] = dict(state['sampler_state'], source_population=1)
        before = state_digest(restored.raw.state_dict())
        with pytest.raises(ValueError, match='Collective preflight'): restored.restore(bad)
        assert state_digest(restored.raw.state_dict()) == before
        (Path(folder) / f'rank-{rank}.json').write_text(json.dumps(dict(actual_four_CPU_workers=True, parameters=sum(p.numel() for p in original.parameters()),
            reference_updates=first.steps, resumed_from=31, resumed_updates=restored.steps-31, all29families=True,
            full_weights_Adam_metrics_cursors_and_per_rank_rng_exact=True, held_rank_rejects_before_update=True,
            failed_save_preserves_previous_checkpoint=True, frozen_vision_and_critic_unchanged=True)))
        dist.barrier()
    finally: dist.destroy_process_group()


def test_four_actual_workers_full_language_update_recovery_and_all29_source_objectives(tmp_path):
    mp.spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    reports=[json.loads((tmp_path/f'rank-{i}.json').read_text()) for i in range(4)]
    assert all(r['full_weights_Adam_metrics_cursors_and_per_rank_rng_exact'] for r in reports)
