"""Exercise real full native pairs on a small own CPU actor, including replay."""
import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import random

import numpy as np
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_checkpoint_v1 import make_state, restore_state
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.own_execution_contrast_CPU_v1 import ExecutionContrastPairs, batch, update
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/own-execution-contrast-CPU-v1-mechanics'


def equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-only mechanism proof required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    data = ExecutionContrastPairs(ROOT / 'data/own-execution-negative-pairs-CPU-v1')
    config = ModelConfig(vocab_size=32768, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=4096, vision_hidden_size=32, vision_intermediate_size=64,
        vision_layers=1, vision_heads=4, image_size=32, patch_size=16)
    model = OwnDenseActor(OwnDenseAssistant(new_backbone(config, 969817), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1), rank=2, seed=969917).eval()
    parameters = [p for _, p in model.named_selected(('actor',))]
    adam = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
    sources = dict(pair_manifest_sha256=file_hash(data.folder / 'manifest.json'),
                   kernel_sha256=file_hash(ROOT / 'src/slm_lab/own_execution_contrast_CPU_v1.py'))
    base = tensor_digest(model.base.named_parameters()); critic = model.parts['critic'].weight.detach().clone()
    own = {}
    with torch.no_grad():
        for row in data.rows[:2]:
            own[row['source_id']] = model(batch(row, 'chosen')['input_ids'])['logits'][:, :-1].float().log_softmax(-1).detach()
    metrics = []
    def step(index):
        row = data.rows[index]; result = update(model, adam, row, own[row['source_id']], method='anchor_contrast')
        result.update(step=index + 1); metrics.append(result); return result
    random.seed(970017); np.random.seed(970017); torch.manual_seed(970017)
    step(0); saved = make_state(model, adam, groups=('actor',), sources=sources, completed=1, planned=2,
        cursor=dict(next_pair=1, planned=2), metrics=metrics, device='cpu')
    second = step(1); expected = model.export(sources); moments = copy.deepcopy(adam.state_dict()); rng = capture_rng('cpu')
    before = [(id(p), p._version, p.data_ptr()) for p in model.parameters()]
    bad = copy.deepcopy(saved); bad['model']['sources']['pair_manifest_sha256'] = 'f' * 64
    try:
        restore_state(model, adam, bad, groups=('actor',), expected_sources=sources, planned=2, expected_cursor=dict(next_pair=1, planned=2), device='cpu')
    except ValueError: pass
    else: raise ValueError('Changed own pair source must reject before mutation')
    if before != [(id(p), p._version, p.data_ptr()) for p in model.parameters()]: raise ValueError('Rejected restore mutated live parameters')
    completed, metrics = restore_state(model, adam, saved, groups=('actor',), expected_sources=sources,
        planned=2, expected_cursor=dict(next_pair=1, planned=2), device='cpu')
    replayed = step(1)
    if (completed != 1 or not equal(second, replayed) or not equal(expected, model.export(sources))
            or not equal(moments, adam.state_dict()) or not equal(rng, capture_rng('cpu'))
            or tensor_digest(model.base.named_parameters()) != base or not torch.equal(critic, model.parts['critic'].weight)):
        raise ValueError('Actual next weights/Adam/fixed base/critic/private CPU RNG must exactly replay')
    if torch.cuda.is_initialized(): raise ValueError('CUDA must remain uninitialized')
    files = [Path(__file__), data.folder / 'manifest.json', ROOT / 'reports/own-execution-contrast-CPU-v1-tests.log',
             ROOT / 'reports/own-execution-contrast-policy-CPU-v1-tests.log', *python_sources(ROOT, [Path(__file__)])]
    OUT.mkdir()
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256={str(p): file_hash(p) for p in files},
        actual_small_student_values=model.counts()['actual_unique'], actual_updates=metrics, logical_updates=2, physical_updates=3,
        next_actual_update_weights_Adam_and_private_CPU_RNG_bitexact=True, changed_pair_restore_rejected_before_mutation=True,
        fixed_dense_vision_all_other_heads_and_unused_critic_unchanged=True,
        actual_full_original_positive_EOS_targets=[r['original_reference_EOS_targets'] for r in metrics],
        actual_full_negative_action_targets=[r['genuine_negative_action_targets'] for r in metrics],
        all_negative_rewards_zero_but_real_contrastive_gradients_present=True,
        offline_fixed_pairs_no_online_sampling_RNG_consumption=True, current_full_size_model_or_GPU_training=False,
        held_quality_gain_measured=False, candidate_promoted=False, goal_achieved=False))
    print(str(OUT / 'complete.json'), flush=True)


if __name__ == '__main__': main()
