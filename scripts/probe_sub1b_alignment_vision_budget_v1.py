"""Actual-source shape budget plus disposable CPU human-alignment/OCR heads.

This is a feasibility probe, not an instantiated/trained whole assistant or a
new performance claim. Every proposed factor and norm is counted, together
with the complete existing base/code/grounding model. Embeddings are shared.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import RMSNorm
from slm_lab.own_code_delta_v1 import matrix_names, norm_names

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/sub1b-alignment-vision-budget-v1'


def factors(weights, names, rank):
    rows = []
    for name in names:
        tensor = weights[name]
        if tensor.ndim != 2 or rank > min(tensor.shape):
            raise ValueError('Every actual original full matrix must support the proposed rank')
        rows.append(dict(original_tensor=name, original_shape=list(tensor.shape), rank=rank,
            added_parameters=rank * (tensor.shape[0]+tensor.shape[1])))
    return rows


def main():
    if REPORTS.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Fresh GPU-hidden CPU architecture-budget probe required')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    manifest = ROOT / 'checkpoints/own-assistant-candidate-v1/manifest.json'
    spec = json.loads(manifest.read_text()); base = (manifest.parent / spec['base_checkpoint']).resolve()
    if file_hash(base) != spec['base_checkpoint_sha256']:
        raise ValueError('Exact actual original backbone bytes required for shape accounting')
    state = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
    weights = state['model']; config = state['config']; rank = 16
    if (weights['embedding.weight'].data_ptr() != weights['lm_head.weight'].data_ptr()
            or sum(t.numel() for name, t in weights.items() if name != 'lm_head.weight') != 706012672):
        raise ValueError('Actual complete own base and shared original tied embedding required')
    actual = json.loads((ROOT / 'reports/own-onpolicy-pilot-v2/CE_control/trained-shared-reload.json').read_text())['actual_counts']
    if actual['total_unique'] != 986450944:
        raise ValueError('Complete actual base/code/grounding budget required; no active-only parameter count')
    text = factors(weights, matrix_names(config['num_layers']), rank)
    text_norms = {name: weights[name].numel() for name in norm_names(config['num_layers'])}
    vision = factors(weights, ['vision.'+name for name in matrix_names(config['vision_layers'])], rank)
    vision_norms = {'vision.'+name: weights['vision.'+name].numel() for name in norm_names(config['vision_layers'])}
    projection = factors(weights, ['vision.project.weight'], rank)
    hidden = config['hidden_size']; visual = config['vision_hidden_size']; norm_eps = config['norm_eps']
    rng = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(947127)
        critic = nn.Linear(hidden, 1, bias=False)
        reward = nn.Sequential(RMSNorm(hidden, norm_eps), nn.Linear(hidden, 128, bias=False), nn.SiLU(), nn.Linear(128, 1, bias=False))
        word = nn.ModuleDict(dict(norm=RMSNorm(visual, norm_eps), local=nn.Conv2d(visual, visual, 3, padding=1, groups=visual),
            postnorm=RMSNorm(visual, norm_eps), height=nn.Linear(visual, 1),
            temporal=nn.GRU(visual, 96, batch_first=True, bidirectional=True), classifier=nn.Linear(192, 257)))
        patches = torch.randn(2, 4, 8, visual)
        x = word['norm'](patches).permute(0, 3, 1, 2)
        x = word['postnorm'](word['local'](x).permute(0, 2, 3, 1))
        attention = word['height'](x).softmax(1); sequence = (x * attention).sum(1)
        features, _ = word['temporal'](sequence); logits = word['classifier'](features)
        loss = F.ctc_loss(logits.log_softmax(-1).transpose(0, 1), torch.tensor([99, 97, 116, 97, 98]),
            torch.tensor([8, 8]), torch.tensor([3, 2]), blank=256, zero_infinity=False)
        latent = torch.randn(2, hidden); scores = reward(latent); values = critic(latent)
        objective = loss + scores.square().mean() + values.square().mean(); objective.backward()
        modules = dict(separate_own_PPO_critic=critic, separate_own_human_preference_reward_head=reward,
                       own_byte_word_CTC_readout=word)
        if (not bool(torch.isfinite(objective)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all())
                for module in modules.values() for p in module.parameters())):
            raise ValueError('Every disposable actual added head parameter must have finite CPU gradients')
        instantiated = {name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()}
    if not torch.equal(rng, torch.get_rng_state()) or torch.cuda.is_initialized():
        raise ValueError('CPU-only disposable head probe must preserve caller RNG and avoid CUDA')
    added = dict(general_rank16_factors=sum(r['added_parameters'] for r in text),
        general_separate_norms=sum(text_norms.values()), vision_rank16_factors=sum(r['added_parameters'] for r in vision),
        vision_separate_norms=sum(vision_norms.values()), vision_projection_rank16_factors=sum(r['added_parameters'] for r in projection),
        **instantiated)
    projected = actual['total_unique'] + sum(added.values())
    if projected >= 10**9:
        raise ValueError('Every complete existing/proposed parameter must fit strictly below1B')
    REPORTS.mkdir()
    report = dict(completed_at=datetime.now(timezone.utc).isoformat(), source_script_sha256=file_hash(Path(__file__).resolve()),
        actual_base_checkpoint_sha256=file_hash(base), actual_existing_shared_counts=actual,
        proposed_added_parameters=added, projected_full_unique_parameters=projected, remaining_strict_parameter_space=10**9-projected,
        actual_source_general_factor_shapes=text, actual_source_vision_factor_shapes=vision,
        actual_source_projection_shapes=projection, actual_source_general_norm_counts=text_norms,
        actual_source_vision_norm_counts=vision_norms, embeddings_shared_without_any_new_vocab_table=True,
        original_code_and_grounding_and_original_general_paths_intended_unchanged=True,
        actual_disposable_CTC_output_shape=list(logits.shape), actual_disposable_reward_shape=list(scores.shape),
        actual_disposable_critic_shape=list(values.shape), synthetic_CTC_loss=float(loss.detach()),
        actual_all_added_head_gradients_finite=True, CPU_RNG_preserved=True, CUDA_initialized=False,
        complete_combined_model_not_instantiated_or_trained=True, matrix_factors_only_shape_accounted=True,
        extra_weights_not_exported=True, no_optimizer_step=True,
        word_CTC_not_full_photo_OCR_or_general_visual_reasoning=True,
        genuine_human_labels_actual_alignment_full_model_capacity_retention_and_capability_assessments_required=True,
        not_a_novelty_or_high_end_capability_claim=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'complete.json', report)
    print(json.dumps({k: report[k] for k in ['proposed_added_parameters','projected_full_unique_parameters','remaining_strict_parameter_space',
        'actual_disposable_CTC_output_shape','actual_all_added_head_gradients_finite','complete_combined_model_not_instantiated_or_trained','goal_achieved']}, indent=2))


if __name__ == '__main__':
    main()
