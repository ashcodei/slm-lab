"""Real native train-source anchored updates and exact CPU recovery on a small NN.

The network is deliberately small and random, with the real own 32,768-token
vocabulary. Short action controls do not admit the full production GPU profile.
"""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/anchored-execution-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or REPORTS.exists():
        raise ValueError('Fresh CPU-only small-network mechanics proof required')
    import numpy as np
    import torch
    from slm_lab.anchored_execution_step_v1 import update
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_actor_checkpoint_v1 import make_state, restore_state
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    random.seed(961117); np.random.seed(961117); torch.manual_seed(961117)
    corpus = ExpandedExecutionCorpus(ROOT)
    REPORTS.mkdir()
    source_paths = [Path(__file__), ROOT / 'src/slm_lab/anchored_execution_step_v1.py',
        ROOT / 'src/slm_lab/anchored_code_objective_v1.py', ROOT / 'src/slm_lab/expanded_execution_corpus_v1.py',
        ROOT / 'reports/anchored-code-objective-v1-tests.log', ROOT / 'data/apps-execution-rewards-v1/admission.json']
    bindings = {str(path): file_hash(path) for path in source_paths}
    positions = [min((i for i, row in enumerate(corpus.rows) if row['kind'] == kind),
                     key=lambda i: corpus.rows[i]['complete_context_positions']) for kind in ['documented', 'apps']]
    config = ModelConfig(vocab_size=32768, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=4096, vision_hidden_size=32,
        vision_intermediate_size=64, vision_layers=1, vision_heads=4, image_size=32, patch_size=16, pool_size=2)
    base = OwnDenseAssistant(new_backbone(config, 961217), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1).eval()
    model = OwnDenseActor(base, rank=2, seed=961317).eval()
    optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))],
        lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
    base_digest = tensor_digest(base.named_parameters())
    critic = model.parts['critic'].weight.detach().clone()
    sources = dict(kernel_sha256=file_hash(ROOT / 'src/slm_lab/anchored_execution_step_v1.py'),
        source_admission_sha256=file_hash(ROOT / 'data/apps-execution-rewards-v1/admission.json'))
    atomic_json(REPORTS / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bindings, actual_counts=model.counts(), complete_native_source_positions=positions,
        logical_updates=2, physical_optimizer_calls_with_recovery=3, CPU_mechanics_action_horizon=2,
        own_small_random_network_not_current_full_policy=True, full_GPU_profile_capacity_proven=False,
        capability_gain_tested=False, goal_achieved=False))
    def step(position):
        return update(model, optimizer, corpus, position, method='anchor_execution',
            local_group_size=2, cache_absolute_bound=.0001, CPU_mechanics_horizon=2)
    first = step(positions[0])
    state = make_state(model, optimizer, groups=('actor',), sources=sources, completed=1, planned=2,
        cursor=dict(next_original_position=positions[1]), metrics=[first], device='cpu')
    torch.save(state, REPORTS / 'after-first-private-state.pt')
    expected = step(positions[1]); weights = model.export(sources)
    adam = copy.deepcopy(optimizer.state_dict()); rng = capture_rng('cpu')
    restored = torch.load(REPORTS / 'after-first-private-state.pt', map_location='cpu', weights_only=True)
    restore_state(model, optimizer, restored, groups=('actor',), expected_sources=sources, planned=2,
        expected_cursor=dict(next_original_position=positions[1]), device='cpu')
    actual = step(positions[1])
    def equal(a, b):
        if isinstance(a, torch.Tensor): return torch.equal(a, b)
        if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    if not equal(weights, model.export(sources)) or not equal(adam, optimizer.state_dict()) or not equal(rng, capture_rng('cpu')):
        raise ValueError('Exact next actual anchored update, every Adam value and private CPU RNG required')
    for key in ['global_rewards', 'global_advantages', 'original_reference_CE_sum', 'policy_surrogate',
                'reference_KL', 'gradient_norm_before_clip', 'cached_full_max_absolute_differences']:
        if expected[key] != actual[key]:
            raise ValueError('Exact next actual semantic objective record required')
    for before, after in zip(expected['trajectories'], actual['trajectories']):
        for key in ['response_ids', 'actual_cached_behavior_logps', 'actual_full_behavior_logps', 'own_response']:
            if before[key] != after[key]:
                raise ValueError('Actual next own sampled actions and probabilities changed after recovery')
    if base_digest != tensor_digest(base.named_parameters()) or not torch.equal(critic, model.parts['critic'].weight):
        raise ValueError('Every own base/head and separate unused critic must remain exact')
    if any(not record['original_reference_used_regardless_of_rewards'] for record in [first, expected, actual]):
        raise ValueError('Every update must use its complete original-code anchor')
    for path, digest in bindings.items():
        if file_hash(path) != digest: raise ValueError('Mechanics proof source changed')
    atomic_json(REPORTS / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), actual_counts=model.counts(),
        actual_complete_native_source_positions=positions, actual_logical_updates=2, actual_physical_optimizer_calls=3,
        actual_next_sample_actions_objectives_weights_Adam_CPU_RNG_exact=True,
        every_original_reference_target_and_EOS_used=True, base_heads_and_unused_critic_unchanged=True,
        actual_original_machine_rewards_not_synthetic=[first['global_rewards'], expected['global_rewards']],
        actual_records=[first, expected, actual], small_random_network_and_short_CPU_horizon=True,
        full_profile_four_GPU_admission_and_production_driver_still_required=True,
        no_full_current_policy_or_live_training_modified=True, capability_gain_tested=False,
        human_preferences=False, genuine_human_RLHF=False, source_files_sha256=bindings,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(counts=model.counts(), logical_updates=2, physical_optimizer_calls=3,
        next_update_recovery_exact=True, capability_gain_tested=False)), flush=True)


if __name__ == '__main__':
    main()
