"""Actual longest native source/horizon backward on a small own CPU network."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/broad-code-capacity-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or REPORTS.exists():
        raise ValueError('Fresh CPU-only native-horizon mechanics proof required')
    import torch
    from slm_lab.broad_code_capacity_v1 import stress
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    corpus = ExpandedExecutionCorpus(ROOT)
    config = ModelConfig(vocab_size=32768, hidden_size=32, intermediate_size=64, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=4096, vision_hidden_size=32,
        vision_intermediate_size=64, vision_layers=1, vision_heads=4, image_size=32, patch_size=16, pool_size=2)
    base = OwnDenseAssistant(new_backbone(config, 963217), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1).eval()
    model = OwnDenseActor(base, rank=2, seed=963317).eval()
    parameters = [p for _, p in model.named_selected(('actor',))]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
    short = min(range(len(corpus)), key=lambda p: corpus.rows[p]['complete_context_positions'])
    batch = corpus.batch(short, 'cpu'); model(**batch)['loss'].backward(inputs=parameters)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    reference = FrozenOwnActorReference(model, sources=dict(own_starting_actor_state=tensor_digest(model.parts.named_parameters())))
    position = max(range(len(corpus)), key=lambda p: corpus.rows[p]['full_training_action_budget'])
    files = [Path(__file__), ROOT / 'src/slm_lab/broad_code_capacity_v1.py',
        ROOT / 'src/slm_lab/frozen_own_actor_reference_v1.py',
        ROOT / 'reports/broad-anchored-execution-four-CPU-v1/complete.json',
        ROOT / 'reports/expanded-execution-corpus-v1/complete.json']
    bindings = {str(p): file_hash(p) for p in files}
    REPORTS.mkdir()
    atomic_json(REPORTS / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256=bindings,
        actual_counts=model.counts(), original_native_source_position=position,
        actual_full_training_action_budget=corpus.row(position)['full_training_action_budget'],
        actual_small_network_reference_CE_initialization_updates=1, stress_optimizer_commits=0,
        small_CPU_network_not_full_size_GPU_admission=True, capability_gain_tested=False, goal_achieved=False))
    proof = stress(model, optimizer, corpus, reference, position, cache_absolute_bound=.0001)
    for path, digest in bindings.items():
        if file_hash(path) != digest: raise ValueError('Capacity proof source changed')
    atomic_json(REPORTS / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), actual_counts=model.counts(),
        proof=proof, resident_training_inventory=reference.resident_inventory(model), source_files_sha256=bindings,
        actual_longest_horizon=3059, full_native_context_and_reference_targets_not_clipped=True,
        small_CPU_network_not_full_size_GPU_admission=True, no_full_current_policy_updated=True,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(counts=model.counts(), full_action_horizon=proof['actual_forced_action_tokens'],
        complete_action_context=proof['actual_full_action_context_positions'], optimizer_commits=0,
        full_GPU_admission=False)), flush=True)


if __name__ == '__main__':
    main()
