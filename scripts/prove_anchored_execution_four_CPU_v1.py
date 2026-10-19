"""Four real CPU replicas: native anchored updates, recovery and fail coordination.

Small random own network and horizon2 only. This cannot admit full-size GPU
production or establish useful RL/coding behavior.
"""
import copy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/anchored-execution-four-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('All GPUs must be hidden for CPU mechanics')
    import numpy as np
    import torch
    from torch import distributed as dist
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
    dist.init_process_group('gloo', timeout=timedelta(minutes=8))
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if world != 4: raise ValueError('All four CPU replicas required')
        if rank == 0:
            if REPORTS.exists(): raise ValueError('Preserve previous complete or partial proof')
            REPORTS.mkdir()
        dist.barrier()
        corpus = ExpandedExecutionCorpus(ROOT)
        positions = [min((i for i, row in enumerate(corpus.rows) if row['kind'] == kind),
                         key=lambda i: corpus.rows[i]['complete_context_positions']) for kind in ['documented', 'apps']]
        config = ModelConfig(vocab_size=32768, hidden_size=32, intermediate_size=64, num_layers=2,
            num_heads=4, num_kv_heads=2, max_text_length=4096, vision_hidden_size=32,
            vision_intermediate_size=64, vision_layers=1, vision_heads=4, image_size=32, patch_size=16, pool_size=2)
        base = OwnDenseAssistant(new_backbone(config, 962117), reward_hidden=8, word_temporal=4,
            pointing_rank=8, router_hidden=8, detector_hidden=4, word_expansion=1).eval()
        model = OwnDenseActor(base, rank=2, seed=962217).eval()
        optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))],
            lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
        random.seed(962317 + rank); np.random.seed(962317 + rank); torch.manual_seed(962317 + rank)
        base_digest = tensor_digest(base.named_parameters())
        sources = dict(kernel_sha256=file_hash(ROOT / 'src/slm_lab/anchored_execution_step_v1.py'),
            source_admission_sha256=file_hash(ROOT / 'data/apps-execution-rewards-v1/admission.json'))
        files = [Path(__file__), ROOT / 'src/slm_lab/anchored_execution_step_v1.py',
            ROOT / 'reports/anchored-execution-CPU-v1/complete.json',
            ROOT / 'reports/expanded-execution-corpus-v1/complete.json',
            ROOT / 'reports/anchored-code-objective-v1-tests.log',
            ROOT / 'data/apps-execution-rewards-v1/admission.json']
        bindings = {str(p): file_hash(p) for p in files}
        if rank == 0:
            atomic_json(REPORTS / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256=bindings,
                world=4, actual_counts=model.counts(), logical_updates=2, physical_optimizer_calls_per_replica=3,
                private_RNGs='Python/NumPy/Torch-CPU; GPU hidden', complete_original_source_positions=positions,
                CPU_action_horizon=2, small_random_network_not_full_policy=True,
                full_profile_GPU_capacity_proven=False, capability_gain_tested=False, goal_achieved=False))
        def step(position):
            return update(model, optimizer, corpus, position, method='anchor_execution', local_group_size=1,
                rank=rank, world=4, cache_absolute_bound=.0001, CPU_mechanics_horizon=2)
        def equal(a, b):
            if isinstance(a, torch.Tensor): return torch.equal(a, b)
            if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
            if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
            return a == b
        first = step(positions[0])
        private = make_state(model, optimizer, groups=('actor',), sources=sources, completed=1, planned=2,
            cursor=dict(next_original_position=positions[1], outer_world_size=4, replica_rank=rank), metrics=[first], device='cpu')
        saved = REPORTS / f'after-first-rank-{rank}.pt'
        torch.save(dict(core=private, outer_world_size=4, replica_rank=rank, all_four_private_states_required=True), saved)
        expected = step(positions[1]); weights = model.export(sources)
        adam = copy.deepcopy(optimizer.state_dict()); rng = capture_rng('cpu')
        raw = torch.load(saved, map_location='cpu', weights_only=True)
        if raw['outer_world_size'] != 4 or raw['replica_rank'] != rank or raw['all_four_private_states_required'] is not True:
            raise ValueError('Exact outer replica/private-state scope required')
        restore_state(model, optimizer, raw['core'], groups=('actor',), expected_sources=sources, planned=2,
            expected_cursor=dict(next_original_position=positions[1], outer_world_size=4, replica_rank=rank), device='cpu')
        actual = step(positions[1])
        if not equal(weights, model.export(sources)) or not equal(adam, optimizer.state_dict()) or not equal(rng, capture_rng('cpu')):
            raise ValueError('Exact real next global update, Adam and every private CPU RNG required')
        for key in ['global_rewards', 'global_advantages', 'original_reference_CE_sum', 'policy_surrogate',
                    'reference_KL', 'gradient_norm_before_clip', 'cached_full_max_absolute_differences']:
            if expected[key] != actual[key]: raise ValueError('Exact next global objective record required')
        for a, b in zip(expected['trajectories'], actual['trajectories']):
            for key in ['response_ids', 'actual_cached_behavior_logps', 'actual_full_behavior_logps', 'own_response']:
                if a[key] != b[key]: raise ValueError('Every actual global own-response token/probability must replay')
        if base_digest != tensor_digest(base.named_parameters()):
            raise ValueError('Own reference, vision and all base heads changed')

        # A single worker receives deliberately corrupted cached logits. No
        # model file, weight, test or reward is changed by the fault injection.
        original_prefill = model.policy.prefill
        if rank == 3:
            def faulty_prefill(*args, **kwargs):
                result = original_prefill(*args, **kwargs)
                logits = result['logits'].clone(); logits[:, :, ::2] += .2
                return {**result, 'logits': logits}
            model.policy.prefill = faulty_prefill
        original_reduce = dist.all_reduce; calls = []
        def counted_reduce(*args, **kwargs):
            calls.append(True)
            return original_reduce(*args, **kwargs)
        dist.all_reduce = counted_reduce
        rejected = None
        try:
            step(positions[1])
        except ValueError as error:
            rejected = str(error)
        finally:
            dist.all_reduce = original_reduce
            model.policy.prefill = original_prefill
        if (rejected is None or 'Cached/full actual behavior' not in rejected or calls
                or not equal(weights, model.export(sources)) or not equal(adam, optimizer.state_dict())
                or any(p.grad is not None for p in model.parameters())):
            raise ValueError('One-worker failure must coordinate before gradient reduction or any optimizer commit')
        for path, digest in bindings.items():
            if file_hash(path) != digest: raise ValueError('Bound proof source changed')
        proof = dict(rank=rank, actual_counts=model.counts(), actual_global_sample_count=4,
            actual_logical_updates=2, physical_optimizer_calls=3, all_private_CPU_RNGs_next_update_exact=True,
            all_global_sampled_actions_probabilities_and_objectives_replayed=True,
            every_selected_Adam_value_replayed=True, complete_native_reference_targets_used_every_update=True,
            base_heads_and_unused_critic_unchanged=True, actual_reward_groups=[first['global_rewards'], actual['global_rewards']],
            deliberate_single_rank_probability_error_rejected=True, fault_rejection=rejected,
            fault_gradient_reduction_calls=len(calls), fault_optimizer_commits=0,
            final_actor_digest=tensor_digest(model.parts.named_parameters()),
            final_Adam_digest=tensor_digest((str(i) + '.' + key, value) for i, state in optimizer.state_dict()['state'].items()
                for key, value in state.items()), private_state_sha256=file_hash(saved),
            CPU_mechanics_only_not_full_GPU_admission=True, source_files_sha256=bindings,
            capability_gain_tested=False, goal_achieved=False)
        atomic_json(REPORTS / f'rank-{rank}.json', proof)
        dist.barrier()
        if rank == 0:
            proofs = [json.loads((REPORTS / f'rank-{r}.json').read_text()) for r in range(4)]
            if len({p['final_actor_digest'] for p in proofs}) != 1 or len({p['final_Adam_digest'] for p in proofs}) != 1:
                raise ValueError('All actual replica values/Adam must be equal')
            atomic_json(REPORTS / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), world=4,
                proofs=proofs, logical_updates=2, physical_optimizer_calls_per_replica=3,
                all_four_next_update_weights_Adam_actions_objectives_and_private_CPU_RNGs_exact=True,
                single_rank_probability_failure_rejected_before_any_gradient_reduce_or_commit=True,
                same_final_actor_and_Adam_all_four_replicas=True, source_files_sha256=bindings,
                small_random_network_short_CPU_horizon_only=True, full_new_profile_GPU_capacity_still_required=True,
                no_full_current_model_updates=True, no_capability_gain_measured=True,
                candidate_promoted=False, goal_achieved=False))
            print(json.dumps(dict(world=4, logical_updates=2, next_update_recovery_exact=True,
                one_rank_fault_coordinated_before_reduce=True, full_GPU_admission=False)), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
