"""Four Gloo replicas, original human train pairs, small-network recovery proof."""
import copy
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/parallel-human-preference-four-CPU-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('All GPUs must be hidden')
    import numpy as np
    import torch
    from torch import distributed as dist
    from probe_dense_human_preference_stage_full_CPU_v1 import equal
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint, capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
    from slm_lab.parallel_human_preference_v2 import ParallelHumanPreferenceStage
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 4:
        raise ValueError('Four actual CPU replicas required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', timeout=timedelta(minutes=5))
    started = time.monotonic()
    try:
        if rank == 0:
            if REPORTS.exists():
                raise ValueError('Preserve all earlier attempts')
            REPORTS.mkdir()
        dist.barrier()
        files = [Path(__file__), ROOT/'src/slm_lab/parallel_human_preference_v2.py',
                 ROOT/'tests/test_parallel_human_preference_v2.py',
                 ROOT/'reports/parallel-human-preference-v2-tests-v2.log',
                 ROOT/'src/slm_lab/dense_human_preference_stage_v1.py',
                 ROOT/'src/slm_lab/dense_human_preference_v1.py',
                 ROOT/'src/slm_lab/own_dense_actor_checkpoint_v1.py',
                 ROOT/'src/slm_lab/whole_human_data_v2.py',
                 ROOT/'data/whole-human-alignment-v2/admission.json']
        bindings = {str(p): file_hash(p) for p in files}
        guard = FileBindings(bindings)
        data = WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'preferences', 'train', 'optimization')
        config = ModelConfig(vocab_size=32768, hidden_size=32, intermediate_size=64,
            num_layers=2, num_heads=4, num_kv_heads=2, max_text_length=4096,
            vision_hidden_size=8, vision_intermediate_size=16, vision_layers=1,
            vision_heads=2, image_size=16, patch_size=4, pool_size=2)
        base = OwnDenseAssistant(new_backbone(config, 958991), reward_hidden=8,
            word_temporal=4, pointing_rank=8, router_hidden=8, detector_hidden=4)
        model = OwnDenseActor(base, rank=2, seed=958991)
        options = dict(seed=958993, epochs=1, steps=math.ceil(len(data)/4), microbatch=1,
            accumulation=4, learning_rate=.001, betas=[.9, .95], weight_decay=.01,
            gradient_clip=1., DPO_beta=.1, DPO_weight=1, chosen_CE_weight=.1)
        optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))],
            lr=.001, betas=(.9, .95), weight_decay=.01, foreach=False)
        sources = dict(human_admission_sha256=file_hash(files[-1]),
                       actual_parallel_kernel_sha256=file_hash(files[1]))
        stage = ParallelHumanPreferenceStage(model, optimizer, data, options, sources,
            world=4, rank=rank, gradient_group=dist.group.WORLD, control_group=dist.group.WORLD)
        random.seed(958997+rank)
        np.random.seed(958997+rank)
        torch.manual_seed(958997+rank)
        frozen = tensor_digest(base.named_parameters())
        initial = tensor_digest(model.parts.named_parameters())
        first = stage.update()
        snapshot = stage.state()
        atomic_checkpoint(REPORTS/f'private-resume-{rank}.pt', snapshot, 16*1024**2, 26*1024**3)
        def draws():
            return [random.random(), float(np.random.random()), float(torch.rand(()))]
        expected_draws = draws()
        expected = stage.update()
        weights = tensor_digest(model.parts.named_parameters())
        moments = copy.deepcopy(optimizer.state_dict())
        RNG = capture_rng('cpu')
        bad = copy.deepcopy(snapshot)
        bad['core']['core']['rng']['torch'] = torch.zeros(3, dtype=torch.uint8)
        try:
            stage.resume(bad)
        except ValueError:
            pass
        else:
            raise ValueError('Bad private RNG must reject')
        if (weights != tensor_digest(model.parts.named_parameters())
                or not equal(moments, optimizer.state_dict()) or not equal(RNG, capture_rng('cpu'))):
            raise ValueError('Invalid resume must not mutate weights, Adam or any RNG')
        stage.resume(snapshot)
        if (draws() != expected_draws or stage.update() != expected
                or weights != tensor_digest(model.parts.named_parameters())
                or not equal(moments, optimizer.state_dict()) or not equal(RNG, capture_rng('cpu'))):
            raise ValueError('Actual next distributed whole-pair update must recover exactly')
        if initial == weights or frozen != tensor_digest(base.named_parameters()):
            raise ValueError('Actor must learn with exact original frozen own reference')
        guard.check(full=True)
        result = dict(rank=rank, actual_counts=model.counts(), actual_Adam_states=len(optimizer.state),
            actual_logical_updates=2, physical_optimizer_calls=3, actual_global_pair_records=[first, expected],
            learned_parts_sha256=weights, Adam_sha256=tensor_digest((str(i)+'.'+k, s[k])
                for i, s in enumerate(optimizer.state.values()) for k in ('step', 'exp_avg', 'exp_avg_sq')),
            exact_next_source_update_weights_Adam_and_private_Python_numpy_torch_RNG=True,
            invalid_private_RNG_rejected_before_mutation=True,
            all_frozen_base_vision_heads_and_unused_critic_exact=True,
            source_files_sha256=bindings, small_random_network=True,
            full_size_GPU_capacity_and_held_quality_not_measured=True,
            production_updates=0, seconds=time.monotonic()-started, goal_achieved=False)
        atomic_json(REPORTS/f'rank-{rank}.json', result)
        results = [None]*4
        dist.all_gather_object(results, result)
        if rank == 0:
            if (len({r['learned_parts_sha256'] for r in results}) != 1
                    or len({r['Adam_sha256'] for r in results}) != 1):
                raise ValueError('Every actual replica must end with identical actor and Adam values')
            atomic_json(REPORTS/'complete.json', dict(at=datetime.now(timezone.utc).isoformat(),
                proofs=results, all_four_CPU_workers_passed=True,
                actual_original_human_train_pairs=len(data), actual_logical_unique_pair_draws=8,
                actual_global_pairs_per_update=4, policy_and_reference_full_response_EOS_masks=True,
                genuine_human_rankings_on_other_model_outputs=True,
                fresh_human_judgments_on_our_outputs=False,
                full_size_production_human_preference_training_not_started=True,
                production_updates=0, goal_achieved=False))
            print(json.dumps(dict(all_four_CPU_workers_passed=True,
                actual_unique_human_pair_draws=8, seconds=time.monotonic()-started)), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
