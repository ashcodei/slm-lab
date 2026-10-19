"""Actual own-small four-replica controller; never full-model capacity proof."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.batched_reasoning_capacity_controller_v1 import _cold, live
from slm_lab.mixed_coding_SFT_capacity_controller_v1 import run_capacity, require_full_capacity_reports
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData, PROFILES
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.parallel_mixed_coding_SFT_stage_v1 import ParallelMixedCodingSFTStage
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from test_parallel_mixed_coding_SFT_stage_v1 import ROOT, OPTIONS, SOURCES, source_positions
from test_public_reasoning_group_gradients_v1 import optimizer


def student():
    config = ModelConfig(vocab_size=32768, hidden_size=8, intermediate_size=16, num_layers=1,
        num_heads=2, num_kv_heads=1, max_text_length=4096, vision_hidden_size=8, vision_intermediate_size=16,
        vision_layers=1, vision_heads=2, image_size=32, patch_size=4, pool_size=2)
    torch.manual_seed(989924)
    base = OwnDenseAssistant(new_backbone(config, 982113), pointing_rank=4, reward_hidden=4,
        word_temporal=4, router_hidden=4, detector_hidden=4, word_expansion=1)
    return OwnDenseActor(base, rank=2).eval()


def worker(rank, folder):
    folder = Path(folder)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'controller-rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=10))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=10))
    try:
        data = MixedCodingSFTData(ROOT)
        ps = source_positions(data)
        entries = [ps[:12], ps[12:17], ps[17:18]]
        ids, _ = data.sequence(ps[0])
        text = torch.tensor([ids[:32]])
        photo = torch.tensor([ids[:8]])
        inputs = dict(text=dict(input_ids=text, attention_mask=torch.ones_like(text, dtype=torch.bool)),
            photo=dict(input_ids=photo, attention_mask=torch.ones_like(photo, dtype=torch.bool),
                pixel_values=torch.linspace(-1., 1., 3*32*32).reshape(1, 3, 32, 32)),
            word=dict(word_pixels=torch.linspace(-1., 1., 3*64*64).reshape(1, 3, 64, 64)))
        def independent():
            with torch.random.fork_rng(devices=[]): return student()
        for profile in PROFILES:
            own = student()
            random.seed(989925 + rank)
            np.random.seed(989925 + rank)
            torch.manual_seed(989925 + rank)
            stage = ParallelMixedCodingSFTStage(own, optimizer(own), data, entries, profile, OPTIONS, SOURCES,
                mode='mechanics', rank=rank, world=4, control_group=control, gradient_group=None)
            plan = dict(capacity_entries=copy.deepcopy(entries), mechanics_only=True, complete_source_selection=False,
                full_original_capacity_plan_not_used_by_this_explicit_small_fixture=True)
            before = live(stage)
            with pytest.raises(ValueError): run_capacity(stage, folder / 'forbidden', capacity_plan=plan,
                independent_actor=independent, cold_inputs=inputs, maximum_bytes=32*1024**2, minimum_free_bytes=0)
            assert live(stage) == before
            bad = copy.deepcopy(plan)
            if rank == 2: bad['mechanics_only'] = False
            with pytest.raises(ValueError): run_capacity(stage, folder / 'bad-source', capacity_plan=bad,
                independent_actor=independent, cold_inputs=inputs, maximum_bytes=32*1024**2, minimum_free_bytes=0,
                allow_mechanics=True)
            assert live(stage) == before
            result = run_capacity(stage, folder / profile, capacity_plan=plan, independent_actor=independent,
                cold_inputs=inputs, maximum_bytes=32*1024**2, minimum_free_bytes=0, allow_mechanics=True)
            assert result['actual_complete_capacity_updates'] == 3 and result['actual_physical_optimizer_calls'] == 4
            assert result['actual_Adam_states'] == 17 and result['actual_source_budget']['source_presentations'] == 18
            assert result['actual_research_actor_parameters'] == 267718 and result['actual_native_parameters'] == 267430
            assert result['exact_next_losses_unclipped_gradients_weights_Adam_private_RNG']
            assert result['independent_complete_actor_cold']['complete_text_photo_word_outputs_bit_exact']
            assert result['independent_complete_actor_cold']['actual_word_shape'] == [64, 64]
            assert not result['full_size_GPU_capacity_passed'] and result['production_updates'] == 0
            proofs = stage.gather(result)
            with pytest.raises(ValueError): require_full_capacity_reports(proofs, capacity_plan=plan, profile=profile,
                expected_sources=SOURCES, expected_options=OPTIONS)
            saved, _, _ = read_latest(folder / profile)
            before = live(stage)
            with pytest.raises(ValueError, match='independent complete actor cold'):
                _cold(stage, (lambda: stage.model) if rank == 2 else independent, saved, inputs)
            assert live(stage) == before
            result.update(peer_only_hot_model_alias_rejected_without_live_mutation=True,
                actual_missing_mechanics_opt_in_and_peer_source_rejected_before_Adam=True)
            assert not torch.cuda.is_initialized()
            (folder / (profile + '-controller-rank-' + str(rank) + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
    finally: dist.destroy_process_group()


def test_actual_four_replica_connected_full_context_CE_capacity_replay_and_cold(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    reports = [json.loads((tmp_path / (p + '-controller-rank-' + str(r) + '.json')).read_text()) for p in PROFILES for r in range(4)]
    for profile in PROFILES:
        assert len({r['final_actual_shared_checkpoint_sha256'] for r in reports if r['profile'] == profile}) == 1
