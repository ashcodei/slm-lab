"""Own-small complete objectives and actual four-worker private replay."""
from dataclasses import replace
from datetime import timedelta
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.joint_visual_cast_equivalence_v1 import prove_prefix
from slm_lab.joint_visual_experiment_v1 import live
from slm_lab.joint_visual_training_stage_v1 import make_plan, JointVisualStage
from slm_lab.joint_visual_private_checkpoint_v2 import FixedGuard, digest
from slm_lab.own_actor_vision_prefix_split_backward_v1 import OwnActorVisionPrefixSplitBackward
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
from slm_lab.deterministic_positions_v1 import install
from test_joint_visual_controller_v1 import nine_stage, NineFamilyMechanics
from test_joint_visual_private_checkpoint_v2 import sources, adam, OPTIONS


def stage35():
    original, batch, _ = nine_stage(bf16=True)
    base = install(OwnDenseAssistant(new_backbone(replace(original.view.actor.config, num_layers=35), 994117),
        reward_hidden=8, word_temporal=4, pointing_rank=8, router_hidden=8, detector_hidden=4))
    view = OwnActorVisionPrefixSplitBackward(OwnDenseActor(base, rank=2, seed=994217))
    optimizer = adam(view)
    fixed = FixedGuard(view, optimizer)
    bound = {k: v for k, v in sources(fixed).items() if k != 'run_plan_sha256'}
    plan = make_plan(view, optimizer, fixed, sources=bound, entries=original._entries,
        options=dict(OPTIONS, precision='bf16'), mode='mechanics')
    bound['run_plan_sha256'] = digest(plan)
    return JointVisualStage(view, optimizer, NineFamilyMechanics(batch), plan=plan, sources=bound,
        entries=original._entries, fixed=fixed), batch


def test_actual35layer_native_photo_and_FP32_CTC_word_Adam_gradients_and_updates_match(tmp_path):
    stage, batch = stage35()
    before = live(stage)
    cache, proof = prove_prefix(stage, tmp_path / 'state', maximum_bytes=8 * 1024**2,
        minimum_free_bytes=0, allow_mechanics=True)
    assert live(stage) == before and stage.completed == 0 and not stage.optimizer.state
    assert proof['all_three_objectives_and_Adam_input_gradients_and_weights_Adam_metrics_private_RNG_exact']
    assert proof['cache_inventory']['actual_conversions'] == 246
    assert proof['cache_inventory']['patched_detach_only_wrappers'] == 492
    assert proof['actual_separate_physical_capacity_updates'] == 6
    assert not proof['full_size_GPU_equivalence'] and not torch.cuda.is_initialized()
    word = stage._entries[2]
    assert proof['actual_prefix_families'][2] == word['family'] == 'words'
    stage.next_update()
    cache.assert_current()
    cache.close()
    (tmp_path / 'actual35layer-visual-cast-equivalence.json').write_text(json.dumps(proof, indent=2))


@pytest.mark.parametrize('case', ['not_explicit_mechanics', 'wrong_precision', 'already_advanced'])
def test_unadmitted_or_wrong_objective_boundary_rejects_before_any_new_forward(case, tmp_path):
    stage, _, _ = nine_stage(bf16=case != 'wrong_precision')
    if case == 'already_advanced':
        stage.next_update()
    before = live(stage)
    calls = []
    hook = stage.view.native.vision.register_forward_hook(lambda *args: calls.append(1))
    with pytest.raises(ValueError):
        prove_prefix(stage, tmp_path / 'state', maximum_bytes=8 * 1024**2,
            minimum_free_bytes=0, allow_mechanics=case != 'not_explicit_mechanics')
    hook.remove()
    assert not calls and live(stage) == before and not (tmp_path / 'state').exists()


def worker(rank, folder):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.nice(0) >= 19
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'rendezvous'), rank=rank,
        world_size=4, timeout=timedelta(minutes=5))
    try:
        stage, _, _ = nine_stage(rank=rank, world=4, bf16=True)
        random.seed(12000 + rank)
        np.random.seed(13000 + rank)
        torch.manual_seed(14000 + rank)
        before = live(stage)
        cache, proof = prove_prefix(stage, Path(folder) / 'private', maximum_bytes=8 * 1024**2,
            minimum_free_bytes=0, allow_mechanics=True)
        assert live(stage) == before and stage.completed == 0
        assert proof['all_three_objectives_and_Adam_input_gradients_and_weights_Adam_metrics_private_RNG_exact']
        assert not proof['full_size_GPU_equivalence'] and not torch.cuda.is_initialized()
        cache.assert_current()
        cache.close()
        (Path(folder) / (str(rank) + '.json')).write_text(json.dumps(proof, indent=2))
    finally:
        dist.destroy_process_group()


def test_four_actual_CPUreplicas_keep_distinct_private_RNG_and_complete_cross_objective_equivalence(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    reports = [json.loads((tmp_path / (str(r) + '.json')).read_text()) for r in range(4)]
    assert all(r['passed'] and r['actual_separate_physical_capacity_updates'] == 6 for r in reports)
    assert len({r['external_original_initializer_pointer_sha256'] for r in reports}) == 1
