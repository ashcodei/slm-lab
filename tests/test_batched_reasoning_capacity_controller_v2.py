"""Full actual source selection; explicitly own-small CPU runtime mechanics."""
from collections import Counter
import copy
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.batched_reasoning_capacity_cases_v1 import choose
from slm_lab.batched_reasoning_capacity_controller_v1 import run_capacity, require_full_capacity_reports, _cold, live
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
from test_parallel_batched_reasoning_stage_v2 import positions, OPTIONS, SOURCES, data
from test_public_reasoning_group_gradients_v1 import optimizer
from slm_lab.parallel_batched_reasoning_stage_v2 import ParallelBatchedReasoningStage


def make_student(data):
    """Own-small32pixel base admits64pixel words without changing runtime guards."""
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    config = ModelConfig(vocab_size=32768, hidden_size=8, intermediate_size=16, num_layers=1,
        num_heads=2, num_kv_heads=1, max_text_length=4096, vision_hidden_size=8, vision_intermediate_size=16,
        vision_layers=1, vision_heads=2, image_size=32, patch_size=4, pool_size=2)
    torch.manual_seed(989218)
    base = OwnDenseAssistant(new_backbone(config, 982113), pointing_rank=4, reward_hidden=4,
        word_temporal=4, router_hidden=4, detector_hidden=4, word_expansion=1)
    p = positions(data)[0]
    ids, _ = data.sequence(p)
    start = data.information(p)['original_prompt_positions']
    network = base.network
    with torch.no_grad():
        network.embedding.weight.zero_()
        network.embedding.weight[ids[start-1], 0] = 1.
        network.embedding.weight[ids[start], 0] = 100.
        network.embedding.weight[ids[start], 1] = 1.
        network.embedding.weight[2, 0] = 100.
        network.embedding.weight[2, 1] = -1.
        for block in network.blocks:
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']: getattr(block.attn, name).weight.zero_()
            block.attn.o_proj.weight.copy_(torch.eye(8))
            for name in ['gate', 'up', 'down']: getattr(block, name).weight.zero_()
    return OwnDenseActor(base, rank=2).eval()


def setup(data, profile='anchor_RL', rank=0, world=1, control=None):
    import random
    import numpy as np
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    student = make_student(data)
    entries = [[p] for p in positions(data)[:3]] if world == 1 else [positions(data)[:4], positions(data)[4:8], positions(data)[8:9]]
    random.seed(989313 + rank)
    np.random.seed(989313 + rank)
    torch.manual_seed(989313 + rank)
    return ParallelBatchedReasoningStage(student, optimizer(student), data, entries, profile, OPTIONS, SOURCES,
        mode='mechanics', rank=rank, world=world, control_group=control, gradient_group=None)


@pytest.mark.parametrize('epochs', [1, 3])
def test_full_original_actual_sources_targets_worst_batches_and_all_rank_rotations(data, tmp_path, epochs):
    plan = choose(data, seed=OPTIONS['seed'], epochs=epochs)
    full = data.complete_schedule(epochs=epochs, width=4, seed=OPTIONS['seed'])
    assert Counter(p for row in full for p in row) == Counter({p: epochs for p in data.slots})
    assert plan['complete_training_budget'] == data.budget(full)
    assert plan['complete_training_budget']['updates'] == 642 * epochs
    assert plan['complete_training_budget']['response_EOS_targets'] == 19080 * epochs
    assert plan['full_original_action_budgets'] == [64, 256] and not plan['actual_original_training_has_empty_tail']
    native = {}
    for p in data.slots:
        ids, labels = data.sequence(p)
        start = next(i for i, x in enumerate(labels) if x != -100)
        native[p] = dict(context=len(ids), prefix=start, targets=sum(x != -100 for x in labels), task=data.row(p)['task'])
        assert ids[-1] == 2 and labels == [-100] * start + ids[start:]
    selected_sources = set()
    for batch in plan['selected_intact_original_batches']:
        original = full[batch['original_global_step'] - 1]
        assert batch['intact_original_positions'] == original
        selected_sources.update(original)
        mappings = [m for m in plan['capacity_case_mapping'] if m.get('original_global_step') == batch['original_global_step']]
        assert len(mappings) == 4
        for rank in range(4):
            assert {m['actual_original_positions'][rank] for m in mappings} == set(original)
        for m in mappings:
            rotation = m['whole_batch_rank_rotation']
            assert m['actual_original_positions'] == original[rotation:] + original[:rotation]
            assert plan['capacity_entries'][m['capacity_step'] - 1] == m['actual_original_positions']
    tasks = sorted({r['task'] for r in native.values()})
    assert len(tasks) == 12 and plan['actual_train_task_families'] == tasks
    # Independently enumerate maxima from actual native input/label arrays,
    # without using the producer's reasons, information values or selector.
    for task in tasks:
        rows = [p for p, r in native.items() if r['task'] == task]
        for key in ['context', 'prefix', 'targets']:
            actual_max = max(native[p][key] for p in rows)
            assert any(native[p][key] == actual_max for p in selected_sources if native[p]['task'] == task)
    for key in ['context', 'targets']:
        worst_sum = max(sum(native[p][key] for p in row) for row in full)
        assert any(sum(native[p][key] for p in b['intact_original_positions']) == worst_sum for b in plan['selected_intact_original_batches'])
    last = plan['capacity_case_mapping'][-1]
    assert last['separate_single_source_zero_gradient_control'] and not last['actual_original_training_tail']
    assert last['actual_rank_source_counts'] == [1, 0, 0, 0]
    assert plan['capacity_budget'] == data.budget(plan['capacity_entries'])
    assert len(plan['capacity_entries']) == 4 * len(plan['selected_intact_original_batches']) + 1
    (tmp_path / 'actual-complete-worst-source-selection.json').write_text(json.dumps(dict(
        actual_original_train_sources=len(native), actual_task_families=tasks, complete_training_budget=data.budget(full),
        actual_selected_intact_original_batches=len(plan['selected_intact_original_batches']),
        actual_capacity_entries=len(plan['capacity_entries']), actual_capacity_budget=plan['capacity_budget'],
        independent_native_ids_labels_all_task_and_whole_batch_maxima_checked=True,
        every_physical_replica_receives_every_selected_worst_source=True,
        separate_zero_gradient_control_not_an_actual_training_tail=True, models_constructed=0,
        GPU_contexts=0, optimizer_updates=0, training_admitted=False), indent=2))


def test_invalid_source_role_and_epoch_settings_cannot_create_a_capacity_plan(data):
    for options in [dict(seed=True, epochs=1), dict(seed=1, epochs=0), dict(seed=0, epochs=1)]:
        with pytest.raises(ValueError): choose(data, **options)
    data.role = 'reserved'
    try:
        with pytest.raises(ValueError): choose(data, seed=OPTIONS['seed'], epochs=3)
    finally: del data.role


def test_single_worker_fixture_cannot_run_four_GPU_controller_or_gain_admission(data, tmp_path):
    stage = setup(data)
    before = live(stage)
    with pytest.raises(ValueError):
        run_capacity(stage, tmp_path, capacity_plan=dict(capacity_entries=stage.entries, mechanics_only=True),
            independent_actor=lambda: make_student(data), cold_inputs={}, maximum_bytes=32 * 1024**2,
            minimum_free_bytes=0, allow_mechanics=True)
    assert live(stage) == before
    with pytest.raises(ValueError):
        require_full_capacity_reports([], capacity_plan=dict(mechanics_only=True), profile='anchor_RL',
            expected_sources=SOURCES, expected_options=OPTIONS, expected_reference=stage.reference_binding)


def cold_inputs(data):
    ids, _ = data.sequence(0)
    text = torch.tensor([ids[:32]], dtype=torch.long)
    photo = torch.tensor([ids[:8]], dtype=torch.long)
    return dict(text=dict(input_ids=text, attention_mask=torch.ones_like(text, dtype=torch.bool)),
        photo=dict(input_ids=photo, attention_mask=torch.ones_like(photo, dtype=torch.bool),
            pixel_values=torch.linspace(-1., 1., 3 * 32 * 32).reshape(1, 3, 32, 32)),
        word=dict(word_pixels=torch.linspace(-1., 1., 3 * 64 * 64).reshape(1, 3, 64, 64)))


def worker(rank, folder):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = Path(folder)
    dist.init_process_group('gloo', init_method='file://' + str(folder / 'rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=5))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=5))
    try:
        public = PublicReasoningTrainData('data/public-reasoning-curriculum-v1',
            'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
        inputs = cold_inputs(public)
        def independent():
            with torch.random.fork_rng(devices=[]): return make_student(public)
        for profile in ['anchor_CE', 'anchor_RL']:
            stage = setup(public, profile, rank, 4, control)
            plan = dict(capacity_entries=copy.deepcopy(stage.entries), mechanics_only=True,
                complete_source_selection=False, actual_full_source_capacity_plan_not_used_by_this_small_fixture=True)
            before = live(stage)
            # Omitting the explicit mechanics opt-in cannot start any Adam call.
            with pytest.raises(ValueError):
                run_capacity(stage, folder / 'forbidden', capacity_plan=plan, independent_actor=independent,
                    cold_inputs=inputs, maximum_bytes=32 * 1024**2, minimum_free_bytes=0)
            assert live(stage) == before
            bad = copy.deepcopy(plan)
            if rank == 2: bad['mechanics_only'] = False
            with pytest.raises(ValueError):
                run_capacity(stage, folder / 'bad-source', capacity_plan=bad, independent_actor=independent,
                    cold_inputs=inputs, maximum_bytes=32 * 1024**2, minimum_free_bytes=0, allow_mechanics=True)
            assert live(stage) == before
            directory = folder / profile / 'durable'
            result = run_capacity(stage, directory, capacity_plan=plan, independent_actor=independent,
                cold_inputs=inputs, maximum_bytes=32 * 1024**2, minimum_free_bytes=0, allow_mechanics=True)
            assert result['actual_complete_capacity_updates'] == 3 and result['actual_physical_optimizer_calls'] == 4
            assert result['actual_source_budget']['source_presentations'] == 9 and result['actual_Adam_states'] == 17
            assert result['actual_committed_prefix_trajectory_count'] == 36
            assert result['actual_all_physical_global_generated_actions_including_replay'] > result['actual_committed_prefix_generated_action_count']
            assert result['actual_research_actor_parameters'] == 267718 and not result['full_size_GPU_capacity_passed']
            assert result['all_actual_next_actions_verifiers_unclipped_gradients_Adam_private_RNG_exact']
            assert result['independent_complete_actor_cold']['complete_text_photo_word_outputs_bit_exact']
            assert result['independent_complete_actor_cold']['actual_photo_side'] == 32
            assert result['independent_complete_actor_cold']['actual_word_shape'] == [64, 64]
            assert result['actual_complete_capacity_source_positions'][-1] == [stage.entries[-1][0]]
            reports = stage.gather(result)
            with pytest.raises(ValueError):
                require_full_capacity_reports(reports, capacity_plan=plan, profile=profile,
                    expected_sources=stage.sources, expected_options=stage.options, expected_reference=stage.reference_binding)
            # A peer returning the hot model instead of an independent own
            # cold loader must fail without copying a single live parameter.
            saved, _, _ = read_latest(directory)
            before = live(stage)
            with pytest.raises(ValueError, match='independent complete actor cold'):
                _cold(stage, (lambda: stage.model) if rank == 2 else independent, saved, inputs)
            assert live(stage) == before
            assert stage.failure_observations[-1]['actual_rank_errors'][2] is not None
            result['actual_peer_only_hot_model_alias_rejected_without_live_mutation'] = True
            result['actual_peer_only_source_mode_error_coordinated_before_any_update'] = True
            result['actual_missing_mechanics_opt_in_rejected_before_any_update'] = True
            assert not torch.cuda.is_initialized()
            (folder / (profile + '-rank-' + str(rank) + '.json')).write_text(json.dumps(result, indent=2, allow_nan=False))
    finally: dist.destroy_process_group()


def test_actual_four_replica_connected_controller_private_replay_cold_and_peer_faults(tmp_path):
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    reports = [json.loads((tmp_path / (p + '-rank-' + str(r) + '.json')).read_text())
        for p in ['anchor_CE', 'anchor_RL'] for r in range(4)]
    assert len(reports) == 8
    for profile in ['anchor_CE', 'anchor_RL']:
        assert len({r['final_actual_shared_checkpoint_sha256'] for r in reports if r['profile'] == profile}) == 1
