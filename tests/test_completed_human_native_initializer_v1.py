"""Three actual small CPU DPO updates, followed by a four-envelope reader control.

The envelopes share that one actually trained CPU trajectory. Their explicit
four-rank metadata exercises inspection, not four-replica neural execution or
full-size completion. Only the separate real full-size run can establish that.
"""
import copy
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.completed_human_native_initializer_v1 import verify_private_states, materialize
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.parallel_human_preference_v2 import ParallelHumanPreferenceStage
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_stage_v1 import pairs
from test_joint_task_gradients_v1 import view, fixture, word_fixture

SOURCES = {'actual_small_reader_control': 'a' * 64}


@pytest.fixture(scope='module')
def completed():
    actor = view().actor; data = pairs(); steps = math.ceil(len(data) / 2)
    options = dict(seed=959117, epochs=1, steps=steps, microbatch=1, accumulation=2,
        learning_rate=.001, betas=[.9, .95], weight_decay=.01, gradient_clip=1.,
        DPO_beta=.1, DPO_weight=1, chosen_CE_weight=.1)
    optimizer = torch.optim.AdamW([p for _, p in actor.named_selected(('actor',))], lr=.001,
        betas=(.9, .95), weight_decay=.01, foreach=False)
    stage = ParallelHumanPreferenceStage(actor, optimizer, data, options, SOURCES)
    for _ in range(steps): stage.update()
    actor.eval(); original = stage.state(); states = []
    for rank in range(4):
        state = copy.deepcopy(original); state.update(replicas=4, replica_rank=rank)
        core = state['core']['core']; core['cursor'].update(world_size=4, replica_rank=rank)
        for row in core['metrics']:
            row['per_replica_original_pairs'] = [len(row['original_source_positions'][r::4]) for r in range(4)]
        rng = core['rng']; rng['python'] = random.Random(100 + rank).getstate()
        rng['torch'] = torch.Generator(device='cpu').manual_seed(100 + rank).get_state()
        n = np.random.RandomState(100 + rank).get_state()
        rng['numpy'] = dict(kind=n[0], keys=n[1].tolist(), position=n[2], has_gauss=n[3], cached_gaussian=n[4])
        states.append(state)
    return actor, states, options, len(data)


def inspect(completed, states=None, **kwargs):
    actor, original, options, population = completed
    return verify_private_states(actor, states if states is not None else original, options=options,
        population=population, sources=SOURCES, allow_mechanics=True, **kwargs)


def test_actual_completed_small_DPO_values_Adam_and_every_pair_reconcile_without_restoring_generators(completed):
    actor = completed[0]; before = tensor_digest(actor.named_parameters()), value_digest(capture_rng('cpu'))
    result = inspect(completed)
    assert result['all_four_final_private_states_match_exported_actor'] and result['new_Adam_calls'] == 0
    assert not result['actual_full_size_completed_human_state_readback'] and not result['GPU_training_admitted']
    assert result['actual_completed_human_updates'] == 3 and result['original_pair_population'] == 5
    assert len({r['actual_private_RNG_sha256'] for r in result['actual_rank_observations']}) == 4
    assert before == (tensor_digest(actor.named_parameters()), value_digest(capture_rng('cpu')))


@pytest.mark.parametrize('kind', ['unused_critic', 'Adam_clock', 'missing_rank', 'lost_pair', 'duplicate_private_RNG'])
def test_corrupted_or_incomplete_final_state_rejects_without_source_or_generator_mutation(completed, kind):
    actor, originals, _, _ = completed; states = copy.deepcopy(originals)
    before = tensor_digest(actor.named_parameters()), value_digest(capture_rng('cpu'))
    if kind == 'unused_critic': states[2]['core']['core']['model']['parts']['critic.weight'].add_(.25)
    elif kind == 'Adam_clock': states[1]['core']['core']['optimizer']['state'][0]['step'].add_(1)
    elif kind == 'missing_rank': states.pop()
    elif kind == 'lost_pair':
        core = states[0]['core']['core']; core['metrics'][-1]['original_source_positions'] = [0]
    else: states[3]['core']['core']['rng'] = copy.deepcopy(states[0]['core']['core']['rng'])
    with pytest.raises(ValueError): inspect(completed, states)
    assert before == (tensor_digest(actor.named_parameters()), value_digest(capture_rng('cpu')))


def test_full_native_materialization_control_keeps_every_value_and_actual_text_photo_word_outputs(completed, tmp_path):
    actor = completed[0]; verified = inspect(completed)
    photo, _ = fixture(1); _, word, _ = word_fixture()
    code = {'input_ids': torch.tensor([[1, 4, 7, 2]]), 'attention_mask': torch.ones(1, 4, dtype=torch.bool)}
    token = Path(__file__).resolve().parents[1] / 'data/tokenizer-corpus/tokenizer.json'
    if not token.exists(): pytest.skip('The original own native tokenizer is local and excluded from Git')
    result = materialize(actor, tmp_path / 'native', token, sources=SOURCES, verified=verified,
        probes=[('synthetic_text_control', code), ('synthetic_photo_control', photo)], word_pixels=word['word_pixels'],
        working_budget_bytes=256 * 1024**2, maximum_checkpoint_bytes=8 * 1024**2,
        minimum_free_bytes=40 * 1024**3, preflight=lambda: None, allow_mechanics=True)
    assert result['all_native_values_and_actual_text_photo_word_cold_outputs_exact']
    assert result['actual_counts']['total_unique'] == actor.base.counts()['total_unique']
    assert result['actual_language_fold']['original_native_critic_preserved']
    assert not result['actual_full_size_CPU_initializer'] and not result['full_size_GPU_capacity_or_training_admitted']
    assert result['inference_needs_only_native_checkpoint_and_packaged_tokenizer']
    assert result['joint_optimizer_updates'] == 0
