"""Actual complete native-head fitting after cached-feature capacity admission."""
import copy

import torch

from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .native_human_reward_v1 import NativeRewardFeatures, NativeRewardHeadStage, evaluate_reward
from .native_reward_bundle_v1 import export
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest


def fit(base, train, held, train_shards, held_shards, sources, options, *, checked=lambda: None,
        progress=lambda row: None, checkpoint=lambda stage: None, allow_mechanics=False):
    if options.get('steps', 0) < 2: raise ValueError('At least two real source updates are required for exact next-update capacity replay')
    training = NativeRewardFeatures(base, train, train_shards, sources, allow_mechanics=allow_mechanics)
    validation = NativeRewardFeatures(base, held, held_shards, sources, allow_mechanics=allow_mechanics)
    if next(base.parameters()).device.type != 'cpu': raise ValueError('The complete cached-feature head optimizer uses CPU FP32')
    def optimizer():
        return torch.optim.AdamW(base.heads['reward'].parameters(), lr=options['learning_rate'],
            betas=tuple(options['betas']), weight_decay=options['weight_decay'], foreach=False)
    initial = copy.deepcopy(base.heads['reward'].state_dict()); rng = capture_rng('cpu')
    probe = NativeRewardHeadStage(base, optimizer(), training, options); checked()
    probe.update(); state = probe.state(); pristine = copy.deepcopy(state); expected = probe.update()
    learned = tensor_digest(base.heads['reward'].named_parameters()); adam = value_digest(probe.optimizer.state_dict())
    next_rng = value_digest(capture_rng('cpu'))
    bad = copy.deepcopy(state); bad['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
    before = value_digest((base.heads['reward'].state_dict(), probe.optimizer.state_dict(), capture_rng('cpu')))
    try: probe.resume(bad)
    except ValueError: pass
    else: raise ValueError('An actually corrupted head Adam state must reject')
    if value_digest((base.heads['reward'].state_dict(), probe.optimizer.state_dict(), capture_rng('cpu'))) != before:
        raise ValueError('Corrupt head recovery cannot change live weights, Adam or private RNG')
    probe.resume(state)
    if (probe.update() != expected or tensor_digest(base.heads['reward'].named_parameters()) != learned
            or value_digest(probe.optimizer.state_dict()) != adam or value_digest(capture_rng('cpu')) != next_rng
            or value_digest(state) != value_digest(pristine)):
        raise ValueError('Actual next full-source head values, Adam, RNG and complete metrics must recover exactly')
    proof = dict(actual_logical_head_updates=2, actual_physical_head_Adam_calls=3, actual_Adam_states=len(probe.optimizer.state),
        exact_next_update_weights_Adam_RNG_and_metric_replay=True, corrupt_Adam_rejected_before_mutation=True,
        saved_recovery_payload_unchanged=True, source_population=len(training), full_original_population=training.full)
    base.heads['reward'].load_state_dict(initial, strict=True); base.heads['reward'].zero_grad(set_to_none=True); restore_rng(rng, 'cpu')
    del probe
    initial_held = evaluate_reward(base, validation)
    stage = NativeRewardHeadStage(base, optimizer(), training, options)
    while stage.completed < options['steps']:
        checked(); row = stage.update(); progress(row)
        if stage.completed % 256 == 0 or stage.completed == options['steps']: checkpoint(stage)
    bundle = export(stage); final_held = evaluate_reward(base, validation); checked()
    return dict(bundle=bundle, initial_held=initial_held, final_held=final_held, actual_head_recovery=proof)
