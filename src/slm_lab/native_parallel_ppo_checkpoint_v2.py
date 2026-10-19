"""Coordinated four-rank PPO recovery before any neural or RNG mutation."""
import copy

import torch

from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .code_residual_checkpoint_v1 import _cpu, _digest, _validate_adam, _validate_rng
from .native_parallel_ppo_history_v2 import read_prefix
from .native_parallel_ppo_v2 import ParallelNativePPO
from .own_dense_actor_checkpoint_v1 import selected
from .parallel_function_repair_stage_v1 import value_digest

FORMAT = 'own-four-rank-native-PPO-private-Adam-RNG-constant-history-tip-v2'
PRIVATE = {'rank', 'private_RNG', 'private_RNG_sha256', 'history', 'common_sha256'}


def common_digest(state): return value_digest({k:v for k,v in state.items() if k not in PRIVATE})


def prefix(stage, saved_prefix, budgets, last_sha256, checked):
    expected = [last_sha256]
    def validate(record):
        stage.validate_record(record)
        if record['global_record_sha256'] != expected[0]:
            raise ValueError('Every actual common mathematical PPO update must match its independent complete history chain')
        expected[0] = record['previous_global_record_sha256']
    result = read_prefix(saved_prefix, binding=stage.history_binding, budgets=budgets,
        validate=validate, guard=lambda: (checked(), stage.check()))
    if expected[0] != stage.initial_global_record_sha256:
        raise ValueError('Every original shared PPO source/action/gradient history must close at the fixed initial recipe')
    return result


def make_state(stage):
    if type(stage) is not ParallelNativePPO: raise ValueError('The actual four-rank shared-native PPO graph is required')
    error = common = state = None
    try:
        stage.check(full=True); kernel = stage.kernel
        if stage.completed < 1 or stage.history is None: raise ValueError('An actual completed four-rank PPO update and writer are required')
        tip = stage.history.prefix()
        if tip is None or tip['completed'] != stage.completed: raise ValueError('The entire actual trajectory prefix must match the private state')
        adam = _cpu(kernel.optimizer.state_dict()); _validate_adam(adam, kernel.parameters, stage.completed)
        rng = capture_rng(kernel.device); _validate_rng(rng, kernel.device)
        state = dict(format=FORMAT, binding=_cpu(stage.binding), binding_sha256=stage.binding_sha256,
            rank=stage.rank, world_size=4, completed=stage.completed, cursor=stage.cursor(stage.completed),
            policy=kernel.model.export(kernel.sources), optimizer=adam,
            private_RNG=_cpu(rng), private_RNG_sha256=value_digest(rng),
            history=tip, history_budgets=dict(stage.history.budgets), last_global_record_sha256=stage.last_global_record_sha256,
            no_resident_trajectory_history=True, useful_own_output_alignment_proved=False, candidate_promoted=False, goal_achieved=False)
        state['common_sha256'] = common_digest(state); common = state['common_sha256']
    except Exception as exc: error = type(exc).__name__+': '+str(exc)
    stage.collective.agree(error, common)
    identities = stage.collective.gather(state['private_RNG_sha256'])
    if len(set(identities)) != 4: raise ValueError('All four actual private policy samplers must have distinct retained generators')
    return state


def restore_state(stage, state, *, expected_common_sha256, expected_private_rng_sha256,
                  expected_completed, allow_completed=False, checked=lambda: None):
    if type(stage) is not ParallelNativePPO: raise ValueError('The actual four-rank shared-native PPO graph is required')
    error = common = result = None
    try:
        stage.check(full=True); kernel = stage.kernel; completed = state.get('completed')
        if (state.get('format') != FORMAT or state.get('binding') != stage.binding
                or state.get('binding_sha256') != stage.binding_sha256 or type(state.get('rank')) is not int
                or state['rank'] != stage.rank or state.get('world_size') != 4
                or type(completed) is not int or not 0 < completed <= stage.options['steps']
                or completed == stage.options['steps'] and not allow_completed
                or completed != expected_completed or state.get('cursor') != stage.cursor(completed)
                or state.get('history', {}).get('completed') != completed
                or not _digest(state.get('last_global_record_sha256'))
                or state.get('no_resident_trajectory_history') is not True
                or any(state.get(k) is not False for k in ['candidate_promoted', 'goal_achieved', 'useful_own_output_alignment_proved'])):
            raise ValueError('The exact actual rank/world/source/complete PPO cursor is required before mutation')
        kernel.model.validate_export(state['policy'], expected_sources=kernel.sources)
        _validate_adam(state.get('optimizer'), kernel.parameters, completed); _validate_rng(state.get('private_RNG'), kernel.device)
        if (state['optimizer']['param_groups'] != kernel.optimizer.state_dict()['param_groups']
                or not _digest(expected_private_rng_sha256)
                or state.get('private_RNG_sha256') != value_digest(state['private_RNG'])
                or state['private_RNG_sha256'] != expected_private_rng_sha256):
            raise ValueError('Every actual ordinary Adam setting and independently pinned private rank generator must match')
        common = common_digest(state)
        if not _digest(expected_common_sha256) or common != expected_common_sha256 or state.get('common_sha256') != common:
            raise ValueError('Every actual common actor/critic value, Adam moment and complete source/metric chain must match')
        result = prefix(stage, state['history'], state['history_budgets'], state['last_global_record_sha256'], checked)
        if result['original_prompt_draws'] != stage.cursor(completed)['actual_prompt_draws']:
            raise ValueError('Every original prompt including all true empty ranks must reconcile before recovery')
    except Exception as exc: error = type(exc).__name__+': '+str(exc)
    # No peer copies weights or advances a private RNG if ANY actual rank's
    # source, history, moment or independent generator binding is rejected.
    stage.collective.agree(error, common)
    named = selected(kernel.model, kernel.optimizer, ('actor', 'critic'))
    with torch.no_grad():
        for name, p in named: p.copy_(state['policy']['parts'][name])
    kernel.optimizer.load_state_dict(_cpu(state['optimizer'])); kernel.optimizer.zero_grad(set_to_none=True)
    restore_rng(state['private_RNG'], kernel.device); stage.completed = completed
    stage.last_global_record_sha256 = state['last_global_record_sha256']
    stage.previous_prefix = copy.deepcopy(state['history']); stage.history = None
    return result


def finish(stage):
    saved = make_state(stage); result = error = None
    try:
        if stage.completed != stage.options['steps']: raise ValueError('The entire original actual four-rank PPO epoch must finish')
        result = prefix(stage, saved['history'], saved['history_budgets'], saved['last_global_record_sha256'], lambda: None)
        if result['original_prompt_draws'] != len(stage.kernel.data): raise ValueError('Every complete original train-only prompt must appear once')
    except Exception as exc: error = type(exc).__name__+': '+str(exc)
    stage.collective.agree(error, saved['last_global_record_sha256'])
    return dict(complete=True, rank=stage.rank, world_size=4, actual_updates=stage.completed,
        actual_original_train_prompts=len(stage.kernel.data), max_new_tokens=stage.options['max_new_tokens'],
        full_original_four_rank_PPO_recipe_complete=stage.kernel.full,
        actual_unique_policy_reward_values=stage.kernel.model.counts()['actual_unique'], additional_reward_encoder_parameters=0,
        actual_all_rank_action_tokens=result['actual_action_tokens'], all_private_states_and_full_trajectories_verified=True,
        common_sha256=saved['common_sha256'], history=saved['history'], useful_own_output_alignment_proved=False, goal_achieved=False)
