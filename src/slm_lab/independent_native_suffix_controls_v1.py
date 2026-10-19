"""Rebuild exact human suffix controls without importing the producer helper.

These checks establish source/control integrity only. They do not run a model,
admit production RL or infer unassisted code quality from public reference hints.
"""
import math

from .batched_native_suffix_group_gradients_v1 import verdict_semantics
from .independent_token_suffix_readback_v1 import rebuild
from .native_code_execution_oracle_v2 import FORMAT, NativeCodeExecutionOracle


def delivery(corpus, oracle, position, native, observed, actions, origin):
    if type(oracle) is not NativeCodeExecutionOracle or oracle.corpus is not corpus:
        raise ValueError('The exact original human corpus and native program oracle are required')
    prefix, horizon = native['original_reference_prefix_ids'], native['metadata']['full_native_action_budget']
    if observed.get('status') == 'preserved_native_control_adapter_exception':
        if (observed.get('reward') is not None or observed.get('requested_source_position') != position
                or observed.get('requested_public_reference_ids') != prefix or observed.get('requested_native_actions') != actions
                or observed.get('requested_action_budget') != horizon or observed.get('requested_origin') != origin
                or not isinstance(observed.get('error'), str)):
            raise ValueError('Every preserved adapter exception must retain its full original attempted request and no invented reward')
        return
    text = corpus.tokenizer.decode(prefix + actions[:-1], skip_special_tokens=False)
    raw = corpus.tokenizer.decode(prefix + actions, skip_special_tokens=False)
    if (observed.get('format') != FORMAT or observed.get('actual_response_origin') != origin
            or observed.get('actual_generated_suffix_ids') != actions or observed.get('actual_supplied_public_reference_ids') != prefix
            or observed.get('complete_actual_response') != text or observed.get('actual_response_with_native_control_tokens_preserved') != raw
            or observed.get('actual_delivered_action_budget') != horizon or observed.get('actual_stopped_on_EOS') is not True
            or observed.get('actual_delivered_EOS') is not True
            or observed.get('terminal_EOS_is_a_declared_boundary_not_executed_Python') is not True
            or observed.get('rendered_response') != oracle.render(position, text)
            or observed.get('human_preferences') is not False or observed.get('unassisted_capability_score') is not False):
        raise ValueError('Every actual native origin/action/prefix/program byte/EOS/full budget must independently reconstruct')
    reward = observed.get('reward')
    if observed['rendered_response']['valid'] is False:
        if (reward != 0 or observed.get('status') != observed['rendered_response']['status']
                or observed.get('actual_original_oracle_observation') is not None):
            raise ValueError('An actually incomplete declared source envelope earns only its observed structural zero reward')
        return
    if reward is not None and (type(reward) not in [int, float] or not math.isfinite(reward) or not 0 <= reward <= 1
            or observed.get('source_reward_informative') is not True or reward != observed.get('original_oracle_reward')):
        raise ValueError('Only the actual bounded informative native reward may be credited')
    result = observed.get('actual_original_oracle_observation')
    worker = None if result is None else result.get('actual_execution', result.get('raw_candidate_execution'))
    if worker is not None:
        actual = worker.get('actual_program_sha256', worker.get('original_program_sha256', worker.get('program_sha256')))
        if actual != observed['rendered_response'].get('exact_execution_program_sha256'):
            raise ValueError('The actual worker must receive the entire declared native program byte exactly')
    if reward is not None and reward > 0 and worker is None:
        raise ValueError('Positive native reward requires actual retained worker program/output observations')


def check(corpus, oracle, position, level, metadata, controls):
    row = corpus.row(position)
    ids, labels = corpus.sequence(position)
    native = rebuild(corpus.tokenizer, row['original'], ids, labels, level, metadata)
    if len(controls) != 2: raise ValueError('Two complete actually observed original native control repeats are required')
    for repeat, observed in enumerate(controls):
        if observed.get('repeat') != repeat: raise ValueError('Every original repeat identity must remain')
        delivery(corpus, oracle, position, native, observed['native_gold'], native['original_missing_suffix_and_EOS_ids'], 'original_reference_control')
        delivery(corpus, oracle, position, native, observed['native_prefix_then_EOS'], [2], 'test_fixture')
        legacy = observed['legacy_prefix_then_EOS'].get('reward')
        if legacy is not None and (type(legacy) not in [int, float] or not math.isfinite(legacy) or not 0 <= legacy <= 1):
            raise ValueError('Every old legacy control reward must remain bounded or explicitly missing')
    qualified = all(c['native_gold'].get('reward') == 1 and c['native_prefix_then_EOS'].get('reward') is not None
        and 0 <= c['native_prefix_then_EOS']['reward'] < 1
        and c['legacy_prefix_then_EOS'].get('reward') is not None and 0 <= c['legacy_prefix_then_EOS']['reward'] < 1 for c in controls)
    return native, qualified


def agrees(controls, fresh_gold, fresh_prefix):
    """Actual returned values/program/tests/verdicts, without timing equality."""
    return all(verdict_semantics(c['native_gold']) == verdict_semantics(fresh_gold)
        and verdict_semantics(c['native_prefix_then_EOS']) == verdict_semantics(fresh_prefix) for c in controls)
