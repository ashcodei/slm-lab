"""Reconstruct actual sampled code and the complete original training prefix.

These are source-train functions, including controls never optimized by this
experiment. Neither group is an independent held benchmark.
"""
import hashlib
import json
import math
import struct

from .own_function_trajectory_preferences_v1 import validate


def effect(observation):
    fields = ['status', 'syntax_valid', 'all_original_examples_observed',
              'all_original_examples_passed', 'original_public_example_count',
              'passed_examples', 'entry_body_calls']
    if 'raw_results' in observation:
        fields += ['raw_results', 'example_passes']
    return {key: observation[key] for key in fields}


def reconstruct(pairs, events, deliveries, decode, extract):
    if (len(pairs) != len(events) or len(pairs) != len(deliveries)
            or len({p['id'] for p in pairs}) != len(pairs)):
        raise ValueError('Every distinct original function and actual delivery required')
    results = []
    for position, (pair, event, delivery) in enumerate(zip(pairs, events, deliveries)):
        validate(pair)
        prompt = pair['original_native_prompt_ids']
        actions = event.get('actual_generated_token_ids', [])
        budget = pair['rejected_observation_budget']
        if (event.get('case_position') != position or event.get('original_input_token_ids') != prompt
                or event.get('complete_input_ids_sha256') != hashlib.sha256(struct.pack('<'+'H'*len(prompt), *prompt)).hexdigest()
                or event.get('requested_native_action_budget') != budget
                or not actions or any(type(t) is not int or not 0 <= t < 32768 for t in actions)
                or event.get('actual_native_action_count') != len(actions) or len(actions) > budget
                or type(event.get('actual_stopped_on_EOS')) is not bool
                or event['actual_stopped_on_EOS'] != (actions[-1] == 2) or 2 in actions[:-1]
                or not event['actual_stopped_on_EOS'] and len(actions) != budget
                or event.get('greedy_native_output_tensor_returned_unchanged') is not True
                or event.get('selected_action_log_probabilities_not_retained') is not True
                or delivery.get('original_source_id') != pair['id']
                or delivery.get('optimization_role') != pair['optimization_role']
                or delivery.get('actual_generated_token_ids') != actions):
            raise ValueError('Exact source role, prompt, bounded actual actions and stopping boundary required')
        decoded = decode(actions); program = extract(decoded)
        observed = delivery['actual_public_example_observations']
        if (delivery.get('raw_decoded_response') != decoded or delivery.get('actual_generated_program') != program
                or observed.get('actual_generated_program_sha256') != hashlib.sha256(program.encode()).hexdigest()
                or observed.get('original_source_id') != pair['id']
                or observed.get('held_benchmark') is not False
                or observed.get('generated_program_not_repaired') is not True):
            raise ValueError('Actual decoded code and original execution observations must reconstruct')
        results.append(dict(source_id=pair['id'], optimization_role=pair['optimization_role'], program=program,
                            actual_generated_tokens=len(actions), actual_stopped_on_EOS=event['actual_stopped_on_EOS'],
                            producer_effect=effect(observed)))
    return results


def validate_training(records, order, pairs, method, options):
    optimizing = {i for i, p in enumerate(pairs) if p['optimization_role'] == 'optimization_source_train'}
    if len(records) != len(order) or len(records) != options['steps']:
        raise ValueError('Complete declared source schedule required, without favorable prefixes')
    for i, (record, positions) in enumerate(zip(records, order)):
        if (record.get('step') != i+1 or record.get('method') != method
                or record.get('original_source_positions') != positions or set(positions)-optimizing
                or len(record.get('original_trajectory_pairs', [])) != len(positions)
                or record.get('learning_rate') != options['learning_rate']
                or record.get('beta') != options['beta'] or record.get('chosen_CE_weight') != options['chosen_CE_weight']
                or record.get('own_starting_actor_reference_queried') != (method == 'trajectory_DPO')
                or record.get('unoptimized_train_controls_not_used') is not True
                or record.get('failed_program_actions_never_positive_CE_targets') is not True
                or record.get('no_unobserved_rejected_EOS_or_other_actions') is not True
                or record.get('human_preferences') is not False):
            raise ValueError('Every original optimization source, method and fixed objective must reconstruct')
        chosen = rejected = 0
        for position, observed in zip(positions, record['original_trajectory_pairs']):
            pair = pairs[position]; validate(pair)
            expected = dict(original_train_trajectory_position=position, id=pair['id'],
                original_source_group=pair['original_source_group'],
                native_pair_sha256=hashlib.sha256(json.dumps(pair,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                original_prompt_tokens=len(pair['original_native_prompt_ids']),
                chosen_response_EOS_targets=len(pair['chosen_original_complete_response_EOS_ids']),
                rejected_actual_action_targets=len(pair['rejected_actual_sampled_action_ids']),
                rejected_is_budget_censored=pair['rejected_is_budget_censored'],
                rejected_actual_stopped_on_EOS=pair['rejected_actual_stopped_on_EOS'])
            if any(observed.get(k) != v for k, v in expected.items()):
                raise ValueError('Every original complete/censored source mask fingerprint required')
            chosen += expected['chosen_response_EOS_targets']; rejected += expected['rejected_actual_action_targets']
        if (record.get('chosen_response_EOS_targets') != chosen or record.get('rejected_actual_action_targets') != rejected
                or record.get('normalization_chosen_targets') != chosen or record.get('normalization_pairs') != len(positions)
                or any(not isinstance(record.get(k), (int,float)) or not math.isfinite(record[k]) for k in
                       ['chosen_token_mean_NLL','pair_mean_preference','summed_objective','gradient_norm_before_clip'])):
            raise ValueError('Every real target denominator and finite recorded objective required')


def summarize(rows):
    result = {}
    for role in ['optimization_source_train', 'unoptimized_source_train_control']:
        members = [row for row in rows if row['optimization_role'] == role]
        result[role] = dict(functions=len(members), syntax_valid=sum(row['producer_effect']['syntax_valid'] for row in members),
            all_original_public_examples_passed=sum(row['producer_effect']['all_original_examples_passed'] for row in members))
    result.update(actual_EOS_stops=sum(row['actual_stopped_on_EOS'] for row in rows),
                  actual_budget_exhaustions=sum(not row['actual_stopped_on_EOS'] for row in rows))
    return result
