"""Four-rank PPO frames using the existing funded immutable frame codec.

Only the reference's actual world size changes. Preserve the prior complete
record/receipt format, failure allocation, constant-size tips and immutable
resume branches. All four ledgers explicitly bind the real four-rank policy.
"""
import copy
import json

from .indexed_native_history_v1 import canonical, limits, RECEIPT_LIMIT, sha
from .joint_indexed_history_v2 import Writer, Reader, read_regular
from .native_shared_ppo_history_v1 import (History as OriginalHistory, reference as original_reference,
    prefix_shape, failure_allowance, remaining)


def reference(binding, previous):
    if binding.get('world_size') != 4 or binding.get('physical_rank') not in range(4):
        raise ValueError('The actual four-rank PPO source and private ledger owner are required')
    result = original_reference(binding, previous); result['world_size'] = 4
    return result


class History(OriginalHistory):
    def __init__(self, directory, *, binding, budgets, previous=None, guard=lambda: None):
        self.binding, self.budgets = copy.deepcopy(binding), limits(budgets)
        if previous is not None: prefix_shape(previous)
        self.previous = copy.deepcopy(previous); self.starting = 0 if previous is None else previous['completed']
        if self.budgets['maximum_records'] != binding['planned_updates']:
            raise ValueError('The entire original four-rank PPO horizon must match its history capacity')
        self.writer = Writer(directory, reference_binding=reference(binding, self.previous),
            starting_completed=self.starting, budgets=remaining(self.budgets, self.previous), guard=guard)


def read_prefix(prefix, *, binding, budgets, validate, guard=lambda: None):
    budgets = limits(budgets); prefix_shape(prefix)
    wanted = prefix['completed']
    if wanted > binding['planned_updates'] or budgets['maximum_records'] != binding['planned_updates']:
        raise ValueError('Only a complete prefix of the original four-rank PPO horizon is admissible')
    result = dict(complete_updates=0, original_prompt_draws=0, actual_action_tokens=0,
        complete_prompt_tokens=0, segments=0, full_trajectory_readback=True)
    tip = copy.deepcopy(prefix)
    while tip is not None:
        prefix_shape(tip)
        if tip['completed'] != wanted: raise ValueError('Every recovered branch must join the exact completed update')
        reader = Reader(tip['directory'], expected_header_sha256=tip['header_sha256'],
            maximum_compact_record_bytes=budgets['maximum_compact_record_bytes'],
            maximum_expanded_record_bytes=budgets['maximum_expanded_record_bytes'], guard=guard)
        previous = reader.header['reference_binding'].get('model_metadata', {}).get('previous_prefix')
        if previous is not None: prefix_shape(previous)
        start = 0 if previous is None else previous['completed']
        if (start >= wanted or reader.header['starting_completed'] != start
                or reader.header['reference_binding'] != reference(binding, previous)
                or reader.budgets != remaining(budgets, previous) or tip['segment_records'] != wanted-start):
            raise ValueError('Every original actual four-rank PPO source/model/reward prefix must remain complete')
        expected = tip['last_receipt_sha256']; logical = len(reader.codec.original_bytes)
        physical = reader.header_physical_bytes+failure_allowance(reader.budgets)
        for sequence in range(tip['segment_records'], 0, -1):
            receipt = json.loads(read_regular(reader.root/f'record-{sequence:012d}.receipt.json', RECEIPT_LIMIT))
            if (sha(canonical(receipt)) != expected or receipt.get('sequence') != sequence
                    or receipt.get('step') != start+sequence):
                raise ValueError('Every exact four-rank trajectory receipt must match its independently retained prefix')
            row = reader.codec.expand(reader.compact(receipt)); validate(row)
            result['complete_updates'] += 1; result['original_prompt_draws'] += len(row['source_positions'])
            result['actual_action_tokens'] += row['actual_action_tokens']; result['complete_prompt_tokens'] += row['actual_complete_prompt_tokens']
            expected = receipt['previous_receipt_sha256']; logical += receipt['original_expanded_bytes']
            physical += receipt['compressed_bytes']+len(canonical(receipt))
        if (expected != tip['header_sha256'] or logical != tip['segment_logical_bytes']
                or physical != tip['segment_charged_physical_bytes']
                or tip['all_logical_bytes'] != logical+(0 if previous is None else previous['all_logical_bytes'])
                or tip['all_charged_physical_bytes'] != physical+(0 if previous is None else previous['all_charged_physical_bytes'])
                or tip['all_logical_bytes'] > budgets['maximum_logical_bytes']
                or tip['all_charged_physical_bytes'] > budgets['maximum_physical_bytes']):
            raise ValueError('Every complete original four-rank history byte, receipt chain and quota must reconcile')
        result['segments'] += 1; wanted, tip = start, previous
    if wanted != 0 or result['complete_updates'] != prefix['completed']:
        raise ValueError('The entire immutable four-rank PPO history must close at its exact original start')
    guard(); return result
