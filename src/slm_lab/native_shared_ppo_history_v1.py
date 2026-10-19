"""Complete PPO trajectories on allocated immutable frames, constant-size tips.

Reuse the existing whole-record gzip/receipt and independent failure storage.
A resumed writer starts a fresh directory and points to its immutable earlier
prefix. Readback walks backward one frame at a time; old future records and
failed attempts are never truncated, replaced or silently treated as updates.
"""
import copy
import json

from .indexed_native_history_v1 import canonical, limits, RECEIPT_LIMIT, sha
from .joint_indexed_history_v2 import Writer, Reader, read_regular

FORMAT = 'own-native-shared-PPO-constant-resident-history-prefix-v1'
BINDING = 'own-native-shared-PPO-full-trajectory-binding-v1'
PREFIX_KEYS = {'format', 'directory', 'header_sha256', 'segment_records', 'completed',
    'last_receipt_sha256', 'segment_logical_bytes', 'segment_charged_physical_bytes',
    'all_logical_bytes', 'all_charged_physical_bytes'}


def failure_allowance(budgets):
    return 2*budgets['maximum_expanded_record_bytes'] + 2*RECEIPT_LIMIT + 64*1024


def remaining(budgets, previous):
    result = limits(budgets)
    if previous is not None:
        result['maximum_logical_bytes'] -= previous['all_logical_bytes']
        result['maximum_physical_bytes'] -= previous['all_charged_physical_bytes']
    return limits(result)


def reference(binding, previous):
    if (type(binding) is not dict or binding.get('format') != BINDING
            or type(binding.get('planned_updates')) is not int or binding['planned_updates'] < 1):
        raise ValueError('The exact shared-native PPO source/model/reward/sampler binding is required')
    return dict(sources=copy.deepcopy(binding['sources']),
        model_metadata=dict(PPO_binding=copy.deepcopy(binding), previous_prefix=copy.deepcopy(previous)),
        own_text_reference=copy.deepcopy(binding['frozen_reference']), options=copy.deepcopy(binding['options']),
        world_size=1, optimizer_settings_sha256=binding['optimizer_settings_sha256'])


def prefix_shape(prefix):
    if (type(prefix) is not dict or set(prefix) != PREFIX_KEYS or prefix.get('format') != FORMAT
            or not isinstance(prefix.get('directory'), str)
            or any(type(prefix.get(k)) is not int or prefix[k] < 1 for k in [
                'segment_records', 'completed', 'segment_logical_bytes', 'segment_charged_physical_bytes',
                'all_logical_bytes', 'all_charged_physical_bytes'])
            or any(not isinstance(prefix.get(k), str) or len(prefix[k]) != 64
                or any(c not in '0123456789abcdef' for c in prefix[k])
                for k in ['header_sha256', 'last_receipt_sha256'])):
        raise ValueError('One exact nonempty complete trajectory prefix and constant-size tip are required')


class History:
    def __init__(self, directory, *, binding, budgets, previous=None, guard=lambda: None):
        self.binding, self.budgets = copy.deepcopy(binding), limits(budgets)
        if previous is not None: prefix_shape(previous)
        self.previous = copy.deepcopy(previous)
        self.starting = 0 if previous is None else previous['completed']
        if self.budgets['maximum_records'] != binding['planned_updates']:
            raise ValueError('The complete original PPO horizon must match its history capacity')
        # The stage validates the complete earlier prefix before this call.
        # Deduct every earlier frame/header/failure allowance, not just this
        # branch's bytes, from the full experiment quotas.
        self.writer = Writer(directory, reference_binding=reference(binding, self.previous),
            starting_completed=self.starting, budgets=remaining(self.budgets, self.previous), guard=guard)

    def reserve_next(self): return self.writer.reserve_next()
    def append(self, row): return self.writer.append(row)
    def acknowledge(self, step): return self.writer.acknowledge_success(step)
    def failure(self, observation): return self.writer.preserve_failure(observation)

    def prefix(self):
        writer = self.writer
        if writer.failed or writer.ticket is not None or writer.pending_model_commit:
            raise ValueError('An interrupted or unacknowledged attempt cannot certify a policy checkpoint')
        if not writer.records: return copy.deepcopy(self.previous)
        physical = writer.physical_bytes + failure_allowance(writer.budgets)
        logical = writer.logical_bytes
        prefix = dict(format=FORMAT, directory=str(writer.root), header_sha256=writer.header_sha256,
            segment_records=writer.records, completed=self.starting+writer.records,
            last_receipt_sha256=writer.last_receipt_sha256,
            segment_logical_bytes=logical, segment_charged_physical_bytes=physical,
            all_logical_bytes=logical+(0 if self.previous is None else self.previous['all_logical_bytes']),
            all_charged_physical_bytes=physical+(0 if self.previous is None else self.previous['all_charged_physical_bytes']))
        if (prefix['all_logical_bytes'] > self.budgets['maximum_logical_bytes']
                or prefix['all_charged_physical_bytes'] > self.budgets['maximum_physical_bytes']):
            raise ValueError('The entire complete trajectory lineage must fit the funded original quotas')
        return prefix


def read_prefix(prefix, *, binding, budgets, validate, guard=lambda: None):
    """Verify every whole frame/source/action/GAE in constant resident space.

    Validation is newest-to-oldest. Receipt hashes chain in that direction,
    and strictly decreasing completed steps prevent cyclic predecessor trees.
    A checkpoint tip deliberately ignores later records in the same directory.
    """
    budgets = limits(budgets); prefix_shape(prefix)
    wanted = prefix['completed']
    if wanted > binding['planned_updates'] or budgets['maximum_records'] != binding['planned_updates']:
        raise ValueError('Only a complete prefix of the original full PPO horizon is admissible')
    result = dict(complete_updates=0, original_prompt_draws=0, actual_action_tokens=0,
        complete_prompt_tokens=0, segments=0, full_trajectory_readback=True)
    tip = copy.deepcopy(prefix)
    while tip is not None:
        prefix_shape(tip)
        if tip['completed'] != wanted:
            raise ValueError('Every recovered branch must join the exact earlier completed update')
        reader = Reader(tip['directory'], expected_header_sha256=tip['header_sha256'],
            maximum_compact_record_bytes=budgets['maximum_compact_record_bytes'],
            maximum_expanded_record_bytes=budgets['maximum_expanded_record_bytes'], guard=guard)
        previous = reader.header['reference_binding'].get('model_metadata', {}).get('previous_prefix')
        if previous is not None: prefix_shape(previous)
        start = 0 if previous is None else previous['completed']
        if (start >= wanted or reader.header['starting_completed'] != start
                or reader.header['reference_binding'] != reference(binding, previous)
                or reader.budgets != remaining(budgets, previous)
                or tip['segment_records'] != wanted-start):
            raise ValueError('Every earlier whole PPO source/model/reward prefix must match without missing updates')
        expected = tip['last_receipt_sha256']
        logical = len(reader.codec.original_bytes)
        physical = reader.header_physical_bytes + failure_allowance(reader.budgets)
        for sequence in range(tip['segment_records'], 0, -1):
            receipt = json.loads(read_regular(reader.root / f'record-{sequence:012d}.receipt.json', RECEIPT_LIMIT))
            if sha(canonical(receipt)) != expected:
                raise ValueError('Every exact trajectory receipt must match its independently retained prefix tip')
            if receipt.get('sequence') != sequence or receipt.get('step') != start+sequence:
                raise ValueError('No completed PPO update may be omitted, duplicated or reordered')
            row = reader.codec.expand(reader.compact(receipt))
            validate(row)
            result['complete_updates'] += 1
            result['original_prompt_draws'] += len(row['source_positions'])
            result['actual_action_tokens'] += row['actual_action_tokens']
            result['complete_prompt_tokens'] += row['actual_complete_prompt_tokens']
            expected = receipt['previous_receipt_sha256']
            logical += receipt['original_expanded_bytes']
            physical += receipt['compressed_bytes'] + len(canonical(receipt))
        if (expected != tip['header_sha256'] or logical != tip['segment_logical_bytes']
                or physical != tip['segment_charged_physical_bytes']
                or tip['all_logical_bytes'] != logical+(0 if previous is None else previous['all_logical_bytes'])
                or tip['all_charged_physical_bytes'] != physical+(0 if previous is None else previous['all_charged_physical_bytes'])
                or tip['all_logical_bytes'] > budgets['maximum_logical_bytes']
                or tip['all_charged_physical_bytes'] > budgets['maximum_physical_bytes']):
            raise ValueError('Every complete original history byte, receipt chain and aggregate quota must reconcile')
        result['segments'] += 1
        wanted, tip = start, previous
    if wanted != 0 or result['complete_updates'] != prefix['completed']:
        raise ValueError('The whole immutable PPO history must close at its exact original start')
    guard()
    return result
