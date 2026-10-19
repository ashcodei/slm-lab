"""Bounded exact accounting of a complete native observation prefix.

Only counters, maxima and an incremental mathematical-history digest remain
resident. This does not replace the original source/trajectory validator or
admit capacity, training, execution correctness or model quality.
"""
import hashlib
import math

from .parallel_function_repair_stage_v1 import value_digest
from .parallel_native_suffix_stage_v1 import CE_ONLY, KERNEL, record_semantics

PROTOCOL = 'own-native-streamed-complete-history-accounting-v1'
FIELDS = [
    'actual_complete_updates', 'actual_source_presentations',
    'actual_execution_eligible_presentations', 'actual_complete_CE_only_presentations',
    'actual_unconditional_original_CE_targets', 'actual_original_context_tokens',
    'actual_single_source_three_empty_rank_steps', 'actual_committed_prefix_trajectory_count',
    'actual_committed_prefix_generated_action_count', 'actual_genuine_EOS_stops',
    'actual_reward1_trajectories', 'actual_reward_varying_groups', 'actual_abstaining_groups',
]


def integer(value, *, positive=False):
    if type(value) is not int or value < int(positive):
        raise ValueError('All original native accounting counts must retain exact integer types and ranges')
    return value


class Summary:
    def __init__(self, *, group_size):
        if type(group_size) is not int or not 2 <= group_size <= 8:
            raise ValueError('The original complete native attempt group is required')
        self.group_size = group_size
        self.counts = dict.fromkeys(FIELDS, 0)
        self.maxima = dict(actual_maximum_cached_full_logp_difference=0.,
            actual_peak_logical_KV_bytes=0, actual_conservative_full_horizon_logical_KV_bytes=0)
        self.digest = hashlib.sha256((PROTOCOL + '\n').encode())

    def append(self, row):
        if type(row) is not dict or type(row.get('step')) is not int or row['step'] != self.counts['actual_complete_updates'] + 1:
            raise ValueError('Every original complete native update must retain its exact prefix position')
        ranks, positions = row.get('actual_rank_observations'), row.get('actual_original_source_positions')
        world = row.get('actual_world_size')
        if (type(world) is not int or not 1 <= world <= 4 or type(ranks) is not list or len(ranks) != world
                or type(positions) is not list or not 1 <= len(positions) <= world
                or any(type(p) is not int or p < 0 for p in positions) or len(set(positions)) != len(positions)
                or type(row.get('global_original_sources')) is not list or len(row['global_original_sources']) != len(positions)):
            raise ValueError('All actual original sources and real empty-rank positions are required')
        delta = dict.fromkeys(FIELDS, 0)
        maxima = dict(self.maxima)
        targets = context = 0
        for rank, observed in enumerate(ranks):
            if (type(observed) is not dict or type(observed.get('rank')) is not int or observed['rank'] != rank
                    or observed.get('actual_original_positions') != positions[rank:rank + 1]):
                raise ValueError('Every actual original rank and source position must remain ordered')
            kernel = observed.get('kernel')
            if rank >= len(positions):
                if kernel is not None: raise ValueError('A genuine empty rank cannot invent a source forward')
                continue
            if type(kernel) is not dict or kernel.get('format') not in [CE_ONLY, KERNEL]:
                raise ValueError('Every original native or CE-only source observation must remain')
            source = row['global_original_sources'][rank]
            count = integer(kernel['unconditional_original_CE_anchor']['local_original_response_EOS_targets'], positive=True)
            if count != integer(source['original_response_EOS_targets'], positive=True):
                raise ValueError('No source may drop its unconditional original response/EOS anchor')
            targets += count
            context += integer(source['original_context_tokens'], positive=True)
            delta['actual_source_presentations'] += 1
            if kernel['format'] == CE_ONLY:
                delta['actual_complete_CE_only_presentations'] += 1
                continue
            delta['actual_execution_eligible_presentations'] += 1
            trajectories = kernel['trajectories']
            rewards = kernel['actual_rewards']
            sampler = kernel['complete_batched_sampling_observation']
            if (type(trajectories) is not list or len(trajectories) != self.group_size
                    or type(rewards) is not list or len(rewards) != self.group_size
                    or any(v is not None and (type(v) not in [float, int] or not math.isfinite(v) or not 0 <= v <= 1) for v in rewards)
                    or type(sampler['trajectories']) is not list or len(sampler['trajectories']) != self.group_size):
                raise ValueError('All actual native attempts, abstentions and finite execution rewards must remain')
            actions = 0
            for actual in trajectories:
                emitted = actual['actual_generated_token_ids']
                if (type(emitted) is not list or not emitted or any(type(token) is not int for token in emitted)
                        or type(actual['actual_stopped_on_EOS']) is not bool
                        or actual['actual_stopped_on_EOS'] != (emitted[-1] == 2)):
                    raise ValueError('Every original complete action sequence and genuine EOS disposition is required')
                actions += len(emitted)
            if actions != integer(kernel['actual_sampled_action_tokens'], positive=True):
                raise ValueError('Every original actual generated action must be counted')
            abstaining = kernel['policy_gradient_abstained_due_to_missing_verdict']
            if type(abstaining) is not bool or abstaining != any(v is None for v in rewards):
                raise ValueError('The original missing-verdict disposition cannot be relabelled')
            delta['actual_committed_prefix_trajectory_count'] += len(trajectories)
            delta['actual_committed_prefix_generated_action_count'] += actions
            delta['actual_genuine_EOS_stops'] += sum(t['actual_stopped_on_EOS'] for t in trajectories)
            delta['actual_reward1_trajectories'] += sum(v == 1. for v in rewards)
            delta['actual_reward_varying_groups'] += int(len({v for v in rewards if v is not None}) > 1)
            delta['actual_abstaining_groups'] += int(abstaining)
            difference = sampler['cached_full_max_absolute_difference']
            if type(difference) not in [float, int] or not math.isfinite(difference) or difference < 0:
                raise ValueError('The complete observed cached/full probability difference must remain finite')
            maxima['actual_maximum_cached_full_logp_difference'] = max(maxima['actual_maximum_cached_full_logp_difference'], difference)
            for output, source_key in [('actual_peak_logical_KV_bytes', 'actual_peak_logical_KVcache_bytes'),
                    ('actual_conservative_full_horizon_logical_KV_bytes', 'conservative_full_horizon_logical_KVcache_bytes')]:
                maxima[output] = max(maxima[output], integer(sampler[source_key], positive=True))
        if (targets != integer(row['original_response_EOS_targets'], positive=True)
                or context != integer(row['complete_original_context_tokens'], positive=True)):
            raise ValueError('Every complete original context token and response/EOS target must reconcile')
        delta['actual_complete_updates'] = 1
        delta['actual_unconditional_original_CE_targets'] = targets
        delta['actual_original_context_tokens'] = context
        delta['actual_single_source_three_empty_rank_steps'] = int(world == 4 and len(positions) == 1)
        # All validation precedes mutation: a rejected row leaves the exact
        # earlier prefix counters/digest usable and unchanged.
        semantic = bytes.fromhex(value_digest(record_semantics(row)))
        self.digest.update(semantic)
        for key, value in delta.items(): self.counts[key] += value
        self.maxima = maxima

    def result(self):
        return dict(protocol=PROTOCOL, **self.counts, **self.maxima,
            actual_mathematical_history_sha256=self.digest.hexdigest(),
            only_scalar_counters_maxima_and_incremental_digest_retained=True,
            production_training_admitted=False, new_quality_gain=False)


def summarize(rows, *, group_size):
    summary = Summary(group_size=group_size)
    for row in rows: summary.append(row)
    return summary.result()
