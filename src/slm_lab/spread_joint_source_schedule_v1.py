"""Space complete source streams across one horizon, preserving endpoints.

This deterministic pacing reduces long modality-free tails without repeating
or dropping a source. It changes global optimizer co-occurrence, so retention
and final quality need measurement; no learning improvement is guaranteed.
"""
from collections import Counter

from .complete_joint_source_schedule_v1 import STREAMS, combine, digest

FORMAT = 'own-complete-joint-source-endpoint-spaced-schedule-v1'


def gaps(entries, name):
    absent = longest = 0; active = []
    for row in entries:
        if any(part['stream'] == name for part in row['components']):
            active.append(row['step']); absent = 0
        else:
            absent += 1; longest = max(longest, absent)
    return dict(first_active_step=active[0], last_active_step=active[-1],
        longest_consecutive_steps_without_task=longest, actual_component_steps=len(active))


def spread(streams):
    original = combine(streams); horizon = original['actual_joint_updates']
    entries = [dict(step=i+1, components=[]) for i in range(horizon)]
    for name in STREAMS:
        rows = streams[name]['entries']; population = len(rows)
        for offset, row in enumerate(rows):
            step = horizon if population == 1 else 1 + offset * (horizon - 1) // (population - 1)
            entries[step-1]['components'].append(dict(stream=name, source_step=offset+1, original_entry_sha256=digest(row)))
    if any(not row['components'] for row in entries): raise ValueError('Every joint step must retain an actual original source')
    used = {name: [] for name in STREAMS}
    for row in entries:
        for part in row['components']: used[part['stream']].append(part['source_step'])
    if any(used[name] != list(range(1, len(streams[name]['entries'])+1)) for name in STREAMS):
        raise ValueError('Every original stream step must remain in its original order exactly once')
    result = {k:v for k,v in original.items() if k not in ['format','entries','task_combinations']}
    result.update(format=FORMAT, entries=entries,
        task_combinations=dict(Counter('+'.join(p['stream'] for p in row['components']) for row in entries)),
        original_prefix_reference_sha256=digest(original),
        original_prefix_gaps={name:gaps(original['entries'],name) for name in STREAMS},
        actual_spaced_gaps={name:gaps(entries,name) for name in STREAMS},
        all_multi_step_streams_include_first_and_final_joint_step=True,
        global_optimizer_task_cooccurrence_changes=True, model_retention_or_quality_gain_unmeasured=True)
    return result
