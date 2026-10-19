"""Select intact joint batches from every original source and source extreme.

This is a capacity workload, not a capacity result. The actual full paced
epoch, individual source lengths, original visual stress cases, corrections,
and genuine empty ranks are checked before choosing whole joint entries.
No source is shortened or reassigned to a different rank.
"""
from collections import Counter
import copy

from .complete_joint_source_schedule_v1 import digest

FORMAT = 'own-complete-joint-original-source-capacity-workload-v1'


def select(summaries, *, required_steps):
    """Deterministic metadata selection; this helper grants no admission."""
    if (type(summaries) is not list or not summaries or type(required_steps) is not dict
            or any(type(s) is not int or not 1 <= s <= len(summaries) for s in required_steps)):
        raise ValueError('Complete ordered source summaries and original required steps are needed')
    choices = {}

    def add(step, reason):
        reasons = choices.setdefault(step, [])
        if reason not in reasons:
            reasons.append(reason)

    names = set()
    combinations = {}
    for step, row in enumerate(summaries, 1):
        if (set(row) != {'step', 'original_entry_sha256', 'components', 'limits', 'empty_rank_tasks', 'corrective_pairs'}
                or type(row['step']) is not int or row['step'] != step
                or type(row['original_entry_sha256']) is not str or len(row['original_entry_sha256']) != 64
                or type(row['limits']) is not dict or not row['limits']
                or any(type(k) is not str or not k or type(v) is not int or v < 0 for k, v in row['limits'].items())
                or type(row['components']) is not list or not row['components']
                or len(set(row['components'])) != len(row['components'])
                or any(n not in ['code', 'visual', 'human'] for n in row['components'])
                or type(row['empty_rank_tasks']) is not list or type(row['corrective_pairs']) is not int or row['corrective_pairs'] < 0):
            raise ValueError('Every complete actual original source summary must remain intact')
        names.update(row['limits'])
        combinations.setdefault(tuple(row['components']), step)
        if row['empty_rank_tasks']:
            add(step, 'actual_original_empty_rank_tasks')
        if row['corrective_pairs']:
            add(step, 'every_original_corrective_pair_batch')
    for name in sorted(names):
        candidates = [r for r in summaries if name in r['limits']]
        chosen = max(candidates, key=lambda r: (r['limits'][name], -r['step']))
        add(chosen['step'], 'maximum_' + name)
    for combination, step in sorted(combinations.items()):
        add(step, 'original_task_combination_' + '+'.join(combination))
    for step, reasons in sorted(required_steps.items()):
        if type(reasons) is not list or not reasons or any(type(r) is not str or not r for r in reasons):
            raise ValueError('Every original required capacity reason must remain explicit')
        for reason in reasons:
            add(step, reason)
    add(1, 'original_first_joint_step')
    add(len(summaries), 'original_final_joint_step')
    return [dict(capacity_step=i, original_joint_step=step, reasons=choices[step],
        source_summary=copy.deepcopy(summaries[step - 1])) for i, step in enumerate(sorted(choices), 1)]


def choose(source, original_visual_capacity):
    """Read the actual admitted sources, including all individual lengths."""
    from .complete_joint_training_data_v1 import CompleteJointTrainingDataV1
    from .complete_joint_objective_bridge_v1 import checkpoint_entry
    from .joint_visual_capacity_cases_v1 import digest as visual_digest
    if (type(source) is not CompleteJointTrainingDataV1 or len(source) != 5362
            or original_visual_capacity.get('actual_original_epoch_entries') != 4733
            or original_visual_capacity.get('actual_original_epoch_source_views') != 302627
            or original_visual_capacity.get('original_whole_epoch_entries_sha256') != visual_digest(source.visual_entries)
            or original_visual_capacity.get('actual_full_size_GPU_capacity_passed') is not False):
        raise ValueError('All original joint sources and the unchanged complete visual workload are required')
    source.guard.check(full=True)
    if any(digest(source.streams[n]) != source.joint['component_streams_sha256'][n] for n in ['code', 'visual', 'human']):
        raise ValueError('All complete original source streams must retain their original identities')
    adapter = source.visual_adapter
    entries, summaries, visual_to_joint = [], [], {}
    code_seen = Counter(); human_seen = Counter(); visual_seen = Counter()
    for index in range(len(source)):
        metadata = source.metadata(index); parts = metadata['components']
        entry = checkpoint_entry(metadata, source_identity_sha256=source.identity_sha256)
        limits = {}; empty = []; corrective = 0
        for stream, part in parts.items():
            row = part['entry']; positions = row['original_positions']
            if any(n == 0 for n in row['rank_source_counts']):
                empty.append(stream)
            limits[stream + ':largest_original_global_batch'] = len(positions)
            limits[stream + ':complete_batch_contexts'] = row['complete_context_tokens']
            limits[stream + ':complete_batch_response_EOS'] = row['response_EOS_targets']
            if stream == 'code':
                info = [source.code.information(p) for p in positions]
                code_seen.update((row['epoch'], p) for p in positions)
                for family in sorted({r['family'] for r in info}):
                    group = [r for r in info if r['family'] == family]
                    limits['code:' + family + ':single_context'] = max(r['complete_context_tokens'] for r in group)
                    limits['code:' + family + ':single_response_EOS'] = max(r['response_EOS_targets'] for r in group)
                lengths = [r['complete_context_tokens'] for r in info]
                targets = [r['response_EOS_targets'] for r in info]
            elif stream == 'human':
                human_seen.update(positions)
                originals = [source.human.rows[p] for p in positions]
                lengths = [sum(r['native'][f]['length'] for f in ['chosen', 'rejected']) for r in originals]
                targets = [sum(r['native'][f]['length'] - r['native'][f]['target_start'] for f in ['chosen', 'rejected']) for r in originals]
                for field in ['chosen', 'rejected']:
                    limits['human:' + field + ':single_context'] = max(r['native'][field]['length'] for r in originals)
                    limits['human:' + field + ':single_response_EOS'] = max(r['native'][field]['length'] - r['native'][field]['target_start'] for r in originals)
                    limits['human:' + field + ':largest_rank_contexts'] = max(sum(r['native'][field]['length'] for r in originals[rank::4]) for rank in range(4))
            else:
                family = row['family']; original_step = row['source_step']
                if original_step in visual_to_joint:
                    raise ValueError('Every original visual batch must occur exactly once')
                visual_to_joint[original_step] = index + 1
                visual_seen.update((family, p) for p in positions)
                lengths = [int(adapter.lengths[family][p]) for p in positions]
                targets = [int(adapter.targets[family][p]) for p in positions]
                limits['visual:' + family + ':single_context'] = max(lengths)
                limits['visual:' + family + ':single_response_EOS'] = max(targets)
                limits['visual:' + family + ':largest_original_global_batch'] = len(positions)
                corrective = len(row['complete_original_correction_indices'])
                limits['visual:complete_corrective_negative_EOS'] = row['complete_original_negative_response_EOS_targets']
                limits['visual:whole_word_UTF8_bytes'] = row['whole_word_UTF8_bytes']
                for number in row['complete_original_correction_indices']:
                    negative = adapter.corrections[number]['complete_original_negative_native']
                    limits['visual:single_corrective_negative_context'] = max(limits.get('visual:single_corrective_negative_context', 0), len(negative['input_ids']))
            if sum(lengths) != row['complete_context_tokens'] or sum(targets) != row['response_EOS_targets']:
                raise ValueError('Every individual actual source length and response/EOS target must reconstruct its complete batch')
            limits[stream + ':largest_rank_contexts'] = max(sum(lengths[rank::4]) for rank in range(4))
            limits[stream + ':largest_rank_response_EOS'] = max(sum(targets[rank::4]) for rank in range(4))
        entries.append(entry)
        summaries.append(dict(step=index + 1, original_entry_sha256=metadata['source_entry_sha256'],
            components=list(parts), limits=limits, empty_rank_tasks=empty, corrective_pairs=corrective))
    expected_visual = Counter((name, p) for name, positions in adapter.positions.items() for p in range(len(positions)))
    if (code_seen != Counter((epoch, p) for epoch in range(3) for p in range(28028))
            or human_seen != Counter(range(85786)) or visual_seen != expected_visual or sum(visual_seen.values()) != 302627):
        raise ValueError('All84084code/85786human/302627visual occurrences must remain without extra repeats')
    required = {}
    cases = original_visual_capacity['capacity_cases']
    if len(cases) != 10:
        raise ValueError('All ten original visual source-extreme capacity cases must remain')
    for case in cases:
        original_step = case['original_complete_epoch_step']
        if visual_digest(source.visual_entries[original_step - 1]) != case['original_complete_entry_sha256']:
            raise ValueError('Every original full visual capacity entry must retain its source bytes')
        required.setdefault(visual_to_joint[original_step], []).extend('original_visual:' + r for r in case['reasons'])
    selected = select(summaries, required_steps=required)
    for case in selected:
        original = entries[case['original_joint_step'] - 1]
        case['original_checkpoint_entry_sha256'] = digest(original)
        case['checkpoint_entry'] = dict(copy.deepcopy(original), step=case['capacity_step'])
    source.guard.check(full=True)
    return dict(format=FORMAT, source_identity_sha256=source.identity_sha256, original_paced_schedule_sha256=digest(source.joint['entries']),
        complete_checkpoint_entries_sha256=digest(entries), actual_original_joint_steps=5362,
        original_stream_budgets=copy.deepcopy(source.joint['original_stream_budgets']),
        all_original_source_summaries_sha256=digest(summaries), original_visual_capacity_plan_sha256=digest(original_visual_capacity),
        all_original_visual_capacity_cases=10, actual_complete_capacity_cases=len(selected), cases=selected,
        original_task_combinations=copy.deepcopy(source.joint['task_combinations']),
        original_full_shape_limits={name: max(r['limits'].get(name, 0) for r in summaries) for name in sorted({k for r in summaries for k in r['limits']})},
        all_original_batches_rank_assignments_and_targets_unchanged=True, only_capacity_cursor_renumbered=True,
        actual_full_size_GPU_capacity_passed=False, production_training_admitted=False, neural_forwards=0, optimizer_updates=0, goal_achieved=False)
