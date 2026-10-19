"""Choose actual whole schedule entries for future visual capacity checks.

This is source selection, never a capacity result or training admission. All
original full-epoch occurrences are checked before choosing worst-shape cases.
Capacity rows preserve every selected occurrence/target and only renumber the
local capacity cursor; original schedule identities remain explicit outside it.
"""
from collections import Counter
import copy
import hashlib
import json

import numpy as np

from .joint_visual_objective_update_v1 import FAMILIES
from .multimodal_source_index_format_v1 import DTYPE


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def choose(family_names, index, entries, word_identities):
    if (not isinstance(family_names, list) or set(family_names) != set(FAMILIES)
            or len(family_names) != len(FAMILIES) or index.dtype != DTYPE or index.ndim != 1
            or not isinstance(entries, list) or not entries or not isinstance(word_identities, list)):
        raise ValueError('Exact original nine-family source schema and complete word identities are required')
    local = {}
    counts = {}
    for number, name in enumerate(family_names):
        rows = index[index['family'] == number]
        if len(rows) < 1 or not np.array_equal(rows['original_local_position'], np.arange(len(rows))):
            raise ValueError('Every original family occurrence must retain its unique ordered local position')
        local[name] = rows
        counts[name] = len(rows)
    if sum(counts.values()) != len(index):
        raise ValueError('No unknown source family may be omitted')
    words = local['words']
    if len(word_identities) != len(words):
        raise ValueError('Every original whole-word identity is required')
    for position, (identity, row) in enumerate(zip(word_identities, words)):
        if (identity['original_local_position'] != position or identity['source_role'] != 'train'
                or type(identity['whole_word_UTF8_bytes']) is not int or not 1 <= identity['whole_word_UTF8_bytes'] <= 64
                or not identity['whole_word_UTF8_bytes'] <= identity['minimum_repeated_byte_CTC_steps'] <= 140
                or identity['complete_context_tokens'] != int(row['context_tokens'])
                or identity['response_EOS_targets'] != int(row['response_EOS_targets'])
                or not isinstance(identity['whole_word_UTF8_sha256'], str)
                or len(identity['whole_word_UTF8_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in identity['whole_word_UTF8_sha256'])):
            raise ValueError('Every whole-word byte/native/CTC identity must agree with the complete index')
    seen = {name: Counter() for name in FAMILIES}
    memberships = {name: {} for name in FAMILIES}
    summaries = []
    for number, entry in enumerate(entries, 1):
        name, positions = entry['family'], entry['original_local_positions']
        if (entry['step'] != number or name not in local or not isinstance(positions, list)
                or not 1 <= len(positions) <= 64 or len(set(positions)) != len(positions)
                or any(type(p) is not int or not 0 <= p < counts[name] for p in positions)
                or entry['proposed_rank_source_counts'] != [len(positions[r::4]) for r in range(4)]):
            raise ValueError('Every complete original schedule entry and actual four-rank assignment is required')
        rows = local[name][positions]
        if (any(int(row['context_tokens']) < 3 or not 0 < int(row['response_EOS_targets']) < int(row['context_tokens']) for row in rows)
                or entry['complete_context_tokens'] != int(rows['context_tokens'].sum())
                or entry['response_EOS_targets'] != int(rows['response_EOS_targets'].sum())):
            raise ValueError('Every complete schedule context and response/EOS target budget must reconstruct')
        seen[name].update(positions)
        for p in positions:
            if p in memberships[name]:
                raise ValueError('No original source occurrence may be repeated in the whole epoch')
            memberships[name][p] = number - 1
        summaries.append(dict(maximum_native_context=int(rows['context_tokens'].max()),
            complete_native_context_tokens=int(rows['context_tokens'].sum()),
            source_count=len(positions), family=name))
    if any(seen[name] != Counter(range(count)) for name, count in counts.items()):
        raise ValueError('Every original whole-epoch source occurrence must be retained once')
    choices = {}

    def select(number, reason):
        choices.setdefault(number, []).append(reason)

    # Include the longest complete native context for every family, using the
    # larger complete batch to break ties. No artificially shortened examples.
    for name in family_names:
        candidates = [n for n, summary in enumerate(summaries) if summary['family'] == name]
        number = max(candidates, key=lambda n: (summaries[n]['maximum_native_context'],
            summaries[n]['source_count'], summaries[n]['complete_native_context_tokens'], -n))
        select(number, 'maximum_complete_native_context_for_' + name)
    photos = [n for n, summary in enumerate(summaries) if summary['family'] == 'images:additional_photo_captions']
    photo = max(photos, key=lambda n: (summaries[n]['source_count'], summaries[n]['maximum_native_context'], -n))
    select(photo, 'largest_original_global_photo_negative_set')
    longest_word = max(range(len(words)), key=lambda p: (word_identities[p]['whole_word_UTF8_bytes'],
        word_identities[p]['minimum_repeated_byte_CTC_steps'], -p))
    select(memberships['words'][longest_word], 'longest_complete_original_UTF8_word')
    real_words = [p for p, identity in enumerate(word_identities) if identity['byte_family'] == 'real_word']
    if not real_words:
        raise ValueError('An actual original real-word source is required for cold-readout capacity')
    real_word = max(real_words, key=lambda p: (word_identities[p]['whole_word_UTF8_bytes'], -p))
    select(memberships['words'][real_word], 'longest_original_real_word_for_cold_readout')
    tails = [n for n, summary in enumerate(summaries) if summary['source_count'] < 4]
    if not tails:
        raise ValueError('The actual empty-rank tail must be included')
    for number in tails:
        select(number, 'actual_original_partial_tail_with_empty_ranks')
    # Recovery must cross all three real objective/group-clock regimes before
    # proceeding to the remaining capacity cases. Keep the true tail last.
    priority = {'native': 0, 'photo': 1, 'word': 2}
    ordered = sorted(choices, key=lambda n: (n in tails, priority[FAMILIES[entries[n]['family']]], n))
    first = []
    for kind in ['native', 'photo', 'word']:
        first.append(next(n for n in ordered if n not in tails and FAMILIES[entries[n]['family']] == kind))
    ordered = first + [n for n in ordered if n not in first]
    selected, cases = [], []
    for step, number in enumerate(ordered, 1):
        entry = entries[number]
        selected.append(dict(copy.deepcopy(entry), step=step))
        cases.append(dict(capacity_step=step, original_complete_epoch_step=entry['step'],
            original_complete_entry_sha256=digest(entry), reasons=choices[number], **summaries[number]))
    return dict(capacity_entries=selected, capacity_cases=cases,
        actual_original_epoch_entries=len(entries), actual_original_family_counts=counts,
        actual_original_epoch_source_views=sum(counts.values()),
        original_whole_epoch_entries_sha256=digest(entries),
        maximum_original_native_context=max(s['maximum_native_context'] for s in summaries),
        maximum_original_photo_negative_set=summaries[photo]['source_count'],
        maximum_original_word_UTF8_bytes=word_identities[longest_word]['whole_word_UTF8_bytes'],
        longest_original_word_local_position=longest_word, real_word_cold_readout_local_position=real_word,
        actual_original_tail_steps=[entries[n]['step'] for n in tails],
        selected_original_occurrences_and_full_native_targets_unchanged=True,
        only_capacity_cursor_steps_renumbered=True, training_admitted=False,
        actual_full_size_GPU_capacity_passed=False, neural_forwards=0)
