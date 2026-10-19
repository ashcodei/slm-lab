"""Identical positive/rehearsal exposure for own-code optimization controls."""
from collections import Counter
import hashlib
import json
import random

from .instruction_generalization_trial_v3 import LANGUAGES, complete_encoding


def sequence_digest(encoder, row, field):
    ids, labels = encoder.sequence(row, field)
    return hashlib.sha256(json.dumps([ids, labels], separators=(',', ':')).encode()).hexdigest()


def schedule(pairs, pools, passes=8, seed=88124):
    if type(passes) is not int or passes < 1 or not pairs:
        raise ValueError('Positive passes and complete nonempty execution pairs required')
    if set(pools) != set([*LANGUAGES, 'human_instruction']) or any(not rows for rows in pools.values()):
        raise ValueError('All six languages and original human rehearsal must remain present')
    if (len({row['original_program_sha256'] for row in pairs}) != len(pairs)
            or len({row['id'] for row in pairs}) != len(pairs)):
        raise ValueError('Repeated source algorithm or ID in execution pairs')
    orders = {name: random.Random(seed + i).sample(range(len(rows)), len(rows))
              for i, (name, rows) in enumerate(pools.items())}
    positions = Counter(); allocation = []
    for epoch in range(passes):
        order = random.Random(seed + 100 + epoch).sample(range(len(pairs)), len(pairs))
        for pair in order:
            language = LANGUAGES[len(allocation) % len(LANGUAGES)]
            module = orders[language][positions[language] % len(orders[language])]
            human = orders['human_instruction'][positions['human_instruction'] % len(orders['human_instruction'])]
            allocation.append({'pair_index': pair, 'module_pool': language,
                               'module_index': module, 'human_index': human})
            positions[language] += 1; positions['human_instruction'] += 1
    return allocation


def validate_targets(pairs, pools, allocation, encoder):
    """Reconstruct full masks and token totals for each actually selected source."""
    records = {}; totals = Counter(); exposures = []
    for update, pick in enumerate(allocation, 1):
        pair = pairs[pick['pair_index']]
        selected = [('chosen', pair, 'chosen'), ('rejected', pair, 'rejected'),
                    ('module', pools[pick['module_pool']][pick['module_index']], 'response'),
                    ('human', pools['human_instruction'][pick['human_index']], 'response')]
        entry = {'update': update, **pick, 'pair_id': pair['id'], 'source_targets': {}}
        for pool, row, field in selected:
            key = (pool, row['id'])
            if key not in records:
                counts = complete_encoding({**row, 'response': row[field]}, encoder)
                if pool in ['chosen', 'rejected']:
                    expected = row[field + '_target_tokens']
                    if counts['target_tokens'] != expected or row['feedback_source'] != 'synthetic':
                        raise ValueError('Actual execution response targets or feedback origin differ')
                records[key] = {'pool': pool, 'id': row['id'], **counts,
                                'sequence_sha256': sequence_digest(encoder, row, field)}
            count = records[key]['target_tokens']; totals[pool] += count
            entry['source_targets'][pool] = count
            entry[pool + '_id'] = row['id']
        exposures.append(entry)
    if encoder.truncated:
        raise ValueError('Optimization input or actual mask was clipped')
    return {'planned_target_exposures': dict(totals), 'source_records': list(records.values()),
            'maximum_complete_context': max(row['context_tokens'] for row in records.values()),
            'exposures': exposures, 'all_complete_masks_verified': True}
