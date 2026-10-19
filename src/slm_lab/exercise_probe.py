"""Validate a frozen multi-language module corpus and account for SFT exposures."""
from collections import Counter
import math
import random


def validate_splits(train, valid):
    if not train or not valid: raise ValueError('Both exercise splits must be nonempty')
    for split, rows in [('train', train), ('valid', valid)]:
        if len({row['id'] for row in rows}) != len(rows): raise ValueError('Repeated exercise implementation ID')
        if any(row['split'] != split or row.get('image') or not row['prompt'] or not row['response'] for row in rows):
            raise ValueError('Expected complete text-only exercise modules in their declared split')
    if {row['group_id'] for row in train} & {row['group_id'] for row in valid}:
        raise ValueError('Cross-language exercise group leaks between training and validation')
    if {row['id'] for row in train} & {row['id'] for row in valid}:
        raise ValueError('Exercise implementation leaks between splits')


def exposure_plan(rows, target_counts, passes=4, batch=2, accumulation=4, seed=42):
    if not rows or len(rows) != len(target_counts) or min(passes, batch, accumulation) < 1:
        raise ValueError('Invalid module-training exposure budget')
    if any(count < 1 for count in target_counts): raise ValueError('Every complete module needs supervised targets')
    width = batch * accumulation
    steps = math.ceil(len(rows) * passes / width)
    order = list(range(len(rows))); random.Random(seed).shuffle(order)
    records = [{'position': position, 'id': rows[order[position % len(order)]]['id'],
                'group_id': rows[order[position % len(order)]]['group_id'],
                'language': rows[order[position % len(order)]]['language'],
                'target_tokens': target_counts[order[position % len(order)]]}
               for position in range(steps * width)]
    counts = Counter(record['id'] for record in records)
    return {'steps': steps, 'seed': seed, 'batch_size': batch, 'gradient_accumulation': accumulation,
            'world_size': 1, 'minimum_passes': passes, 'examples': len(rows),
            'sample_exposures': len(records), 'remainder_exposures': len(records) - passes * len(rows),
            'targets_per_complete_pass': sum(target_counts),
            'planned_target_exposures': sum(record['target_tokens'] for record in records),
            'exposures_by_language': dict(Counter(record['language'] for record in records)),
            'minimum_implementation_exposures': min(counts.values()),
            'maximum_implementation_exposures': max(counts.values())}, records
