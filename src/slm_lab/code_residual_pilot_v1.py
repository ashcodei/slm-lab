"""Deterministic complete-programme learning-rate pilot selection and schedule."""
from collections import Counter, defaultdict
import math
import random


PROFILES = (
    dict(name='constant_3e-5', peak_learning_rate=3e-5, warmup_updates=0),
    dict(name='warmup16_3e-5', peak_learning_rate=3e-5, warmup_updates=16),
    dict(name='warmup16_1e-5', peak_learning_rate=1e-5, warmup_updates=16),
)


def select_complete_rows(rows, *, number, per_language, seed, role):
    if (role not in {'train', 'validation'} or type(number) is not int or number < 1
            or type(per_language) is not int or per_language < 1 or type(seed) is not int):
        raise ValueError('Explicit train/validation role and positive whole-row sampling budget required')
    if (not rows or len(rows) < number or any(r['role'] != role for r in rows)
            or len({r['global_position'] for r in rows}) != len(rows)
            or len({r['id'] for r in rows}) != len(rows)
            or any(not r['group_id'] or r['context_tokens'] > 4096 or r['target_tokens'] < 1
                   or r['target_tokens'] >= r['context_tokens'] for r in rows)):
        raise ValueError('Unique complete original rows from the one sealed role required')
    rows = sorted(rows, key=lambda r:r['global_position'])
    counts = Counter(r['language'] for r in rows)
    if sum(min(per_language,n) for n in counts.values()) + 2 > number:
        raise ValueError('Declared budget cannot guarantee every language plus both size extremes')
    rng = random.Random(seed); shuffled = rows.copy(); rng.shuffle(shuffled)
    buckets = defaultdict(list)
    for row in shuffled:buckets[row['language']].append(row)
    chosen = []; ids = set(); groups = set(); selected_counts = Counter()
    def add(row):
        if row['global_position'] in ids:return
        chosen.append(row); ids.add(row['global_position']); groups.add(row['group_id'])
        selected_counts[row['language']] += 1
    # Include genuine whole-source context/response maxima, without clipping.
    add(max(rows,key=lambda r:r['context_tokens'])); add(max(rows,key=lambda r:r['target_tokens']))
    for language in sorted(buckets):
        while selected_counts[language] < min(per_language,counts[language]):
            available = [r for r in buckets[language] if r['global_position'] not in ids]
            add(next((r for r in available if r['group_id'] not in groups),available[0]))
    for prefer_new_group in [True, False]:
        for row in shuffled:
            if len(chosen) == number:break
            if not prefer_new_group or row['group_id'] not in groups:add(row)
    if len(chosen) != number:raise ValueError('Every declared whole-row exposure must exist')
    rng.shuffle(chosen)
    return chosen


def learning_rate(profile, step):
    if (set(profile) != {'name','peak_learning_rate','warmup_updates'} or type(step) is not int or step < 0
            or type(profile['warmup_updates']) is not int or profile['warmup_updates'] < 0
            or not math.isfinite(profile['peak_learning_rate']) or profile['peak_learning_rate'] <= 0):
        raise ValueError('Explicit finite positive learning rate and nonnegative update/warmup required')
    return profile['peak_learning_rate'] * min(1.,(step+1)/max(1,profile['warmup_updates']))
