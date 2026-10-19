"""Schedule verified algorithms independently of their source-file split groups."""
from collections import Counter, defaultdict
import random

from .instruction_generalization_trial_v3 import LANGUAGES, order


def schedule(pools, steps, seed):
    if type(steps) is not int or steps < 1:
        raise ValueError('Positive complete update budget required')
    required = [*LANGUAGES, 'verified_python', 'human_instruction']
    if set(pools) != set(required):
        raise ValueError('Every declared language and public instruction pool required')
    orders = {name: order(len(pools[name]), seed + i) for i, name in enumerate(required)}
    programs = defaultdict(dict)
    for index, row in enumerate(pools['verified_python']):
        program = row.get('original_program_sha256')
        variant = row.get('variant_index')
        if not isinstance(program, str) or not program:
            raise ValueError('Original algorithm identity required separately from file split group')
        if type(variant) is not int or variant < 0 or variant in programs[program]:
            raise ValueError('Each algorithm must have distinct nonnegative variant indices')
        programs[program][variant] = index
    if any(0 not in variants for variants in programs.values()):
        raise ValueError('Every scheduled algorithm requires its original-name variant')
    names = sorted(programs)
    random.Random(seed + 71).shuffle(names)
    variants = {name: [index for _, index in sorted(programs[name].items())] for name in names}
    orders['verified_python'] = [variants[name][visit]
        for visit in range(max(map(len, variants.values())))
        for name in names if visit < len(variants[name])]
    positions = Counter()
    updates = []
    for step in range(steps):
        names = [LANGUAGES[(2 * step) % 6], LANGUAGES[(2 * step + 1) % 6],
                 'verified_python', 'human_instruction']
        update = []
        for name in names:
            index = orders[name][positions[name] % len(orders[name])]
            update.append({'pool': name, 'index': index})
            positions[name] += 1
        updates.append(update)
    return updates
