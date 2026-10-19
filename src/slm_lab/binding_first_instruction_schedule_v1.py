"""Visit originals, argument-name variants, then function-name variants.

Only scheduling ranks are projected. Returned indices address the unchanged
source rows, with their real variant/program/file identities intact.
"""
from collections import defaultdict

from .instruction_program_schedule_v4 import schedule as original_schedule


def schedule(pools, steps, seed):
    # Validate actual original identities and duplicate variant indices first.
    original_schedule(pools, 1, seed)
    rows = pools['verified_python']
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        flag = row.get('binding_variant', False)
        if type(flag) is not bool or (flag and (row['variant_index'] == 0 or row.get('parent_variant_index') != 0)):
            raise ValueError('Binding variants require an original parent and a distinct real variant index')
        groups[row['original_program_sha256']].append(index)
    projected = [dict(row) for row in rows]
    for indices in groups.values():
        def priority(index):
            row = rows[index]
            return (0 if row['variant_index'] == 0 else 1 if row.get('binding_variant') else 2,
                    row['variant_index'])
        for rank, index in enumerate(sorted(indices, key=priority)):
            projected[index]['variant_index'] = rank
    return original_schedule({**pools, 'verified_python': projected}, steps, seed)
