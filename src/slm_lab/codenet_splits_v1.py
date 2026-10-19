"""Fixed problem-component roles and exact complete-example deduplication.

These are internal source partitions, not a claim of independent benchmark
cleanliness. In particular, the publisher's Python clone coverage is partial.
"""
from collections import Counter
import hashlib
import re


SALT = 'slm-lab:codenet-original:problem-components:v1'
ROLES = ('train', 'validation', 'test')


def component_role(component, salt=SALT):
    if not re.fullmatch(r'p\d{5}', component):
        raise ValueError('Invalid original problem component')
    value = int.from_bytes(hashlib.sha256((salt + '\0' + component).encode()).digest()[:8], 'big')
    return 'train' if value * 100 < 90 * 2**64 else ('validation' if value * 100 < 95 * 2**64 else 'test')


def partition(records, problem_to_group, salt=SALT):
    """Retain every candidate role; mark one representative per full token pair.

    Identical token ids with identical response-mask start must share a component,
    even when a random hash would coincidentally put both components in one role.
    Changing the partition to conceal a missed duplicate is forbidden.
    """
    if not problem_to_group or any(problem_to_group.get(g) != g for g in problem_to_group.values()):
        raise ValueError('Complete canonical problem-component map required')
    output, seen, copies = [], set(), {}
    for source in records:
        row = dict(source)
        member, problem = row['archive_member'], row['problem_id']
        if member in seen:
            raise ValueError('Repeated original source member')
        seen.add(member)
        expected_prefix = 'Project_CodeNet/data/' + problem + '/' + row['language'] + '/'
        if not member.startswith(expected_prefix) or '/' in member[len(expected_prefix):]:
            raise ValueError('Original member/problem/language identity differs')
        group = problem_to_group[problem]
        if row['proposed_problem_group'] != group or row.get('split') is not None:
            raise ValueError('Unassigned context audit and exact proposed grouping required')
        row.update(group_id=group, split=component_role(group, salt),
                   dataset_admitted_to_training=False, complete_example_representative=None)
        if row['context_qualification'] == 'complete_original_pair_fits':
            digest, targets, length = row['complete_input_ids_sha256'], row['target_tokens'], row['context_tokens']
            if not re.fullmatch(r'[0-9a-f]{64}', digest) or not 1 <= targets < length <= 4096:
                raise ValueError('Invalid complete-token/mask identity')
            key = digest, targets
            if key in copies and copies[key][0] != group:
                raise ValueError('Identical complete token/mask pair crosses problem components')
            copies[key] = (group, min(member, copies.get(key, (None, member))[1]))
        output.append(row)
    for row in output:
        if row['context_qualification'] == 'complete_original_pair_fits':
            row['complete_example_representative'] = copies[(row['complete_input_ids_sha256'], row['target_tokens'])][1]
            row['duplicate_complete_example'] = row['archive_member'] != row['complete_example_representative']
        else:
            row['duplicate_complete_example'] = None
    output.sort(key=lambda row: row['archive_member'])
    counts = Counter((row['split'], row['context_qualification']) for row in output)
    roles = {}
    for role in ROLES:
        fit = [row for row in output if row['split'] == role and row['context_qualification'] == 'complete_original_pair_fits']
        unique = [row for row in fit if not row['duplicate_complete_example']]
        roles[role] = {
            'original_problems': sum(component_role(g, salt) == role for g in problem_to_group.values()),
            'problem_components': sum(component_role(g, salt) == role for g in set(problem_to_group.values())),
            'source_candidates': sum(value for (r, _), value in counts.items() if r == role),
            'complete_fit_candidates': len(fit), 'unique_complete_examples': len(unique),
            'duplicate_complete_examples': len(fit) - len(unique),
            'unique_context_tokens': sum(row['context_tokens'] for row in unique),
            'unique_target_tokens': sum(row['target_tokens'] for row in unique),
        }
    return output, roles
