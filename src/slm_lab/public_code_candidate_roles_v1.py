"""Reserve complete connected question/body groups before fresh execution.

This prevents the declared exact-normalized-input and docstring-free AST
duplicates from crossing roles. It does not certify semantic independence or
absence from previous pretraining. Every candidate keeps its original source
position; no examples are removed to improve a result.
"""
import hashlib
import json
import re

SEED = 'public-code-shard-source-roles-v1:2026-10-18'
HEX = re.compile(r'[0-9a-f]{64}')


def assign_roles(candidates):
    records = list(candidates)
    parents = {}
    first = {}

    def find(position):
        while parents[position] != position:
            parents[position] = parents[parents[position]]
            position = parents[position]
        return position

    def union(left, right):
        left, right = find(left), find(right)
        parents[max(left, right)] = min(left, right)

    for record in records:
        position = record['original_shard_position']
        if type(position) is not int or position < 0 or position in parents:
            raise ValueError('Every candidate requires a unique original position')
        parents[position] = position
        for field in ['input_only_group_sha256', 'program_AST_without_docstrings_sha256']:
            value = record[field]
            if not isinstance(value, str) or HEX.fullmatch(value) is None:
                raise ValueError('Both declared complete-input and exact-AST hashes required')
            key = (field, value)
            if key in first:
                union(position, first[key])
            else:
                first[key] = position
    components = {}
    for position in sorted(parents):
        components.setdefault(find(position), []).append(position)
    result = {}
    for members in components.values():
        identity = hashlib.sha256(json.dumps(members, separators=(',', ':')).encode()).hexdigest()
        draw = hashlib.sha256((SEED + ':' + identity).encode()).digest()
        # A whole component has a fixed 1/10 reservation probability. This
        # happens before any fresh replay outcomes are known.
        reserved = int.from_bytes(draw, 'big') % 10 == 0
        for position in members:
            result[position] = dict(
                source_role='reserved' if reserved else 'train',
                declared_duplicate_component_sha256=identity,
                declared_duplicate_component_size=len(members),
                role_fixed_before_fresh_replay=True,
                optimization_forbidden=reserved,
            )
    return result


def verify_role_separation(candidates, roles):
    records = list(candidates)
    if set(roles) != {r['original_shard_position'] for r in records}:
        raise ValueError('No original candidate role may be omitted or added')
    if roles != assign_roles(records):
        raise ValueError('Original deterministic connected-component assignment differs')
    ownership = {}
    for record in records:
        role = roles[record['original_shard_position']]['source_role']
        for field in ['input_only_group_sha256', 'program_AST_without_docstrings_sha256']:
            key = (field, record[field])
            if key in ownership and ownership[key] != role:
                raise ValueError('A declared duplicate crosses training and reserved roles')
            ownership[key] = role
    return dict(
        declared_duplicate_components=len({r['declared_duplicate_component_sha256'] for r in roles.values()}),
        train_candidates=sum(r['source_role'] == 'train' for r in roles.values()),
        reserved_candidates=sum(r['source_role'] == 'reserved' for r in roles.values()),
        largest_component=max((r['declared_duplicate_component_size'] for r in roles.values()), default=0),
        declared_input_and_AST_cross_role_intersections=0,
        semantic_equivalence_and_historical_independence_not_certified=True,
    )
