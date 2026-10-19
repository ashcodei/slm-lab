"""Input-derived curriculum roles, including explicit equivalent-input groups.

These keys cover declared transformations, not arbitrary semantic equivalence
or historical-model source exposure. No published answer enters the key.
"""
import ast
from collections import Counter
import hashlib
import json

from .reasoning_source_oracles_v1 import expected


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)


def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def _signed_terms(node,sign=1):
    if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Add,ast.Sub)):
        return _signed_terms(node.left,sign)+_signed_terms(node.right,-sign if isinstance(node.op,ast.Sub) else sign)
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        return _signed_terms(node.operand,-sign if isinstance(node.op,ast.USub) else sign)
    if isinstance(node,ast.Constant) and type(node.value) is int:return [sign*node.value]
    raise ValueError('Only complete literal signed additive inputs are allowed')


def _factors(node):
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Mult):return _factors(node.left)+_factors(node.right)
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        values=_factors(node.operand)
        return [-values[0],*values[1:]] if isinstance(node.op,ast.USub) else values
    if isinstance(node,ast.Constant) and type(node.value) is int:return [node.value]
    raise ValueError('Only complete literal integer multiplicative inputs are allowed')


def _arithmetic_tree(node):
    if isinstance(node,ast.Constant) and type(node.value) is int:return ['integer',node.value]
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
        return [type(node.op).__name__,_arithmetic_tree(node.operand)]
    if isinstance(node,ast.BinOp) and type(node.op) in (ast.Add,ast.Sub,ast.Mult,ast.LShift,ast.RShift):
        children=[_arithmetic_tree(node.left),_arithmetic_tree(node.right)]
        if isinstance(node.op,(ast.Add,ast.Mult)):children.sort(key=canonical)
        return [type(node.op).__name__,*children]
    raise ValueError('Only the verified bounded arithmetic operations are allowed')


def _grid_orbit(matrix,reflections=True):
    current=matrix;values=[]
    for _ in range(4):
        values.append(current)
        if reflections:values.append([row[::-1] for row in current])
        current=[list(row) for row in zip(*current[::-1])]
    return min(values,key=canonical)


def _pattern(text):
    names={};result=[]
    for letter in text:
        if letter not in names:names[letter]=len(names)
        result.append(names[letter])
    return result


def input_key(entry):
    # This binds the actual question inputs to the metadata and checks bounds;
    # expected deliberately ignores the supplied answer and solution metadata.
    expected(entry)
    meta=entry['metadata'];task=meta['source_dataset']
    if task=='count_bits':value=meta['number']
    elif task in ('gcd','lcm'):value=sorted(meta['numbers'])
    elif task=='chain_sum':value=sorted(_signed_terms(ast.parse(meta['expression'],mode='eval').body))
    elif task=='products':
        factors=_factors(ast.parse(meta['expression'],mode='eval').body)
        value=[bool(sum(v<0 for v in factors)%2) if all(factors) else False,sorted(abs(v) for v in factors)]
    elif task=='bitwise_arithmetic':value=_arithmetic_tree(ast.parse(meta['problem'],mode='eval').body)
    elif task=='base_conversion':value=[meta['source_base'],meta['target_base'],int(meta['source_repr'],meta['source_base'])]
    elif task in ('binary_matrix','rotten_oranges'):value=_grid_orbit(meta['matrix'])
    elif task=='rotate_matrix':value=[meta['num_rotations']%4,_grid_orbit(meta['matrix'],reflections=False)]
    elif task=='ransom_note':
        note,magazine=Counter(meta['ransom_note']),Counter(meta['magazine'])
        value=sorted([note[letter],magazine[letter]] for letter in note.keys()|magazine.keys())
    elif task=='isomorphic_strings':
        patterns=[_pattern(text) for text in meta['words']]
        value=min([patterns,patterns[::-1]],key=canonical)
    else:raise ValueError('Every original task needs an explicit input-only group')
    return [task,value]


def group_id(entry):return digest(input_key(entry))


def role(group):
    if not isinstance(group,str) or len(group)!=64:raise ValueError('A complete fixed SHA256 input group is required')
    return 'validation' if int(hashlib.sha256(('public-reasoning-source-role-v1\0'+group).encode()).hexdigest(),16)%10==0 else 'train'


def assignments(rows,overlap_by_id):
    if len({r['id'] for r in rows})!=len(rows) or set(overlap_by_id)!={r['id'] for r in rows}:
        raise ValueError('Every original draw and complete overlap disposition is required')
    groups={r['id']:group_id(r['original_entry']) for r in rows}
    excluded={groups[key] for key,value in overlap_by_id.items() if value}
    representatives={};result=[]
    for position,row in enumerate(rows):
        group=groups[row['id']];first=representatives.setdefault(group,row['id'])
        split='excluded_protected_text_overlap' if group in excluded else role(group)
        result.append(dict(original_position=position,id=row['id'],group_id=group,split=split,
            representative_id=first,representative=row['id']==first,
            disposition='excluded_whole_input_group' if group in excluded else 'retained_representative' if row['id']==first else 'equivalent_input_duplicate',
            exact_original_overlap_hits=overlap_by_id[row['id']]))
    return result
