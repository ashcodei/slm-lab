"""Checked argument-name transformations of admitted public training programs.

The source descriptions, examples and algorithms are retained. These are synthetic
static transformations, not new human solutions or human preference labels.
"""
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import keyword
from pathlib import Path

from .code_quality import execute, instruction_code
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .instruction_generalization_trial_v3 import complete_encoding, TargetDoesNotFit


class UnsupportedBinding(ValueError):
    """The program falls outside this deliberately limited static transform."""


def _function(tree, name):
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != name:
        raise UnsupportedBinding('requires one top-level function')
    fn = functions[0]
    if any(not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))
           for node in tree.body):
        raise UnsupportedBinding('module statements beyond imports and one function')
    if fn.decorator_list or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs:
        raise UnsupportedBinding('decorated or variadic/keyword-only interface')
    if not fn.args.posonlyargs + fn.args.args:
        raise UnsupportedBinding('no positional parameters to transform')
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                          ast.Lambda, ast.Global, ast.Nonlocal, ast.Import,
                          ast.ImportFrom, ast.ExceptHandler, ast.Match))
           for stmt in fn.body for n in ast.walk(stmt)):
        raise UnsupportedBinding('unsupported local scope or binding form')
    if fn.name in {a.arg for a in fn.args.posonlyargs + fn.args.args}:
        raise UnsupportedBinding('parameter shadows the entry point')
    forbidden = {'eval', 'exec', 'locals', 'globals', 'vars', 'dir', 'getattr',
                 'setattr', 'delattr', 'compile', '__import__', 'inspect'}
    if any(isinstance(n, ast.Name) and (n.id in forbidden or n.id.startswith('__'))
           for n in ast.walk(tree)):
        raise UnsupportedBinding('reflective name access')
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            parent = parents[node]
            if not isinstance(parent, ast.Call) or parent.func is not node:
                raise UnsupportedBinding('entry point is rebound or used as a value')
            if any(kw.arg is None for kw in parent.keywords):
                raise UnsupportedBinding('dynamic recursive keyword binding')
    return fn


def _rewrite(tree, name, mapping):
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

    class Body(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = mapping.get(node.id, node.id)
            return node

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == name:
                for kw in node.keywords:
                    kw.arg = mapping.get(kw.arg, kw.arg)
            return self.generic_visit(node)

    for arg in fn.args.posonlyargs + fn.args.args:
        arg.arg = mapping.get(arg.arg, arg.arg)
    # Defaults and annotations are evaluated outside the function's local scope.
    fn.body = [Body().visit(stmt) for stmt in fn.body]
    return ast.fix_missing_locations(tree)


def argument_names(row, style):
    tree = ast.parse(instruction_code(row['response']))
    fn = _function(tree, row['function'])
    args = fn.args.posonlyargs + fn.args.args
    if style not in {'positional', 'example_type'}:
        raise ValueError('Unknown argument-name style')
    cases = [ast.literal_eval(case['args_literal']) for case in row['verification_cases']]
    if not cases:
        raise ValueError('Published positional examples are required')
    names = []
    for index, _ in enumerate(args):
        values = [case[index] for case in cases if len(case) > index]
        kind = 'value'
        if values and all(type(v) in {int, float} for v in values): kind = 'number'
        elif values and all(type(v) is str for v in values): kind = 'text'
        elif values and all(type(v) in {list, tuple} for v in values): kind = 'sequence'
        elif values and all(type(v) is dict for v in values): kind = 'mapping'
        names.append(('input' if style == 'positional' else kind) + '_' + str(index))
    return {arg.arg: new for arg, new in zip(args, names)}


def transform(row, mapping):
    original = ast.parse(instruction_code(row['response']))
    fn = _function(original, row['function'])
    args = fn.args.posonlyargs + fn.args.args
    existing = {n.id for n in ast.walk(original) if isinstance(n, ast.Name)}
    existing.update(n.arg for n in ast.walk(original) if isinstance(n, ast.arg))
    existing.add(fn.name)
    if (set(mapping) != {a.arg for a in args} or len(set(mapping.values())) != len(mapping)
            or any(not isinstance(v, str) or not v.isidentifier() or keyword.iskeyword(v) or v.startswith('__')
                   or v in existing for v in mapping.values())):
        raise UnsupportedBinding('missing, invalid, colliding or unchanged argument names')
    changed = _rewrite(deepcopy(original), fn.name, mapping)
    inverse = _rewrite(deepcopy(changed), fn.name, {v: k for k, v in mapping.items()})
    if ast.dump(inverse) != ast.dump(original):
        raise ValueError('Argument transformation does not invert to the original AST')
    signature = ast.unparse(fn).splitlines()[0]
    if row['prompt'].splitlines().count(signature) != 1:
        raise UnsupportedBinding('original complete signature is not unique in prompt')
    replacement = ast.unparse(next(n for n in changed.body if isinstance(n, ast.FunctionDef))).splitlines()[0]
    binding = 'Argument names in the original description map to this signature as follows: '
    binding += ', '.join(k + ' -> ' + v for k, v in mapping.items()) + '.'
    # Keep the original description, published calls and source/license metadata.
    result = deepcopy(row)
    result['prompt'] = row['prompt'].replace(signature, replacement + '\n\n' + binding, 1)
    result['response'] = '```python\n' + ast.unparse(changed) + '\n```'
    result['parent_training_id'] = row['id']
    result['parent_variant_index'] = result.pop('variant_index')
    result['parent_response_sha256'] = hashlib.sha256(row['response'].encode()).hexdigest()
    result['argument_mapping'] = dict(mapping)
    result['binding_variant'] = True
    result['id'] = 'binding-' + hashlib.sha256((row['id'] + json.dumps(mapping, sort_keys=True)).encode()).hexdigest()
    result['annotation_origin'] = ('Synthetic static argument renaming of an admitted public human program; '
        'original description/examples retained, inverse AST checked, published examples reexecuted. '
        'Not human preference feedback or a new human-authored solution.')
    return result


def check(row):
    return execute({'language': 'Python', 'code': instruction_code(row['response']),
                    'function': row['function'], 'cases': row['verification_cases'], 'stress': []})


def prepare(source, output, encoder, workers=4):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Use a fresh output; preserve earlier evidence')
    manifest = source.parent / 'manifest.json'
    manifest_digest = file_hash(manifest)
    metadata = json.loads(manifest.read_text())
    digest = file_hash(source)
    if digest != metadata['files_sha256'][source.name]:
        raise ValueError('Admitted original instruction source differs')
    source_rows = list(jsonl(source))
    rows = [r for r in source_rows if r['variant_index'] == 0]
    if len({r['original_program_sha256'] for r in rows}) != len(rows):
        raise ValueError('Expected one admitted original-name row per algorithm')
    last_variant = {}
    for row in source_rows:
        program = row['original_program_sha256']
        last_variant[program] = max(last_variant.get(program, -1), row['variant_index'])
    proposals, excluded = [], []
    for row in rows:
        for offset, style in enumerate(['positional', 'example_type'], start=1):
            try:
                variant = transform(row, argument_names(row, style))
                variant['variant_index'] = last_variant[row['original_program_sha256']] + offset
                variant['binding_style'] = style
                counts = complete_encoding(variant, encoder)
                proposals.append((row, variant, counts, style))
            except (UnsupportedBinding, TargetDoesNotFit) as error:
                excluded.append({'id': row['id'], 'style': style, 'reason': str(error)})
    # Recheck each original and each new program through the existing constrained worker.
    originals = {row['id']: row for row, _, _, _ in proposals}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verified_originals = dict(zip(originals, pool.map(check, originals.values())))
        outcomes = list(pool.map(check, (variant for _, variant, _, _ in proposals)))
    admitted, evidence = [], []
    for (original, variant, counts, style), outcome in zip(proposals, outcomes):
        gold = verified_originals[original['id']]
        if not gold.get('functional_passed') or not outcome.get('functional_passed'):
            excluded.append({'id': original['id'], 'style': style, 'reason': 'original or transformed public examples failed',
                             'original': gold, 'transformed': outcome})
            continue
        admitted.append(variant)
        evidence.append({'id': variant['id'], 'parent_training_id': original['id'], 'style': style,
                         'original_program_sha256': original['original_program_sha256'],
                         'group_id': original['group_id'], 'argument_mapping': variant['argument_mapping'],
                         'original': gold, 'transformed': outcome, **counts})
    if not admitted:
        raise ValueError('No transformed programs passed complete source example checks')
    if file_hash(source) != digest or file_hash(manifest) != manifest_digest:
        raise ValueError('Original source changed during preparation')
    output.mkdir(parents=True)
    for name, items in [('train.jsonl', admitted), ('verification.jsonl', evidence), ('excluded.jsonl', excluded)]:
        with (output / name).open('w') as stream:
            for row in items: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    receipt = {'source': str(source.resolve()), 'source_sha256': digest,
        'source_manifest_sha256': manifest_digest,
        'original_algorithms_available': len(rows), 'original_algorithms_reexecuted': len(originals),
        'transformed_training_rows': len(admitted),
        'distinct_original_algorithms': len({r['original_program_sha256'] for r in admitted}),
        'styles': dict(Counter(r['style'] for r in evidence)),
        'complete_response_targets': sum(r['target_tokens'] for r in evidence),
        'maximum_complete_context': max(r['context_tokens'] for r in evidence),
        'all_admitted_targets_complete': True,
        'files_sha256': {p.name: file_hash(p) for p in output.glob('*.jsonl')},
        'source_groups_and_original_algorithm_identities_retained': True,
        'trained': False, 'automatically_admitted_to_live_training': False, 'goal_achieved': False,
        'limitations': ['Published training examples only; no held-out or general correctness/efficiency proof.',
                       'Argument names change; algorithm, original description and published examples are retained.',
                       'Restricted static transform excludes complex scopes/interfaces/reflection.',
                       'Source semantic contamination is not exhaustively audited; this reuses an existing admitted training pool.',
                       'Data preparation does not establish learning or generalization gains.']}
    atomic_json(output / 'manifest.json', receipt)
    return receipt
