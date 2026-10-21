"""Single-token operator interventions and complete reference-target weights.

Only supported operators in a function body are considered. All other source
bytes, signatures and docstrings stay unchanged. Test-insensitive mutations
are not certified equivalent; these candidate weights require separate runtime
evidence and admission. Original target tokens and genuine EOS are retained.
"""
import ast
from functools import lru_cache
import hashlib
import io
import json
import math
import token
import tokenize

OPERATORS = {ast.Add: ('+', '-'), ast.Sub: ('-', '+'), ast.Mult: ('*', '//'),
    ast.Div: ('/', '*'), ast.FloorDiv: ('//', '%'), ast.Mod: ('%', '//'),
    ast.Eq: ('==', '!='), ast.NotEq: ('!=', '=='), ast.Lt: ('<', '<='), ast.LtE: ('<=', '<'),
    ast.Gt: ('>', '>='), ast.GtE: ('>=', '>'), ast.And: ('and', 'or'), ast.Or: ('or', 'and'),
    ast.UAdd: ('+', '-'), ast.USub: ('-', '+')}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _docs(tree):
    return [(type(n).__name__, getattr(n, 'name', None), ast.get_docstring(n, clean=False))
            for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]


@lru_cache(maxsize=256)
def _inventory(program):
    if not isinstance(program, str) or not 0 < len(program.encode()) <= 65536:
        raise ValueError('A whole bounded original Python function is required')
    tree = ast.parse(program)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('One complete original function entrypoint is required')
    function = tree.body[0]; lines = program.splitlines(keepends=True); starts = []; offset = 0
    for line in lines:
        starts.append(offset); offset += len(line)
    def position(row, column, utf8=False):
        if row > len(lines):
            return len(program)
        return starts[row-1] + (len(lines[row-1].encode()[:column].decode()) if utf8 else column)
    def start(node):return position(node.lineno, node.col_offset, True)
    def end(node):return position(node.end_lineno, node.end_col_offset, True)
    lexical = [(t.string, position(*t.start), position(*t.end))
               for t in tokenize.generate_tokens(io.StringIO(program).readline) if t.type in (token.OP, token.NAME)]
    requests = []
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.BinOp):
                requests.append((node.op, end(node.left), start(node.right)))
            elif isinstance(node, ast.Compare):
                previous = node.left
                for op, following in zip(node.ops, node.comparators):
                    requests.append((op, end(previous), start(following))); previous = following
            elif isinstance(node, ast.BoolOp):
                requests.extend((node.op, end(a), start(b)) for a, b in zip(node.values, node.values[1:]))
            elif isinstance(node, ast.UnaryOp):
                requests.append((node.op, start(node), start(node.operand)))
    rows = []; qualified = []; seen = set()
    for op, begin, finish in requests:
        if type(op) not in OPERATORS:
            continue
        original, replacement = OPERATORS[type(op)]
        matches = [(a, b) for text, a, b in lexical if text == original and begin <= a < b <= finish]
        if len(matches) != 1:
            qualified.append(dict(operator=type(op).__name__, search_char_start=begin, search_char_end=finish,
                                  disposition='supported_AST_operator_without_one_lexical_token'))
            continue
        a, b = matches[0]
        if (a, b) in seen:
            continue
        seen.add((a, b)); changed = program[:a]+replacement+program[b:]; rebuilt = ast.parse(changed)
        if (rebuilt.body[0].name != function.name or ast.dump(rebuilt.body[0].args) != ast.dump(function.args)
                or _docs(rebuilt) != _docs(tree)):
            raise ValueError('Every original signature and docstring must remain unchanged')
        item = dict(original_program_sha256=sha(program), operator=type(op).__name__,
            program_char_start=a, program_char_end=b, program_UTF8_byte_start=len(program[:a].encode()),
            program_UTF8_byte_end=len(program[:b].encode()), original_operator=original, replacement_operator=replacement,
            derivative_program_sha256=sha(changed), only_one_original_operator_token_changed=True)
        item['mutation_id'] = sha(json.dumps(item, sort_keys=True, separators=(',', ':')))
        rows.append(item)
    rows.sort(key=lambda r:(r['program_char_start'], r['program_char_end']))
    return tuple(json.dumps(r, sort_keys=True) for r in rows), tuple(json.dumps(q, sort_keys=True) for q in qualified)


def catalogue(program):
    rows, qualified = _inventory(program)
    return dict(mutations=[json.loads(r) for r in rows], lexical_qualifications=[json.loads(q) for q in qualified],
                entire_declared_operator_population_enumerated=True)


def derivative(program, mutation):
    rows, _ = _inventory(program)
    if not any(json.loads(row) == mutation for row in rows):
        raise ValueError('An exact source-bound supported single-operator intervention is required')
    a, b = mutation['program_char_start'], mutation['program_char_end']
    changed = program[:a]+mutation['replacement_operator']+program[b:]
    if sha(changed) != mutation['derivative_program_sha256']:
        raise ValueError('The complete derivative bytes must reconstruct')
    return changed


def candidate_target_weights(response, program, mutations, selected_ids, tokenizer, *, operator_weight=4.):
    if (not math.isfinite(operator_weight) or not 1 <= operator_weight <= 16
            or response.count(program) != 1 or len(selected_ids) != len(set(selected_ids))):
        raise ValueError('One exact complete source target and distinct bounded operator interventions required')
    by_id = {m['mutation_id']: m for m in mutations}
    if len(by_id) != len(mutations) or any(key not in by_id for key in selected_ids):
        raise ValueError('Every selected intervention must belong to the full original source inventory')
    mask = bytearray(len(response)); shift = response.index(program)
    for key in selected_ids:
        m = by_id[key]; derivative(program, m)
        a, b = shift+m['program_char_start'], shift+m['program_char_end']
        mask[a:b] = b'\1'*(b-a)
    encoded = tokenizer.encode(response, add_special_tokens=False)
    if (tokenizer.decode(encoded.ids, skip_special_tokens=False) != response
            or len(encoded.ids) != len(encoded.offsets) or not encoded.ids
            or any(not 4 <= i < 32768 for i in encoded.ids)):
        raise ValueError('Every complete original target byte and actual tokenizer offset must round-trip')
    weights = []
    for a, b in encoded.offsets:
        if not 0 <= a <= b <= len(response):
            raise ValueError('Offsets must refer to original Unicode character positions')
        weights.append(1+(operator_weight-1)*(sum(mask[a:b])/(b-a) if b > a else 0))
    weights.append(1.)
    return dict(response_EOS_ids=[*encoded.ids, 2], positive_target_weights=weights,
        response_EOS_targets=len(weights), weighted_denominator=math.fsum(weights), operator_weight=operator_weight,
        selected_mutation_ids=list(selected_ids), all_original_targets_retained=True, true_EOS_weight=1.,
        candidate_weights_only_runtime_readback_and_admission_required=True)
