"""Source-preserving code negatives that escape all original public examples.

Single lexical edits retain comments, docstrings, signatures and all other
source bytes. Admission requires observed published-example success and a
fully observed failure under an already qualified finite contract. These are
machine negatives, not policy rollouts, human preferences or correctness proofs.
"""
import ast
import hashlib
import io
import json
import tokenize

from .contract_source_contrasts_v2 import encode, source_only
from .public_function_contract_census_v2 import cases, digest, family, passes


REPLACEMENTS = {'==': ('!=',), '!=': ('==',), '<': ('<=', '>'), '<=': ('<', '>='),
                '>': ('>=', '<'), '>=': ('>', '<='), '+': ('-',), '-': ('+',),
                '*': ('+',), '/': ('*',), '//': ('/',), '%': ('//',), '**': ('*',),
                '&': ('|',), '|': ('&',), '^': ('|',), '<<': ('>>',), '>>': ('<<',),
                'and': ('or',), 'or': ('and',), 'True': ('False',), 'False': ('True',)}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def signature(tree):
    function = tree.body[0]
    return (function.name, ast.dump(function.args),
            None if function.returns is None else ast.dump(function.returns),
            [ast.dump(n) for n in function.decorator_list],
            [ast.get_docstring(n, clean=False) for n in ast.walk(tree)
             if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))])


def replacements(token):
    if token.type not in (tokenize.OP, tokenize.NAME, tokenize.NUMBER):
        return ()
    if token.type != tokenize.NUMBER:
        return REPLACEMENTS.get(token.string, ())
    try:
        value = ast.literal_eval(token.string)
    except (SyntaxError, ValueError):
        return ()
    return (str(value + 1),) if type(value) is int and abs(value) <= 1000000 else ()


def candidates(row, *, maximum=16):
    source_only(row)
    if type(maximum) is not int or not 1 <= maximum <= 64:
        raise ValueError('Explicit bounded lexical candidate budget required')
    program = row['response']
    tree = ast.parse(program)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != row['source_candidate']['function']):
        raise ValueError('One complete reviewed original function required')
    function = tree.body[0]
    lines = program.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    records = []
    seen = set()
    for token in tokenize.generate_tokens(io.StringIO(program).readline):
        if (token.type not in (tokenize.OP, tokenize.NAME, tokenize.NUMBER)
                or token.start < (function.body[0].lineno, function.body[0].col_offset)):
            continue
        choices = replacements(token)
        start = offsets[token.start[0] - 1] + token.start[1]
        end = offsets[token.end[0] - 1] + token.end[1]
        for replacement in choices:
            negative = program[:start] + replacement + program[end:]
            try:
                rebuilt = ast.parse(negative)
            except (SyntaxError, ValueError, RecursionError):
                continue
            if ast.dump(rebuilt) == ast.dump(tree) or signature(rebuilt) != signature(tree):
                continue
            if sha(negative) in seen:
                continue
            seen.add(sha(negative))
            patch = dict(start_character=start, end_character=end, old=token.string, new=replacement,
                         original_prefix_bytes=len(program[:start].encode()),
                         original_replaced_bytes=len(program[start:end].encode()))
            records.append(dict(source_id=row['id'], original_program_sha256=sha(program),
                                program_sha256=sha(negative), program=negative, patch=patch,
                                single_lexical_token_edit_all_other_source_bytes_unchanged=True))
    records.sort(key=lambda value: sha(json.dumps([row['id'], value['patch']], sort_keys=True)))
    return [dict(value, candidate_position=i) for i, value in enumerate(records[:maximum])]


def pair(row, mutation, positive_contract, negative_contract, negative_public, tokenizer):
    source_only(row)
    original = row['response']
    patch = mutation['patch']
    start, end = patch['start_character'], patch['end_character']
    expected = original[:start] + patch['new'] + original[end:]
    offsets = [0]
    for line in original.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    matching = [token for token in tokenize.generate_tokens(io.StringIO(original).readline)
                if offsets[token.start[0] - 1] + token.start[1] == start
                and offsets[token.end[0] - 1] + token.end[1] == end]
    tree = ast.parse(original)
    if (original[start:end] != patch['old'] or mutation['program'] != expected
            or mutation['original_program_sha256'] != sha(original)
            or mutation['program_sha256'] != sha(expected)
            or mutation['source_id'] != row['id'] or len(matching) != 1
            or matching[0].start < (tree.body[0].body[0].lineno, tree.body[0].body[0].col_offset)
            or patch['new'] not in replacements(matching[0])
            or patch['original_prefix_bytes'] != len(original[:start].encode())
            or patch['original_replaced_bytes'] != len(original[start:end].encode())
            or signature(ast.parse(expected)) != signature(tree)):
        raise ValueError('Exact original source and single preserved lexical patch required')
    selected = cases(row)
    for observed, program in [(positive_contract, original), (negative_contract, expected)]:
        if (observed.get('source_id') != row['id'] or observed.get('program_sha256') != sha(program)
                or observed.get('cases_sha256') != digest(selected)
                or observed.get('all_cases_observed') is not True
                or len(observed.get('raw_results', [])) != len(selected)):
            raise ValueError('Every finite case must actually finish before any preference label')
        checks = [passes(case, raw) for case, raw in zip(selected, observed['raw_results'])]
        if checks != observed.get('case_passes') or all(checks) != observed.get('all_cases_passed'):
            raise ValueError('Every finite return/type/mutation judgment must reconstruct')
    if (not positive_contract['all_cases_passed'] or negative_contract['all_cases_passed']
            or negative_public.get('original_source_id') != row['id']
            or negative_public.get('actual_generated_program_sha256') != sha(expected)
            or negative_public.get('all_original_examples_observed') is not True
            or negative_public.get('all_original_examples_passed') is not True
            or not all(negative_public.get('example_passes', []))
            or negative_public.get('entry_body_calls', 0) < 1):
        raise ValueError('A valid original and a public-example-passing finite counterexample required')
    ids, labels = encode(row['prompt'], original, tokenizer)
    negative_ids, negative_labels = encode(row['prompt'], expected, tokenizer)
    boundary = next(i for i, value in enumerate(labels) if value != -100)
    if ids[:boundary] != negative_ids[:boundary] or negative_labels[:boundary] != labels[:boundary]:
        raise ValueError('Exact same original task prompt and complete native masks required')
    witness_position = next(i for i, value in enumerate(negative_contract['case_passes']) if not value)
    return dict(id='public-edge-negative:' + digest([row['id'], mutation['program_sha256']]),
                source_id=row['id'], source_group=row['group_id'], source_role='train', contract_family=family(row),
                prompt=row['prompt'], chosen=original, rejected=expected, mutation=mutation,
                original_public_URL=row['original_public_URL'], rights=row['rights'],
                native_prompt_ids=ids[:boundary], chosen_complete_response_EOS_ids=ids[boundary:],
                rejected_complete_response_EOS_ids=negative_ids[boundary:],
                complete_positive_targets=len(ids) - boundary, complete_negative_targets=len(negative_ids) - boundary,
                published_examples_passed=negative_public['original_public_example_count'],
                finite_cases=len(selected), rejected_finite_cases_passed=sum(negative_contract['case_passes']),
                counterexample=dict(case=selected[witness_position], raw=negative_contract['raw_results'][witness_position]),
                whole_cases_sha256=digest(selected), counterexample_not_added_to_original_model_prompt=True,
                source_train_diagnostic_not_independent_benchmark=True, finite_checks_not_general_correctness=True,
                machine_derived_negative_not_public_original_code=True, human_preferences=False, on_policy_RL=False,
                external_neural_teacher=False, optimization_admitted=False)
