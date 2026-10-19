"""Teacher-free function contracts from exact permissively labelled public files.

Descriptions/examples come from the function's own literal docstring. The
complete original function is the response. A declared existing stdlib prelude
supplies context; decorators and unresolved globals are excluded. Execution of
every published example and independent source/native readback remain required
before admission. Finite examples are not a general correctness certificate.
"""
import ast
import doctest
import hashlib
import io
import tokenize

from .documented_doctest_v1 import PRELUDE, contract, compare, execute, sha, unresolved_globals


def signature(program):
    lines = program.splitlines(keepends=True)
    depth = 0
    for token in tokenize.generate_tokens(io.StringIO(program).readline):
        if token.type != tokenize.OP:continue
        if token.string in '([{':depth += 1
        elif token.string in ')]}':depth -= 1
        elif token.string == ':' and depth == 0:
            row, column = token.end
            return ''.join(lines[:row-1])+lines[row-1][:column]
    raise ValueError('Exact complete original function signature required')


def from_function(original_entry, node, tokenizer, overlap):
    text = original_entry['original_publisher_row']['code']
    if (original_entry['family'] != 'code' or original_entry['language'] != 'Python'
            or original_entry['role'] not in {'train','valid'}):
        raise ValueError('A whole independently inspectable original Python file with a fixed source role is required')
    source_sha = sha(text)
    if (original_entry['original_UTF8_sha256'] != source_sha
            or original_entry['original_public_text'] != text
            or original_entry['role'] != ('valid' if int(source_sha[-8:],16)%100 == 0 else 'train')):
        raise ValueError('Original complete source body and deterministic raw-document role must remain exact')
    if not isinstance(node,ast.FunctionDef) or node.decorator_list:
        return None,'decorated_async_or_nonordinary_function'
    program = ast.get_source_segment(text,node)
    description = ast.get_docstring(node,clean=False)
    if not program or not isinstance(description,str) or not 24 <= len(description.encode()) <= 20000:
        return None,'missing_or_unbounded_original_literal_docstring'
    try:
        examples = doctest.DocTestParser().get_examples(description)
        missing = unresolved_globals(program)
    except (SyntaxError,ValueError,RecursionError):
        return None,'unsupported_original_docstring_or_function_context'
    if not 2 <= len(examples) <= 128 or any(t.options.get(doctest.SKIP,False) for t in examples):
        return None,'requires_all_original_examples_at_least_two_and_no_SKIP'
    if missing:return None,'unresolved_function_context_not_in_declared_stdlib'
    header = signature(program)
    prompt = ('Implement this Python function according to its original public documentation. '
        'Return the complete function only. The declared standard-library context is available.\n\n'
        'Declared standard-library context:\n'+PRELUDE+'\nOriginal function signature:\n'+header+
        '\n\nOriginal literal docstring, including every published example:\n'+description)
    prefix = [1,*tokenizer.encode('User: '+prompt+'\nAssistant:',add_special_tokens=False).ids]
    target = [*tokenizer.encode(program,add_special_tokens=False).ids,2]
    ids = prefix+target
    if (len(ids) > 4096 or len(target) < 2 or any(not 4 <= v < 32768 for v in ids[1:-1])
            or tokenizer.decode(prefix[1:],skip_special_tokens=False) != 'User: '+prompt+'\nAssistant:'
            or tokenizer.decode(target[:-1],skip_special_tokens=False) != program):
        return None,'whole_original_function_contract_not_lossless_within4096'
    native_text = tokenizer.decode(ids[1:-1],skip_special_tokens=False)
    if overlap.literal(native_text) or overlap.lexical_copies(native_text):
        return None,'actual_function_native_input_or_response_has_declared_held_overlap'
    source_sha = original_entry['original_UTF8_sha256']
    identity = 'public-function:'+sha(source_sha+':'+str(node.lineno)+':'+sha(program))
    runtime = dict(source=dict(id=identity,original_partition='train',language='Python',prompt=description,response=program),
        function=node.name,program=program,program_sha256=sha(program),doctest_text=description,
        doctest_text_sha256=sha(description),existing_formatted_source_target_sha256=sha(program),
        whole_original_example_count=len(examples))
    contract(runtime)
    return dict(id=identity,source_original_file=original_entry,function=node.name,
        original_function_span=dict(lineno=node.lineno,end_lineno=node.end_lineno,col_offset=node.col_offset,end_col_offset=node.end_col_offset),
        original_function_program=program,original_function_program_sha256=sha(program),
        original_literal_docstring=description,original_literal_docstring_sha256=sha(description),
        original_signature=header,prompt=prompt,response=program,runtime_candidate=runtime,
        role=original_entry['role'],whole_original_example_count=len(examples),
        whole_native_ids=ids,whole_native_labels=[-100]*len(prefix)+target,
        native_target_start=len(prefix),complete_native_context_tokens=len(ids),response_EOS_targets=len(target),
        declared_prelude=PRELUDE,declared_prelude_sha256=sha(PRELUDE),
        prompt_derived_from_original_signature_and_literal_docstring_not_publisher_authored_instruction=True,
        complete_original_function_AST_segment_is_unchanged_response=True,
        worker_adapter_legacy_formatted_target_field_points_to_original_function_not_a_prior_training_row=True,
        original_publisher_role_is_train_derived_source_role_stays_sealed=True,
        model_generated_target=False,external_neural_teacher=False,human_preferences=False,
        optimization_admitted=False,all_original_published_examples_must_still_pass=True,
        source_semantic_and_all_history_independence_not_certified=True),None


def execute_candidate(candidate,seconds=3):
    observed = execute(candidate['runtime_candidate'],seconds=seconds)
    # The original program itself contains its literal docstring/example answers.
    # The legacy runner withholds separate expected-output comparison artifacts;
    # its old flag cannot mean the answers are absent from this source program.
    observed['expected_text_not_written_to_worker'] = False
    observed['original_program_docstring_includes_public_example_answers'] = True
    observed['expected_outputs_not_sent_as_separate_assertions_or_comparison_artifacts'] = True
    return observed


def verified_training_row(candidate, observations, tokenizer, overlap):
    original = candidate['source_original_file']
    text = original['original_publisher_row']['code']
    tree = ast.parse(text)
    node = next((n for n in tree.body if isinstance(n,ast.FunctionDef)
        and n.lineno == candidate['original_function_span']['lineno']),None)
    if node is None:raise ValueError('Exact complete original top-level function source position required')
    rebuilt,reason = from_function(original,node,tokenizer,overlap)
    if rebuilt is None or rebuilt != candidate:
        raise ValueError('Every original function/source field, literal example, native mask and EOS must rebuild')
    tests = contract(candidate['runtime_candidate'])
    if len(observations) != 2:
        raise ValueError('Two actual fresh complete published-example observations are required')
    for observed in observations:
        if (observed.get('status') != 'all_original_examples_passed'
                or observed.get('source_id') != candidate['id']
                or observed.get('program_sha256') != candidate['original_function_program_sha256']
                or observed.get('doctest_text_sha256') != candidate['original_literal_docstring_sha256']
                or observed.get('declared_prelude_sha256') != candidate['declared_prelude_sha256']
                or observed.get('expected_text_not_written_to_worker') is not False
                or observed.get('original_program_docstring_includes_public_example_answers') is not True
                or observed.get('expected_outputs_not_sent_as_separate_assertions_or_comparison_artifacts') is not True
                or not observed.get('all_original_examples_observed') or not observed.get('all_original_examples_passed')
                or observed.get('original_example_count') != len(tests)
                or observed.get('entry_body_calls',0) < 1 or observed.get('example_passes') != [True]*len(tests)
                or len(observed.get('raw_results',[])) != len(tests)
                or not all(compare(t,r) for t,r in zip(tests,observed['raw_results']))):
            raise ValueError('Every original published example must pass in both actual fresh interpreters')
    if observations[0]['raw_results'] != observations[1]['raw_results']:
        raise ValueError('Original raw outputs and actual function-call observations must reproduce exactly')
    role = 'train' if candidate['role'] == 'train' else 'validation'
    row = dict(id=candidate['id'],group_id='public-file:'+original['original_UTF8_sha256'],split=role,publisher_split='train',language='Python',
        prompt=candidate['prompt'],response=candidate['response'],source_candidate=candidate,
        original_public_URL=original['original_public_URL'],rights=original['rights'],
        function_contract_from_original_public_docstring=True,complete_original_function_response_unchanged=True,
        actual_fresh_complete_example_repetitions=2,whole_original_example_count=len(tests),
        explicit_declared_stdlib_context=True,finite_examples_not_general_correctness_or_efficiency=True,
        original_public_example_answers_visible_inside_original_function_docstring=True,
        separate_expected_output_assertions_or_comparison_files_not_supplied_to_worker=True,
        human_preferences=False,external_neural_teacher_used=False,model_generated_target=False,
        dataset_admitted_to_training=False)
    return row,candidate['whole_native_ids'],candidate['whole_native_labels']
