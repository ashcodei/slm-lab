"""Source-declared execution contracts; never execute or rewrite public code."""
import ast
import hashlib


CONTRACT = (
    "Execution contract: this is a two-stage Python judge environment with NumPy "
    "and Numba ahead-of-time compilation. A preparation invocation uses "
    "ONLINE_JUDGE as the final command-line argument to build a local native "
    "module named my_module. That module is retained for subsequent invocations "
    "without the preparation argument. Preparation may continue into input "
    "handling, so provide valid problem input during that invocation too. "
    "Subsequent invocations read problem input and write the required output."
)


def native_inventory(row):
    """Report syntax evidence, including optional branches, without running it."""
    if row['language'] != 'Python':
        return None
    try:
        tree = ast.parse(row['response'])
    except (SyntaxError, ValueError):
        return None  # Python2 and other historical syntax are not reclassified.
    nodes = list(ast.walk(tree))
    if not any(isinstance(n, ast.ImportFrom) and n.module == 'numba.pycc' for n in nodes):
        return None
    modules = sorted({n.args[0].value for n in nodes
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == 'CC' and n.args and isinstance(n.args[0], ast.Constant)
        and isinstance(n.args[0].value, str)})
    imports = sorted({n.module for n in nodes if isinstance(n, ast.ImportFrom) and n.module in modules})
    return {
        'original_archive_member': row['original_archive_member'],
        'original_source_body_sha256': hashlib.sha256(row['response'].encode()).hexdigest(),
        'split': row['split'], 'group_id': row['group_id'],
        'declared_CC_modules': modules, 'generated_module_imports': imports,
        'compile_call_lines': sorted(n.lineno for n in nodes if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == 'compile'),
        'ONLINE_JUDGE_literal_present': any(isinstance(n, ast.Constant) and n.value == 'ONLINE_JUDGE' for n in nodes),
        'scope': 'Static syntax inventory only; branch reachability and compilation success are not established.'}


def with_native_contract(row, qualification, qualification_sha256):
    """Annotate exactly the source/role qualified earlier, preserving its answer."""
    keys = ['original_archive_member', 'original_source_body_sha256', 'source_problem_id', 'group_id', 'split']
    if any(row.get(k) != qualification.get(k) for k in keys):
        raise ValueError('Exact qualified original source identity and sealed role required')
    if (row['split'] != 'train' or row['language'] != 'Python'
            or row['original_judge_status'] != 'Accepted'
            or qualification['source_declared_preparation_flag'] != 'ONLINE_JUDGE'
            or qualification['numba_pycc_generated_modules'] != ['my_module']
            or qualification['generated_modules_imported_at_runtime'] != ['my_module']):
        raise ValueError('Only the qualified original training-source contract is supported')
    facts = native_inventory(row)
    if (facts is None or facts['original_source_body_sha256'] != row['original_source_body_sha256']
            or facts['declared_CC_modules'] != ['my_module']
            or facts['generated_module_imports'] != ['my_module']
            or not facts['compile_call_lines'] or not facts['ONLINE_JUDGE_literal_present']):
        raise ValueError('Original source no longer supports the recorded native contract')
    head, separator, statement = row['prompt'].partition('\n\n')
    if not separator or not statement or not head.startswith('Write a complete Python program.\nTarget language/runtime: '):
        raise ValueError('Original complete prompt framing required')
    if 'execution_contract' in row or CONTRACT in row['prompt']:
        raise ValueError('Preserve original input; do not apply a contract twice')
    return {**row, 'prompt': head + '\n' + CONTRACT + separator + statement,
        'execution_contract': {
            'kind': 'source-declared-numba-aot-v1',
            'preparation_final_argument': 'ONLINE_JUDGE', 'generated_module': 'my_module',
            'preparation_may_continue_into_input_handling': True,
            'source_qualification_report_sha256': qualification_sha256,
            'actual_native_preparation_verified': False,
            'original_response_modified': False}}
