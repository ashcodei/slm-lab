"""Rebuild retained execution verdicts directly from complete publisher cases.

Does not call the execution producer's comparator or case-contract function.
It checks the original source and every retained output; it does not rerun the
program or independently establish the original OS confinement.
"""
import base64
import hashlib
import math

KINDS=('public_tests','private_tests','generated_tests')
SUCCESS='every_original_finite_case_passed_under_declared_local_protocol'


def sha(value):return hashlib.sha256(value.encode()).hexdigest()


def verify(row,problem,result):
    counts={kind:len(getattr(problem,kind)) for kind in KINDS}
    total=sum(counts.values());program=row['response']
    if (row['language']!='Python3' or row['split']!='train' or row['publisher_split']!='train'
            or row['original_program_sha256']!=sha(program)
            or result['original_problem_id']!=row['source_problem_id']
            or result['original_program_sha256']!=sha(program)
            or result['original_solution_position']!=row['original_solution_position']
            or result['original_pack_position']!=row['token_pack_index']
            or result['complete_original_test_counts']!=counts
            or result['complete_original_test_population']!=total
            or result['no_original_program_or_case_clipped'] is not True
            or result['original_publisher_split']!='train' or result['admitted_source_role']!='train'
            or result['execution_reward_training_admitted'] is not False):
        raise ValueError('Exact whole original screened training source and population required')
    cases=[(kind,i,case) for kind in KINDS for i,case in enumerate(getattr(problem,kind))]
    population_bytes=sum(len(c.input.encode())+len(c.output.encode()) for _,_,c in cases)
    execution=result.get('actual_execution')
    if execution is None:
        if result['all_original_finite_cases_pass'] is not False:raise ValueError('An unexecuted reference cannot be qualified')
        status=result['status']
        if status=='original_file_interface_not_supported_by_this_stdin_protocol':
            if not any(problem.HasField(k) and getattr(problem,k) for k in ('input_file','output_file')):
                raise ValueError('Original named-file interface qualification lacks its source')
        elif status=='whole_original_case_count_requires_review':
            if 1<=total<=1000:raise ValueError('Whole original count qualification differs')
        elif status=='whole_original_case_payload_or_encoding_requires_review':
            if population_bytes<=8*1024**2 and all(len(v.encode())<=4*1024**2 for _,_,c in cases for v in (c.input,c.output)):
                raise ValueError('Whole original payload qualification differs')
        elif status!='original_program_local_runtime_qualification':raise ValueError('Unknown unexecuted source qualification')
        return dict(actual_retained_case_outputs_checked=0,fully_qualified=False,status=status,
            actual_runtime_not_reexecuted=True)
    digest=hashlib.sha256()
    for kind,i,c in cases:
        for value in (kind.encode(),i.to_bytes(8,'little'),c.input.encode(),c.output.encode()):
            digest.update(len(value).to_bytes(8,'little'));digest.update(value)
    if (execution['complete_original_tests_sha256']!=digest.hexdigest()
            or execution['total_cases']!=total or execution['counts_by_original_kind']!=counts
            or execution['complete_original_input_and_expected_UTF8_bytes']!=population_bytes
            or execution['informative_nonempty_expected_cases']!=sum(bool(c.output.encode().split()) for _,_,c in cases)
            or execution['original_program_sha256']!=sha(program)
            or execution['status']!=result['status']
            or execution['expected_outputs_never_written_to_candidate_readable_filesystem'] is not True
            or not 1<=total<=1000 or population_bytes>8*1024**2
            or any(len(v.encode())>4*1024**2 for _,_,c in cases for v in (c.input,c.output))):
        raise ValueError('Complete original case hash, shape, bounds or source identity differs')
    try:compile(program,'<independent-original-source-readback>','exec');syntax=True
    except (SyntaxError,ValueError,RecursionError):syntax=False
    if execution['syntax_valid'] is not syntax:raise ValueError('Fresh original syntax check differs')
    actual=execution['original_case_verdicts']
    if (type(execution['actual_observed_case_count']) is not int or len(actual)!=execution['actual_observed_case_count']
            or len(actual)>total or (not syntax and actual)):
        raise ValueError('Every retained original-case output must retain its prefix position')
    passed=[];output_bytes=0
    for ordinal,observed in enumerate(actual):
        kind,i,c=cases[ordinal]
        stdout=base64.b64decode(observed['actual_stdout_b64'],validate=True)
        stderr=base64.b64decode(observed['actual_stderr_b64'],validate=True)
        output_bytes+=len(stdout)+len(stderr)
        equal=stdout.split()==c.output.encode().split()
        if (observed['original_case_ordinal']!=ordinal or observed['kind']!=kind or observed['original_kind_position']!=i
                or observed['original_input_sha256']!=sha(c.input) or observed['original_expected_sha256']!=sha(c.output)
                or observed['input_bytes']!=len(c.input.encode())
                or not 0<=observed['input_bytes_sent']<=len(c.input.encode())
                or observed['actual_output_tokens_equal_expected'] is not equal
                or not math.isfinite(observed['seconds']) or observed['seconds']<0
                or (observed['process_status']=='completed' and observed['returncode']!=0)):
            raise ValueError('Actual output, fresh original-token verdict or complete case position differs')
        passed.append(observed['process_status']=='completed' and equal)
        if ordinal+1<len(actual) and not passed[-1]:raise ValueError('Execution must stop at its first local qualification')
    status=execution['status'];qualified=status==SUCCESS
    if qualified:
        if (not syntax or len(actual)!=total or not all(passed) or output_bytes>8*1024**2
                or execution['all_cases_observed'] is not True
                or execution['all_original_finite_cases_pass'] is not True
                or result['all_original_finite_cases_pass'] is not True):
            raise ValueError('A full-case pass requires every actual original output and no resource qualification')
    else:
        if execution['all_original_finite_cases_pass'] is not False or result['all_original_finite_cases_pass'] is not False:
            raise ValueError('A partial or qualified source cannot claim full-case success')
        if status=='native_reference_runtime_or_output_qualification':
            if not actual or passed[-1] or execution['first_unqualified_original_case_ordinal']!=len(actual)-1:
                raise ValueError('The retained last output must support the actual runtime/output qualification')
            all_observed=len(actual)==total and actual[-1]['process_status']=='completed'
        elif status=='source_syntax_or_version_qualification':
            if syntax or actual:raise ValueError('Syntax qualification lacks actual syntax failure')
            all_observed=False
        elif status=='whole_actual_output_budget_qualification':
            if output_bytes<=8*1024**2 or not all(passed):raise ValueError('Output budget qualification lacks retained evidence')
            all_observed=False
        elif status=='whole_program_wall_budget_qualification':
            if len(actual)>=total or not all(passed):raise ValueError('Whole-program wall qualification must preserve a passing prefix')
            all_observed=False
        else:raise ValueError('Unknown retained execution qualification')
        if execution['all_cases_observed'] is not all_observed:raise ValueError('Actual complete-case observation status differs')
    return dict(actual_retained_case_outputs_checked=len(actual),fully_qualified=qualified,status=status,
        actual_runtime_not_reexecuted=True,original_publisher_judge_not_reproduced=True)
