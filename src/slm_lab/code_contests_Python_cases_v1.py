"""Fresh-interpreter original stdin/stdout reference checks in OS confinement.

Every original case in the supplied public/private/generated population is
retained. Whole populations outside the declared bounds require review. A
reference that fails or exceeds local limits is unqualified for this local
runtime, not newly labelled algorithmically incorrect by this module.
"""
import base64
import hashlib
import math
from pathlib import Path
import tempfile
import time

from .standalone_program_v1 import prepare_python, run_bounded

MAX_POPULATION_BYTES=8*1024**2
MAX_CASE_BYTES=4*1024**2
MAX_CASES=1000


def case_contract(cases):
    if not isinstance(cases,list) or not 1<=len(cases)<=MAX_CASES:
        raise ValueError('Whole original case population size requires review')
    digest=hashlib.sha256();total=0;counts={k:0 for k in ['public_tests','private_tests','generated_tests']}
    for case in cases:
        if (set(case)!={'kind','position','input','expected'} or case['kind'] not in counts
                or type(case['position']) is not int or case['position']!=counts[case['kind']]
                or not isinstance(case['input'],str) or not isinstance(case['expected'],str)):
            raise ValueError('Every original case, kind and contiguous local position required')
        counts[case['kind']]+=1
        for value in [case['kind'].encode(),case['position'].to_bytes(8,'little'),case['input'].encode(),case['expected'].encode()]:
            digest.update(len(value).to_bytes(8,'little'));digest.update(value)
        a,b=len(case['input'].encode()),len(case['expected'].encode());total+=a+b
        if a>MAX_CASE_BYTES or b>MAX_CASE_BYTES or total>MAX_POPULATION_BYTES:
            raise ValueError('Complete original case bytes exceed declared local protocol; no case clipping permitted')
    return dict(complete_original_tests_sha256=digest.hexdigest(),total_cases=len(cases),counts_by_original_kind=counts,
        complete_original_input_and_expected_UTF8_bytes=total,
        informative_nonempty_expected_cases=sum(bool(c['expected'].encode().split()) for c in cases))


def token_equal(observed,expected):
    if not isinstance(observed,bytes) or not isinstance(expected,str):raise ValueError('Actual raw output and original expected text required')
    # Exact ordered byte tokens: whitespace layout may differ, float tolerance,
    # unordered outputs and problem-specific constructive checkers are absent.
    return observed.split()==expected.encode().split()


def execute_reference(program,cases,*,seconds_per_case=3.,seconds_total=120.):
    if (not isinstance(program,str) or not program or '\0' in program
            or not 0<seconds_per_case<=30 or not 0<seconds_total<=300):
        raise ValueError('Unchanged original source and explicit bounded execution required')
    contract=case_contract(cases)
    result=dict(**contract,original_program_sha256=hashlib.sha256(program.encode()).hexdigest(),
        syntax_valid=False,all_cases_observed=False,all_original_finite_cases_pass=False,
        original_case_verdicts=[],actual_observed_case_count=0,
        expected_outputs_never_written_to_candidate_readable_filesystem=True,
        runtime='Fresh CPython3.11 standard-library interpreter per original case; pre-exec Landlock/seccomp;512MiB',
        output_comparator='Exact ordered byte tokens; no float tolerance or custom constructive checker',
        original_publisher_judge_not_reproduced=True,model_updates=0)
    try:compile(program,'<original-CodeContests-program>','exec')
    except (SyntaxError,ValueError,RecursionError) as error:
        return dict(result,status='source_syntax_or_version_qualification',diagnostic=str(error)[:500])
    started=time.monotonic();deadline=started+seconds_total;output_observed=0
    with tempfile.TemporaryDirectory(prefix='slm-CodeContests-Python-') as temporary:
        policy=prepare_python(program.encode(),Path(temporary));policy['memory_bytes']=512*1024**2
        policy['cpu_seconds']=max(1,math.ceil(seconds_per_case))
        result['syntax_valid']=True
        for ordinal,case in enumerate(cases):
            remaining=deadline-time.monotonic()
            if remaining<=0:
                return dict(result,status='whole_program_wall_budget_qualification',elapsed_seconds=time.monotonic()-started)
            observed=run_bounded(policy,case['input'].encode(),seconds=min(seconds_per_case,remaining),
                output_bytes=MAX_CASE_BYTES,python=True)
            output_observed+=len(observed['stdout'])+len(observed['stderr'])
            verdict=dict(original_case_ordinal=ordinal,kind=case['kind'],original_kind_position=case['position'],
                original_input_sha256=hashlib.sha256(case['input'].encode()).hexdigest(),
                original_expected_sha256=hashlib.sha256(case['expected'].encode()).hexdigest(),
                process_status=observed['status'],returncode=observed['returncode'],seconds=observed['seconds'],
                input_bytes=observed['input_bytes'],input_bytes_sent=observed['input_bytes_sent'],
                actual_stdout_b64=base64.b64encode(observed['stdout']).decode(),
                actual_stderr_b64=base64.b64encode(observed['stderr']).decode(),
                actual_output_tokens_equal_expected=token_equal(observed['stdout'],case['expected']))
            result['original_case_verdicts'].append(verdict);result['actual_observed_case_count']+=1
            passed=observed['status']=='completed' and verdict['actual_output_tokens_equal_expected']
            if not passed:
                return dict(result,all_cases_observed=(ordinal+1==len(cases) and observed['status']=='completed'),
                    status='native_reference_runtime_or_output_qualification',
                    first_unqualified_original_case_ordinal=ordinal,elapsed_seconds=time.monotonic()-started,
                    observed_prefix_is_not_full_case_success=True)
            if output_observed>MAX_POPULATION_BYTES:
                return dict(result,status='whole_actual_output_budget_qualification',elapsed_seconds=time.monotonic()-started)
    result.update(all_cases_observed=True,all_original_finite_cases_pass=True,
        elapsed_seconds=time.monotonic()-started,status='every_original_finite_case_passed_under_declared_local_protocol',
        full_execution_reward_admission_not_granted=True)
    return result
