"""Execute every original reserved MBPP equality with golds kept in the parent.

The confined child receives the unchanged program, original input setup and
left-hand expressions only. Original expected RHS values are computed by the
trusted parent and never written to the candidate's filesystem or process.
Every original test/challenge case retains its kind, position and denominator.
"""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

from .corpus import file_hash
from .finite_Python_values_v2 import decode
from .standalone_program_v1 import python_policy,run_bounded

ROOT=Path(__file__).resolve().parents[2]
HARNESS=ROOT/'scripts/mbpp_original_observations_v2.py'
VALUES=ROOT/'src/slm_lab/finite_Python_values_v2.py'
FORMAT='own-complete-original-MBPP-function-execution-v2'


def cases(row):
    result=[]
    for kind in ('test_list','challenge_test_list'):
        for position,text in enumerate(row.get(kind,[])):
            tree=ast.parse(text)
            if (len(tree.body)!=1 or not isinstance(tree.body[0],ast.Assert)
                    or not isinstance(tree.body[0].test,ast.Compare) or len(tree.body[0].test.ops)!=1
                    or not isinstance(tree.body[0].test.ops[0],ast.Eq)):
                raise ValueError('Every original test must be one whole publisher equality; no filtering')
            compare=tree.body[0].test;right=compare.comparators[0]
            # All original expected values are literal data, except three
            # explicit sys.getsizeof(literal tuple) cases. No generated code
            # or arbitrary publisher expression is executed in the parent.
            try:expected=ast.literal_eval(right)
            except (ValueError,TypeError):
                if (not isinstance(right,ast.Call) or right.keywords or len(right.args)!=1
                        or not isinstance(right.func,ast.Attribute) or right.func.attr!='getsizeof'
                        or not isinstance(right.func.value,ast.Name) or right.func.value.id!='sys'):
                    raise ValueError('Original expected value outside declared literal/sys.getsizeof protocol')
                expected=sys.getsizeof(ast.literal_eval(right.args[0]))
            left=ast.get_source_segment(text,compare.left)
            if left is None:raise ValueError('Exact original left expression must remain available')
            result.append(dict(kind=kind,original_position=position,original_test=text,
                original_test_sha256=hashlib.sha256(text.encode()).hexdigest(),left_expression=left,expected=expected))
    if not result or len(result)>1000:raise ValueError('Every original nonempty finite case population required')
    return result


def execute(program,row,*,seconds=4.):
    if (not isinstance(program,str) or not program or len(program.encode())>32768 or '\0' in program
            or type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=30):
        raise ValueError('Whole bounded unchanged program and explicit runtime required')
    original=cases(row);setup=row.get('test_setup_code','')
    base=dict(format=FORMAT,program_sha256=hashlib.sha256(program.encode()).hexdigest(),original_task_id=row['task_id'],
        original_cases=len(original),original_test_cases=len(row['test_list']),original_challenge_cases=len(row.get('challenge_test_list',[])),
        original_case_SHA256s=[c['original_test_sha256'] for c in original],syntax_valid=False,
        actual_observed_cases=0,passed_cases=0,all_cases_observed=False,functional_passed=False,
        original_program_or_case_not_rewritten=True,expected_RHS_not_sent_to_confined_worker=True,
        ordinary_import_style_module_name=True,standard_library_only=True,
        protocol_is_custom_native_not_official_few_shot_MBPP=True,
        finite_original_examples_not_general_or_asymptotic_efficiency_proof=True)
    try:compile(program,'<unchanged-candidate-module>','exec')
    except (SyntaxError,ValueError,RecursionError) as error:return dict(base,status='syntax_error',diagnostic=str(error)[:300])
    base['syntax_valid']=True
    with tempfile.TemporaryDirectory(prefix='slm-original-MBPP-v2-') as temporary:
        work=Path(temporary);(work/'program.py').write_text(program);(work/'harness.py').write_bytes(HARNESS.read_bytes())
        (work/'values.py').write_bytes(VALUES.read_bytes())
        settings=dict(program=str(work/'program.py'),setup=setup,expressions=[c['left_expression'] for c in original])
        raw_settings=json.dumps(settings,allow_nan=False)
        if len(raw_settings.encode())>4*1024**2:raise ValueError('Entire original input/setup population exceeds declared runtime')
        (work/'settings.json').write_text(raw_settings)
        policy=python_policy(work);policy['command'][-1]=str(work/'harness.py');policy['command'].append(str(work/'settings.json'))
        policy['memory_bytes']=512*1024**2;policy['cpu_seconds']=max(1,math.ceil(seconds))
        observed=run_bounded(policy,b'',seconds=seconds,output_bytes=4*1024**2,python=True)
        execution={k:v for k,v in observed.items() if k not in ('stdout','stderr')}
        execution.update(stdout_sha256=hashlib.sha256(observed['stdout']).hexdigest(),stderr=observed['stderr'][-1600:].decode(errors='replace'))
        base['execution']=execution
        if observed['status']!='completed':return dict(base,status='runtime_or_resource_failure')
        try:
            raw=json.loads(observed['stdout']);actual=raw['results']
            if (set(raw)!={'format','results','module_failure'} or raw['format']!='own-original-MBPP-raw-left-values-v2'
                    or not isinstance(actual,list) or raw['module_failure'] is not None or len(actual)!=len(original)):
                return dict(base,status='incomplete_module_or_input_observations',raw_observations=raw)
            comparisons=[]
            for source,result in zip(original,actual):
                if not math.isfinite(result['seconds']) or result['seconds']<0:raise ValueError('Actual finite runtime required')
                if result['status']=='value':
                    value=decode(result['value']);passed=value==source['expected']
                elif result['status']=='expression_error':passed=False
                else:raise ValueError('Every actual original expression needs an explicit value/error')
                comparisons.append(dict(kind=source['kind'],original_position=source['original_position'],
                    original_test_sha256=source['original_test_sha256'],passed=bool(passed)))
        except (KeyError,TypeError,ValueError,RecursionError,OverflowError):return dict(base,status='invalid_raw_value_observations')
        return dict(base,status='all_original_finite_cases_pass' if all(c['passed'] for c in comparisons) else 'original_finite_case_failure',
            actual_observed_cases=len(original),all_cases_observed=True,passed_cases=sum(c['passed'] for c in comparisons),
            functional_passed=all(c['passed'] for c in comparisons),original_case_verdicts=comparisons,raw_observations=raw)


def implementation_sources():
    return {str(p):file_hash(p) for p in [Path(__file__),HARNESS,VALUES,ROOT/'src/slm_lab/standalone_program_v1.py',
        ROOT/'src/slm_lab/native_code.py',ROOT/'scripts/python_program_sandbox_v1.py',ROOT/'scripts/native_sandbox.py']}
