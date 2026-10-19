"""Rebuild one declared source disposition and any complete repeated target.

Full original reference-output verification is a separate required caller
gate. This helper never executes a candidate or imports the target formatter.
"""
import base64
import hashlib
import json

from .public_python_state_trace_v1 import validate
from .public_state_trace_readback_v2 import rebuild


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def assess(row,problem,reference,disposition,observation,repeated,tokenizer):
    if (row.get('split')!='train' or row.get('publisher_split')!='train' or row.get('language')!='Python3'
            or row['original_program_sha256']!=digest(row['response'])
            or reference['original_problem_id']!=row['source_problem_id']
            or reference['original_program_sha256']!=row['original_program_sha256']
            or reference['original_solution_position']!=row['original_solution_position']):
        raise ValueError('Exact original screened training source required')
    def result(status,target=None):
        if disposition['status']!=status:raise ValueError('Independently reconstructed source disposition differs: '+status)
        return dict(status=status,target=target)
    if reference['all_original_finite_cases_pass'] is False:
        if observation is not None or repeated is not None:raise ValueError('Unqualified source cannot supply a state target')
        return result('original_reference_not_fully_qualified')
    if reference['all_original_finite_cases_pass'] is not True:raise ValueError('Explicit complete original reference qualification required')
    execution=reference['actual_execution'];counts={k:len(getattr(problem,k)) for k in ('public_tests','private_tests','generated_tests')}
    if (execution.get('all_cases_observed') is not True or execution.get('all_original_finite_cases_pass') is not True
            or execution.get('actual_observed_case_count')!=sum(counts.values())
            or execution.get('counts_by_original_kind')!=counts or len(execution['original_case_verdicts'])!=sum(counts.values())):
        raise ValueError('Every original public/private/generated case must qualify before source-state supervision')
    if not problem.public_tests:
        if observation is not None or repeated is not None:raise ValueError('A source without public inputs cannot supply a state target')
        return result('no_original_public_input')
    public=problem.public_tests[0];program=row['response']
    if (observation is None or observation['program_sha256']!=digest(program)
            or observation['public_input_sha256']!=digest(public.input)
            or observation.get('expected_or_private_values_not_supplied_to_worker') is not True
            or disposition['original_public_case_position']!=0 or disposition['original_public_input_sha256']!=digest(public.input)):
        raise ValueError('Entire unchanged original source and public case0 observer receipt required')
    if observation.get('eligible_complete_trace') is False:
        if repeated is not None:raise ValueError('A partial observation cannot have a native-candidate repeat')
        status=observation['status']
        if status not in {'observer_runtime_qualification','invalid_observer_transport_qualification','incomplete_or_bounded_trace_qualification'}:
            raise ValueError('Unknown trace qualification')
        if status=='incomplete_or_bounded_trace_qualification':
            raw=json.loads(base64.b64decode(observation['raw_observation_stdout_b64'],validate=True))
            if raw!=observation['actual_observations'] or raw['whole_trace_complete'] is not False or raw['qualification'] is None:
                raise ValueError('Whole retained partial-trace observation must reconstruct')
        return result(status)
    if observation.get('eligible_complete_trace') is not True:raise ValueError('Explicit whole trace or qualification required')
    raw=json.loads(base64.b64decode(observation['raw_observation_stdout_b64'],validate=True))
    if raw!=observation['actual_observations'] or observation['returncode']!=0:raise ValueError('Raw source-state transport changed')
    validate(raw,program,maximum_events=256,maximum_trace_bytes=256*1024)
    prior=execution['original_case_verdicts'][0]
    if (prior['kind']!='public_tests' or prior['original_kind_position']!=0
            or prior['original_input_sha256']!=digest(public.input) or prior['original_expected_sha256']!=digest(public.output)):
        raise ValueError('Exact original first public-case result required')
    stdout=base64.b64decode(raw['actual_stdout_b64'],validate=True)
    if stdout!=base64.b64decode(prior['actual_stdout_b64'],validate=True):
        if repeated is not None:raise ValueError('Changed instrumented stdout cannot become a native target')
        return result('instrumented_stdout_differs_from_original_retained_execution')
    if stdout.split()!=public.output.encode().split():raise ValueError('Actual reference stdout must retain the original token verdict')
    try:native=rebuild(program,public.input,observation,tokenizer)
    except UnicodeDecodeError:native=dict(eligible=False,reason='whole_observed_stdout_not_UTF8')
    if not native['eligible']:
        if repeated is not None:raise ValueError('A nonfitting whole native target cannot have a candidate repeat')
        return result(native['reason'])
    if (repeated is None or repeated['program_sha256']!=digest(program) or repeated['public_input_sha256']!=digest(public.input)
            or repeated.get('expected_or_private_values_not_supplied_to_worker') is not True):
        raise ValueError('Second fresh whole-source/public-input observation required')
    same=repeated['eligible_complete_trace'] and repeated['actual_observations']==raw
    if repeated['eligible_complete_trace']:
        repeat_raw=json.loads(base64.b64decode(repeated['raw_observation_stdout_b64'],validate=True))
        if repeat_raw!=repeated['actual_observations'] or repeated['returncode']!=0:raise ValueError('Repeated worker transport changed')
    if not same:return result('whole_state_or_stdout_not_reproduced_in_second_fresh_interpreter')
    return result('whole_trace_native_candidate',native)
