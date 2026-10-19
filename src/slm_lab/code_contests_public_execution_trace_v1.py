"""Build complete public-input execution targets from actual reference stdout.

This auxiliary task teaches program execution through the same language-model
head. It adds no parameters or pretrained teacher. Private/generated cases can
qualify the reference, but their input/output values never enter this prompt.
"""
import base64
import hashlib

from .code_contests_curriculum_candidates_v1 import native_sequence
from .code_contests_Python_cases_v1 import token_equal


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def public_trace(row,problem,reference,tokenizer):
    if (row['split']!='train' or row['publisher_split']!='train' or row['language']!='Python3'
            or row['original_program_sha256']!=digest(row['response'])
            or reference['original_program_sha256']!=row['original_program_sha256']
            or reference['original_problem_id']!=row['source_problem_id']
            or reference['original_solution_position']!=row['original_solution_position']):
        raise ValueError('Exact original screened training program required')
    if not reference['all_original_finite_cases_pass']:
        return None,dict(status='reference_not_qualified_for_all_original_finite_cases')
    execution=reference['actual_execution']
    expected_counts={kind:len(getattr(problem,kind)) for kind in ['public_tests','private_tests','generated_tests']}
    if (not execution['all_cases_observed'] or not execution['all_original_finite_cases_pass']
            or execution['counts_by_original_kind']!=expected_counts
            or execution['actual_observed_case_count']!=sum(expected_counts.values())
            or len(execution['original_case_verdicts'])!=sum(expected_counts.values())):
        raise ValueError('Reference must cover every original public/private/generated case without clipping')
    skipped=[]
    for actual in execution['original_case_verdicts']:
        if actual['kind']!='public_tests':continue
        position=actual['original_kind_position']
        if type(position) is not int or not 0<=position<len(problem.public_tests):raise ValueError('Exact original public case position required')
        source=problem.public_tests[position]
        stdout=base64.b64decode(actual['actual_stdout_b64'],validate=True)
        if (actual['process_status']!='completed' or actual['original_input_sha256']!=digest(source.input)
                or actual['original_expected_sha256']!=digest(source.output) or not token_equal(stdout,source.output)):
            raise ValueError('Public trace must reconstruct from actual retained stdout and original public case')
        try:response=stdout.decode('utf-8',errors='strict')
        except UnicodeDecodeError:
            skipped.append(dict(public_case_position=position,reason='actual_stdout_not_UTF8'));continue
        if not response:
            skipped.append(dict(public_case_position=position,reason='empty_stdout_has_no_native_output_target'));continue
        if '```' in row['response'] or '```' in source.input:
            skipped.append(dict(public_case_position=position,reason='original_program_or_input_conflicts_with_prompt_fence'));continue
        prompt=('Predict the exact standard output of this Python 3 program for the given standard input. '
            'Return only the output.\n\nProgram:\n```python\n'+row['response']+'\n```\n\nStandard input:\n```text\n'+source.input+'\n```')
        sequence,reason=native_sequence(tokenizer,prompt,response)
        if sequence is None:
            skipped.append(dict(public_case_position=position,reason=reason));continue
        ids,labels=sequence
        return dict(id=row['source_problem_id']+':public-execution:'+str(row['original_solution_position'])+':'+str(position),
            group_id=row['group_id'],split='train',publisher_split='train',language='Python3_execution_prediction',
            prompt=prompt,response=response,source_problem_id=row['source_problem_id'],parent_original_presentation_id=row['id'],
            original_solution_position=row['original_solution_position'],original_program_sha256=row['original_program_sha256'],
            original_public_case_position=position,original_public_input_sha256=digest(source.input),
            actual_stdout_sha256=hashlib.sha256(stdout).hexdigest(),
            original_public_expected_sha256=digest(source.output),
            target_is_actual_observed_stdout_not_generated_teacher_text=True,
            no_private_or_generated_test_values_in_model_prompt_or_target=True,
            no_additional_neural_parameters_or_teacher=True,human_preferences=False,
            ids=ids,labels=labels),dict(status='first_complete_fitting_public_execution_trace_selected',
                original_public_cases=len(problem.public_tests),earlier_public_case_dispositions=skipped,
                selected_public_case_position=position,all_original_cases_retained_in_raw_source=True)
    return None,dict(status='no_complete_fitting_nonempty_public_execution_trace',
        original_public_cases=len(problem.public_tests),public_case_dispositions=skipped,
        all_original_cases_retained_in_raw_source=True)
