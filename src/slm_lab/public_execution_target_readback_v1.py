"""Independent reconstruction of public-only native execution supervision.

Uses retained stdout and original public cases directly. Does not import the
candidate formatter or its token-sequence helper. Full original reference
output verification is a separate prerequisite for admission.
"""
import base64
import hashlib


def rebuild(row,problem,reference,tokenizer):
    sha=lambda text:hashlib.sha256(text.encode()).hexdigest()
    if (row['split']!='train' or row['publisher_split']!='train' or row['language']!='Python3'
            or reference['original_problem_id']!=row['source_problem_id']
            or reference['original_program_sha256']!=sha(row['response'])
            or row['original_program_sha256']!=sha(row['response'])
            or reference['original_solution_position']!=row['original_solution_position']):
        raise ValueError('Exact whole screened training source and original program required')
    if reference['all_original_finite_cases_pass'] is not True:return None
    execution=reference['actual_execution']
    counts={k:len(getattr(problem,k)) for k in ('public_tests','private_tests','generated_tests')}
    if (execution['counts_by_original_kind']!=counts or execution['actual_observed_case_count']!=sum(counts.values())
            or len(execution['original_case_verdicts'])!=sum(counts.values())
            or execution['all_cases_observed'] is not True or execution['all_original_finite_cases_pass'] is not True):
        raise ValueError('Every complete original case is required before constructing public-only targets')
    for position,source in enumerate(problem.public_tests):
        actual=execution['original_case_verdicts'][position]
        if (actual['kind']!='public_tests' or actual['original_kind_position']!=position
                or actual['process_status']!='completed' or actual['original_input_sha256']!=sha(source.input)
                or actual['original_expected_sha256']!=sha(source.output)):
            raise ValueError('Actual observed original public case identity differs')
        stdout=base64.b64decode(actual['actual_stdout_b64'],validate=True)
        if stdout.split()!=source.output.encode().split():raise ValueError('Actual public stdout does not match the original token protocol')
        try:response=stdout.decode('utf-8',errors='strict')
        except UnicodeDecodeError:continue
        if not response or '```' in row['response'] or '```' in source.input:continue
        prompt=('Predict the exact standard output of this Python 3 program for the given standard input. '
            'Return only the output.\n\nProgram:\n```python\n'+row['response']+'\n```\n\nStandard input:\n```text\n'+source.input+'\n```')
        prefix=[1]+tokenizer.encode('User: '+prompt+'\nAssistant:').ids
        target=tokenizer.encode(response).ids
        ids=prefix+target+[2]
        if not target or any(v<4 or v>=32768 for v in prefix[1:]+target) or len(ids)>4096:continue
        return dict(prompt=prompt,response=response,ids=ids,labels=[-100]*len(prefix)+target+[2],
            original_public_case_position=position,original_public_input_sha256=sha(source.input),
            actual_stdout_sha256=hashlib.sha256(stdout).hexdigest(),original_public_expected_sha256=sha(source.output))
    return None
