"""Generate one unchanged program per held question, then score sealed outputs."""
import ast
from collections import Counter
import hashlib
import time

import torch


QUESTION_FIELDS = {'id','problem_id','group_id','split','language','target_runtime','prompt',
    'original_html_sha256','source','license','use_for_training_or_preferences','complete_prompt_tokens'}


def validate_question(row):
    if (set(row) != QUESTION_FIELDS or row['split'] != 'test'
        or row['use_for_training_or_preferences'] is not False
        or row['language'] not in {'C++','Python'} or not row['prompt'].strip()
        or type(row['complete_prompt_tokens']) is not int or row['complete_prompt_tokens'] < 2):
        raise ValueError('Only the complete predeclared public held question may enter generation')


def generate_record(model, tokenizer, encoder, question, budget, device, amp):
    """No case/reference parameter: model inputs are public questions only."""
    validate_question(question)
    if type(budget) is not int or budget < 1:
        raise ValueError('Positive declared generation budget required')
    prompt = [encoder.bos] + encoder.tokens(encoder.prompt(question))
    if len(prompt) != question['complete_prompt_tokens'] or len(prompt) + budget > model.config.max_text_length:
        raise ValueError('Whole original prompt and complete generation budget must fit; no clipping')
    ids = torch.tensor([prompt], device=device)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    tick = time.monotonic()
    with torch.inference_mode(), amp():
        generated = model.generate(ids, max_new_tokens=budget, temperature=0.,
                                   eos_token_id=encoder.eos, use_cache=True)
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)
    if (generated.ndim != 2 or generated.shape[0] != 1 or generated.shape[1] <= len(prompt)
        or generated.shape[1] > len(prompt) + budget or generated[0,:len(prompt)].tolist() != prompt):
        raise ValueError('Actual generated prefix/budget differs from requested model input')
    raw = generated[0,len(prompt):].tolist()
    terminal_eos = raw[-1] == encoder.eos
    if encoder.eos in raw[:-1]:
        raise ValueError('Generated output continued after its first declared stop token')
    if not terminal_eos and len(raw) != budget:
        raise ValueError('Generation stopped without EOS before the declared token budget')
    body = raw[:-1] if terminal_eos else raw
    # Preserve intermediate BOS/PAD and other special tokens; remove only the actual final stop.
    text = tokenizer.decode(body, skip_special_tokens=False)
    return {'id':question['id'],'problem_id':question['problem_id'],'group_id':question['group_id'],
        'split':'test','language':question['language'],'prompt_tokens':len(prompt),
        'prompt_input_ids_sha256':hashlib.sha256(ids.detach().cpu().numpy().astype('<i8').tobytes()).hexdigest(),
        'generated_token_ids':raw,'program_text':text,'program_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'generated_tokens_including_stop':len(raw),'terminal_eos':terminal_eos,
        'stop_reason':'eos' if terminal_eos else 'token_budget','generation_budget':budget,
        'literal_reserved_body_token_count':sum(value in {encoder.pad,encoder.bos,encoder.eos} for value in body),
        'generation_seconds':time.monotonic()-tick,'temperature':0.,'sampling_or_repairs':False}


def score_record(question, generation, cases, tools, assessor):
    validate_question(question)
    identity = ('id','problem_id','group_id','split','language')
    if (any(question[key] != generation[key] for key in identity) or not cases
        or any(case['problem_id'] != question['problem_id'] or case['split'] != 'test'
               or case['use_for_training_or_preferences'] for case in cases)):
        raise ValueError('Sealed completion and exact complete held case population required')
    code = generation['program_text'].encode()
    if hashlib.sha256(code).hexdigest() != generation['program_sha256']:
        raise ValueError('Sealed model-generated program changed before execution')
    syntax = None
    syntax_error = None
    if not code or len(code) > 32768 or b'\0' in code:
        execution = {'qualification':'generated_program_outside_execution_profile','cases':[]}
    else:
        if question['language'] == 'Python':
            try:
                ast.parse(generation['program_text'])
                syntax = True
            except (SyntaxError, ValueError, RecursionError) as error:
                syntax,syntax_error = False,str(error)[:1000]
        execution = assessor(code,question['language'],cases,tools)
    if question['language'] == 'C++':
        syntax = execution.get('compilation',{}).get('status') == 'completed'
    observed = {row['case_id']:row for row in execution['cases']}
    expected_ids = {case['id'] for case in cases}
    if len(observed) != len(execution['cases']) or not observed.keys() <= expected_ids:
        raise ValueError('Executor returned duplicate/foreign case results')
    if execution['cases'] and set(observed) != expected_ids:
        raise ValueError('A partial executable-case result cannot become a completed score')
    counts = Counter(case['kind'] for case in cases)
    passed = {kind:sum(bool(observed.get(case['id'],{}).get('oracle_passed'))
        for case in cases if case['kind']==kind) for kind in counts}
    functional = all(passed[kind]==count for kind,count in counts.items() if kind!='stress')
    stress = passed.get('stress',0)==counts['stress'] if counts['stress'] else None
    return {**{key:question[key] for key in identity},'program_sha256':generation['program_sha256'],
        'python_syntax_or_CPP_compilation_passed':syntax,'python_syntax_error':syntax_error,
        'case_counts':dict(counts),'passed_case_counts':passed,
        'functional_passed':functional,'bounded_stress_passed':stress,
        'overall_passed':functional and stress is not False,'execution':execution,
        'primary_program_is_raw_model_output':True,'generation_budget':generation['generation_budget'],
        'generation_stop_reason':generation['stop_reason']}


def summarize(questions, generations, results):
    expected = [row['id'] for row in questions]
    if (not expected or len(set(expected)) != len(expected)
        or [row['id'] for row in generations] != expected or [row['id'] for row in results] != expected):
        raise ValueError('Every predeclared question and its single unchanged completion/result must be present')
    summary = {'questions':len(results),'tasks':len({r['problem_id'] for r in questions}),
        'sealed_components':len({r['group_id'] for r in questions}),
        'functional_correct':sum(r['functional_passed'] for r in results),
        'overall_correct_including_defined_stress':sum(r['overall_passed'] for r in results),
        'syntax_or_compilation_passed':sum(r['python_syntax_or_CPP_compilation_passed'] is True for r in results),
        'stress_eligible_questions':sum(r['bounded_stress_passed'] is not None for r in results),
        'bounded_stress_passed':sum(r['bounded_stress_passed'] is True for r in results),
        'generation_budget_exhausted_questions':sum(r['stop_reason']=='token_budget' for r in generations),
        'case_counts':dict(sum((Counter(r['case_counts']) for r in results),Counter())),
        'passed_case_counts':dict(sum((Counter(r['passed_case_counts']) for r in results),Counter()))}
    summary['functional_pass_at_1']=summary['functional_correct']/len(results)
    summary['overall_pass_at_1_with_stress_where_defined']=summary['overall_correct_including_defined_stress']/len(results)
    summary['bounded_stress_pass_rate']=summary['bounded_stress_passed']/summary['stress_eligible_questions'] if summary['stress_eligible_questions'] else None
    summary['per_language']={language:{'questions':sum(r['language']==language for r in results),
        'functional_correct':sum(r['language']==language and r['functional_passed'] for r in results),
        'overall_correct':sum(r['language']==language and r['overall_passed'] for r in results)} for language in sorted({r['language'] for r in results})}
    return summary
