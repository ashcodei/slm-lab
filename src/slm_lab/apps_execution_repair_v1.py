"""Complete public-program repair candidates with actual finite counterexamples.

The correct target is always an unchanged original human program. These are
machine execution signals, not human preferences or model-generated reasoning.
"""
import hashlib
import io
import json
import random
import tokenize

from .apps_published_cases_v2 import PRELUDE, compare

OPS = {'True':('False','boolean'), 'False':('True','boolean'),
    'and':('or','boolean'), 'or':('and','boolean'),
    '==':('!=','comparison'), '!=':('==','comparison'),
    '<':('<=','comparison'), '<=':('<','comparison'),
    '>':('>=','comparison'), '>=':('>','comparison'),
    '+':('-','arithmetic'), '-':('+','arithmetic'), '*':('+','arithmetic'),
    '0':('1','integer'), '1':('0','integer')}


def mutations(program, source_id, count=4, seed=932021):
    if not isinstance(program,str) or not isinstance(source_id,str) or not source_id or type(count) is not int or not 1<=count<=4:
        raise ValueError('Complete original source and explicit maximum-four mutation budget required')
    starts = [0]
    for line in program.splitlines(keepends=True):
        starts.append(starts[-1]+len(line))
    candidates = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(program).readline))
    except (tokenize.TokenError,IndentationError,SyntaxError):
        return []
    for token in tokens:
        if token.type not in {tokenize.OP,tokenize.NAME,tokenize.NUMBER} or token.string not in OPS:
            continue
        old = token.string
        new, family = OPS[old]
        start = starts[token.start[0]-1]+token.start[1]
        end = starts[token.end[0]-1]+token.end[1]
        if program[start:end] != old:
            raise ValueError('Actual Unicode/token source offsets differ')
        candidates.append(dict(start=start,end=end,old=old,new=new,family=family))
    rng = random.Random(seed+int.from_bytes(hashlib.sha256(source_id.encode()).digest()[:8],'big'))
    rng.shuffle(candidates)
    selected = []
    for family in ['boolean','comparison','arithmetic','integer']:
        first = next((p for p in candidates if p['family']==family),None)
        if first is not None and len(selected)<count:
            selected.append(first)
    for patch in candidates:
        if len(selected)>=count:
            break
        if patch not in selected:
            selected.append(patch)
    result = []
    for ordinal, patch in enumerate(selected):
        changed = program[:patch['start']]+patch['new']+program[patch['end']:]
        if changed == program:
            raise ValueError('Actual nonempty source edit required')
        result.append(dict(ordinal=ordinal,patch=patch,program=changed,
            program_sha256=hashlib.sha256(changed.encode()).hexdigest()))
    return result


def counterexample(positive, negative, cases):
    if (positive.get('status')!='all_finite_published_cases_pass' or not positive.get('all_cases_observed')
            or not positive.get('all_finite_published_cases_pass')):
        raise ValueError('Actual full-case passing original human source required')
    if (negative.get('status')!='completed_published_case_mismatch' or not negative.get('all_cases_observed')
            or not negative.get('syntax_valid') or negative.get('execution',{}).get('status')!='completed'):
        return None
    for observed in [positive,negative]:
        if (observed['original_tests_sha256']!=cases['original_tests_sha256']
                or observed['total_cases']!=cases['total_cases'] or observed['mode']!=cases['mode']
                or len(observed['raw_results']['results'])!=cases['total_cases']):
            raise ValueError('Every unchanged original case/result required for a preference or repair')
        flags = [compare(row['value'],expected,cases['mode'])
            for row,expected in zip(observed['raw_results']['results'],cases['outputs'])]
        if flags!=observed['case_passes'] or sum(flags)!=observed['passed_cases']:
            raise ValueError('Actual raw source comparisons differ from retained evidence')
    if not all(positive['case_passes']) or all(negative['case_passes']):
        raise ValueError('Actual passing original and concrete failed value required')
    position = negative['case_passes'].index(False)
    return dict(case_index=position,complete_original_case_count=cases['total_cases'],
        mode=cases['mode'],function=cases['function'],input=cases['inputs'][position],
        original_expected=cases['outputs'][position],observed=negative['raw_results']['results'][position]['value'],
        original_tests_sha256=cases['original_tests_sha256'])


def repair_example(source_row, question, starter, original, mutation, witness, tokenizer):
    # Reject held roles before inspecting any code, labels, or tokenizer.
    if source_row.get('split')!='train' or source_row.get('publisher_split')!='train':
        raise ValueError('Only explicitly admitted original publisher-train roles can produce repair targets')
    if not witness or witness['original_tests_sha256']!=source_row['original_tests_sha256']:
        raise ValueError('Concrete complete-source training counterexample required')
    if (hashlib.sha256(original.encode()).hexdigest()!=source_row['original_source_body_sha256']
            or hashlib.sha256(question.encode()).hexdigest()!=source_row['original_question_sha256']):
        raise ValueError('Complete unchanged original human target and question bytes required')
    patch = mutation['patch']
    if (patch['old'] not in OPS or OPS[patch['old']]!=(patch['new'],patch['family'])
            or type(patch['start']) is not int or type(patch['end']) is not int
            or not 0<=patch['start']<patch['end']<=len(original)
            or type(mutation['ordinal']) is not int or not 0<=mutation['ordinal']<4):
        raise ValueError('Exact declared one-token mutation required')
    expected = original[:mutation['patch']['start']]+mutation['patch']['new']+original[mutation['patch']['end']:]
    if (original[mutation['patch']['start']:mutation['patch']['end']]!=mutation['patch']['old']
            or expected!=mutation['program'] or hashlib.sha256(expected.encode()).hexdigest()!=mutation['program_sha256']):
        raise ValueError('Exactly one retained original token edit required')
    interface = ('Read standard input and write standard output.' if witness['mode']=='stdin' else
        'Use the original function or Solution method named '+witness['function']+'.')
    prompt = ('Fix the following Python program to solve the original problem. Return only the complete corrected code.\n\n'
        'Original problem:\n'+question+'\n\nRequired interface: '+interface)
    if starter:
        prompt += '\nOriginal starter code:\n'+starter
    prompt += ('\n\nThe runtime supplies this standard-library prelude:\n```python\n'+PRELUDE+
        '```\n\nProgram to repair:\n```python\n'+mutation['program']+
        '\n```\n\nA published training case produced a wrong result (JSON):\n'+
        json.dumps(dict(input=witness['input'],expected=witness['original_expected'],actual=witness['observed']),
            ensure_ascii=False,sort_keys=True,separators=(',',':')))
    reasons = []
    if any(marker in prompt or marker in original for marker in ['<pad>','<bos>','<eos>']):
        reasons.append('reserved_token_literal_requires_review')
    prefix = [tokenizer.token_to_id('<bos>')]+tokenizer.encode('User: '+prompt+'\nAssistant:').ids
    body = tokenizer.encode(original).ids+[tokenizer.token_to_id('<eos>')]
    if len(prefix)+len(body)>4096:
        reasons.append('complete_repair_context_exceeds4096')
    identity = source_row['id']+':execution-repair-v1:'+str(mutation['ordinal'])
    row = dict(id=identity,split='train',publisher_split='train',language='Python',
        group_id=source_row['group_id'],source_problem_id=source_row['source_problem_id'],
        original_source_program_id=source_row['id'],original_solution_index=source_row['original_solution_index'],
        prompt=prompt,response=original,task='repair_code_with_published_training_counterexample',
        original_question_sha256=source_row['original_question_sha256'],
        original_source_body_sha256=hashlib.sha256(original.encode()).hexdigest(),
        original_tests_sha256=source_row['original_tests_sha256'],mutation=mutation,witness=witness,
        original_target_program_bytes_modified=False,external_model_teacher_used=False,human_preferences=False,
        feedback_origin='Machine execution of a deterministic single-token mutation on all original published training cases',
        finite_source_cases_not_algorithmic_correctness_proof=True,dataset_admitted_to_training=False,
        complete_context_tokens=len(prefix)+len(body),complete_response_targets=len(body),review_reasons=reasons)
    return row,prefix+body,[-100]*len(prefix)+body
