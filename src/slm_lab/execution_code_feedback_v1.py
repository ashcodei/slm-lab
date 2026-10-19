"""Deterministic original-Java mutations and conservative published-test labels.

These machine labels concern finite published cases. They are not human
preferences, universally correct programs, or broad model capability scores.
"""
import hashlib
import json
import random
import re

from .exercise_java_runner import BLOCK, extract_files, inventory


# Comments and all Java string/character forms are opaque to mutation. The
# scanner consumes longest operators/numerals/identifiers before single chars.
LEX = re.compile(r'(?P<opaque>"""(?:\\[\s\S]|(?!""")[\s\S])*"""|"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\'|//[^\r\n]*|/\*[\s\S]*?\*/)'
    r'|(?P<number>0[xX](?:[0-9a-fA-F_]+(?:\.[0-9a-fA-F_]*)?|\.[0-9a-fA-F_]+)(?:[pP][+-]?[0-9_]+[fFdD]?|[lL]?)'
    r'|0[bB][01_]+[lL]?|(?:[0-9][0-9_]*\.[0-9_]*|\.[0-9][0-9_]*)(?:[eE][+-]?[0-9_]+)?[fFdD]?'
    r'|[0-9][0-9_]*[eE][+-]?[0-9_]+[fFdD]?|[0-9][0-9_]*[fFdDlL]?)'
    r'|(?P<word>[A-Za-z_$][\w$]*)|(?P<op>>>>=|>>=|<<=|>>>|>>|<<|\+\+|--|&&|\|\||==|!=|<=|>=|->|::|[+*/%<>=!&|^-]=?)|(?P<other>[\s\S])')
OPS = {'true': ('false','boolean'), 'false': ('true','boolean'),
    '&&': ('||','boolean'), '||': ('&&','boolean'),
    '==': ('!=','comparison'), '!=': ('==','comparison'), '<': ('<=','comparison'),
    '<=': ('<','comparison'), '>': ('>=','comparison'), '>=': ('>','comparison'),
    '+': ('-','arithmetic'), '-': ('+','arithmetic'), '*': ('+','arithmetic'),
    '0': ('1','integer'), '1': ('0','integer')}
ASSERTION = re.compile(r'TEST_FAILURE: (?:java\.lang\.AssertionError|org\.opentest4j\.AssertionFailedError|org\.junit\.ComparisonFailure|junit\.framework\.(?:AssertionFailedError|ComparisonFailure))\b')


def mutations(response, source_id, count=4, seed=930031):
    if type(count) is not int or count<1 or type(seed) is not int:
        raise ValueError('Explicit positive mutation budget and deterministic seed required')
    files=extract_files(response)
    choices=[]
    for block in BLOCK.finditer(response):
        name=block.group(1);body=block.group(2);offset=block.start(2)
        if files[name]!=body:raise ValueError('Exact original complete Java body required')
        # Java preprocesses Unicode escapes before tokenization. Conservatively
        # qualify such files instead of confusing escaped delimiters with code.
        if re.search(r'\\u+[0-9a-fA-F]{4}',body):continue
        for token in LEX.finditer(body):
            if token.lastgroup=='opaque' or token.group() not in OPS:continue
            old=token.group();new,family=OPS[old]
            choices.append(dict(file=name,start=offset+token.start(),end=offset+token.end(),
                old=old,new=new,family=family))
    stable=int.from_bytes(hashlib.sha256(source_id.encode()).digest()[:8],'big')
    rng=random.Random(seed+stable);rng.shuffle(choices)
    selected=[]
    for family in ['boolean','comparison','arithmetic','integer']:
        match=next((c for c in choices if c['family']==family),None)
        if match is not None and len(selected)<count:selected.append(match)
    for c in choices:
        if len(selected)==count:break
        if c not in selected:selected.append(c)
    result=[]
    for patch in selected:
        other=response[:patch['start']]+patch['new']+response[patch['end']:]
        extract_files(other,files)
        result.append(dict(patch=patch,response=other,response_sha256=hashlib.sha256(other.encode()).hexdigest()))
    return result


def classify(observed, reference=None):
    """Do not convert process, inventory or compiler failures to preferences."""
    if not observed.get('compilation_passed'):
        return 'compiler_or_environment_qualification'
    runtime=observed.get('execution',{})
    if runtime.get('returncode')!=0 or runtime.get('limit') is not None:
        return 'runtime_or_environment_qualification'
    raw=observed.get('raw_inventory_output')
    if not isinstance(raw,str):return 'incomplete_test_inventory_qualification'
    try:measured=inventory(raw.encode())
    except (ValueError,TypeError,KeyError,json.JSONDecodeError):return 'incomplete_test_inventory_qualification'
    if measured!=observed.get('inventory') or hashlib.sha256(raw.encode()).hexdigest()!=observed.get('report_sha256'):
        raise ValueError('Complete raw Java inventory/hash differs from its parsed evidence')
    if (measured['skipped'] or measured['container_errors'] or measured['started_cases']!=measured['cases']
            or sorted(measured['passed_cases']+measured['failed'])!=measured['cases']):
        return 'incomplete_test_inventory_qualification'
    if reference is not None and measured['cases']!=reference['cases']:
        return 'changed_case_population_qualification'
    if measured['all_passed']:
        if observed.get('published_tests_passed') is not True:raise ValueError('Complete passing test flags differ')
        return 'all_finite_published_cases_pass'
    if reference is None:return 'original_reference_test_failure_qualification'
    if measured['failed'] and ASSERTION.search(runtime.get('diagnostic','')):
        if observed.get('published_tests_passed') is not False:raise ValueError('Failed test flags differ')
        return 'completed_published_assertion_counterexample'
    return 'test_failure_without_retained_assertion_qualification'


def sequence(tokenizer,prompt,response,limit=4096):
    prefix=[tokenizer.token_to_id('<bos>')]+tokenizer.encode(prompt).ids
    body=tokenizer.encode(response).ids+[tokenizer.token_to_id('<eos>')]
    if any(v is None for v in prefix+body) or len(prefix)+len(body)>limit:
        raise ValueError('Complete original question and program/EOS must fit without clipping')
    return prefix+body,[-100]*len(prefix)+body
