"""Whole published Python examples in the existing OS-confined runtime.

No model generation or gradients. The original comments, retained formatted
target and all example commands are unchanged. Expected output stays in the
parent process. Passing these finite examples is not general correctness.
"""
import ast
import doctest
import hashlib
import json
import math
import builtins
from pathlib import Path
import symtable
import tempfile
from functools import lru_cache

from .standalone_program_v1 import python_policy,run_bounded
from .corpus import file_hash
from .code_quality import instruction_code

ROOT=Path(__file__).resolve().parents[2]
HARNESS=ROOT/'scripts/documented_doctest_raw_harness_v1.py'
PRELUDE='''import math, itertools, collections, functools, typing, re, json, string, struct, base64, binascii, hashlib, ipaddress, operator
from math import *
from itertools import *
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap, namedtuple
from functools import lru_cache, reduce, partial
from typing import *
from datetime import datetime, date, timedelta
from urllib.parse import urlparse, urlsplit, urljoin, quote, unquote, urlencode, parse_qs
'''


def sha(value):return hashlib.sha256(value.encode()).hexdigest()


@lru_cache(maxsize=1)
def declared_names():
    namespace={};exec(PRELUDE,namespace) # Our fixed trusted stdlib prelude, never source code.
    return set(namespace)|set(vars(builtins))|{'__name__','__file__'}


def unresolved_globals(program):
    table=symtable.symtable(program,'<retained-source>','exec');module={s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    pending=[table];needed=set()
    while pending:
        current=pending.pop();pending.extend(current.get_children())
        needed.update(s.get_name() for s in current.get_symbols() if s.is_referenced() and s.is_global())
    return sorted(needed-module-declared_names())


def contract(candidate):
    source=candidate['source']
    if source.get('original_partition')!='train' or source.get('language')!='Python':
        raise ValueError('Only original publisher-train Python functions are eligible')
    program,text=candidate['program'],candidate['doctest_text']
    if (not isinstance(program,str) or not 0<len(program.encode())<=65536 or not isinstance(text,str) or len(text)>20000
            or sha(program)!=candidate['program_sha256'] or sha(text)!=candidate['doctest_text_sha256']
            or text!=source['prompt'] or sha(source['response'])!=candidate['existing_formatted_source_target_sha256']
            or instruction_code(source['response'])!=program):
        raise ValueError('Every unchanged retained code/comment/example byte must reconstruct')
    tree=ast.parse(program);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if len(tree.body)!=1 or len(nodes)!=1 or nodes[0].name!=candidate['function']:
        raise ValueError('One complete actual standalone entry point is required')
    missing=unresolved_globals(program)
    if missing:raise ValueError('Missing undeclared global context: '+', '.join(missing[:12]))
    tests=doctest.DocTestParser().get_examples(text)
    if len(tests)!=candidate['whole_original_example_count'] or not 2<=len(tests)<=128 or any(t.options.get(doctest.SKIP,False) for t in tests):
        raise ValueError('Every original example, at least two and no skipped tests, must execute')
    return tests


def flags_for(example):
    flags=0
    for bit,enabled in example.options.items():
        if enabled:flags|=bit
    return flags


def compare(example,observed):
    checker=doctest.OutputChecker();flags=flags_for(example);exception=observed['exception_only']
    if exception is None:return example.exc_msg is None and checker.check_output(example.want,observed['stdout'],flags)
    if example.exc_msg is None:return False
    if checker.check_output(example.exc_msg,exception,flags):return True
    if flags&doctest.IGNORE_EXCEPTION_DETAIL:
        # The documented flag ignores exception message and module qualifier,
        # retaining the actual exception class name.
        expected=example.exc_msg.split(':',1)[0].split('.')[-1].strip()
        actual=exception.split(':',1)[0].split('.')[-1].strip()
        return expected==actual
    return False


def execute(candidate,seconds=5):
    tests=contract(candidate)
    if not 0<seconds<=30:raise ValueError('An explicit bounded whole-example wall budget is required')
    base=dict(source_id=candidate['source']['id'],program_sha256=candidate['program_sha256'],doctest_text_sha256=candidate['doctest_text_sha256'],
        declared_prelude_sha256=sha(PRELUDE),original_example_count=len(tests),expected_text_not_written_to_worker=True,
        all_original_examples_observed=False,all_original_examples_passed=False)
    with tempfile.TemporaryDirectory(prefix='slm-document-doctest-') as tmp:
        work=Path(tmp);(work/'program.py').write_text(candidate['program']);(work/'prelude.py').write_text(PRELUDE)
        (work/'harness.py').write_bytes(HARNESS.read_bytes())
        settings=dict(program=str(work/'program.py'),prelude=str(work/'prelude.py'),function=candidate['function'],examples=[t.source for t in tests])
        (work/'settings.json').write_text(json.dumps(settings))
        policy=python_policy(work);policy['command'][-1]=str(work/'harness.py');policy['command'].append(str(work/'settings.json'))
        policy['memory_bytes']=512*1024**2;policy['cpu_seconds']=max(1,math.ceil(seconds))
        raw=run_bounded(policy,b'',seconds=seconds,output_bytes=1024**2,python=True)
    execution={k:v for k,v in raw.items() if k not in ('stdout','stderr')};execution['stderr']=raw['stderr'][-2000:].decode(errors='replace')
    if raw['status']!='completed':return dict(**base,status='runtime_or_missing_context_qualification',execution=execution)
    try:
        result=json.loads(raw['stdout']);rows=result['results']
        if (set(result)!={'format','results'} or result['format']!='own-documented-doctest-raw-output-v1' or len(rows)!=len(tests)
                or any(set(r)!={'stdout','exception_only','entry_body_calls'} or not isinstance(r['stdout'],str)
                    or type(r['entry_body_calls']) is not int or r['entry_body_calls']<0
                    or (r['exception_only'] is not None and not isinstance(r['exception_only'],str)) for r in rows)):
            raise ValueError('Every original raw output record required')
    except (ValueError,TypeError,KeyError):return dict(**base,status='incomplete_raw_output_qualification',execution=execution)
    checks=[compare(t,r) for t,r in zip(tests,rows)];calls=sum(r['entry_body_calls'] for r in rows);passed=all(checks) and calls>0
    base.update(all_original_examples_observed=True,all_original_examples_passed=passed)
    status='all_original_examples_passed' if passed else ('published_examples_do_not_exercise_entrypoint' if all(checks) and not calls else 'completed_source_example_mismatch_or_context_qualification')
    return dict(**base,status=status,execution=execution,raw_results=rows,example_passes=checks,passed_examples=sum(checks),entry_body_calls=calls)


def implementation_sources():
    paths=[Path(__file__),HARNESS,ROOT/'src/slm_lab/standalone_program_v1.py',ROOT/'src/slm_lab/native_code.py',
        ROOT/'scripts/python_program_sandbox_v1.py',ROOT/'scripts/native_sandbox.py']
    return {str(p.resolve()):file_hash(p) for p in paths}


def training_example(candidate,observed,tokenizer):
    tests=contract(candidate);source=candidate['source'];raw=observed.get('raw_results',[])
    if (observed.get('source_id')!=source['id'] or observed.get('program_sha256')!=candidate['program_sha256']
            or observed.get('doctest_text_sha256')!=candidate['doctest_text_sha256'] or observed.get('declared_prelude_sha256')!=sha(PRELUDE)
            or observed.get('status')!='all_original_examples_passed' or observed.get('all_original_examples_observed') is not True
            or observed.get('all_original_examples_passed') is not True or len(raw)!=len(tests)
            or observed.get('entry_body_calls',0)<=0 or sum(r['entry_body_calls'] for r in raw)!=observed['entry_body_calls']
            or observed.get('example_passes')!=[True]*len(tests) or not all(compare(t,r) for t,r in zip(tests,raw))):
        raise ValueError('Every actual whole source example and exercised entry-point result must pass before training')
    footer='The surrounding module or class supplies its existing imports, types and helpers. Return only the implementation.'
    if not source['prompt'].endswith(footer):raise ValueError('Known original formatter footer required; never rewrite source comments')
    original_description=source['prompt'][:-len(footer)]
    prompt=('Implement the documented Python function using only its declared surrounding context. '
        'Return the complete function and any necessary helpers.\n\nProvided standard-library context:\n```python\n'+PRELUDE+
        '```\n\nOriginal source description and signature:\n'+original_description)
    response=source['response']
    if any(s in prompt or s in response for s in ('<pad>','<bos>','<eos>')):raise ValueError('Native token marker collision in source')
    prefix=[1]+tokenizer.encode('User: '+prompt+'\nAssistant:').ids;targets=tokenizer.encode(response).ids+[2]
    if len(prefix)+len(targets)>4096:raise ValueError('Whole declared context and original target exceed the native budget')
    row=dict(id='whole-doctest:'+source['id'],group_id=source['group_id'],split='train',publisher_split='train',language='Python',
        prompt=prompt,response=response,parent_source=source,original_source_segment_row=candidate['original_source_segment_row'],
        original_retained_prompt_sha256=sha(source['prompt']),unchanged_retained_target_sha256=sha(response),
        executable_program_sha256=candidate['program_sha256'],declared_prelude_sha256=sha(PRELUDE),
        whole_original_example_count=len(tests),verified_entry_body_calls=observed['entry_body_calls'],
        verification='Every complete original published example passes under the explicitly declared stdlib context and OS confinement',
        machine_execution_feedback=True,human_preferences=False,genuine_RLHF_labels=False,model_generated_examples=False,
        external_model_teacher_used=False,complete_original_formatted_target_unchanged=True,
        finite_examples_not_general_correctness_or_algorithmic_efficiency=True,dataset_admitted_to_training=False)
    return row,prefix+targets,[-100]*len(prefix)+targets
