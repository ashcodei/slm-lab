"""Post-hoc fresh-input diagnostics of all fourteen original train functions.

Numeric specifications are independent of the retained source algorithms.
This is a source-learning diagnostic, not a new independent model benchmark.
All generated programs execute unchanged in the existing confined runtime.
"""
import ast
import doctest
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

from .documented_doctest_v1 import PRELUDE,HARNESS
from .standalone_program_v1 import python_policy,run_bounded

NAMES=['not_gate','is_operand','price_plus_tax','abs_val','expectation','equation','perfect_cube','factorial',
    'increment_score','is_safe','str_eval','digit_sum','is_prime','is_sq']
DONORS={'perfect_cube':'maths/perfect_cube.py:perfect_cube_binary_search',
    'is_prime':'maths/next_prime_number.py:is_prime','is_sq':'maths/perfect_square.py:perfect_square_binary_search'}
DOMAINS=dict(not_gate='binary integer inputs0and1; public examples already exhaust this domain',
    is_operand='single ASCII character; digit membership is0through9',price_plus_tax='finite dyadic prices and nonnegative dyadic tax rates',
    abs_val='finite dyadic numbers',expectation='equal-length dyadic values and nonnegative dyadic probabilities summing to1; empty pair allowed',
    equation='finite dyadic x; mathematical value10minus x squared',perfect_cube='nonnegative integers at most256cubed',
    factorial='integers0through32',increment_score='nonnegative integers at most1000',
    is_safe='nonnegative grid dimensions at most16 and integer coordinates around the boundary',
    str_eval='nonempty ASCII digit strings at most32characters',digit_sum='nonnegative integers at most10to30',
    is_prime='nonnegative integers0through4096',is_sq='nonnegative integers including exact squares around roots2to53')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def primes():
    sieve=[True]*4097;sieve[:2]=[False,False]
    for i in range(2,65):
        if sieve[i]:
            for j in range(i*i,4097,i):sieve[j]=False
    return frozenset(i for i,v in enumerate(sieve) if v)


def expected(name,args):
    if name=='not_gate':return [1,0][args[0]]
    if name=='is_operand':return args[0] in '0123456789'
    if name=='price_plus_tax':return args[0]+args[0]*args[1]
    if name=='abs_val':return math.fabs(args[0])
    if name=='expectation':return math.fsum(v*p for v,p in zip(*args))
    if name=='equation':return 10-math.pow(args[0],2)
    if name=='perfect_cube':return args[0] in {k**3 for k in range(257)}
    if name=='factorial':return math.factorial(args[0])
    if name=='increment_score':return sum(range(1,args[0]+1))
    if name=='is_safe':return (args[0],args[1]) in {(r,c) for r in range(args[2]) for c in range(args[3])}
    if name=='str_eval':return math.prod(map(int,args[0]))
    if name=='digit_sum':
        n=args[0];total=0
        while n:n,d=divmod(n,10);total+=d
        return total
    if name=='is_prime':return args[0] in primes()
    if name=='is_sq':return math.isqrt(args[0])**2==args[0]
    raise ValueError('Every original function needs an explicit bounded numerical domain')


def published_arguments(row):
    name=row['source_candidate']['function'];result=set()
    for example in doctest.DocTestParser().get_examples(row['source_candidate']['original_literal_docstring']):
        for node in ast.walk(ast.parse(example.source)):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id==name and not node.keywords:
                try:args=[ast.literal_eval(arg) for arg in node.args]
                except (ValueError,TypeError):continue
                result.add(digest(args))
    return result


def cases(row):
    source=row['source_candidate'];name=source['function']
    if row['split']!='train' or source['role']!='train' or source['id']!=row['id'] or name not in NAMES:
        raise ValueError('Only the fourteen exact original source-train functions are eligible')
    rng=random.Random('source-semantic-stress-v1:'+row['id']);proposals=[]
    if name=='not_gate':proposals=[[0],[1]]
    elif name=='is_operand':proposals=[[chr(i)] for i in range(32,127)]
    elif name=='factorial':proposals=[[i] for i in range(33)]
    elif name in {'perfect_cube','is_prime','is_sq','increment_score','digit_sum'}:
        proposals=[[n] for n in range(17)]
        if name=='perfect_cube':proposals += [[n] for k in [4,5,7,10,31,64,127,255,256] for n in [k**3-1,k**3,k**3+1] if n<=256**3]
        elif name=='is_sq':proposals += [[n] for k in [17,1001,2**26+1,2**53-1,2**53,2**53+1] for n in [k*k-1,k*k,k*k+1]]
        limit={'perfect_cube':256**3,'is_prime':4096,'is_sq':10**12,'increment_score':1000,'digit_sum':10**30}[name]
        proposals += [[rng.randrange(limit+1)] for _ in range(256)]
    elif name in {'abs_val','equation'}:proposals=[[n/8] for n in [-1023,-64,-9,-1,0,1,9,64,1023]]+[[rng.randrange(-1024,1025)/8] for _ in range(256)]
    elif name=='price_plus_tax':proposals=[[rng.randrange(-1024,1025)/8,rng.randrange(33)/32] for _ in range(256)]
    elif name=='expectation':
        proposals=[[[],[]]]
        for _ in range(256):
            n=rng.randrange(1,9);proposals.append([[rng.randrange(-128,129)/8 for _ in range(n)],[1/n]*n])
        # Powers of two keep every supplied probability exactly dyadic.
        proposals=[args for args in proposals if len(args[0]) in [0,1,2,4,8]]
    elif name=='is_safe':
        proposals=[[r,c,rows,cols] for rows,cols in [(0,0),(0,3),(3,0),(1,1),(2,3),(16,16)]
            for r,c in [(-1,-1),(0,0),(rows-1,cols-1),(rows,cols),(rows-1,cols)]]
        proposals += [[rng.randrange(-2,19),rng.randrange(-2,19),rng.randrange(17),rng.randrange(17)] for _ in range(256)]
    elif name=='str_eval':proposals=[[s] for s in ['0','1','9','10','11','99','0000','10203']]+[
        [''.join(rng.choice('0123456789') for _ in range(rng.randrange(1,33)))] for _ in range(256)]
    original=published_arguments(row);selected=[];seen=set()
    for args in proposals:
        identity=digest(args)
        if identity in seen or identity in original and name!='not_gate':continue
        seen.add(identity);selected.append(dict(case_id=identity,args=args,expected=expected(name,args),
            arguments_not_in_original_public_examples=identity not in original))
        if len(selected)==64:break
    required=2 if name=='not_gate' else 30 if name=='factorial' else 64
    if len(selected)!=required:raise ValueError('Every fixed complete bounded case set must be present: '+name)
    return selected


def matches(expected_value,raw):
    if raw['exception_only'] is not None or raw['entry_body_calls']<1:return False
    try:actual=ast.literal_eval(raw['stdout'].strip())
    except (ValueError,SyntaxError,RecursionError,TypeError):return False
    if type(expected_value) is bool:return type(actual) is bool and actual==expected_value
    if type(expected_value) is int:return type(actual) is int and actual==expected_value
    return type(actual) in {int,float} and math.isfinite(actual) and math.isclose(actual,expected_value,rel_tol=1e-12,abs_tol=1e-12)


def observe(program,name,selected,*,seconds=10):
    if (name not in NAMES or not isinstance(selected,list) or not selected or len(selected)>64
            or not isinstance(program,str) or len(program.encode())>50000 or '\0' in program or not 0<seconds<=30):
        raise ValueError('Explicit complete bounded source-train diagnostic required')
    for case in selected:
        if case['case_id']!=digest(case['args']) or case['expected']!=expected(name,case['args']):
            raise ValueError('Independent fixed specification/input identity must reconstruct')
    base=dict(program_sha256=hashlib.sha256(program.encode()).hexdigest(),cases=len(selected),actual_program_launched=False,
        syntax_valid=False,all_cases_observed=False,all_cases_passed=False,case_passes=[False]*len(selected),
        expected_outputs_not_sent_to_worker=True,program_not_repaired=True,held_benchmark=False)
    try:ast.parse(program)
    except (SyntaxError,ValueError,RecursionError) as exc:return dict(base,status='syntax_error',diagnostic=str(exc)[:500])
    base['syntax_valid']=True
    with tempfile.TemporaryDirectory(prefix='slm-function-semantic-stress-') as tmp:
        work=Path(tmp);(work/'program.py').write_text(program);(work/'prelude.py').write_text(PRELUDE);(work/'harness.py').write_bytes(HARNESS.read_bytes())
        settings=dict(program=str(work/'program.py'),prelude=str(work/'prelude.py'),function=name,
            examples=[name+'(*'+repr(case['args'])+')\n' for case in selected])
        (work/'settings.json').write_text(json.dumps(settings));policy=python_policy(work)
        policy['command'][-1]=str(work/'harness.py');policy['command'].append(str(work/'settings.json'))
        policy['memory_bytes']=512*1024**2;policy['cpu_seconds']=math.ceil(seconds)
        raw=run_bounded(policy,b'',seconds=seconds,output_bytes=1024**2,python=True)
    base['actual_program_launched']=True;base['execution']={k:v for k,v in raw.items() if k not in ['stdout','stderr']}
    base['execution']['stderr']=raw['stderr'][-1000:].decode(errors='replace')
    if raw['status']!='completed':return dict(base,status='runtime_or_resource_failure')
    try:
        result=json.loads(raw['stdout']);results=result['results']
        if (set(result)!={'format','results'} or result['format']!='own-documented-doctest-raw-output-v1' or len(results)!=len(selected)
                or any(set(r)!={'stdout','exception_only','entry_body_calls'} or not isinstance(r['stdout'],str)
                    or type(r['entry_body_calls']) is not int or r['entry_body_calls']<0
                    or r['exception_only'] is not None and not isinstance(r['exception_only'],str) for r in results)):
            raise ValueError('Every complete raw result must reconstruct')
    except (ValueError,TypeError,KeyError):return dict(base,status='incomplete_raw_observations')
    checks=[matches(case['expected'],r) for case,r in zip(selected,results)]
    return dict(base,status='all_cases_passed' if all(checks) else 'semantic_case_failure',all_cases_observed=True,
        all_cases_passed=all(checks),case_passes=checks,raw_results=results)


def public_alternative(row,donor):
    """Keep every original donor byte; add only a static argument-forwarding alias."""
    name=row['source_candidate']['function'];other=donor['source_candidate']
    for source in [row,donor]:
        if (source['split']!='train' or source['source_candidate']['role']!='train'
                or source['response']!=source['source_candidate']['original_function_program']):
            raise ValueError('Only complete original admitted public train implementations may provide a contrast')
    if DONORS.get(name)!=other['original_source_git_path']+':'+other['function']:
        raise ValueError('The fixed independent public implementation must match the diagnosed target')
    target=ast.parse(row['response']).body[0];original=ast.parse(donor['response']).body
    if (not isinstance(target,ast.FunctionDef) or len(original)!=1 or not isinstance(original[0],ast.FunctionDef)
            or len(target.args.args)!=1 or len(original[0].args.args)!=1):raise ValueError('Exact single-argument public functions required')
    if name==other['function']:return donor['response']
    doc=ast.get_docstring(target,clean=False);prefix='def '+name+'('+target.args.args[0].arg+'):\n'
    if doc is not None:prefix+='    '+repr(doc)+'\n'
    alias=prefix+'    return '+other['function']+'('+target.args.args[0].arg+')\n'
    return donor['response']+'\n\n'+alias
