"""Explicit finite contracts across public train implementations.

Only exact reviewed source paths/interfaces are grouped. Equal function names
never imply equivalent behavior. Unsupported functions remain in the inventory;
the original training data and every held role remain untouched.
"""
import ast
import doctest
from collections import Counter
from functools import lru_cache
import ipaddress
import json
import math
from pathlib import Path
import random
import statistics
import tempfile

from .documented_doctest_v1 import PRELUDE,HARNESS
from .function_semantic_stress_v1 import digest,expected as old_expected,published_arguments
from .standalone_program_v1 import python_policy,run_bounded

ROOT=Path(__file__).resolve().parents[2]
OLD_CASES=ROOT/'reports/function-semantic-stress-CPU-v1/fixed-cases.json'


@lru_cache(maxsize=1)
def registry():
    result={}
    def add(family,names):
        for path,name in names:
            key=path+':'+name
            if key in result:raise ValueError('One exact public source can have only one contract')
            result[key]=family
    # Exact fourteen source paths come from their frozen original records.
    originals=json.loads((ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/all14-original-source-train-records.json').read_text())
    for item in originals:
        source=item['row']['source_candidate'];add(source['function'],[(source['original_source_git_path'],source['function'])])
    for path in ['data_structures/hashing/number_theory/prime_numbers.py','maths/next_prime_number.py','maths/prime_check.py',
            'maths/primelib.py','project_euler/problem_007/sol1.py','project_euler/problem_007/sol3.py','project_euler/problem_010/sol1.py',
            'project_euler/problem_027/sol1.py','project_euler/problem_037/sol1.py','project_euler/problem_049/sol1.py']:
        add('is_prime',[(path,'is_prime')])
    add('factorial',[(p,'factorial') for p in ['data_structures/binary_tree/number_of_possible_binary_trees.py','maths/factorial.py','maths/primelib.py']])
    add('factorial',[('maths/factorial.py','factorial_recursive')])
    add('perfect_cube',[('maths/perfect_cube.py','perfect_cube_binary_search')])
    add('is_sq',[('maths/perfect_square.py','perfect_square'),('maths/perfect_square.py','perfect_square_binary_search')])
    add('integer_root',[('maths/integer_square_root.py','integer_square_root')])
    add('parity',[('bit_manipulation/parity.py','parity')])
    add('digital_root',[('project_euler/problem_060/sol1.py','sum_digits')])
    add('digit_sum',[('project_euler/problem_065/sol1.py','sum_digits')])
    add('palindrome_int',[('maths/is_int_palindrome.py','is_int_palindrome'),*[(p,'is_palindrome') for p in
        ['project_euler/problem_036/sol1.py','project_euler/problem_055/sol1.py','project_euler/problem_125/sol1.py']]])
    add('prime_factors',[(p,'prime_factors') for p in ['maths/basic_maths.py','maths/prime_factors.py']])
    add('unique_factors_list',[('maths/prime_factors.py','unique_prime_factors')])
    add('unique_factors_set',[('project_euler/problem_047/sol1.py','unique_prime_factors')])
    add('mean',[('maths/average_mean.py','mean')]);add('median',[('maths/average_median.py','median')]);add('mode',[('maths/average_mode.py','mode')])
    add('ipv4_integer',[('conversions/ipv4_conversion.py',n) for n in ['ipv4_to_decimal','alt_ipv4_to_decimal']])
    add('ipv4_string',[('conversions/ipv4_conversion.py','decimal_to_ipv4')])
    add('sort_nonnegative',[('sorts/bead_sort.py','bead_sort')])
    add('sort_int',[(p,n) for p,n in [('sorts/counting_sort.py','counting_sort'),('sorts/pigeon_sort.py','pigeon_sort'),('sorts/flash_sort.py','flash_sort')]])
    add('sort_inplace',[('sorts/pigeonhole_sort.py','pigeonhole_sort')])
    add('search_first',[('searches/binary_search.py','binary_search'),('searches/linear_search.py','linear_search')])
    add('search_bool',[('searches/simple_binary_search.py','binary_search')])
    add('search_unique_optional',[('searches/interpolation_search.py','interpolation_search')])
    add('search_first_optional',[('searches/sentinel_linear_search.py','sentinel_linear_search')])
    add('hamming',[('strings/hamming_distance.py','hamming_distance')])
    add('pattern',[('strings/naive_string_search.py','naive_pattern_search')])
    add('normalized_palindrome',[('strings/palindrome.py','is_palindrome_ignore_case_and_spaces')])
    return result


def identity(row):return row['source_candidate']['original_source_git_path']+':'+row['source_candidate']['function']


def family(row):
    source=row['source_candidate']
    if row['split']!='train' or source['role']!='train' or source['id']!=row['id']:
        raise ValueError('Only original sealed train roles may supply qualification or future correction targets')
    return registry().get(identity(row))


@lru_cache(maxsize=1)
def old_arguments():return {c['function']:[v['args'] for v in c['cases']] for c in json.loads(OLD_CASES.read_text())}


def arguments(profile):
    if profile in old_arguments():return old_arguments()[profile]
    if profile=='digital_root':return old_arguments()['digit_sum']
    rng=random.Random('public-contract-census-v1:'+profile)
    if profile=='integer_root':return [[n] for n in [0,1,2,3,4,8,9,15,16,17,2**106-1,2**106,2**106+1]]+[[rng.randrange(10**30)] for _ in range(51)]
    if profile=='parity':return [[n] for n in [0,1,2,3,255,256,2**128-1,2**128]]+[[rng.getrandbits(128)] for _ in range(56)]
    if profile=='palindrome_int':
        values=[0,1,10,11,100,101,1001,1010,12321,12345]
        values += [int((s:=str(rng.randrange(1,10**8)))+s[::-1]) for _ in range(27)]
        values += [rng.randrange(10**16) for _ in range(27)]
        return [[n] for n in values]
    if profile in ['prime_factors','unique_factors_list','unique_factors_set']:
        return [[n] for n in [1,2,3,4,6,8,12,16,31,64,127,256]]+[[rng.randrange(1,4097)] for _ in range(52)]
    if profile in ['mean','median','mode','sort_nonnegative','sort_int','sort_inplace']:
        arrays=[[0],[1,0],[3,1,3,2],[5,5,5],[9,8,7,6],list(range(16))]
        if profile not in ['mean','median']:arrays.insert(0,[])
        while len(arrays)<64:
            low=0 if profile=='sort_nonnegative' else -24
            arrays.append([rng.randrange(low,25) for _ in range(rng.randrange(1,33))])
        return [[a] for a in arrays]
    if profile in ['ipv4_integer','ipv4_string']:
        nums=[0,1,255,256,65535,65536,2**31,2**32-1]+[rng.randrange(2**32) for _ in range(56)]
        return [[str(ipaddress.IPv4Address(n)) if profile=='ipv4_integer' else n] for n in nums]
    if profile.startswith('search_'):
        result=[[[],0],[[1],1],[[1],2],[[1,2,2,2,3],2],[[0,1,2],-1],[[0,1,2],3]]
        if profile=='search_unique_optional':result[3]=[[1,2,3],2]
        while len(result)<64:
            a=sorted(rng.randrange(-24,25) for _ in range(rng.randrange(33)))
            if profile=='search_unique_optional':a=sorted(set(a))
            result.append([a,rng.randrange(-25,26)])
        return result
    if profile=='hamming':
        result=[['',''],['abc','abc'],['abc','xyz'],['0101','1010']]
        while len(result)<64:
            n=rng.randrange(33);result.append([''.join(rng.choice('abc012') for _ in range(n)) for _ in range(2)])
        return result
    if profile in ['pattern','normalized_palindrome']:
        result=[['',''],['aaa','a'],['aaa','aa'],['abababa','aba'],['xyz','longer']]
        if profile=='normalized_palindrome':result=[[''],['a'],['aa'],['ab'],['A, b! a'],['Never odd or even']]
        while len(result)<64:
            text=''.join(rng.choice('abAB01 ,.!?') for _ in range(rng.randrange(33)))
            result.append([text] if profile=='normalized_palindrome' else [text,''.join(rng.choice('abAB01') for _ in range(rng.randrange(1,9)))])
        return result
    raise ValueError('Every supported explicit finite contract needs its complete argument list')


def expected(profile,args):
    if profile in old_arguments():return old_expected(profile,args)
    value=args[0]
    if profile=='digital_root':return 0 if value==0 else 1+(value-1)%9
    if profile=='integer_root':return math.isqrt(value)
    if profile=='parity':return int(value.bit_count()%2)
    if profile=='palindrome_int':
        n=value;reverse=0
        while n:n,d=divmod(n,10);reverse=reverse*10+d
        return value==reverse
    if profile in ['prime_factors','unique_factors_list','unique_factors_set']:
        n=value;factors=[]
        # Independent divisibility specification over primes from a sieve.
        from .function_semantic_stress_v1 import primes
        for p in sorted(primes()):
            while n%p==0:factors.append(p);n//=p
            if n==1:break
        if n!=1:raise ValueError('Every bounded positive input must factor completely')
        return factors if profile=='prime_factors' else sorted(set(factors)) if profile=='unique_factors_list' else set(factors)
    if profile=='mean':return statistics.fmean(value)
    if profile=='median':return statistics.median(value)
    if profile=='mode':
        counts=Counter(value);return sorted(k for k,v in counts.items() if v==max(counts.values())) if counts else []
    if profile in ['sort_int','sort_nonnegative']:return sorted(value)
    if profile=='sort_inplace':return None
    if profile=='ipv4_integer':return int(ipaddress.IPv4Address(value))
    if profile=='ipv4_string':return str(ipaddress.IPv4Address(value))
    if profile.startswith('search_'):
        index=next((i for i,v in enumerate(value) if v==args[1]),None)
        return index is not None if profile=='search_bool' else index if index is not None else -1 if profile=='search_first' else None
    if profile=='hamming':return sum(a!=b for a,b in zip(*args))
    if profile=='pattern':return [i for i in range(len(value)+1) if value.startswith(args[1],i)]
    if profile=='normalized_palindrome':
        letters=[c.lower() for c in value if c.isalnum()];return all(letters[i]==letters[-i-1] for i in range(len(letters)//2))
    raise ValueError('An explicit numeric/structural specification is required')


def tagged(value):
    if isinstance(value,set):return {'set':sorted(value)}
    return value


def cases(row):
    profile=family(row)
    if profile is None:raise ValueError('Unsupported functions need an explicit unqualified disposition, never an invented contract')
    node=ast.parse(row['response']).body[0];items=arguments(profile);result=[];published=published_arguments(row)
    if (not isinstance(node,ast.FunctionDef) or node.name!=row['source_candidate']['function'] or node.args.vararg or node.args.kwarg
            or node.args.kwonlyargs or any(len(a)!=len(node.args.posonlyargs)+len(node.args.args) for a in items)):
        raise ValueError('Every exact reviewed public interface and complete positional argument set must match')
    for position,args in enumerate(items):
        value=expected(profile,args);case=dict(position=position,args=args,expected=tagged(value),
            literal_arguments_not_matched_in_public_examples=digest(args) not in published)
        if profile=='sort_inplace':case['expected_post_call_args']=[sorted(args[0])]
        case['case_id']=digest(dict(profile=profile,position=position,args=args));result.append(case)
    return result


def equal(actual,wanted):
    if isinstance(wanted,dict) and set(wanted)=={'set'}:return type(actual) is set and sorted(actual)==wanted['set']
    if type(wanted) in [list,tuple]:return type(actual) is type(wanted) and len(actual)==len(wanted) and all(equal(a,b) for a,b in zip(actual,wanted))
    if type(wanted) is float:return type(actual) in [int,float] and math.isfinite(actual) and math.isclose(actual,wanted,rel_tol=1e-12,abs_tol=1e-12)
    return type(actual) is type(wanted) and actual==wanted


def in_domain(profile,args):
    """Only compare published literals covered by the explicit finite domain."""
    if len(args)!=len(arguments(profile)[0]):return False
    integer=lambda v:type(v) is int
    numeric=lambda v:type(v) in [int,float] and math.isfinite(v)
    ints=lambda values:isinstance(values,list) and all(integer(v) and -24<=v<=24 for v in values)
    x=args[0]
    bounds={'not_gate':1,'factorial':32,'perfect_cube':256**3,'is_prime':4096,'is_sq':2**110,
        'integer_root':2**110,'increment_score':1000,'digit_sum':10**30,'digital_root':10**30,'parity':2**128}
    if profile in bounds:return integer(x) and 0<=x<=bounds[profile]
    if profile in ['prime_factors','unique_factors_list','unique_factors_set']:return integer(x) and 1<=x<=4096
    if profile=='palindrome_int':return integer(x) and 0<=x<=10**16
    if profile in ['abs_val','equation']:return numeric(x) and abs(x)<=128
    if profile=='price_plus_tax':return all(numeric(v) for v in args) and abs(x)<=128 and 0<=args[1]<=1
    if profile=='is_operand':return isinstance(x,str) and len(x)==1 and 32<=ord(x)<127
    if profile=='str_eval':return isinstance(x,str) and 0<len(x)<=32 and all(c in '0123456789' for c in x)
    if profile=='is_safe':return all(integer(v) for v in args) and all(0<=v<=16 for v in args[2:])
    if profile=='expectation':return (all(isinstance(v,list) for v in args) and len(x)==len(args[1])
        and all(numeric(v) for values in args for v in values) and all(p>=0 for p in args[1])
        and (not x or math.isclose(sum(args[1]),1.)))
    if profile in ['mean','median','mode','sort_int','sort_inplace','sort_nonnegative']:
        return ints(x) and (bool(x) or profile not in ['mean','median']) and (profile!='sort_nonnegative' or all(v>=0 for v in x))
    if profile=='ipv4_integer':
        try:return isinstance(x,str) and str(ipaddress.IPv4Address(x))==x
        except ipaddress.AddressValueError:return False
    if profile=='ipv4_string':return integer(x) and 0<=x<2**32
    if profile.startswith('search_'):
        return (ints(x) and integer(args[1]) and x==sorted(x)
            and (profile!='search_unique_optional' or len(x)==len(set(x))))
    if profile in ['hamming','pattern','normalized_palindrome']:
        return all(isinstance(v,str) and len(v)<=128 and v.isascii() for v in args) and (profile!='hamming' or len(x)==len(args[1]))
    return False


def literal_document_checks(row,profile):
    """Fail an incompatible registry before calling any source code wrong."""
    name=row['source_candidate']['function'];checks=[]
    for example in doctest.DocTestParser().get_examples(row['source_candidate']['original_literal_docstring']):
        try:
            statements=ast.parse(example.source).body
            if len(statements)!=1 or not isinstance(statements[0],ast.Expr):continue
            call=statements[0].value
            if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!=name or call.keywords:continue
            args=[ast.literal_eval(v) for v in call.args];published=ast.literal_eval(example.want.strip())
            if not in_domain(profile,args):continue
        except (ValueError,TypeError,SyntaxError,RecursionError):continue
        oracle=expected(profile,args)
        same=(math.isclose(oracle,published,rel_tol=1e-12,abs_tol=1e-12)
            if type(oracle) in [int,float] and type(published) in [int,float] else equal(oracle,tagged(published)))
        checks.append(dict(arguments=args,published_expected=tagged(published),independent_contract_expected=tagged(oracle),matches=same))
    return checks


def passes(case,raw):
    if raw['exception_only'] is not None or raw['entry_body_calls']<1:return False
    try:value=ast.literal_eval(raw['stdout'].strip())
    except (ValueError,SyntaxError,TypeError,RecursionError):return False
    return (type(value) is tuple and len(value)==2 and equal(value[0],case['expected'])
        and ('expected_post_call_args' not in case or equal(value[1],case['expected_post_call_args'])))


def observe(row,program=None,*,seconds=10):
    selected=cases(row);program=row['response'] if program is None else program;name=row['source_candidate']['function']
    if not isinstance(program,str) or len(program.encode())>50000 or '\0' in program or not 0<seconds<=30:
        raise ValueError('Complete bounded original or explicitly declared comparison program required')
    base=dict(source_id=row['id'],contract_family=family(row),program_sha256=__import__('hashlib').sha256(program.encode()).hexdigest(),
        cases_sha256=digest(selected),cases=len(selected),actual_program_launched=False,all_cases_observed=False,
        all_cases_passed=False,case_passes=[False]*len(selected),expected_values_not_sent_to_worker=True,
        source_train_not_held_benchmark=True,generated_program_not_repaired=True)
    try:ast.parse(program)
    except (ValueError,SyntaxError,RecursionError) as exc:return dict(base,status='syntax_error',diagnostic=str(exc)[:500])
    with tempfile.TemporaryDirectory(prefix='slm-public-contract-census-') as tmp:
        work=Path(tmp);(work/'program.py').write_text(program);(work/'prelude.py').write_text(PRELUDE);(work/'harness.py').write_bytes(HARNESS.read_bytes())
        settings=dict(program=str(work/'program.py'),prelude=str(work/'prelude.py'),function=name,examples=[
            '__slm_contract_args='+repr(c['args'])+'; __slm_contract_result='+name+'(*__slm_contract_args); print(repr((__slm_contract_result,__slm_contract_args)))\n' for c in selected])
        (work/'settings.json').write_text(json.dumps(settings));policy=python_policy(work)
        policy['command'][-1]=str(work/'harness.py');policy['command'].append(str(work/'settings.json'))
        policy['memory_bytes']=512*1024**2;policy['cpu_seconds']=math.ceil(seconds)
        result=run_bounded(policy,b'',seconds=seconds,output_bytes=1024**2,python=True)
    base['actual_program_launched']=True;base['execution']={k:v for k,v in result.items() if k not in ['stdout','stderr']}
    base['execution']['stderr']=result['stderr'][-1000:].decode(errors='replace')
    if result['status']!='completed':return dict(base,status='runtime_or_resource_failure')
    try:
        raw=json.loads(result['stdout']);rows=raw['results']
        if (set(raw)!={'format','results'} or raw['format']!='own-documented-doctest-raw-output-v1' or len(rows)!=len(selected)
                or any(set(r)!={'stdout','exception_only','entry_body_calls'} or not isinstance(r['stdout'],str)
                    or type(r['entry_body_calls']) is not int or r['entry_body_calls']<0
                    or r['exception_only'] is not None and not isinstance(r['exception_only'],str) for r in rows)):
            raise ValueError('Every complete raw return/argument-mutation result required')
    except (ValueError,TypeError,KeyError):return dict(base,status='incomplete_raw_observations')
    checks=[passes(c,r) for c,r in zip(selected,rows)]
    return dict(base,status='all_cases_passed' if all(checks) else 'contract_counterexample',all_cases_observed=True,
        all_cases_passed=all(checks),case_passes=checks,raw_results=rows)
