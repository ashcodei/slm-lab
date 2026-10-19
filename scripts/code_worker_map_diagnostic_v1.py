"""Diagnostic fork adding only the pure map builtin; original benchmark unchanged.
Candidate Python gets the same restricted namespace;
JavaScript runs inside QuickJS with no Python/OS/network callbacks.
"""
import ast
import copy
import json
import math
import os
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
import tracemalloc

request=json.load(sys.stdin);code=request['code'];language=request['language'];name=request['function']
state={'syntax_valid':False,'functional_passed':False,'passed_cases':0,'total_cases':len(request['cases']),
       'efficiency_completed':False,'stress':[]}


def emit():print(json.dumps(state),flush=True)


class EvaluationTimeout(BaseException):pass


def expired(*_):raise EvaluationTimeout('Per-call time limit exceeded')


def safe_import(name,globals=None,locals=None,fromlist=(),level=0):
    if name!='math' or level or any(item.startswith('_') for item in fromlist):
        raise ValueError('Only public math imports are supported in this Python benchmark')
    return math


try:
    if len(code)>50000:raise ValueError('Candidate code exceeds size limit')
    if language=='Python':
        tree=ast.parse(code);state['syntax_valid']=True
        methods={'append','extend','pop','insert','remove','clear','copy','count','index','sort','reverse',
                 'add','discard','get','items','keys','values','setdefault','update','split','strip','lstrip',
                 'rstrip','lower','upper','join','replace','find','startswith','endswith','isdigit','isalpha',
                 'bit_count','bit_length'}|{n for n in dir(math) if not n.startswith('_')}
        for node in ast.walk(tree):
            if isinstance(node,(ast.ClassDef,ast.Global,ast.Nonlocal,ast.With,ast.AsyncWith)):
                raise ValueError('Outside the supported Python subset')
            if isinstance(node,ast.Name) and node.id.startswith('__'):raise ValueError('Dunder names are unsupported')
            if isinstance(node,ast.Attribute) and node.attr not in methods:raise ValueError('Unsupported Python attribute: '+node.attr)
        for statement in tree.body:
            if not isinstance(statement,(ast.FunctionDef,ast.Import,ast.ImportFrom)):
                raise ValueError('Module-level code must contain functions or approved imports')
            if isinstance(statement,ast.FunctionDef) and statement.decorator_list:raise ValueError('Decorators are unsupported')
        # Type annotations are parsed for syntax but do not affect the runtime contract.
        for node in ast.walk(tree):
            if isinstance(node,ast.arg):node.annotation=None
            if isinstance(node,ast.FunctionDef):node.returns=None
            if isinstance(node,ast.AnnAssign):node.annotation=ast.Constant(value=None)
        ast.fix_missing_locations(tree)
    else:
        node=shutil.which('node')
        checked=subprocess.run([node,'--max-old-space-size=64','--check','-'],input=code,text=True,
                               capture_output=True,timeout=5)
        if checked.returncode:raise SyntaxError(checked.stderr[-200:])
        state['syntax_valid']=True
    # Node's V8 reserves substantial virtual address space; enforce the worker
    # address-space limit after its parse-only syntax check has completed.
    resource.setrlimit(resource.RLIMIT_AS,(384*1024*1024,384*1024*1024))
    resource.setrlimit(resource.RLIMIT_CPU,(8,8));resource.setrlimit(resource.RLIMIT_FSIZE,(0,0))
    if language=='Python':
        import builtins
        allowed={k:getattr(builtins,k) for k in ['range','len','sum','sorted','min','max','abs','enumerate',
            'zip','reversed','list','tuple','set','dict','int','float','str','bool','all','any','pow','divmod','map']}
        allowed['__import__']=safe_import
        namespace={'__builtins__':allowed};exec(compile(tree,'<candidate>','exec'),namespace);fn=namespace[name]
        signal.signal(signal.SIGALRM,expired)
        def invoke(args):
            args=copy.deepcopy(args);signal.setitimer(signal.ITIMER_REAL,.75)
            try:
                start=time.perf_counter();answer=fn(*args);elapsed=time.perf_counter()-start
            finally:signal.setitimer(signal.ITIMER_REAL,0)
            return answer,elapsed
    else:
        import quickjs
        context=quickjs.Context();context.set_memory_limit(64*1024*1024)
        context.set_max_stack_size(1024*1024);context.set_time_limit(.75)
        context.eval(code);fn=context.get(name)
        def invoke(args):
            prepared=[context.parse_json(json.dumps(arg)) if isinstance(arg,(list,dict)) else arg for arg in args]
            start=time.perf_counter();answer=fn(*prepared);elapsed=time.perf_counter()-start
            if isinstance(answer,quickjs.Object):answer=json.loads(answer.json())
            return answer,elapsed
    state['phase']='correctness'
    for case in request['cases']:
        # Curated public examples may use tuples, whose type JSON cannot retain.
        # These optional literals are parsed as data, never evaluated as code.
        args=ast.literal_eval(case['args_literal']) if 'args_literal' in case else case['args']
        expected=ast.literal_eval(case['expected_literal']) if 'expected_literal' in case else case['expected']
        answer,_=invoke(args)
        if answer!=expected:
            state['error']='Hidden case mismatch';emit();sys.exit(0)
        state['passed_cases']+=1
    state['functional_passed']=True;state['phase']='efficiency';emit()
    for case in request['stress']:
        timings=[];peak=None
        for _ in range(3):
            total=0
            for _ in range(request.get('repetitions',1)):
                answer,elapsed=invoke(case['args']);total+=elapsed
                if answer!=case['expected']:raise ValueError('Large-input result mismatch')
            timings.append(total/request.get('repetitions',1))
        if language=='Python':
            tracemalloc.start();invoke(case['args']);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        else:peak=context.memory()['memory_used_size']
        state['stress'].append({'input_size':len(case['args'][0]),'median_seconds':statistics.median(timings),
                                'memory_bytes':peak,'memory_measure':'Python tracemalloc peak including input copy' if language=='Python' else 'QuickJS total heap after call'})
    state['efficiency_completed']=True;state['phase']='complete';emit()
except BaseException as error:
    if isinstance(error,SystemExit):raise
    state['error']=type(error).__name__+': '+str(error)[:220]
    if isinstance(error,SyntaxError):state['syntax_valid']=False
    emit()
