"""Distinct complete parent/subtest evidence and original Python helper staging.

v1 artifacts remain immutable. Subtest ordinal identities retain their complete
publisher context; a changed context cannot supply a preference label.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import time

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .native_code import NativeBackendError,stop
from .python_execution_feedback_v1 import PublishedPythonSuites,ROOT,LAUNCHER,SOURCE_PATH

HARNESS=ROOT/'scripts/python_publisher_harness_v2.py'
PREFIX=b'SLM_PYTHON_PUBLISHER_INVENTORY_V2\t'


def inventory(raw):
    if not isinstance(raw,bytes) or len(raw)>4*1024**2:raise ValueError('Bounded complete Python events required')
    payload=[line[len(PREFIX):] for line in raw.splitlines() if line.startswith(PREFIX)]
    if len(payload)!=1:raise ValueError('Exactly one complete v2 inventory required')
    r=json.loads(payload[0])
    if (set(r)!={'format','framework','framework_version','registered','started','events','collection_errors','session_exitstatus'}
            or r['format']!='own-complete-python-publisher-events-v2' or r['framework']!='pytest'
            or not isinstance(r['framework_version'],str) or type(r['session_exitstatus']) is not int):
        raise ValueError('Exact v2 original framework schema required')
    for name in ['registered','started']:
        v=r[name]
        if not isinstance(v,list) or len(v)>8192 or len(v)!=len(set(v)) or any(not isinstance(x,str) or not x or len(x)>4096 for x in v):
            raise ValueError('Distinct complete original parent identities required')
    registered=set(r['registered']);parents={};subtests={};contexts={};indices={}
    if not set(r['started'])<=registered:raise ValueError('Only original registered parents can start')
    if not isinstance(r['events'],list) or len(r['events'])>65536:raise ValueError('Bounded complete phases required')
    for e in r['events']:
        if (set(e)!={'nodeid','phase','outcome','expected_failure','exception_type','assertion_failure','kind','subtest_index','subtest_context'}
                or e['nodeid'] not in registered or e['phase'] not in {'setup','call','teardown'}
                or e['outcome'] not in {'passed','failed','skipped'} or type(e['assertion_failure']) is not bool
                or any(e[k] is not None and not isinstance(e[k],str) for k in ['expected_failure','exception_type'])):
            raise ValueError('Exact declared original case event required')
        if e['assertion_failure'] and (e['outcome']!='failed' or not e['exception_type']):raise ValueError('Actual assertion exception required')
        if e['kind']=='parent':
            key=(e['nodeid'],e['phase'])
            if e['subtest_index'] is not None or e['subtest_context'] is not None or key in parents:raise ValueError('Distinct complete parent phases required')
            parents[key]=e
        elif e['kind']=='subtest':
            i=e['subtest_index'];context=e['subtest_context']
            if (e['phase']!='call' or type(i) is not int or i<1 or i!=indices.get(e['nodeid'],0)+1
                    or not isinstance(context,dict) or set(context)!={'msg','kwargs'}
                    or context['msg'] is not None and not isinstance(context['msg'],str)
                    or not isinstance(context['kwargs'],dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in context['kwargs'].items())):
                raise ValueError('Every ordered original subtest and complete context required')
            indices[e['nodeid']]=i
            key=e['nodeid']+'::owned-subtest:'+str(i)
            subtests[key]=e;contexts[key]=context
        else:raise ValueError('Explicit parent/subtest event kind required')
    errors=r['collection_errors']
    if not isinstance(errors,list) or any(set(e)!={'nodeid','diagnostic'} for e in errors):raise ValueError('Full collection errors required')
    complete=bool(registered) and set(r['started'])==registered and set(parents)=={(x,p) for x in registered for p in ['setup','call','teardown']}
    calls={name:parents[(name,'call')] for name in registered if (name,'call') in parents};calls.update(subtests)
    skipped=[e for e in r['events'] if e['outcome']=='skipped' or e['expected_failure'] is not None]
    outer=[e for e in r['events'] if e['phase']!='call' and e['outcome']!='passed']
    failed={name:e for name,e in calls.items() if e['outcome']=='failed'}
    cases=sorted(registered|set(subtests))
    return dict(cases=cases,parent_cases=sorted(registered),subtest_contexts=contexts,
        started_cases=sorted(set(r['started'])|set(subtests)),complete_case_phases=complete,
        passed_cases=sorted(name for name,e in calls.items() if e['outcome']=='passed'),failed_cases=sorted(failed),
        assertion_failed_cases=sorted(name for name,e in failed.items() if e['assertion_failure']),
        nonassertion_call_failures=sorted(name for name,e in failed.items() if not e['assertion_failure']),
        skipped_or_expected_failures=skipped,outer_phase_failures=outer,collection_errors=errors,
        session_exitstatus=r['session_exitstatus'],framework_version=r['framework_version'],
        all_passed=complete and not errors and not skipped and not outer and not failed and r['session_exitstatus']==0)


def classify(observed,reference=None):
    run=observed['execution']
    if run['returncode']!=0 or run['limit'] is not None:return 'runtime_or_environment_qualification'
    raw=observed.get('raw_inventory_output')
    if not isinstance(raw,str):return 'incomplete_test_inventory_qualification'
    try:v=inventory(raw.encode())
    except (ValueError,TypeError,KeyError,json.JSONDecodeError):return 'incomplete_test_inventory_qualification'
    if v!=observed.get('inventory') or hashlib.sha256(raw.encode()).hexdigest()!=observed.get('report_sha256'):
        raise ValueError('Full raw v2 events/hash differ')
    if (not v['complete_case_phases'] or v['collection_errors'] or v['skipped_or_expected_failures']
            or v['outer_phase_failures'] or v['session_exitstatus'] not in {0,1}):return 'incomplete_test_inventory_qualification'
    if reference is not None and (v['cases']!=reference['cases'] or v['framework_version']!=reference['framework_version']):
        return 'changed_case_population_qualification'
    if reference is not None and v['subtest_contexts']!=reference['subtest_contexts']:
        return 'changed_subtest_context_qualification'
    if v['all_passed']:return 'all_finite_published_cases_pass'
    if reference is None:return 'original_reference_test_failure_qualification'
    if v['nonassertion_call_failures']:return 'nonassertion_test_exception_qualification'
    if v['assertion_failed_cases'] and v['session_exitstatus']==1:return 'completed_published_assertion_counterexample'
    return 'test_failure_without_retained_assertion_qualification'


class PublishedPythonSuitesV2(PublishedPythonSuites):
    def __init__(self,cache):
        super().__init__(cache)
        self.guard=FileBindings({**{str(p):h for p,h in self.guard.files.items()},
            str(HARNESS):file_hash(HARNESS),str(Path(__file__).resolve()):file_hash(__file__)})

    def suite(self,row):
        result=super().suite(row)
        if row['id']=='exercism:python:error-handling':
            path='exercises/practice/error-handling/test_utils.py'
            raw=self.read(path)
            result['test_files']['test_utils.py']=raw
            result['test_sha256']=hashlib.sha256(b''.join(n.encode()+b'\0'+v for n,v in sorted(result['test_files'].items()))).hexdigest()
            result['original_provided_helpers']={path:hashlib.sha256(raw).hexdigest()}
        return result

    def run(self,files,suite):
        self.guard.check()
        if set(files)!=set(suite['expected_solution_files']):raise ValueError('Complete original expected Python solution set required')
        with tempfile.TemporaryDirectory(prefix='slm-python-publisher-') as temporary:
            work=Path(temporary);src=work/'src';cache=work/'cache';src.mkdir();cache.mkdir()
            for name,body in {**files,**suite['test_files']}.items():
                if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Safe exact Python solution/test path required')
                path=src/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(body.encode() if isinstance(body,str) else body)
            settings=work/'settings.json';settings.write_text(json.dumps(dict(packages=str(self.packages),source=str(src),
                temporary=str(cache/'pytest'),tests=[str(src/n) for n in suite['test_modules']])))
            paths=[(str(Path(sys.executable).resolve()),5),(str(Path(sysconfig.get_path('stdlib')).parent),13),
                (str(self.packages),12),(str(src),12),(str(cache),32766),(str(HARNESS),4),(str(settings),4)]
            for name in ['/lib/x86_64-linux-gnu','/usr/lib/x86_64-linux-gnu','/lib64','/etc/ld.so.cache','/dev/null','/dev/urandom']:
                p=Path(name)
                if p.exists():paths.append((name,13 if p.is_dir() else 6 if name=='/dev/null' else 4))
            policy=dict(mode='runtime',cwd=str(cache),paths=paths,address_bytes=1024**3,cpu_seconds=20,file_bytes=8*1024**2,
                command=[str(Path(sys.executable).resolve()),'-I','-S','-B',str(HARNESS),str(settings)],
                environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',TZ='UTC',TMPDIR=str(cache),PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'))
            output,errors=work/'stdout',work/'stderr'
            with output.open('wb') as stdout,errors.open('wb') as stderr:
                process=subprocess.Popen([sys.executable,'-I',str(LAUNCHER),json.dumps(policy)],stdout=stdout,stderr=stderr,
                    stdin=subprocess.DEVNULL,close_fds=True,start_new_session=True,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
                tick=time.monotonic();peak=0;limit=None
                while process.poll() is None:
                    try:
                        status=Path(f'/proc/{process.pid}/status').read_text()
                        peak=max(peak,next(int(x.split()[1])*1024 for x in status.splitlines() if x.startswith('VmRSS:')))
                    except (OSError,StopIteration):pass
                    if time.monotonic()-tick>30:limit='Python publisher wall-time budget exceeded'
                    if peak>512*1024**2:limit='Python publisher RSS budget exceeded'
                    if output.stat().st_size+errors.stat().st_size>4*1024**2:limit='Python publisher output budget exceeded'
                    if limit:stop(process);break
                    time.sleep(.02)
            raw=output.read_bytes();diagnostic=errors.read_bytes()[-6000:].decode(errors='replace')
            if 'SANDBOX_ERROR:' in diagnostic:raise NativeBackendError(diagnostic)
            result=dict(execution=dict(returncode=process.returncode,limit=limit,peak_rss_bytes=peak,diagnostic=diagnostic),
                raw_inventory_output=raw.decode(errors='replace'),report_sha256=hashlib.sha256(raw).hexdigest(),runtime=self.runtime)
            try:result['inventory']=inventory(raw)
            except (ValueError,TypeError,KeyError,json.JSONDecodeError):result['inventory']=None
            self.guard.check()
            return result

