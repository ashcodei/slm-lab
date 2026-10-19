"""Complete original Python publisher suites and conservative token mutations.

This module admits no data, alters no public tests and trains no model. The
pre-exec policy keeps the original Landlock/network/process limits and permits
pytest's directory changes inside the confined filesystem.
"""
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import sysconfig
import tempfile
import time
import tokenize
import zipfile

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .exercise_code import target_for
from .native_code import NativeBackendError,stop

ROOT=Path(__file__).resolve().parents[2]
HARNESS=ROOT/'scripts/python_publisher_harness_v1.py'
LAUNCHER=ROOT/'scripts/python_publisher_sandbox_v1.py'
PREFIX=b'SLM_PYTHON_PUBLISHER_INVENTORY\t'
BLOCK=re.compile(r'^File: ([^\r\n]+)\r?\n```python[ \t]*\r?\n(.*?)^```[ \t]*$',re.M|re.S)
SOURCE_PATH=re.compile(r'(?:[A-Za-z_]\w*/)*[A-Za-z_]\w*\.py',re.ASCII)
OPS={'True':('False','boolean'),'False':('True','boolean'),'and':('or','boolean'),'or':('and','boolean'),
    '==':('!=','comparison'),'!=':('==','comparison'),'<':('<=','comparison'),'<=':('<','comparison'),
    '>':('>=','comparison'),'>=':('>','comparison'),'+':('-','arithmetic'),'-':('+','arithmetic'),
    '*':('+','arithmetic'),'0':('1','integer'),'1':('0','integer')}


def extract_files(response,expected=None):
    if not isinstance(response,str) or len(response.encode())>100000:raise ValueError('Bounded complete Python files required')
    files={}
    for match in BLOCK.finditer(response):
        path,body=match.groups()
        if not SOURCE_PATH.fullmatch(path) or path in files or not body.strip():raise ValueError('Exact safe unique Python source paths required')
        files[path]=body
    if not files or BLOCK.sub('',response).strip():raise ValueError('Complete labeled Python files with no additional prose required')
    if expected is not None and set(files)!=set(expected):raise ValueError('Every original expected complete Python file required')
    return files


def mutations(response,source_id,count=4,seed=930031):
    if type(count) is not int or count<1 or type(seed) is not int:raise ValueError('Explicit deterministic mutation budget required')
    files=extract_files(response);choices=[]
    for block in BLOCK.finditer(response):
        name,body=block.groups();offset=block.start(2)
        if files[name]!=body:raise ValueError('Exact original Python body required')
        lines=body.splitlines(keepends=True);starts=[0]
        for line in lines:starts.append(starts[-1]+len(line))
        try:tokens=list(tokenize.generate_tokens(io.StringIO(body).readline))
        except (tokenize.TokenError,IndentationError,SyntaxError):continue
        for token in tokens:
            if token.type not in {tokenize.OP,tokenize.NAME,tokenize.NUMBER} or token.string not in OPS:continue
            old=token.string;new,family=OPS[old]
            start=offset+starts[token.start[0]-1]+token.start[1]
            end=offset+starts[token.end[0]-1]+token.end[1]
            if response[start:end]!=old:raise ValueError('Unicode/line Python token offsets differ')
            choices.append(dict(file=name,start=start,end=end,old=old,new=new,family=family))
    rng=random.Random(seed+int.from_bytes(hashlib.sha256(source_id.encode()).digest()[:8],'big'));rng.shuffle(choices)
    selected=[]
    for family in ['boolean','comparison','arithmetic','integer']:
        first=next((p for p in choices if p['family']==family),None)
        if first is not None and len(selected)<count:selected.append(first)
    for p in choices:
        if len(selected)>=count:break
        if p not in selected:selected.append(p)
    result=[]
    for patch in selected:
        other=response[:patch['start']]+patch['new']+response[patch['end']:]
        extract_files(other,files)
        result.append(dict(response=other,patch=patch,response_sha256=hashlib.sha256(other.encode()).hexdigest()))
    return result


def inventory(raw):
    if not isinstance(raw,bytes) or len(raw)>4*1024**2:raise ValueError('Bounded complete Python test inventory required')
    rows=[line[len(PREFIX):] for line in raw.splitlines() if line.startswith(PREFIX)]
    if len(rows)!=1:raise ValueError('Exactly one complete Python test inventory required')
    record=json.loads(rows[0])
    if (set(record)!={'format','framework','framework_version','registered','started','events','collection_errors','session_exitstatus'}
            or record['format']!='own-complete-python-publisher-events-v1' or record['framework']!='pytest'
            or not isinstance(record['framework_version'],str) or type(record['session_exitstatus']) is not int):
        raise ValueError('Exact complete original pytest event schema required')
    for name in ['registered','started']:
        ids=record[name]
        if (not isinstance(ids,list) or len(ids)>8192 or any(not isinstance(x,str) or not x or len(x)>4096 for x in ids)
                or len(ids)!=len(set(ids))):raise ValueError('Distinct bounded original case identities required')
    registered=set(record['registered'])
    if not set(record['started'])<=registered:raise ValueError('Started case was not registered')
    if not isinstance(record['events'],list) or len(record['events'])>3*8192:raise ValueError('Bounded full phase events required')
    events={}
    for event in record['events']:
        if (set(event)!={'nodeid','phase','outcome','expected_failure','exception_type','assertion_failure'}
                or event['nodeid'] not in registered or event['phase'] not in {'setup','call','teardown'}
                or event['outcome'] not in {'passed','failed','skipped'} or type(event['assertion_failure']) is not bool
                or any(event[k] is not None and not isinstance(event[k],str) for k in ['expected_failure','exception_type'])):
            raise ValueError('Exact declared case/phase/outcome evidence required')
        key=(event['nodeid'],event['phase'])
        if key in events:raise ValueError('Repeated Python test phase event')
        if event['assertion_failure'] and (event['outcome']!='failed' or not event['exception_type']):
            raise ValueError('Assertion marker needs actual failed exception evidence')
        events[key]=event
    errors=record['collection_errors']
    if not isinstance(errors,list) or any(set(e)!={'nodeid','diagnostic'} or any(not isinstance(v,str) for v in e.values()) for e in errors):
        raise ValueError('Complete collection error evidence required')
    complete=bool(registered) and set(record['started'])==registered and set(events)=={(x,p) for x in registered for p in ['setup','call','teardown']}
    phases=list(events.values())
    skipped=[e for e in phases if e['outcome']=='skipped' or e['expected_failure'] is not None]
    outer=[e for e in phases if e['phase']!='call' and e['outcome']!='passed']
    failed=[e for e in phases if e['phase']=='call' and e['outcome']=='failed']
    passed=sorted(e['nodeid'] for e in phases if e['phase']=='call' and e['outcome']=='passed')
    assertion=sorted(e['nodeid'] for e in failed if e['assertion_failure'])
    return dict(cases=sorted(registered),started_cases=sorted(record['started']),complete_case_phases=complete,
        passed_cases=passed,failed_cases=sorted(e['nodeid'] for e in failed),assertion_failed_cases=assertion,
        nonassertion_call_failures=sorted(e['nodeid'] for e in failed if not e['assertion_failure']),
        skipped_or_expected_failures=skipped,outer_phase_failures=outer,collection_errors=errors,
        session_exitstatus=record['session_exitstatus'],framework_version=record['framework_version'],
        all_passed=complete and not errors and not skipped and not outer and not failed and record['session_exitstatus']==0)


def classify(observed,reference=None):
    run=observed['execution']
    if run['returncode']!=0 or run['limit'] is not None:return 'runtime_or_environment_qualification'
    raw=observed.get('raw_inventory_output')
    if not isinstance(raw,str):return 'incomplete_test_inventory_qualification'
    try:value=inventory(raw.encode())
    except (ValueError,TypeError,KeyError,json.JSONDecodeError):return 'incomplete_test_inventory_qualification'
    if value!=observed.get('inventory') or hashlib.sha256(raw.encode()).hexdigest()!=observed.get('report_sha256'):
        raise ValueError('Raw complete Python inventory/hash differs from parsed evidence')
    if (not value['complete_case_phases'] or value['collection_errors'] or value['skipped_or_expected_failures']
            or value['outer_phase_failures'] or value['session_exitstatus'] not in {0,1}):return 'incomplete_test_inventory_qualification'
    if reference is not None and (value['cases']!=reference['cases'] or value['framework_version']!=reference['framework_version']):
        return 'changed_case_population_qualification'
    if value['all_passed']:return 'all_finite_published_cases_pass'
    if reference is None:return 'original_reference_test_failure_qualification'
    if value['nonassertion_call_failures']:return 'nonassertion_test_exception_qualification'
    if value['assertion_failed_cases'] and value['session_exitstatus']==1:return 'completed_published_assertion_counterexample'
    return 'test_failure_without_retained_assertion_qualification'


class PublishedPythonSuites:
    def __init__(self,cache):
        self.cache=Path(cache).resolve();self.source=json.loads((self.cache/'source.json').read_text())
        if file_hash(self.cache/'source.zip')!=self.source['archive_sha256']:raise ValueError('Original Python source archive changed')
        self.archive=zipfile.ZipFile(self.cache/'source.zip');self.prefix=self.archive.namelist()[0].split('/')[0]+'/'
        specs=ROOT/'data/exercism-source-cache-v1/problem-specifications'
        self.spec_source=json.loads((specs/'source.json').read_text())
        if file_hash(specs/'source.zip')!=self.spec_source['archive_sha256']:raise ValueError('Original canonical-data archive changed')
        self.spec_archive=zipfile.ZipFile(specs/'source.zip');self.spec_prefix=self.spec_archive.namelist()[0].split('/')[0]+'/'
        self.packages=Path(importlib.metadata.distribution('pytest').locate_file('')).resolve()
        self.runtime=dict(python=sys.version,executable_sha256=file_hash(sys.executable),pytest=importlib.metadata.version('pytest'))
        files=[self.cache/'source.json',self.cache/'source.zip',specs/'source.json',specs/'source.zip',HARNESS,LAUNCHER,ROOT/'scripts/native_sandbox.py',
            Path(__file__).resolve(),ROOT/'src/slm_lab/exercise_code.py',Path(sys.executable).resolve()]
        for package in ['_pytest','pytest','pluggy','iniconfig','packaging']:
            files+=list((self.packages/package).rglob('*.py'))
        self.guard=FileBindings({str(p):file_hash(p) for p in files})

    def close(self):
        self.archive.close();self.spec_archive.close()

    def read(self,path):
        if Path(path).is_absolute() or '..' in Path(path).parts:raise ValueError('Exact safe archived Python source required')
        info=self.archive.getinfo(self.prefix+path)
        if info.file_size>1024**2:raise ValueError('Bounded publisher Python file required')
        return self.archive.read(info)

    def read_item(self,item):
        repository=item.get('repository','exercism/python')
        if repository=='exercism/python':return self.read(item['path'])
        if (repository!='exercism/problem-specifications' or Path(item['path']).is_absolute()
                or '..' in Path(item['path']).parts or '/blob/'+self.spec_source['revision']+'/' not in item['url']):
            raise ValueError('Exact original canonical repository/revision/path required')
        info=self.spec_archive.getinfo(self.spec_prefix+item['path'])
        if info.file_size>1024**2:raise ValueError('Bounded original canonical test artifact required')
        return self.spec_archive.read(info)

    def suite(self,row):
        if (row['language']!='Python' or row['split']!='train' or row['revision']!=self.source['revision']
                or row['source_archive_sha256']!=self.source['archive_sha256']):raise ValueError('Original training-only pinned Python publisher source required')
        for item in row['source_files']+row['published_test_artifacts']:
            if hashlib.sha256(self.read_item(item)).hexdigest()!=item['sha256']:raise ValueError('Original Python source/test bytes differ')
        config_path=next(i['path'] for i in row['source_files'] if i['path'].endswith('/.meta/config.json'))
        base=config_path.split('/.meta/')[0]+'/';config=json.loads(self.read(config_path))['files']
        solution=config['solution'];expected={}
        for source in config.get('example',config.get('exemplar',[])):
            target=target_for(source,solution)
            if target in expected:raise ValueError('Ambiguous original Python reference-to-solution mapping')
            expected[target]=self.read(base+source).decode().rstrip()+'\n'
        if set(expected)!=set(solution) or extract_files(row['response'],solution)!=expected:
            raise ValueError('Every complete exact original Python reference must match the training response')
        tests={name:self.read(base+name) for name in config['test']}
        if any(Path(name).is_absolute() or '..' in Path(name).parts for name in tests) or set(tests)&set(solution):
            raise ValueError('Original distinct Python solution and test paths required')
        modules=[name for name in tests if name.endswith('.py')]
        if not modules:raise ValueError('Full declared publisher test modules required')
        return dict(expected_solution_files=solution,test_files=tests,test_modules=modules,
            test_sha256=hashlib.sha256(b''.join(name.encode()+b'\0'+raw for name,raw in sorted(tests.items()))).hexdigest(),
            source_files=expected)

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
