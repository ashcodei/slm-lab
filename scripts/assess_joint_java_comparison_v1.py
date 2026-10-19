"""Two idle GPUs compare own checkpoints; restore the idle queued trainer on exit."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/joint-vs-language-Java-v1'
WAITER = dict(pid=2754743, ticks='27602681')


def identity(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
        if fields[0] in {'Z','X'}:
            return None
        return dict(pid=pid,ticks=fields[19],state=fields[0])
    except FileNotFoundError:
        return None


def same(actor):
    actual=identity(actor['pid'])
    return actual if actual and actual['ticks']==actor['ticks'] else None


def stop_child(actor):
    if same(actor):
        os.kill(actor['pid'],signal.SIGTERM)
        deadline=time.monotonic()+8
        while same(actor) and time.monotonic()<deadline:
            time.sleep(.1)
        if same(actor):
            os.kill(actor['pid'],signal.SIGKILL)


def restore(plan, reason):
    registry=REPORTS/'children.json'
    for child in (json.loads(registry.read_text()) if registry.exists() else []):
        stop_child(child)
    waiter=same(plan['waiter'])
    if waiter and waiter['state'] in {'T','t'}:
        os.kill(waiter['pid'],signal.SIGCONT)
    atomic_json(REPORTS/'restored.json',dict(at=datetime.now(timezone.utc).isoformat(),reason=reason,
        waiter=same(plan['waiter']),same_original_waiter_resumed=bool(same(plan['waiter'])),goal_achieved=False))


def watchdog():
    plan=json.loads((REPORTS/'plan.json').read_text())
    while same(plan['owner']) and not (REPORTS/'restored.json').exists():
        time.sleep(.5)
    if not (REPORTS/'restored.json').exists():
        restore(plan,'owner ended; exact registered inference workers stopped before resuming original waiter')


def worker(name):
    plan=json.loads((REPORTS/'plan.json').read_text())
    me=identity(os.getpid())
    # The owner publishes exact child birth identity before this process may
    # initialize CUDA. Its watchdog can therefore identify every GPU child.
    while True:
        if not same(plan['owner']):
            raise ValueError('Checked comparison owner ended before worker registration')
        registry=json.loads((REPORTS/'children.json').read_text())
        if any(a['pid']==me['pid'] and a['ticks']==me['ticks'] and a['subject']==name for a in registry):
            break
        time.sleep(.1)
    subject=next(s for s in plan['subjects'] if s['name']==name)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(subject['physical_gpu']):
        raise ValueError('Exact owned isolated GPU required')
    import slm_lab.exercise_java_evaluation as evaluation
    original=evaluation.FileBindings
    class OwnedGuard:
        def __init__(self,files):
            self.files=original(files)
        def check(self,full=False):
            waiter=same(plan['waiter'])
            if not same(plan['owner']) or not waiter or waiter['state'] not in {'T','t'}:
                raise ValueError('Checked comparison owner or held idle waiter changed')
            self.files.check(full=full)
    guard=OwnedGuard({**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    guard.check(full=True)
    evaluation.FileBindings=OwnedGuard
    try:
        evaluation.evaluate(subject['checkpoint'],str(ROOT/'data/exercism-support-context-v3/valid.jsonl'),
            str(REPORTS/(name+'.json')),device='cuda:0')
        guard.check(full=True)
    finally:
        evaluation.FileBindings=original


def comparison_report(before,after,plan_sha256,before_sha256,after_sha256):
    from slm_lab.exercise_java_evaluation import compare_reports
    report=compare_reports(before,after)
    report.update(driver_plan_sha256=plan_sha256,baseline_report_sha256=before_sha256,
        joint_report_sha256=after_sha256,candidate_promoted=False,
        scope='Own completed language versus completed joint development assessment; neither is the later final policy. Full matched35-module source/tests, not universal code or frontier evidence.')
    return report


def parent():
    if REPORTS.exists():
        raise ValueError('Preserve prior generations/results; no implicit restart')
    waiter=same(WAITER)
    if not waiter or waiter['state']!='S':
        raise ValueError('Exact original idle queued trainer must be sleeping')
    status=json.loads((ROOT/'reports/dialogue-retention-v1/status.json').read_text())
    if status['status']!='waiting for verified live extended curriculum':
        raise ValueError('Queued trainer is no longer in its original idle wait')
    children_path=Path('/proc')/str(WAITER['pid'])/'task'/str(WAITER['pid'])/'children'
    if children_path.read_text().strip():
        raise ValueError('Queued trainer already has live children')
    main=identity(2754742)
    if not main or main['ticks']!='27602681':
        raise ValueError('Original main evaluator must still be live before holding its waiter')
    main_status=json.loads((ROOT/'reports/extended-curriculum-v1/status.json').read_text())
    if not main_status['status'].startswith('evaluating'):
        raise ValueError('Original training has not handed off to inference')
    memory=subprocess.run(['nvidia-smi','--id=2,3','--query-gpu=memory.used','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True).stdout.splitlines()
    if len(memory)!=2 or any(int(v.strip())>128 for v in memory):
        raise ValueError('Both assigned comparison GPUs must be idle')
    from slm_lab.evaluation_provenance import capture
    paths=[Path(__file__),ROOT/'src/slm_lab/exercise_java_evaluation.py',
        ROOT/'reports/exercism-java-publisher-references-v3/summary.json',
        ROOT/'reports/exercise-java-runner-v1-release-tests.log',ROOT/'reports/exercise-java-evaluation-v1-release-tests.log',
        ROOT/'reports/completed-original-joint-and-local-code-v1.json']
    sources={str(ROOT/p):v for p,v in capture()['files_sha256'].items()}
    sources.update({str(p):file_hash(p) for p in paths})
    subjects=[]
    for name,folder,gpu,digest in [('language','language-pretraining',2,'4c05d99b0c3fc775bf89e70c6cbac9c63017eb36f67e17c738b5be268fb191ca'),
            ('joint','joint',3,'2dacd490e4a6ed20527a8393629d560e7230c1b6fdec54c1a49dc2eedf7f4e49')]:
        checkpoint=ROOT/'checkpoints/extended-curriculum-v1'/folder/'model.pt'
        sources[str(checkpoint)]=digest
        sources[str(checkpoint.parent/'tokenizer.json')]=file_hash(checkpoint.parent/'tokenizer.json')
        subjects.append(dict(name=name,checkpoint=str(checkpoint),checkpoint_sha256=digest,physical_gpu=gpu))
    FileBindings(sources).check(full=True)
    REPORTS.mkdir()
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),owner=identity(os.getpid()),waiter=WAITER,
        subjects=subjects,source_files_sha256=sources,model_updates=0,candidate_promoted=False,goal_achieved=False,
        protocol='Two full35-module matched Java assessments; complete generations sealed before tests. Exact idle later trainer held, watchdog restores it after owner exit. Original main evaluations continue separately. No final-baseline certificate.')
    atomic_json(REPORTS/'plan.json',plan)
    atomic_json(REPORTS/'children.json',[])
    watcher=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--watchdog'],cwd=ROOT,start_new_session=True)
    children=[]
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Comparison owner interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:
        os.kill(WAITER['pid'],signal.SIGSTOP)
        deadline=time.monotonic()+5
        while (same(WAITER) or {}).get('state') not in {'T','t'}:
            if time.monotonic()>deadline:
                raise ValueError('Original idle waiter did not stop')
            time.sleep(.05)
        if children_path.read_text().strip():
            raise ValueError('Original waiter spawned a child before it was held')
        registry=[]
        for subject in subjects:
            log=(REPORTS/(subject['name']+'.log')).open('xb')
            process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'--subject',subject['name']],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(subject['physical_gpu']),'OMP_NUM_THREADS':'1'})
            log.close()
            child={**identity(process.pid), 'subject':subject['name']}
            registry.append(child)
            atomic_json(REPORTS/'children.json',registry)
            children.append(process)
        while any(p.poll() is None for p in children):
            atomic_json(REPORTS/'status.json',dict(status='matched original Java generation/scoring active',
                updated_at=datetime.now(timezone.utc).isoformat(),children=[dict(**a,exit_code=p.poll()) for a,p in zip(registry,children)],
                exact_idle_waiter_held=(same(WAITER) or {}).get('state') in {'T','t'},goal_achieved=False))
            time.sleep(1)
        if any(p.returncode!=0 for p in children):
            raise ValueError('One or more Java assessments stopped; inspect retained generations/logs')
        before=json.loads((REPORTS/'language.json').read_text())
        after=json.loads((REPORTS/'joint.json').read_text())
        atomic_json(REPORTS/'comparison.json',comparison_report(before,after,file_hash(REPORTS/'plan.json'),
            file_hash(REPORTS/'language.json'),file_hash(REPORTS/'joint.json')))
    except BaseException as error:
        atomic_json(REPORTS/'failure.json',dict(error=str(error),automatic_retry=False,goal_achieved=False))
        raise
    finally:
        restore(plan,'assessment owner completed or failed; inference workers stopped before original waiter resumes')
        watcher.wait(timeout=10)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--subject',choices=['language','joint'])
    args=parser.parse_args()
    if args.watchdog:
        watchdog()
    elif args.subject:
        worker(args.subject)
    else:
        parent()
