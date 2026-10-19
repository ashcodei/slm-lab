"""Explicitly extend eight never-started queues with admitted state supervision; preserve live work."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import observe,retire,terminal

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/code-first-state-v2-handoff'
PRIOR=ROOT/'reports/code-first-v1-handoff/complete.json'
OLD_NAMES=('full-original-MBPP-pipeline-v2','AI2D-fixed-exposure-strata-v2','full-AI2D-pipeline-v2',
    'full-held-QA-pipeline-v2','code-first-final-standalone-CPU-v1','broad-anchored-code-v4',
    'phase-human-preference-v5','code-first-continuation-v1')
NEW=('code-first-continuation-v2','phase-human-preference-v6','broad-anchored-code-v5',
    'code-first-final-standalone-CPU-v2','full-held-QA-pipeline-v3','full-AI2D-pipeline-v3',
    'full-original-MBPP-pipeline-v3','AI2D-fixed-exposure-strata-v3')
SOURCES=('CodeContests-full-Python-references-v1','CodeContests-public-execution-traces-candidate-v1',
    'CodeContests-reference-outputs-v1-readback','public-execution-targets-v1-readback-admission',
    'public-state-traces-full-v1','public-state-traces-full-v1-readback-admission',
    'focused-code-continuation-sources-v3-preparation')


def birth(identity):
    proc=Path('/proc')/str(identity['pid'])
    fields=(proc/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=identity['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Every original live main/source birth must remain exact')
    return dict(pid=identity['pid'],ticks=fields[19],state=fields[0])


def main(apply=False):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only explicit handoff required')
    log=ROOT/'reports/exact-waiting-process-retirement-v1-final-tests.log'
    if '1 passed' not in log.read_text():raise ValueError('Actual pidfd/birth/other-process-preservation test required')
    main_folder=ROOT/'reports/phase-joint-v3';main_plan=json.loads((main_folder/'plan.json').read_text())
    main_owner=main_plan['owner'];workers=[p for p in json.loads((main_folder/'children.json').read_text()) if p['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('All four exact original main GPU workers required')
    original=[birth(main_owner),*[birth(p) for p in workers]]
    source_owners={name:json.loads((ROOT/'reports'/name/'plan.json').read_text())['owner'] for name in SOURCES}
    source_before={}
    for name,identity in source_owners.items():
        folder=ROOT/'reports'/name
        if not terminal(identity) or not (folder/'complete.json').exists() or (folder/'failure.json').exists():
            raise ValueError('All original reference/output/state data jobs must finish completely before this handoff')
        source_before[name]=dict(owner=identity,complete=True,completion_sha256=file_hash(folder/'complete.json'))
    prior=json.loads(PRIOR.read_text())
    gpu_pids={int(p.strip()) for p in subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).splitlines() if p.strip()}
    if gpu_pids!={p['pid'] for p in workers}:raise ValueError('Only the original four main workers may own any GPU')
    old={};protected=dict(main_plan['source_files_sha256'])
    for name in OLD_NAMES:
        folder=ROOT/'reports'/name;path=ROOT/'reports'/(name+'-queue.json')
        queue=json.loads(path.read_text());identity=queue['owner'];expected=prior['new_waiting_pipeline'][name]['owner']
        if identity!=expected:raise ValueError('Exact previously declared waiting owner required')
        observation=observe(identity)
        if identity['pid'] in gpu_pids:raise ValueError('No GPU worker may be retired')
        if name=='AI2D-fixed-exposure-strata-v2':
            if not (folder/'plan.json').is_file() or (folder/'complete.json').exists():raise ValueError('Only the unstarted original source-group outcome reader may be retired')
        elif folder.exists() or (ROOT/'checkpoints'/name).exists():raise ValueError('Never retire a controller that started training or assessment')
        old[name]=dict(owner=observation,queue=str(path),queue_sha256=file_hash(path),source_files_sha256=queue['source_files_sha256'])
        protected.update(queue['source_files_sha256'])
    new={}
    for name in NEW:
        path=ROOT/'reports'/(name+'-queue.json');q=json.loads(path.read_text());observation=observe(q['owner'])
        if observation['pid'] in gpu_pids:raise ValueError('New queues must remain waiting without GPU children')
        if name!='AI2D-fixed-exposure-strata-v3' and (ROOT/'reports'/name).exists():raise ValueError('New scientific production must wait behind the closed handoff')
        new[name]=dict(owner=q['owner'],actual_observation=observation,queue=str(path),queue_sha256=file_hash(path))
        protected.update(q['source_files_sha256'])
    dependencies=json.loads(Path(new['code-first-continuation-v2']['queue']).read_text())['exact_dependencies']
    if dependencies['main']!=main_owner:raise ValueError('New early CE must wait for the exact original main owner')
    # Validate the combined immutable source closure once. Root documentation
    # may subsequently change only when absent from these protected inputs.
    FileBindings(protected).check(full=True)
    at=datetime.now(timezone.utc).isoformat()
    receipt=dict(at=at,original_main_owner=main_owner,original_main_and_four_worker_births=original,
        original_public_source_owners=source_before,new_waiting_pipeline=new,
        old_controller_sources={name:dict(owner=v['owner'],queue=v['queue'],queue_sha256=v['queue_sha256'],
            immutable_source_bindings_count=len(v['source_files_sha256']),
            source_bindings_sha256=hashlib.sha256(json.dumps(v['source_files_sha256'],sort_keys=True).encode()).hexdigest()) for name,v in old.items()},
        old_source_files_and_reports_preserved=True,live_main_and_completed_public_source_jobs_not_signalled=True,
        no_old_scientific_training_or_evaluation_preempted=True,no_original_source_budget_or_benchmark_population_removed=True,
        new_production_updates_at_handoff=0,all_ten_final_requirements_incomplete=True,goal_achieved=False)
    if not apply:
        print(json.dumps(dict(dry_run=True,old_hidden_childless_controllers=len(old),new_hidden_childless_controllers=len(new),original_GPU_pids=sorted(gpu_pids))));return
    OUT.mkdir();atomic_json(OUT/'before-retirement.json',receipt)
    for name,value in old.items():
        path=Path(value['queue'])
        if file_hash(path)!=value['queue_sha256']:raise ValueError('Original waiting queue changed before retirement')
        (OUT/(name+'-original-queue.json')).write_bytes(path.read_bytes())
    retire([v['owner'] for v in old.values()])
    if not all(terminal(v['owner']) for v in old.values()):raise ValueError('Handoff stays closed until every exact retired controller is terminal')
    for identity in [main_owner,*workers]:birth(identity)
    for name,value in source_before.items():
        if file_hash(ROOT/'reports'/name/'complete.json')!=value['completion_sha256']:raise ValueError('Every completed source result must remain preserved')
    for name,value in new.items():observe(value['owner'])
    for name,value in old.items():
        if file_hash(value['queue'])!=value['queue_sha256']:raise ValueError('Old queue bytes must remain preserved after terminal retirement')
    receipt.update(old_unstarted_controllers_all_terminal=True,explicit_code_first_handoff_complete=True,
        retired_exact_processes=[dict(pid=v['owner']['pid'],ticks=v['owner']['ticks']) for v in old.values()],
        source_files_sha256={str(p):file_hash(p) for p in [Path(__file__),ROOT/'src/slm_lab/exact_waiting_process_retirement_v1.py',
            ROOT/'tests/test_exact_waiting_process_retirement_v1.py',log]})
    atomic_json(OUT/'complete.json',receipt)
    print(json.dumps(dict(handoff=str(OUT/'complete.json'),retired_unstarted_controllers=8,new_queued_controllers=8,
        current_main_and_all_four_GPU_workers_preserved=True,public_source_owners_preserved=True,quality_gain_not_yet_established=True)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--apply',action='store_true');args=parser.parse_args();main(args.apply)
