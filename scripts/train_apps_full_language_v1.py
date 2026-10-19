"""Own706M full human-code epoch, actual4-L40 capacity and fixed assessments.

Own language weights learn; original vision/critic remain frozen. Every APPS
train row is visited once plus9rounding repeats; earlier29-family rehearsal is
sampled, not claimed to be full prior-source/final production coverage.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import assess_joint_java_comparison_v1 as resources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/apps-full-language-v1'
OUTPUT = ROOT / 'checkpoints/apps-full-language-v1'
BASE = ROOT / 'checkpoints/dialogue-retention-v1/systems-variety-code/model.pt'
TOKEN = BASE.parent / 'tokenizer.json'
SOURCE = ROOT / 'data/public-apps-human-solution-view-v1'
SOURCE_CHECK = ROOT / 'reports/code-residual-stage-v3-source-check/results.json'
FOLDERS = {k: ROOT / 'data' / v for k, v in [('joint', 'codenet-replay-v1'), ('facts', 'codenet-fact-replay-v1'),
    ('dialogue', 'codenet-dialogue-replay-v1'), ('images', 'codenet-image-replay-v1'),
    ('later', 'codenet-later-instruction-replay-v2'), ('packed', 'codenet-packed-pretraining-replay-v1')]}
WAITER = dict(pid=2754743, ticks='27602681')
WAIT_STATE = 'waiting for finalized original paired real-photo questions'
RESERVE, MAX_STATE = 26 * 1024**3, 10 * 1024**3
resources.REPORTS = REPORTS


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message, updated_at=datetime.now(timezone.utc).isoformat(),
        candidate_promoted=False, goal_achieved=False, **fields))


def ownership(plan):
    waiter = resources.same(plan['waiter'])
    if not resources.same(plan['owner']) or not waiter or waiter['state'] not in ['T', 't']:
        raise ValueError('Exact live training owner and held original childless photo-data waiter required')


def register(name):
    plan = json.loads((REPORTS / 'plan.json').read_text()); ownership(plan)
    with (REPORTS / 'children.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        actors = json.loads((REPORTS / 'children.json').read_text())
        actor = dict(**resources.identity(os.getpid()), subject=name)
        actors.append(actor); atomic_json(REPORTS / 'children.json', actors)
    return plan


def open_inputs(plan):
    from tokenizers import Tokenizer
    from slm_lab.apps_human_solution_view_v1 import AppsHumanSolutions
    from slm_lab.code_residual_stage_v3 import open_expanded_replay, ExpandedReplayEncoder
    from slm_lab.codenet_fact_replay_v1 import CombinedReplay
    from slm_lab.apps_full_language_v1 import HumanProgramEncoder
    from slm_lab.model import ModelConfig
    source = AppsHumanSolutions(SOURCE)
    replay = open_expanded_replay(FOLDERS, plan['tokenizer_sha256'])
    joint = replay
    while isinstance(joint, CombinedReplay): joint = joint.owners['human_instruction']
    tokenizer = Tokenizer.from_file(str(TOKEN)); config = ModelConfig.load(source.pack.folder / 'model-config.json')
    return source, replay, HumanProgramEncoder(source, tokenizer, config), ExpandedReplayEncoder(tokenizer, config, joint.source.parent, 448)


def worker(phase):
    import gc
    import hashlib
    import random
    import shutil
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.apps_full_language_v1 import FullLanguageStage
    from slm_lab.code_residual_checkpoint_v1 import _cpu
    from slm_lab.code_residual_collective_checkpoint_v2 import _validated
    from slm_lab.code_residual_stage_v3 import replay_field
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.codenet_stage_v1 import freeze_for_code_stage
    from slm_lab.recurrent_ablation import state_digest
    from slm_lab.training import load_checkpoint
    local = int(os.environ['LOCAL_RANK']); rank = int(os.environ['RANK'])
    if int(os.environ['WORLD_SIZE']) != 4: raise ValueError('Four actual workers required')
    plan = register(phase + '-rank-' + str(rank))
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    guard.check(full=True)
    source, replay, code_encoder, replay_encoder = open_inputs(plan)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda', local); torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != 'NVIDIA L40' or not torch.cuda.is_bf16_supported():
        raise ValueError('Exact owned BF16 L40 required')
    torch.cuda.set_per_process_memory_fraction(.9, device)
    dist.init_process_group('nccl')
    try:
        random.seed(931001+rank); np.random.seed(931001+rank); torch.manual_seed(931001+rank)
        raw, metadata = load_checkpoint(BASE, device, training_state=False)
        if sum(p.numel() for p in raw.parameters()) != 706012672: raise ValueError('Exact own706M full model required')
        parameters, frozen = freeze_for_code_stage(raw)
        if sum(p.numel() for p in parameters) != 654386688: raise ValueError('All original language weights must be trainable')
        optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.01, foreach=False)
        amp = lambda: torch.autocast('cuda',dtype=torch.bfloat16)
        sources = dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
            run_plan_sha256=file_hash(REPORTS/'plan.json'), human_source_admission_sha256=file_hash(SOURCE/'admission.json'))
        kernel = FullLanguageStage(raw, optimizer, source, code_encoder, replay, replay_encoder,
            plan['training_options'], sources, device, amp)
        if kernel.steps != 3203 or len(replay.families) != 29: raise ValueError('Actual full102487-row epoch and all29prepared replay families required')
        if phase == 'capacity':
            # Allocate real Adam in the actual update path, then exercise every
            # retained original per-family extreme. Disposable weights only.
            kernel.update(); torch.cuda.synchronize(device)
            index = source.pack.index[source.pack.positions[source.positions]]
            candidates = [dict(kind='human', position=int(np.argmax(index['length']))),
                dict(kind='human', position=int(np.argmax(index['length']-index['target_start'])))]
            checks = json.loads(SOURCE_CHECK.read_text())['selected_actual_replay_batches']
            candidates += [dict(kind='replay', family=f['family'], check=c) for f in checks for c in f['checks']]
            observations=[]
            for candidate in candidates:
                def encode():
                    if candidate['kind']=='human':
                        return code_encoder.batch([dict(human_program_index=candidate['position'])],'response',device)
                    c=candidate['check'];family=candidate['family'];row=replay.row(family,c['position'])
                    batch=replay_encoder.batch([row],replay_field(family),device)
                    ids=batch['input_ids'][0].tolist()
                    if hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()!=c['input_sha256']:
                        raise ValueError('Actual original source-capacity tokens differ')
                    return batch
                batch=_validated(encode); optimizer.zero_grad(set_to_none=True)
                with amp(): loss=raw(**batch)['loss']
                if not bool(torch.isfinite(loss)):raise ValueError('Finite full original capacity loss required')
                loss.backward()
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):
                    raise ValueError('Every original language weight needs a finite capacity gradient')
                if any(p.grad is not None for p in frozen):raise ValueError('Own vision/critic remain frozen')
                observations.append(dict(**candidate, context=batch['input_ids'].shape[1],
                    targets=int((batch['labels'][:,1:]!=-100).sum()), image_size=list(batch['pixel_values'].shape[-2:]) if batch['pixel_values'] is not None else None))
                optimizer.zero_grad(set_to_none=True)
            proof=dict(rank=rank, actual_GPU=torch.cuda.get_device_name(device), total_parameters=706012672,
                trainable_language_parameters=654386688, full_Adam_allocated=True,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                original_source_probes=observations, model_exported=False, training_candidate_updates=0,goal_achieved=False)
            proofs=[None]*4;dist.all_gather_object(proofs,proof)
            if rank==0:atomic_json(REPORTS/'capacity.json',dict(passed=True,actual_four_L40_workers=True,proofs=proofs,model_exported=False,goal_achieved=False))
            return
        capacity=json.loads((REPORTS/'capacity.json').read_text())
        if capacity.get('passed') is not True or not (REPORTS/'baseline-complete.json').exists():
            raise ValueError('Actual own-model capacity and fixed matching baseline assessments required')
        if rank==0: shutil.copy2(TOKEN,OUTPUT/'tokenizer.json')
        dist.barrier()
        started=time.monotonic()
        for step in range(kernel.steps):
            ownership(plan);_validated(lambda:guard.check())
            tick=time.monotonic();record=kernel.update();torch.cuda.synchronize(device)
            record['step_seconds']=time.monotonic()-tick
            if rank==0:
                # Full per-update history is also retained in each resumable save.
                with (REPORTS/'training-metrics.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                if step<2 or (step+1)%8==0:
                    status('training full own language model on all admitted original human programs',completed_steps=step+1,
                        planned_steps=kernel.steps,source_exposures=(step+1)*32,latest_metric=record,
                        training_seconds=time.monotonic()-started)
                    print('actual_APPS_full_language_update',step+1,record['summed_objective'],flush=True)
            if step+1 in [1,16] or (step+1)%128==0 or step+1==kernel.steps:
                _validated(lambda:guard.check(full=True))
                kernel.save(OUTPUT/'resume.pt',MAX_STATE,RESERVE+MAX_STATE)
                if rank==0:atomic_json(REPORTS/'checkpoint-receipt.json',dict(steps=step+1,checkpoint_sha256=file_hash(OUTPUT/'resume.pt'),
                    bytes=(OUTPUT/'resume.pt').stat().st_size,full_language_weights_Adam_all_worker_RNG_and_cursors=True,goal_achieved=False))
        _validated(lambda:guard.check(full=True))
        if rank==0:
            final=dict(model=_cpu(raw.state_dict()),config=metadata['config'],stage='sft',training_kind='complete_APPS_human_solution_language_v1',
                tokenizer_sha256=plan['tokenizer_sha256'],steps=kernel.completed,planned_steps=kernel.steps,
                training_options=plan['training_options'],source_checkpoint_sha256=plan['base_checkpoint_sha256'],
                run_plan_sha256=file_hash(REPORTS/'plan.json'),data_sha256=file_hash(SOURCE/'admission.json'),
                image_prompt_coordinates='letterbox input',truncated_sequences=0,candidate_promoted=False,goal_achieved=False)
            atomic_checkpoint(OUTPUT/'model.pt',final,4*1024**3,RESERVE)
            atomic_json(REPORTS/'training-complete.json',dict(checkpoint_sha256=file_hash(OUTPUT/'model.pt'),steps=kernel.steps,
                actual_source_exposures=kernel.budget['actual_source_exposures'],training_seconds=time.monotonic()-started,
                whole_APPS_source_pass=True,full_prior_source_pass=False,candidate_promoted=False,goal_achieved=False))
        dist.barrier()
    finally: dist.destroy_process_group()


def assess(phase, suite):
    import torch
    from slm_lab import code_quality,cpp_evaluation,knowledge_evaluation,object_evaluation,ocr_evaluation,vision_evaluation,public_code_benchmark
    plan=register(phase+'-assessment-'+str(suite))
    guard=FileBindings({**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    guard.check(full=True);torch.set_num_threads(1)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(suite):raise ValueError('Exact isolated assessment GPU required')
    checkpoint=BASE if phase=='baseline' else OUTPUT/'model.pt';folder=REPORTS/phase;folder.mkdir(exist_ok=True)
    def run(function,name,*args,**kwargs):
        ownership(plan);function(checkpoint,*args,folder/name,device='cuda:0',**kwargs);guard.check(full=True)
    if suite==0:
        if phase!='baseline':
            for mode in ['completion','instruction']:run(code_quality.evaluate,'code-'+mode+'.json',max_new_tokens=512,prompt_mode=mode)
        run(public_code_benchmark.evaluate,'public-Python.json',ROOT/'data/public-code-assessment-v1/questions.jsonl',max_new_tokens=512)
        run(cpp_evaluation.evaluate,'C++.json',max_new_tokens=512)
    elif suite==1:
        if phase!='baseline':run(knowledge_evaluation.evaluate,'knowledge.json',ROOT/'data/knowledge-eval-v1/questions.jsonl',max_examples=1000,max_new_tokens=32)
        # The existing complete actual-systems knowledge/code reports are
        # already runtime-bound and reused for baseline; no model selection.
    elif suite==2:
        run(ocr_evaluation.evaluate,'ocr.json',ROOT/'data/alignment-curriculum-v1/images.valid.jsonl',max_examples=256,max_new_tokens=128,image_size=448)
    else:
        run(object_evaluation.evaluate,'objects.json',ROOT/'data/object-qa-v1/valid.jsonl',max_examples=180,max_new_tokens=32)
        run(vision_evaluation.evaluate,'vision.json',ROOT/'data/alignment-curriculum-v1/contrastive.valid.jsonl',max_examples=128,image_size=448)
    atomic_json(folder/('suite-'+str(suite)+'-complete.json'),dict(checkpoint_sha256=file_hash(checkpoint),suite=suite,goal_achieved=False))


def parent():
    import shutil
    if REPORTS.exists() or OUTPUT.exists():raise ValueError('Preserve existing full epoch; no implicit restart')
    def idle():
        actor=resources.same(WAITER)
        children=Path('/proc')/str(WAITER['pid'])/'task'/str(WAITER['pid'])/'children'
        if (not actor or actor['state']!='S' or children.read_text().strip()
                or json.loads((ROOT/'reports/dialogue-retention-v1/status.json').read_text())['status']!=WAIT_STATE):
            raise ValueError('Never preempt active original training; exact childless photo-data wait required')
    idle()
    if '1 passed' not in (ROOT/'reports/apps-full-language-v1-kernel-final-tests.log').read_text():raise ValueError('Actual four-worker recovery/objective checks required')
    if '1 passed' not in (ROOT/'reports/apps-full-language-v1-driver-release-tests.log').read_text():raise ValueError('Active original work rejection check required')
    if shutil.disk_usage(ROOT).free<RESERVE+3*MAX_STATE:raise ValueError('Fund complete own model/Adam slots above26GiB reserve')
    old=json.loads((ROOT/'reports/attention-allocation-pilot-v1/plan.json').read_text())
    files=dict(old['source_files_sha256'])
    from slm_lab.apps_human_solution_view_v1 import AppsHumanSolutions
    from slm_lab.evaluation_provenance import capture
    source=AppsHumanSolutions(SOURCE);files.update({str(p):h for p,h in source.guard.files.items()})
    check=json.loads(SOURCE_CHECK.read_text());files.update(check['source_files_sha256'])
    if file_hash(BASE)!='fc8e248eb8304cfd3895ed103f5cf172641e6151017b9aa0d83e19a443e444f0':raise ValueError('Exact own completed systems continuation required')
    paths=[BASE,TOKEN,SOURCE_CHECK,Path(__file__).resolve(),ROOT/'src/slm_lab/apps_full_language_v1.py',
        ROOT/'tests/test_apps_full_language_v1.py',ROOT/'reports/apps-full-language-v1-kernel-final-tests.log',
        ROOT/'tests/test_apps_full_language_driver_v1.py',ROOT/'reports/apps-full-language-v1-driver-release-tests.log']
    base_reports=ROOT/'reports/execution-code-pilot-v1/completed_systems_assessment'
    runtime=capture()
    for name in ['code-completion.json','code-instruction.json','knowledge.json']:
        path=base_reports/name;r=json.loads(path.read_text())
        if r['checkpoint_sha256']!=file_hash(BASE) or r['evaluation_provenance']!=runtime:raise ValueError('Exact completed native baseline/runtime required')
        paths.append(path)
    for p in paths:files[str(p)]=file_hash(p)
    FileBindings(files).check(full=True)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);idle()
        memory=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.splitlines()
        if len(memory)!=4 or any(int(v.strip())>128 for v in memory):raise ValueError('All four physical GPUs must be idle')
        REPORTS.mkdir();OUTPUT.mkdir()
        plan=dict(format='own706M-complete-APPS-human-language-epoch-v1',created_at=datetime.now(timezone.utc).isoformat(),
            owner=resources.identity(os.getpid()),waiter=WAITER,source_files_sha256=files,base_checkpoint_sha256=file_hash(BASE),
            tokenizer_sha256=file_hash(TOKEN),source_population=102487,training_options=dict(seed=931001,code_accumulation=8,
                replay_accumulation=2,image_size=448,learning_rate=1e-5,weight_decay=.01,replay_coefficient=.5,gradient_clip=1.),
            world_size=4,total_parameters=706012672,trainable_parameters=654386688,optimizer_steps=3203,
            original_human_program_exposures=102496,rounding_repeats=9,prepared_replay_families=29,full_prior_source_pass=False,
            final_all_source_production=False,maximum_seconds=8*3600,matched_full_math_CPU_assessment_separate=True,
            fixed_budget_before_assessment=True,automatic_promotion=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        watcher=subprocess.Popen(['.venv/bin/python','-u',str(Path(__file__).resolve()),'--watchdog'],cwd=ROOT,start_new_session=True)
        started=time.monotonic()
        def interrupted(signum,frame):raise KeyboardInterrupt('Full epoch owner interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        def launch(commands,phase):
            children=[]
            for i,(args,env) in enumerate(commands):
                with (REPORTS/(phase+'-'+str(i)+'.log')).open('xb') as stream:
                    p=subprocess.Popen(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                        env={**os.environ,'OMP_NUM_THREADS':'1','TOKENIZERS_PARALLELISM':'false','CUBLAS_WORKSPACE_CONFIG':':4096:8',**env})
                with (REPORTS/'children.lock').open('a+') as lock:
                    fcntl.flock(lock.fileno(),fcntl.LOCK_EX);registry=json.loads((REPORTS/'children.json').read_text())
                    registry.append(dict(**resources.identity(p.pid),subject=phase+'-launcher-'+str(i)));atomic_json(REPORTS/'children.json',registry)
                children.append(p)
            while any(p.poll() is None for p in children):
                if time.monotonic()-started>plan['maximum_seconds']:raise ValueError('Bounded8hour GPU lease expired; inspect complete saved progress before explicit recovery')
                if any(p.poll() not in [None,0] for p in children):raise ValueError('An owned worker failed; preserve state, no automatic retry')
                status('full APPS data experiment active',phase=phase,launchers=[dict(pid=p.pid,exit_code=p.poll()) for p in children])
                time.sleep(2)
            if any(p.returncode!=0 for p in children):raise ValueError('An owned phase did not complete')
        def four(phase):return [(['.venv/bin/python','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4',str(Path(__file__).resolve()),'--worker',phase],{})]
        def suites(phase):return [(['.venv/bin/python','-u',str(Path(__file__).resolve()),'--assess',phase,'--suite',str(i)],{'CUDA_VISIBLE_DEVICES':str(i)}) for i in range(4)]
        try:
            idle();os.kill(WAITER['pid'],signal.SIGSTOP)
            deadline=time.monotonic()+5
            while (resources.same(WAITER) or {}).get('state') not in ['T','t']:
                if time.monotonic()>deadline:raise ValueError('Original idle waiter did not enter the owned hold')
                time.sleep(.05)
            launch(four('capacity'),'capacity')
            baseline=REPORTS/'baseline';baseline.mkdir()
            for name in ['code-completion.json','code-instruction.json','knowledge.json']:shutil.copy2(base_reports/name,baseline/name)
            launch(suites('baseline'),'baseline')
            atomic_json(REPORTS/'baseline-complete.json',dict(checkpoint_sha256=file_hash(BASE),native_reports={p.name:file_hash(p) for p in baseline.glob('*.json')},goal_achieved=False))
            launch(four('train'),'train')
            # The full matched1319-question/512-token CPU protocol does not
            # occupy or extend the GPU lease. Its original baseline is already
            # running independently; preserve both native complete outputs.
            math_folder=ROOT/'reports/apps-full-language-CPU-math-v1'
            if (math_folder/'candidate-launch.json').exists() or (math_folder/'candidate.json').exists():
                raise ValueError('Preserve earlier full native math candidate assessment')
            with (math_folder/'candidate.log').open('xb') as stream:
                child=subprocess.Popen(['.venv/bin/python','-u','-m','slm_lab.math_evaluation','--checkpoint',str(OUTPUT/'model.pt'),
                    '--data','data/math-reasoning-v2/test.jsonl','--output',str(math_folder/'candidate.json'),'--device','cpu','--max-new-tokens','512'],
                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1'})
            atomic_json(math_folder/'candidate-launch.json',dict(**resources.identity(child.pid),checkpoint_sha256=file_hash(OUTPUT/'model.pt'),
                device='cpu',precision='fp32',full_official1319=True,generation_budget=512,model_updates=0,goal_achieved=False))
            launch(suites('candidate'),'candidate')
            status('complete human-code epoch and fixed GPU assessments completed; inspect results; no promotion')
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=str(error),automatic_retry=False,goal_achieved=False));raise
        finally:
            resources.restore(plan,'full APPS experiment ended; exact original image-data waiter restored')
            watcher.wait(timeout=20)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--worker',choices=['capacity','train']);p.add_argument('--assess',choices=['baseline','candidate']);p.add_argument('--suite',type=int);p.add_argument('--watchdog',action='store_true');a=p.parse_args()
    if a.watchdog:resources.watchdog()
    elif a.worker:worker(a.worker)
    elif a.assess:assess(a.assess,a.suite)
    else:parent()
