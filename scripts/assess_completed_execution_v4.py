"""Evaluate completed173-update policies; never rerun or modify their training."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/completed-execution-assessment-v4'
OUTPUT = ROOT/'checkpoints/execution-single-update-recovery-v3'
PREVIOUS = ROOT/'reports/execution-single-update-recovery-v3'
owner = legacy.owner


def configure():
    owner.REPORTS = REPORTS
    owner.resources.REPORTS = REPORTS
    legacy.REPORTS = REPORTS
    legacy.OUTPUT = OUTPUT


def previous_ready(prior):
    if owner.resources.same(prior['owner']):
        return False
    if any(owner.resources.same(p) for p in json.loads((PREVIOUS/'children.json').read_text())):
        return False
    if not (PREVIOUS/'restored.json').exists():
        raise ValueError('Every preceding exact worker must end and original work must be restored')
    for arm in ['verified_solution','execution_hybrid']:
        path = PREVIOUS/(arm+'-training-complete.json')
        if not path.exists() or json.loads(path.read_text()).get('actual_updates') != 173:
            raise ValueError('Both actual training arms must already complete all173updates; evaluation cannot replace training')
    return True


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Evaluation owner must hide every GPU')
    if REPORTS.exists():
        raise ValueError('Preserve previous complete or partial evaluation delivery')
    prior = json.loads((PREVIOUS/'plan.json').read_text())
    if not previous_ready(prior):
        raise ValueError('Never preempt a live training owner')
    from train_execution_single_update_recovery_v3 import policy_sources
    from slm_lab.own_tokenizer_package_v1 import ensure
    proof_path = ROOT/'reports/execution-final173-readback-v1/complete.json'
    proof = json.loads(proof_path.read_text())
    FileBindings(proof['source_files_sha256']).check(full=True)
    if (proof['actual_completed_updates_per_arm'] != 173
            or not proof['all_final_native_export_and_private_actor_tensors_match']):
        raise ValueError('All final actor/Adam/private state and actual native weights must reconcile first')
    tests = {'completed-execution-assessment-v4-driver-tests':2,'own-tokenizer-package-v1-tests':2}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():
            raise ValueError('Actual quiescence/completion and missing/corrupt tokenizer package checks required')
    package = ensure(ROOT/'data/tokenizer-corpus/tokenizer.json',OUTPUT,
        expected_sha256=prior['initial_dense_settings']['tokenizer_sha256'])
    files = dict(prior['source_files_sha256'])
    files.update(proof['source_files_sha256'])
    additions = [Path(__file__),proof_path,PREVIOUS/'plan.json',PREVIOUS/'restored.json',PREVIOUS/'failure.json',
        ROOT/'src/slm_lab/own_tokenizer_package_v1.py',ROOT/'tests/test_own_tokenizer_package_v1.py',
        ROOT/'tests/test_completed_execution_assessment_v4_driver.py',Path(package['path']),
        ROOT/'reports/completed-execution-tokenizer-packaging-v1.json']
    for arm in ['verified_solution','execution_hybrid']:
        additions += [OUTPUT/(arm+'-actor.pt'),PREVIOUS/(arm+'-training-complete.json')]
    for name in tests:
        additions.append(ROOT/'reports'/(name+'.log'))
    for path in additions:
        files[str(path)] = file_hash(path)
    me = owner.resources.identity(os.getpid())
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        owner.idle_original()
        used = subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(used) != 4 or any(int(v) > 128 for v in used):
            raise ValueError('Every original local GPU must be idle before evaluation ownership')
        guard = FileBindings(files)
        guard.check(full=True)
        REPORTS.mkdir()
        plan = dict(initial_dense_settings=prior['initial_dense_settings'],actual_counts=999706442,
            trained_run_plan=str(PREVIOUS/'plan.json'),trained_run_plan_sha256=file_hash(PREVIOUS/'plan.json'),
            trained_policy_sources=policy_sources(prior,file_hash(PREVIOUS/'plan.json')),
            completed_training_updates_per_arm=173,all_training_and_optimizer_artifacts_reused_unchanged=True,
            tokenizer_package=package,code_and_broad_precision='bf16_autocast',pulse_sources=prior['pulse_sources'],
            owner=me,waiter=owner.WAITER,source_files_sha256=files,expires_at_unix=time.time()+8*3600,
            evaluation_only=True,training_updates=0,automatic_retry=False,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan)
        atomic_json(REPORTS/'children.json',[])
        driver = str(Path(__file__).resolve())
        def interrupted(signum,frame):
            raise KeyboardInterrupt('Owned completed-policy evaluation interrupted')
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        watcher = subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        try:
            owner.idle_original()
            os.kill(owner.WAITER['pid'],signal.SIGSTOP)
            deadline = time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic() > deadline:
                    raise ValueError('Exact original idle waiter hold failed')
                time.sleep(.05)
            for arm in ['verified_solution','execution_hybrid']:
                for suite in ['code','broad']:
                    children = [owner.launch(plan,['-u',driver,'--assess',arm,'--suite',suite,'--gpu',str(gpu)],
                        'assess-'+arm+'-'+suite+'-'+str(gpu),str(gpu)) for gpu in range(4)]
                    owner.wait_children(plan,children)
                from slm_lab.native_code_shards_v1 import reconcile
                for frame in ['completion','instruction']:
                    reconcile([REPORTS/arm/f'code-{frame}-shard-{rank}.json' for rank in range(4)],REPORTS/arm/f'code-{frame}.json')
            guard.check(full=True)
            atomic_json(REPORTS/'training-complete.json',dict(matched_native_development_assessments_finished=True,
                original_completed_training_updates_per_arm=173,evaluation_only_training_updates=0,
                actual_training_state_readback=str(proof_path),code_and_broad_precision='bf16_autocast',
                independent_output_quality_readback_and_final_benchmarks_pending=True,candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,training_updates=0,goal_achieved=False))
            raise
        finally:
            owner.restore_owned(plan,'Completed-policy native evaluation ended; original work restored')
            watcher.wait(timeout=20)


def assess(arm, gpu, suite):
    plan = legacy.registered('assess-'+arm+'-'+suite+'-'+str(gpu),gpu)
    import torch
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.85,0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base,_ = fresh(plan,'cuda:0')
    model = OwnDenseActor(base)
    checkpoint = OUTPUT/(arm+'-actor.pt')
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if not torch.equal(model.parts['critic'].weight.cpu(),payload['parts']['critic.weight']):
        raise ValueError('Separate unused critic must match original own policy')
    model.import_export(payload,expected_sources=plan['trained_policy_sources'])
    del payload
    model.eval()
    metadata = dict(tokenizer_sha256=plan['initial_dense_settings']['tokenizer_sha256'],training_options=dict(image_size=448))
    folder = REPORTS/arm
    folder.mkdir(exist_ok=True)
    guard = FileBindings(plan['source_files_sha256'])
    def evaluate(module,name,data=None,**kwargs):
        saved = module.load_checkpoint
        module.load_checkpoint = lambda *a,**kw:(model.policy,metadata)
        try:
            owner.owned(plan)
            if data is None:
                module.evaluate(checkpoint,folder/name,device='cuda:0',**kwargs)
            else:
                module.evaluate(checkpoint,data,folder/name,device='cuda:0',**kwargs)
        finally:
            module.load_checkpoint = saved
    previous_threads = torch.set_num_threads
    torch.set_num_threads = lambda count:None
    try:
        with inference_cache():
            if suite == 'code':
                from slm_lab.native_code_shards_v1 import evaluate_shard
                for frame in ['completion','instruction']:
                    evaluate_shard(model.policy,metadata,checkpoint,folder,rank=gpu,world=4,frame=frame)
            elif gpu == 0:
                from slm_lab import knowledge_evaluation
                evaluate(knowledge_evaluation,'knowledge.json',Path(plan['pulse_sources']['knowledge']),max_examples=56,max_new_tokens=32)
            elif gpu == 1:
                from slm_lab import math_evaluation
                evaluate(math_evaluation,'math.json',ROOT/'data/math-reasoning-v2/test.jsonl',max_examples=16,max_new_tokens=256)
            elif gpu == 2:
                from slm_lab import ocr_evaluation,object_evaluation
                evaluate(ocr_evaluation,'ocr.json',Path(plan['pulse_sources']['ocr']),max_examples=32,max_new_tokens=128,image_size=448)
                evaluate(object_evaluation,'objects.json',Path(plan['pulse_sources']['objects']),max_examples=32,max_new_tokens=32)
            else:
                from slm_lab import vision_evaluation
                evaluate(vision_evaluation,'vision.json',Path(plan['pulse_sources']['vision']),max_examples=32,image_size=448)
    finally:
        torch.set_num_threads = previous_threads
    guard.check(full=True)
    atomic_json(folder/f'assessment-{suite}-{gpu}-complete.json',dict(physical_GPU=gpu,
        actual_counts=model.counts(),checkpoint_sha256=file_hash(checkpoint),
        native_development_comparison_not_final_independent_benchmark=True,goal_achieved=False))


if __name__ == '__main__':
    configure()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--assess',choices=['verified_solution','execution_hybrid'])
    parser.add_argument('--gpu',type=int,choices=range(4))
    parser.add_argument('--suite',choices=['code','broad'])
    args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.assess:assess(args.assess,args.gpu,args.suite)
    else:parent()
