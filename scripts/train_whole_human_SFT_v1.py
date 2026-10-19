"""Owned full-source human SFT learning-rate comparison and native assessment."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import train_point_grounding_joint_v1 as owner
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/whole-human-SFT-v1'
OUTPUT=ROOT/'checkpoints/whole-human-SFT-v1'
DATA=ROOT/'data/whole-human-alignment-v2'
MAX_STATE=192*1024**2
RESERVE=26*1024**3


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS


def previous_ended():
    folder=ROOT/'reports/own-alignment-four-L40-v2'
    plan=json.loads((folder/'plan.json').read_text())
    if (owner.resources.same(plan['owner']) or any(owner.resources.same(a) for a in json.loads((folder/'children.json').read_text()))
            or not (folder/'restored.json').exists()):
        raise ValueError('Every exact previous GPU owner/worker must end and restore the original waiter')
    if not json.loads((folder/'complete.json').read_text())['all_four_original_L40_capacity_and_exact_recovery_passed']:
        raise ValueError('Actual full-size four-L40 source-extreme capacity evidence required')


def registered(subject,gpu):
    plan=json.loads((REPORTS/'plan.json').read_text());me=owner.resources.identity(os.getpid())
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(gpu):
        raise ValueError('Exact isolated owned physical GPU required before CUDA initialization')
    while True:
        owner.owned(plan)
        if any(a['pid']==me['pid'] and a['ticks']==me['ticks'] and a['subject']==subject
               for a in json.loads((REPORTS/'children.json').read_text())):
            return plan
        time.sleep(.05)


def runtime(plan):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.85,0)
    if torch.cuda.get_device_name(0)!='NVIDIA L40':raise ValueError('Actual original L40 required')
    guard=FileBindings({**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    return torch,guard,lambda:torch.autocast('cuda',dtype=torch.bfloat16)


def signature(parameters):
    return [(id(p),p._version,p.data_ptr()) for p in parameters]


def probe_batch():
    from slm_lab.point_grounding_data_v1 import PointerData
    pointer=PointerData(ROOT/'data/localized-pointer-candidate-v1','train','audit')
    position=json.loads((ROOT/'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())['original_source_position']
    return pointer.batch([position],'cuda:0'),pointer.rows[position]['id']


def native_original(model,batch,amp):
    import torch
    model.eval()
    with torch.no_grad(),amp():
        return {r:model.original(batch['input_ids'],batch['pixel_values'],route=r)['logits'].cpu()
                for r in ('general','code','grounding')}


def worker(phase,arm,gpu):
    subject=('capacity-'+str(gpu)) if phase=='capacity' else ('initial-human' if phase=='initial' else 'train-'+arm)
    plan=registered(subject,gpu);torch,guard,amp=runtime(plan)
    import random
    import numpy as np
    from slm_lab.codenet_checkpoint_v1 import capture_rng,atomic_checkpoint
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_stage_v1 import WholeHumanSFTStage,make_optimizer
    from slm_lab.whole_human_sft_inference_v1 import fresh,sources,export,export_initial,load
    from slm_lab.whole_human_sft_evidence_v1 import fixed_digest,language_digest,tensor_digest,evaluate_human
    from slm_lab.own_aligned_assistant_v1 import byte_ctc_loss
    if phase=='initial':
        checkpoint=OUTPUT/'initial/model.pt'
        model,metadata=load(checkpoint,REPORTS/'plan.json','initial',expected_checkpoint_sha256=file_hash(checkpoint),
            expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
        data=WholeHumanData(DATA,'dialogue','valid','evaluation')
        evaluate_human(model.aligned,data,REPORTS/'initial/human-validation.json',amp,lambda:owner.owned(plan))
        native_suite(plan,model,metadata,'initial','knowledge',gpu,guard)
        guard.check(full=True);return
    model,metadata=fresh(plan,'cuda:0')
    if tensor_digest(model.parts.state_dict().items())!=plan['initial_parts_digest']:
        raise ValueError('Fresh original zero-factor/own-norm seed initialization required; all capacity weights are discarded')
    selected_arm='rate_2e5' if phase=='capacity' else arm
    options={**plan['training_options'],'learning_rate':plan['arms'][selected_arm]['learning_rate']}
    data=WholeHumanData(DATA,'dialogue','train','optimization')
    optimizer=make_optimizer(model,options);model.aligned.gradient_checkpointing=True
    random.seed(options['seed']);np.random.seed(options['seed']);torch.manual_seed(options['seed'])
    kernel=WholeHumanSFTStage(model,optimizer,data,options,sources(plan,file_hash(REPORTS/'plan.json'),selected_arm),'cuda:0',amp)
    folder=OUTPUT/('capacity-'+str(gpu) if phase=='capacity' else arm)
    if phase=='capacity':folder.mkdir()
    elif {p.name for p in folder.iterdir()}!={'tokenizer.json'}:
        raise ValueError('A fresh exact-tokenizer-only profile folder is required before training')
    batch,photo_id=probe_batch();before=native_original(model,batch,amp)
    with torch.no_grad(),amp():
        cache=model.original.language.code.prefill(batch['input_ids'])['cache']
        cached_before=model.original.language.code.decode(torch.tensor([[4]],device='cuda:0'),cache)['logits'].cpu()
    fixed_before=fixed_digest(model);versions=signature(kernel.fixed);language_before=language_digest(model)
    if phase=='capacity' and gpu==0:
        atomic_checkpoint(OUTPUT/'initial/model.pt',export_initial(model,plan,file_hash(REPORTS/'plan.json')),64*1024**2,RESERVE)
    started=time.monotonic()
    if phase=='capacity':
        kernel.update();kernel.save(folder/'resume.pt',MAX_STATE,RESERVE)
        def random_draws():
            return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device='cuda:0'))]
        expected_draws=random_draws();expected=kernel.update();expected_language=language_digest(model)
        def Adam_digest():
            values=[]
            for i,state in enumerate(optimizer.state.values()):
                values.extend((str(i)+'.'+k,state[k]) for k in ('step','exp_avg','exp_avg_sq'))
            return tensor_digest(values)
        expected_Adam=Adam_digest()
        saved=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True,mmap=True)
        # Corrupt actual private CUDA RNG must reject before any live tensor,
        # optimizer, cursor, metrics or caller RNG changes.
        bad={**saved,'core':{**saved['core'],'rng':{**saved['core']['rng'],'cuda':torch.zeros(4,dtype=torch.uint8)}}}
        rng=capture_rng('cuda:0');live_signature=signature(model.parameters())
        try:kernel.resume(bad)
        except ValueError:pass
        else:raise ValueError('Corrupt actual private CUDA generator state must reject')
        if (live_signature!=signature(model.parameters()) or not torch.equal(rng['torch'],torch.get_rng_state())
                or not torch.equal(rng['cuda'],torch.cuda.get_rng_state()) or expected_Adam!=Adam_digest()):
            raise ValueError('Invalid CUDA recovery changed actual live state')
        kernel.resume(saved);actual_draws=random_draws();actual=kernel.update()
        if (expected!=actual or expected_draws!=actual_draws or expected_language!=language_digest(model) or expected_Adam!=Adam_digest()):
            raise ValueError('Actual whole-human accumulation/source/Adam/RNG recovery must be exact')
    else:
        with (REPORTS/(arm+'-metrics.jsonl')).open('x') as stream:
            for _ in range(kernel.options['steps']):
                owner.owned(plan);tick=time.monotonic();record=kernel.update()
                stream.write(json.dumps(record)+'\n');stream.flush()
                if kernel.completed%25==0 or kernel.completed==1 or kernel.completed==kernel.options['steps']:
                    print(json.dumps(dict(**record,elapsed_seconds=time.monotonic()-started,step_seconds=time.monotonic()-tick)),flush=True)
                    atomic_json(REPORTS/(arm+'-progress.json'),dict(step=kernel.completed,planned_steps=kernel.options['steps'],
                        latest=record,elapsed_seconds=time.monotonic()-started,goal_achieved=False))
                if kernel.completed%256==0 or kernel.completed==kernel.options['steps']:
                    kernel.save(folder/'resume.pt',MAX_STATE,RESERVE)
        atomic_checkpoint(folder/'model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),arm,kernel.completed),64*1024**2,RESERVE)
    after=native_original(model,batch,amp)
    with torch.no_grad(),amp():
        cached_after=model.original.language.code.decode(torch.tensor([[4]],device='cuda:0'),cache)['logits'].cpu()
        aligned=model.aligned(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
    if (any(not torch.equal(before[r],after[r]) for r in before) or not torch.equal(cached_before,cached_after)
            or fixed_before!=fixed_digest(model) or versions!=signature(kernel.fixed) or language_before==language_digest(model)):
        raise ValueError('Every fixed original/unused byte/version/native route/cache must remain exact while actual language weights learn')
    common=dict(actual_counts=model.counts(),actual_updates=kernel.completed,complete_Adam_states=len(optimizer.state),
        all_fixed_bytes_versions_and_original_native_routes_exact=True,pre_existing_original_code_cache_exact=True,
        initial_language_digest=language_before,trained_language_digest=language_digest(model),fixed_parameter_digest=fixed_before,
        original_native_probe_photo_id=photo_id,actual_peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        actual_peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_seconds=time.monotonic()-started,
        genuine_human_dialogue_SFT_not_reward_DPO_PPO=True,candidate_promoted=False,goal_achieved=False)
    if phase=='capacity':
        atomic_json(REPORTS/f'capacity-{gpu}.json',dict(**common,physical_GPU=gpu,
            actual_next_update_all_language_tensors_Adam_cursor_and_all4RNG_exact=True,
            actual_invalid_private_CUDA_RNG_rejected_before_mutation=True,logical_disposable_updates=2,physical_optimizer_calls=3,
            production_updates=0,resume_sha256=file_hash(folder/'resume.pt')))
        return
    checkpoint=folder/'model.pt';probe=REPORTS/arm/'trained-probe.pt'
    atomic_checkpoint(probe,dict(aligned=aligned,original=after),8*1024**2,RESERVE)
    receipt=dict(**common,checkpoint_sha256=file_hash(checkpoint),resume_sha256=file_hash(folder/'resume.pt'),
        original_backbones_loaded=metadata['original_backbones_loaded'],source_exposure_budget=plan['source_exposure_budget'])
    atomic_json(REPORTS/(arm+'-training-complete.json'),receipt)
    valid=WholeHumanData(DATA,'dialogue','valid','evaluation')
    evaluate_human(model.aligned,valid,REPORTS/arm/'human-validation.json',amp,lambda:owner.owned(plan))
    # Actual completed inference factory reload, with every new tensor and
    # all original/aligned native routes checked on the full model.
    cold,cold_metadata=load(checkpoint,REPORTS/'plan.json',arm,expected_checkpoint_sha256=receipt['checkpoint_sha256'],
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    with torch.no_grad(),amp():
        cold_aligned=cold.aligned(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
    if (not torch.equal(aligned,cold_aligned) or any(not torch.equal(after[r],value) for r,value in native_original(cold,batch,amp).items())
            or language_digest(cold)!=language_digest(model) or cold_metadata['original_backbones_loaded']!=1):
        raise ValueError('Actual complete trained single-backbone export and all native routes must cold-reload exactly')
    atomic_json(REPORTS/arm/'trained-shared-reload.json',dict(actual_counts=cold.counts(),
        full_trained_aligned_and_original_3routes_exact=True,one_backbone_loaded=True,goal_achieved=False))
    guard.check(full=True)


def native_suite(plan,model,metadata,arm,suite,gpu,guard,shard=None):
    checkpoint=OUTPUT/arm/'model.pt';destination=REPORTS/arm
    def evaluate(module,name,*args,**kwargs):
        real=module.load_checkpoint
        def loader(path,device,training_state=False):
            if Path(path)!=checkpoint or training_state:raise ValueError('Only exact source-bound own aligned inference may use this native loader')
            return model.aligned,dict(metadata,explicit_route='aligned',training_options=dict(image_size=448))
        module.load_checkpoint=loader
        try:
            owner.owned(plan);module.evaluate(checkpoint,*args,destination/name,device='cuda:0',**kwargs)
        finally:module.load_checkpoint=real
        guard.check(full=True)
    if suite=='knowledge':
        from slm_lab import knowledge_evaluation as module
        evaluate(module,'knowledge.json',ROOT/'data/knowledge-eval-v1/questions.jsonl',max_examples=1000,max_new_tokens=32)
    elif suite=='code':
        from slm_lab import code_quality as module
        for mode in ('completion','instruction'):evaluate(module,'code-'+mode+'.json',max_new_tokens=512,prompt_mode=mode)
    elif suite=='cpp':
        from slm_lab import cpp_evaluation as module
        evaluate(module,'C++.json',max_new_tokens=512)
    elif suite=='vision':
        from slm_lab import ocr_evaluation,object_evaluation,vision_evaluation
        evaluate(ocr_evaluation,'ocr.json',ROOT/'data/alignment-curriculum-v1/images.valid.jsonl',max_examples=256,max_new_tokens=128,image_size=448)
        evaluate(object_evaluation,'objects.json',ROOT/'data/object-qa-v1/valid.jsonl',max_examples=180,max_new_tokens=32)
        evaluate(vision_evaluation,'vision.json',ROOT/'data/alignment-curriculum-v1/contrastive.valid.jsonl',max_examples=128,image_size=448)
    elif suite=='public':
        from slm_lab import public_code_benchmark as module
        evaluate(module,f'public-{shard}.json',ROOT/'reports/own-code-delta-public-v1/inputs'/f'shard-{shard}'/'questions.jsonl',max_new_tokens=512)
    elif suite=='math':
        from slm_lab import full_math_shards_v1 as module
        real=module.load_checkpoint
        module.load_checkpoint=lambda path,device,training_state=False:(model.aligned,metadata)
        try:
            module.evaluate_shard(checkpoint,ROOT/'data/math-reasoning-v2/test.jsonl',destination/f'math-{shard}.json',
                shard=shard,shards=4,device='cuda:0',max_new_tokens=512,checked=lambda:(owner.owned(plan),guard.check()))
        finally:module.load_checkpoint=real
        guard.check(full=True)
    else:raise ValueError('Exact predeclared native capability suite required')


def assess(arm,suite,gpu,shard):
    subject='assess-'+arm+'-'+suite+'-'+str(shard if shard is not None else gpu)
    plan=registered(subject,gpu);torch,guard,amp=runtime(plan)
    from slm_lab.whole_human_sft_inference_v1 import load
    checkpoint=OUTPUT/arm/'model.pt'
    model,metadata=load(checkpoint,REPORTS/'plan.json',arm,expected_checkpoint_sha256=file_hash(checkpoint),
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    native_suite(plan,model,metadata,arm,suite,gpu,guard,shard)


def parent():
    if REPORTS.exists() or OUTPUT.exists():raise ValueError('Preserve every prior attempt; no automatic restart')
    previous_ended();owner.idle_original()
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import budget,tensor_digest
    from slm_lab.whole_human_sft_inference_v1 import fresh
    from slm_lab.evaluation_provenance import capture
    data=WholeHumanData(DATA,'dialogue','train','optimization')
    if '4 passed' not in (ROOT/'reports/whole-human-SFT-kernel-and-loader-v1-tests.log').read_text():
        raise ValueError('Actual source normalization, frozen cache/version, recovery/corruption and completed loader checks required')
    old_path=ROOT/'reports/own-onpolicy-pilot-v2/plan.json';old=json.loads(old_path.read_text())
    receipt=json.loads((ROOT/'reports/own-onpolicy-pilot-v2/CE_control-training-complete.json').read_text())
    options=dict(seed=947171,steps=math.ceil(len(data)/4),microbatch=1,accumulation=4,betas=[.9,.95],weight_decay=.01,gradient_clip=1.)
    capacity=json.loads((ROOT/'reports/own-alignment-four-L40-v2/complete.json').read_text())
    settings=dict(own_code_plan=str(old_path),own_code_plan_sha256=file_hash(old_path),
        own_code_checkpoint=str(ROOT/'checkpoints/own-onpolicy-pilot-v2/CE_control/model.pt'),own_code_checkpoint_sha256=receipt['trained_components_sha256'],
        own_code_reference=str(ROOT/'reports/own-onpolicy-pilot-v2/reference.json'),own_code_reference_sha256=receipt['reference_scores_sha256'],
        model_initialization=dict(rank=16,reward_hidden=128,word_temporal=96,seed=947131,parameter_limit=10**9,word_frame_expansion=5),
        actual_counts=capacity['proofs'][0]['actual_counts'],training_options=options,
        arms=dict(rate_1e5=dict(learning_rate=1e-5,physical_GPU=0),rate_2e5=dict(learning_rate=2e-5,physical_GPU=1),rate_5e5=dict(learning_rate=5e-5,physical_GPU=2)),
        source_manifest_sha256=file_hash(DATA/'manifest.json'),source_admission_sha256=file_hash(DATA/'admission.json'),
        source_exposure_budget=budget(data,options),selection='minimum_complete_held_human_token_NLL; retain initial if no arm improves; lexicographic arm tie break; no capability-based arm selection')
    model,_=fresh(settings,'cpu');settings['initial_parts_digest']=tensor_digest(model.parts.state_dict().items());del model
    files=dict(json.loads((ROOT/'reports/own-alignment-four-L40-v2/plan.json').read_text())['source_files_sha256'])
    # Bind current new code and exact immutable evaluation inputs. Old bound
    # source files are rechecked, never silently resealed around an edit.
    for name,sha in capture()['files_sha256'].items():
        path=str(ROOT/name)
        if path in files and files[path]!=sha:raise ValueError('Existing bound neural/evaluation source changed: '+path)
        files[path]=sha
    names=['scripts/train_whole_human_SFT_v1.py','src/slm_lab/whole_human_sft_stage_v1.py',
        'src/slm_lab/whole_human_sft_inference_v1.py','src/slm_lab/whole_human_sft_evidence_v1.py',
        'tests/test_whole_human_sft_stage_v1.py','tests/test_whole_human_sft_inference_v1.py',
        'reports/whole-human-SFT-kernel-and-loader-v1-tests.log','tests/test_whole_human_SFT_driver_v1.py','reports/whole-human-SFT-driver-v1-tests.log',
        'src/slm_lab/codenet_runner_v1.py','src/slm_lab/codenet_stage_v1.py','src/slm_lab/full_math_shards_v1.py',
        'src/slm_lab/point_grounding_data_v1.py','reports/frozen-base-pointer-full-CPU-v1/complete.json',
        'reports/localized-pointer-candidate-v1/independent-source-readback.json','data/localized-pointer-candidate-v1/manifest.json',
        'reports/own-alignment-four-L40-v2/complete.json','reports/own-alignment-four-L40-v2/independent-complete-readback.json',
        'reports/own-alignment-four-L40-v2/restored.json','reports/own-alignment-four-L40-v2/next-full-source-training-hypothesis.json',
        'data/knowledge-eval-v1/questions.jsonl','data/knowledge-eval-v1/questions.audit.json','data/math-reasoning-v2/manifest.json',
        'data/public-code-assessment-v1/manifest.json','data/alignment-curriculum-v1/images.valid.jsonl',
        'data/object-qa-v1/valid.jsonl','data/alignment-curriculum-v1/contrastive.valid.jsonl']
    for name in names:files[str(ROOT/name)]=file_hash(ROOT/name)
    for path in (ROOT/'reports/own-code-delta-public-v1/inputs').rglob('*'):
        if path.is_file():files[str(path)]=file_hash(path)
    pointer=ROOT/'data/localized-pointer-candidate-v1';pm=json.loads((pointer/'manifest.json').read_text())
    for name,sha in pm['files_sha256'].items():files[str(pointer/name)]=sha
    guard=FileBindings(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);previous_ended();owner.idle_original()
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(memory)!=4 or any(int(v.strip())>128 for v in memory):raise ValueError('Every actual original L40 must be idle')
        if shutil.disk_usage(ROOT).free<RESERVE+16*MAX_STATE:raise ValueError('Full source/optimizer atomic checkpoint reserve required')
        guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        for name in ['initial',*settings['arms']]:
            (REPORTS/name).mkdir();(OUTPUT/name).mkdir();shutil.copyfile(old['tokenizer'],OUTPUT/name/'tokenizer.json')
        plan=dict(**settings,created_at=datetime.now(timezone.utc).isoformat(),owner=owner.resources.identity(os.getpid()),waiter=owner.WAITER,
            source_files_sha256=files,maximum_seconds=8*3600,expires_at_unix=time.time()+8*3600,automatic_retry=False,
            all_capacity_updates_discarded=True,genuine_human_conversation_SFT_not_ranked_RLHF=True,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned full human SFT comparison interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        driver=str(Path(__file__).resolve())
        watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(REPORTS/'watchdog.log').open('xb'),stderr=subprocess.STDOUT,start_new_session=True)
        def launch(args,subject,gpu):return owner.launch(plan,['-u',driver,*args],subject,str(gpu))
        def tasks(tasks):
            children=[launch(args,subject,gpu) for args,subject,gpu in tasks];owner.wait_children(plan,children)
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);until=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>until:raise ValueError('Original sleeping childless waiter did not enter owned hold')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():raise ValueError('Original waiter began work; never preempt it')
            tasks([(['--worker','capacity','--gpu',str(i)],'capacity-'+str(i),i) for i in range(4)])
            proofs=[json.loads((REPORTS/f'capacity-{i}.json').read_text()) for i in range(4)]
            if not all(p['actual_next_update_all_language_tensors_Adam_cursor_and_all4RNG_exact'] for p in proofs):raise ValueError('Every actual accumulated-human SFT capacity/recovery probe must pass')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,all_four_L40=True,proofs=proofs,production_updates=0,goal_achieved=False))
            jobs=[launch(['--worker','initial','--gpu','3'],'initial-human',3)]
            for arm,spec in plan['arms'].items():jobs.append(launch(['--worker','train','--arm',arm,'--gpu',str(spec['physical_GPU'])],'train-'+arm,spec['physical_GPU']))
            owner.wait_children(plan,jobs)
            scores={arm:json.loads((REPORTS/arm/'human-validation.json').read_text())['overall']['token_mean_nll'] for arm in ['initial',*plan['arms']]}
            best=min(plan['arms'],key=lambda arm:(scores[arm],arm));selected=best if scores[best]<scores['initial'] else 'initial'
            atomic_json(REPORTS/'selection.json',dict(selected_arm=selected,held_human_token_NLL=scores,criterion=plan['selection'],capability_feedback_used=False,candidate_promoted=False,goal_achieved=False))
            policies=list(dict.fromkeys(['initial',selected]))
            if selected!='initial':tasks([(['--assess','knowledge','--arm',selected,'--gpu','0'],'assess-'+selected+'-knowledge-0',0)])
            work=[]
            for arm in policies:
                for suite in ('code','cpp','vision'):work.append((arm,suite))
            for start in range(0,len(work),4):
                tasks([(['--assess',suite,'--arm',arm,'--gpu',str(gpu)],'assess-'+arm+'-'+suite+'-'+str(gpu),gpu)
                       for gpu,(arm,suite) in enumerate(work[start:start+4])])
            from slm_lab.full_math_shards_v1 import reconcile as math_reconcile
            for arm in policies:
                for suite in ('public','math'):
                    tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i),'--shard',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i) for i in range(4)])
                math_reconcile([REPORTS/arm/f'math-{i}.json' for i in range(4)],ROOT/'data/math-reasoning-v2/test.jsonl',
                    OUTPUT/arm/'tokenizer.json',REPORTS/arm/'math-complete.json')
            from slm_lab.whole_human_sft_evidence_v1 import reconcile
            reconcile(ROOT,REPORTS,OUTPUT,data);guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(status='Three complete original human SFT arms and full matched native comparisons finished; inspect quality evidence',candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Owned full human SFT comparison ended; original waiter restored');watcher.wait(timeout=20)


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog',action='store_true');parser.add_argument('--worker',choices=['capacity','initial','train'])
    parser.add_argument('--assess',choices=['knowledge','code','cpp','vision','public','math'])
    parser.add_argument('--arm',default='initial');parser.add_argument('--gpu',type=int,choices=range(4),default=0);parser.add_argument('--shard',type=int,choices=range(4))
    args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker:worker(args.worker,args.arm,args.gpu)
    elif args.assess:assess(args.arm,args.assess,args.gpu,args.shard)
    else:parent()
