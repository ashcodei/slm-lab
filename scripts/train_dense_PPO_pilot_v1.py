"""Owned bounded human-PPO/KL-only comparison after complete production DPO.

The predeclared 1,024-source prefix is not a full corpus epoch. Training uses
FP32 cached sampling and the exact separately counted original reward encoder.
Every native/held comparison is retained; no candidate is automatically promoted.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

owner=legacy.owner;ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-PPO-pilot-v1';OUTPUT=ROOT/'checkpoints/dense-PPO-pilot-v1';QUEUE=ROOT/'reports/dense-PPO-pilot-v1-queue.json'
PREVIOUS=ROOT/'reports/dense-human-preference-v2';PREVIOUS_QUEUE=ROOT/'reports/dense-human-preference-v2-queue.json'
CPU_PROOF=ROOT/'reports/dense-ppo-cached-full-CPU-v2';PROPOSAL=ROOT/'reports/dense-ppo-controlled-pilot-proposed-v2.json'
RESERVE=26*1024**3;MAX_STATE=512*1024**2


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS;legacy.REPORTS=REPORTS;legacy.OUTPUT=OUTPUT


def previous_ended(identity):
    if owner.resources.same(identity):return False
    children=PREVIOUS/'children.json'
    if children.exists() and any(owner.resources.same(a) for a in json.loads(children.read_text())):return False
    if (not all((PREVIOUS/n).exists() for n in ('complete.json','restored.json','independent-comparison.json'))
            or (PREVIOUS/'failure.json').exists()):
        raise ValueError('Exact complete production DPO owner/workers and successful original restoration required')
    return True


def require_idle_devices():
    owner.idle_original()
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    if len(used)!=4 or any(int(v.strip())>128 for v in used):raise ValueError('All four local L40s must be idle before this owned run')
    if shutil.disk_usage(ROOT).free<RESERVE+16*MAX_STATE:raise ValueError('All capacity/trained states and atomic-writer reserve required')


def validate_capacity(proofs,plan):
    if len(proofs)!=4:raise ValueError('Every physical local L40 must supply its own capacity proof')
    for i,(arm,proof) in enumerate(zip(plan['arms'],proofs)):
        full=proof['full_maximum_prompt_and_fixed_response_stress']
        if (proof['arm']!=arm or proof['physical_GPU']!=i or proof['actual_counts']!=plan['actual_counts']
                or proof['separate_training_only_reward']!=plan['reward_counts']
                or proof['actual_actor_critic_Adam_states']!=562 or proof['actual_actor_critic_values']!=15306752
                or proof['actual_updates']!=2 or not proof['exact_next_update_and_private_CUDA_RNG_rejection']
                or full['original_source']['original_train_position']!=plan['capacity_maximum_prompt_position']
                or full['full_fixed_action_tokens']!=128 or full['actual_prefills']!=1 or full['actual_decodes']!=127
                or not full['actor_critic_finite_gradients'] or not full['no_optimizer_step']):
            raise ValueError('Every actual configured profile, exact recovery and complete maximum-source/full128-action proof required')


def evaluate_held(model,plan,arm,amp):
    from tokenizers import Tokenizer
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.dense_human_preference_evaluation_v1 import evaluate as preferences
    from slm_lab.dense_word_evaluation_v1 import evaluate as words
    held=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation')
    preferences(model,held,REPORTS/arm/'held-preferences.json',beta=.1,amp=amp,checked=lambda:owner.owned(plan))
    data=SharedWordData(ROOT/'data/shared-byte-words-v1','valid','evaluation')
    words(model,data,Tokenizer.from_file(str(OUTPUT/arm/'tokenizer.json')),REPORTS/arm/'held-words.json',batch_size=16,generation_budget=64,
        amp=amp,checked=lambda:owner.owned(plan),provenance=dict(run_plan_sha256=file_hash(REPORTS/'plan.json'),
            checkpoint_sha256=file_hash(OUTPUT/arm/'model.pt'),inference_device_type='cuda',inference_precision='bf16'))


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def worker(phase,arm,gpu):
    subject='capacity-'+str(gpu) if phase=='capacity' else ('initial-policy' if phase=='initial' else 'train-'+arm)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    import numpy as np
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.dense_ppo_inference_v1 import fresh,options,sources,export,load
    from slm_lab.dense_ppo_reward_v1 import load as load_reward,signature
    from slm_lab.dense_ppo_control_v1 import DenseControlledPPOStage
    from slm_lab.dense_ppo_capacity_v1 import stress
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    model,metadata=fresh(plan,'cuda:0')
    if phase=='initial':
        atomic_checkpoint(OUTPUT/'initial/model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),'initial',0),96*1024**2,RESERVE)
        evaluate_held(model,plan,'initial',amp);guard.check(full=True);return
    if phase=='capacity':arm=list(plan['arms'])[gpu]
    setting=options(plan,arm);scale=plan['arms'][arm]['human_reward_scale']
    scorer=load_reward(plan['reward_inputs'],'cuda:0')
    if scorer.provenance!=plan['reward_provenance'] or scorer.counts()!=plan['reward_counts']:
        raise ValueError('The actual exact original-feature training-only reward basis and parameter inventory required')
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
    random.seed(setting['seed']);np.random.seed(setting['seed']);torch.manual_seed(setting['seed'])
    params=[p for _,p in model.named_selected(('actor','critic'))]
    optimizer=torch.optim.AdamW(params,lr=setting['learning_rate'],betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    kernel=DenseControlledPPOStage(model,optimizer,data,scorer,setting,sources(plan,file_hash(REPORTS/'plan.json')),'cuda:0',human_reward_scale=scale)
    folder=OUTPUT/('capacity-'+str(gpu) if phase=='capacity' else arm)
    if phase=='capacity':folder.mkdir()
    elif {p.name for p in folder.iterdir()}!={'tokenizer.json'}:raise ValueError('Fresh tokenizer-only bounded PPO arm required')
    fixed=signature(kernel.fixed);reward_fixed=signature(scorer.parameters());reward_digest=tensor_digest(scorer.named_parameters())
    initial=tensor_digest(model.parts.named_parameters());started=time.monotonic()
    def moments():return tensor_digest((str(i)+'.'+k,v[k]) for i,v in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
    if phase=='capacity':
        kernel.update();atomic_checkpoint(folder/'resume-step1.pt',kernel.state(),MAX_STATE,RESERVE)
        def draws():return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device='cuda:0'))]
        expected_draws=draws();expected=kernel.update();learned=tensor_digest(model.parts.named_parameters());adam=moments()
        saved=torch.load(folder/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
        bad={**saved,'core':{**saved['core'],'rng':{**saved['core']['rng'],'cuda':torch.zeros(4,dtype=torch.uint8)}}}
        before=capture_rng('cuda:0');versions=signature(model.parameters())
        try:kernel.resume(bad)
        except ValueError:pass
        else:raise ValueError('Invalid private CUDA RNG must reject before mutation')
        if versions!=signature(model.parameters()) or not equal(before,capture_rng('cuda:0')) or adam!=moments():
            raise ValueError('Invalid private recovery changed live policy/Adam or an RNG stream')
        kernel.resume(saved)
        if draws()!=expected_draws or kernel.update()!=expected or learned!=tensor_digest(model.parts.named_parameters()) or adam!=moments():
            raise ValueError('Actual configured own samples/rewards/actor/critic/Adam/all4RNG must recover exactly')
        atomic_checkpoint(folder/'resume-step2.pt',kernel.state(),MAX_STATE,RESERVE);del saved,bad
        full=stress(kernel,plan['capacity_maximum_prompt_position'])
    else:
        with (REPORTS/(arm+'-metrics.jsonl')).open('x') as journal:
            for _ in range(setting['steps']):
                owner.owned(plan);tick=time.monotonic();record=kernel.update();journal.write(json.dumps(record)+'\n');journal.flush()
                if kernel.completed==1 or kernel.completed%16==0 or kernel.completed==setting['steps']:
                    print(json.dumps(dict(step=kernel.completed,source_positions=record['source_positions'],actual_action_tokens=record['actual_action_tokens'],
                        token_weighted_PPO_loss=record['token_weighted_PPO_loss'],gradient_norm_before_clip=record['gradient_norm_before_clip'],
                        step_seconds=time.monotonic()-tick,seconds=time.monotonic()-started)),flush=True)
                    atomic_json(REPORTS/(arm+'-progress.json'),dict(step=kernel.completed,planned_steps=setting['steps'],latest=record,goal_achieved=False))
                if kernel.completed%128==0 or kernel.completed==setting['steps']:atomic_checkpoint(folder/'resume.pt',kernel.state(),MAX_STATE,RESERVE)
        atomic_checkpoint(folder/'model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),arm,kernel.completed),96*1024**2,RESERVE)
    scorer.check()
    if (fixed!=signature(kernel.fixed) or reward_fixed!=signature(scorer.parameters()) or reward_digest!=tensor_digest(scorer.named_parameters())
            or tensor_digest(model.base.named_parameters())!=plan['frozen_dense_sha256'] or initial==tensor_digest(model.parts.named_parameters())):
        raise ValueError('Policy must change with every original frozen base/word/auxiliary and separate reward value/metadata fixed')
    common=dict(actual_counts=model.counts(),separate_training_only_reward=scorer.counts(),actual_actor_critic_values=sum(p.numel() for p in params),
        actual_actor_critic_Adam_states=len(optimizer.state),actual_updates=kernel.completed,all_frozen_base_word_auxiliary_and_training_reward_values_metadata_exact=True,
        training_precision='FP32_TF32_disabled',peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
        candidate_promoted=False,goal_achieved=False)
    if phase=='capacity':
        atomic_json(REPORTS/f'capacity-{gpu}.json',dict(**common,arm=arm,physical_GPU=gpu,exact_next_update_and_private_CUDA_RNG_rejection=True,
            full_maximum_prompt_and_fixed_response_stress=full,logical_disposable_updates=2,physical_optimizer_calls=3,production_updates=0))
        guard.check(full=True);return
    model.eval();batch,_=legacy.probe_batch()
    with torch.no_grad():before_text=model(batch['input_ids']);before_image=model(batch['input_ids'],batch['pixel_values'])
    learned=tensor_digest(model.parts.named_parameters());del kernel,optimizer,scorer;torch.cuda.empty_cache()
    checkpoint_sha=file_hash(folder/'model.pt');cold,_=load(folder/'model.pt',REPORTS/'plan.json',arm,expected_checkpoint_sha256=checkpoint_sha,
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    with torch.no_grad():
        text=cold(batch['input_ids']);image=cold(batch['input_ids'],batch['pixel_values'])
        if (any(not torch.equal(before_text[k],text[k]) or not torch.equal(before_image[k],image[k]) for k in ('logits','values'))
                or learned!=tensor_digest(cold.parts.named_parameters())):raise ValueError('Trained policy text/image/critic cold reload must be exact without a reward encoder')
    del cold;torch.cuda.empty_cache()
    atomic_json(REPORTS/(arm+'-training-complete.json'),dict(**common,checkpoint_sha256=checkpoint_sha,resume_sha256=file_hash(folder/'resume.pt'),
        actual_trained_policy_text_image_critic_cold_reload_exact=True,declared_source_exposures=plan['source_exposures'][arm]))
    evaluate_held(model,plan,arm,amp);guard.check(full=True)


def assess(arm,suite,gpu,shard):
    subject='assess-'+arm+'-'+suite+'-'+str(shard if shard is not None else gpu)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    from slm_lab.dense_ppo_inference_v1 import load
    path=OUTPUT/arm/'model.pt';model,metadata=load(path,REPORTS/'plan.json',arm,expected_checkpoint_sha256=file_hash(path),
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    legacy.native_suite(plan,SimpleNamespace(aligned=model.policy),metadata,arm,suite,gpu,guard,shard)


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Waiting CPU owner must explicitly hide every GPU')
    if any(p.exists() for p in (REPORTS,OUTPUT,QUEUE)):raise ValueError('Preserve earlier attempts; no implicit restart')
    tests={'reports/dense-ppo-v1-tests.log':'6 passed','reports/dense-ppo-cached-v2-tests.log':'4 passed',
        'reports/dense-ppo-control-v1-tests.log':'3 passed','reports/dense-ppo-inference-v1-tests.log':'1 passed',
        'reports/dense-ppo-capacity-v1-tests.log':'2 passed','reports/dense-ppo-evidence-v1-tests.log':'3 passed',
        'reports/dense-PPO-driver-v1-tests.log':'3 passed'}
    for path,passed in tests.items():
        if passed not in (ROOT/path).read_text():raise ValueError('Actual sampler/control/recovery/capacity/source/ownership checks required')
    proof=json.loads((CPU_PROOF/'complete.json').read_text());reader=json.loads((CPU_PROOF/'independent-complete-neural-readback.json').read_text())
    if not proof['exact_next_own_actions_rewards_actor_critic_Adam_all_CPU_RNG_and_metrics'] or not reader['every_prefill_decode_behavior_and_full_causal_comparison_neurally_replayed']:
        raise ValueError('Actual full-size cached actor/critic recovery and complete neural replay required')
    files=dict(json.loads((CPU_PROOF/'plan.json').read_text())['source_files_sha256'])
    names=['scripts/train_dense_PPO_pilot_v1.py','src/slm_lab/dense_ppo_capacity_v1.py','src/slm_lab/dense_ppo_control_v1.py',
        'src/slm_lab/dense_ppo_inference_v1.py','src/slm_lab/dense_ppo_evidence_v1.py','src/slm_lab/dense_policy_native_readback_v1.py',
        'src/slm_lab/dense_human_preference_evaluation_v1.py','src/slm_lab/dense_word_evaluation_v1.py','src/slm_lab/cuda_rng_readback_v1.py',
        'tests/test_dense_ppo_capacity_v1.py','tests/test_dense_ppo_control_v1.py','tests/test_dense_ppo_inference_v1.py',
        'tests/test_dense_ppo_evidence_v1.py','tests/test_dense_PPO_driver_v1.py',str(PROPOSAL.relative_to(ROOT)),
        'scripts/review_dense_ppo_cached_CPU_v2.py','reports/dense-ppo-full-prompt-source-v1.json','reports/dense-ppo-human-reward-offline-gate-v1.json',*tests]
    for name in names:
        p=ROOT/name;value=file_hash(p)
        if str(p) in files and files[str(p)]!=value:raise ValueError('Never reseal a changed bound source')
        files[str(p)]=value
    for name in ('plan.json','complete.json','independent-complete-neural-readback.json','resume-step1.pt','resume-step2.pt','trained-disposable-policy.pt'):
        p=CPU_PROOF/name;files[str(p)]=file_hash(p)
    predecessor=json.loads(PREVIOUS_QUEUE.read_text())['owner'];me=owner.resources.identity(os.getpid());guard=FileBindings(files);deadline=time.monotonic()+168*3600
    while not previous_ended(predecessor):
        guard.check()
        if time.monotonic()>deadline:raise ValueError('Original production preference owner remains live; expire without touching GPUs')
        atomic_json(QUEUE,dict(status='waiting for exact complete production preference owner/workers and successful restoration',owner=me,predecessor=predecessor,
            observed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,production_updates=0,goal_achieved=False));time.sleep(10)
    guard.check(full=True)
    import torch
    from slm_lab.dense_preference_inference_v1 import load as initial_load
    from slm_lab.dense_ppo_reward_v1 import load as load_reward
    from slm_lab.dense_ppo_evidence_v1 import budget
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);proposal=json.loads(PROPOSAL.read_text())
    for name in ('plan.json','complete.json','restored.json','independent-comparison.json'):
        p=PREVIOUS/name;files[str(p)]=file_hash(p)
    for path,value in json.loads((PREVIOUS/'plan.json').read_text())['source_files_sha256'].items():
        if path in files and files[path]!=value:raise ValueError('Never reseal changed predecessor inputs')
        files[path]=value
    inputs=dict(checkpoint=str(ROOT/'checkpoints/dense-human-preference-v2/rate_3e6_DPO/model.pt'),plan=str(PREVIOUS/'plan.json'),arm='rate_3e6_DPO')
    for name in ('checkpoint','plan'):inputs[name+'_sha256']=file_hash(inputs[name]);files[inputs[name]]=inputs[name+'_sha256']
    model,metadata=initial_load(inputs['checkpoint'],inputs['plan'],inputs['arm'],expected_checkpoint_sha256=inputs['checkpoint_sha256'],expected_plan_sha256=inputs['plan_sha256'])
    reward_root=ROOT/'reports/whole-human-reward-v1';summary=json.loads((reward_root/'complete.json').read_text());reward_arm=summary['selected_arm']
    reward_inputs=dict(checkpoint=str(ROOT/'checkpoints/whole-human-reward-v1'/reward_arm/'model.pt'),plan=str(reward_root/'plan.json'),
        feature_admission=str(ROOT/'data/whole-human-reward-features-v1/admission.json'),arm=reward_arm)
    for name in ('checkpoint','plan','feature_admission'):
        reward_inputs[name+'_sha256']=file_hash(reward_inputs[name]);files[reward_inputs[name]]=reward_inputs[name+'_sha256']
    gate=json.loads((ROOT/'reports/dense-ppo-human-reward-offline-gate-v1.json').read_text())
    if not gate['offline_gate']['passed'] or gate['original_complete_held_input']['checkpoint_sha256']!=reward_inputs['checkpoint_sha256']:
        raise ValueError('The same complete held human-ranking head and passed bounded offline gate required')
    scorer=load_reward(reward_inputs)
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization');held=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation')
    words=SharedWordData(ROOT/'data/shared-byte-words-v1','valid','evaluation')
    settings=dict(seed=952931,steps=512,prompts_per_update=2,max_new_tokens=128,betas=[.9,.95],weight_decay=.01,gradient_clip=1.,
        KL_coefficient=.05,gamma=1.,GAE_lambda=.95,clip=.2,value_coefficient=.5)
    arms=proposal['profiles'];exposures={a:budget(data,{**settings,'learning_rate':profile['learning_rate']}) for a,profile in arms.items()}
    expected=next(iter(exposures.values()))
    if expected['whole_original_prompt_tokens']!=proposal['complete_original_prompt_tokens'] or expected['ordered_full_prompt_ledger_sha256']!=proposal['ordered_full_prompt_ledger_sha256']:
        raise ValueError('Exact predeclared whole original training-source prefix required')
    plan_settings=dict(initial_policy=inputs,actual_counts=model.counts(),initial_parts_sha256=tensor_digest(model.parts.named_parameters()),
        frozen_dense_sha256=tensor_digest(model.base.named_parameters()),tokenizer_sha256=metadata['tokenizer_sha256'],reward_inputs=reward_inputs,
        reward_provenance=scorer.provenance,reward_counts=scorer.counts(),human_admission_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json'),
        training_options=settings,arms=arms,arm_GPUs={a:i for i,a in enumerate(arms)},source_exposures=exposures,training_device='cuda',
        vocab_size=model.config.vocab_size,max_text_length=model.config.max_text_length,
        capacity_maximum_prompt_position=max(range(len(data)),key=lambda p:data.rows[p]['native']['chosen']['target_start']))
    if model.counts()['actual_unique']!=999706442 or scorer.counts()['actual_training_only_total']!=706210944:
        raise ValueError('Exact actual deployed and separately counted training-only model inventories required')
    del model,scorer;guard=FileBindings(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not previous_ended(predecessor):raise ValueError('Exact predecessor became live')
        require_idle_devices();guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        source_tokenizer=ROOT/'checkpoints/dense-human-preference-v2/rate_3e6_DPO/tokenizer.json'
        if file_hash(source_tokenizer)!=plan_settings['tokenizer_sha256']:raise ValueError('Exact original own tokenizer required')
        for arm in ['initial',*arms]:(REPORTS/arm).mkdir();(OUTPUT/arm).mkdir();shutil.copyfile(source_tokenizer,OUTPUT/arm/'tokenizer.json')
        plan=dict(**plan_settings,owner=me,waiter=owner.WAITER,source_files_sha256=files,created_at=datetime.now(timezone.utc).isoformat(),expires_at_unix=time.time()+48*3600,
            all_capacity_updates_discarded=True,automatic_retry=False,no_automatic_promotion=True,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned PPO experiment interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        driver=str(Path(__file__).resolve());watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
        def tasks(items):owner.wait_children(plan,[owner.launch(plan,['-u',driver,*args],subject,str(gpu)) for args,subject,gpu in items])
        def native(arm):
            tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i) for i,suite in enumerate(('knowledge','code','cpp','vision'))])
            for suite in ('public','math'):
                tasks([(['--assess',suite,'--arm',arm,'--gpu',str(i),'--shard',str(i)],'assess-'+arm+'-'+suite+'-'+str(i),i) for i in range(4)])
            from slm_lab.full_math_shards_v1 import reconcile
            reconcile([REPORTS/arm/f'math-{i}.json' for i in range(4)],ROOT/'data/math-reasoning-v2/test.jsonl',OUTPUT/arm/'tokenizer.json',REPORTS/arm/'math-complete.json')
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);until=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>until:raise ValueError('Original idle waiter did not enter owned hold')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():raise ValueError('Original waiter started real work; do not preempt it')
            tasks([(['--worker','capacity','--gpu',str(i)],'capacity-'+str(i),i) for i in range(4)])
            proofs=[json.loads((REPORTS/f'capacity-{i}.json').read_text()) for i in range(4)]
            validate_capacity(proofs,plan)
            atomic_json(REPORTS/'capacity.json',dict(passed=True,all_four_actual_L40s=True,proofs=proofs,production_updates=0,goal_achieved=False))
            tasks([(['--worker','initial','--gpu','0'],'initial-policy',0)]);native('initial')
            tasks([(['--worker','train','--arm',a,'--gpu',str(plan['arm_GPUs'][a])],'train-'+a,plan['arm_GPUs'][a]) for a in arms])
            for a in arms:native(a)
            from tokenizers import Tokenizer
            from slm_lab.dense_ppo_evidence_v1 import reconcile
            reconcile(ROOT,REPORTS,OUTPUT,data,held,words,Tokenizer.from_file(str(OUTPUT/'initial/tokenizer.json')));guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(status='Bounded matched human-PPO/KL-only experiment and every held/native comparison completed; inspect measured quality',candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:owner.restore_owned(plan,'Bounded matched PPO comparison ended; original work restored');watcher.wait(timeout=20)


if __name__=='__main__':
    configure();p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watchdog',action='store_true');p.add_argument('--worker',choices=['capacity','initial','train'])
    p.add_argument('--assess',choices=['knowledge','code','cpp','vision','public','math']);p.add_argument('--arm',default='initial');p.add_argument('--gpu',type=int,choices=range(4),default=0);p.add_argument('--shard',type=int,choices=range(4))
    a=p.parse_args()
    if a.watchdog:owner.watchdog()
    elif a.worker:worker(a.worker,a.arm,a.gpu)
    elif a.assess:assess(a.arm,a.assess,a.gpu,a.shard)
    else:parent()
