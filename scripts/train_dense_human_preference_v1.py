"""Owned full-source chosen-CE/DPO comparison after complete dense foundation.

The source arms are predeclared research inputs, not automatically promoted
winners. Every human pair is consumed once per arm. Native code/math/knowledge,
all held human preferences and all held word/image controls stay visible.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from types import SimpleNamespace

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

owner=legacy.owner
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-human-preference-v1'
OUTPUT=ROOT/'checkpoints/dense-human-preference-v1'
QUEUE=ROOT/'reports/dense-human-preference-v1-queue.json'
PREVIOUS=ROOT/'reports/dense-foundation-v1'
PREVIOUS_QUEUE=ROOT/'reports/dense-foundation-v1-queue.json'
WORD_REPORTS=ROOT/'reports/shared-words-v1'
CPU_PROOF=ROOT/'reports/dense-human-preference-stage-full-CPU-v1'
MAX_STATE=512*1024**2
RESERVE=26*1024**3


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS
    legacy.REPORTS=REPORTS;legacy.OUTPUT=OUTPUT


def previous_ended(identity):
    if owner.resources.same(identity):return False
    children=PREVIOUS/'children.json'
    if children.exists() and any(owner.resources.same(a) for a in json.loads(children.read_text())):return False
    if (not all((PREVIOUS/n).exists() for n in ('complete.json','restored.json','independent-comparison.json'))
            or (PREVIOUS/'failure.json').exists()):
        raise ValueError('Exact full foundation owner/workers must finish successfully and restore original work')
    return True


def require_idle_devices():
    owner.idle_original()
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    if len(used)!=4 or any(int(v.strip())>128 for v in used):raise ValueError('All four actual local L40s must be idle')
    if shutil.disk_usage(ROOT).free<RESERVE+5*1024**3+16*MAX_STATE:
        raise ValueError('Complete common base, all policy/Adam histories and atomic writers reserve required')


def signature(parameters):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]


def word_merge_quality(source,fused,expected_ids):
    if (not expected_ids or [r['id'] for r in source['results']]!=expected_ids
            or [r['id'] for r in fused['results']]!=expected_ids
            or source['overall']['examples']!=len(expected_ids) or fused['overall']['examples']!=len(expected_ids)):
        raise ValueError('Every original matched held word must remain in the source/fused comparison')
    before=source['overall']['conditions']['proper']['CTC_exact'];after=fused['overall']['conditions']['proper']['CTC_exact']
    return dict(original_held_words=len(expected_ids),source_CTC_exact=before,fused_CTC_exact=after,net_correct_change=after-before,
        maximum_allowed_net_loss=math.floor(.01*len(expected_ids)),passed=before-after<=math.floor(.01*len(expected_ids)),
        native_language_has_changed_and_native_word_quality_not_inferred=True,goal_achieved=False)


def gpu_merge_parity(model,plan,amp):
    import torch
    from torch import nn
    from slm_lab.model import TinyMultimodalLM
    from slm_lab.shared_word_inference_v1 import load
    from slm_lab.shared_word_data_v1 import SharedWordData
    inputs=plan['base_inputs']
    source,_=load(inputs['word_checkpoint'],inputs['word_plan'],inputs['word_arm'],expected_checkpoint_sha256=inputs['word_checkpoint_sha256'],
        expected_plan_sha256=inputs['word_plan_sha256'],device='cpu')
    # Extract only the source vision view. Its modules have no reference back
    # to the source language model; no second language backbone goes on GPU.
    vision=source.aligned.vision;del source;vision=vision.to('cuda:0').eval()
    class Reference(TinyMultimodalLM):
        def __init__(self):
            nn.Module.__init__(self);network=model.base.network;self.config=model.config
            self.embedding,self.lm_head,self.blocks=network.embedding,network.lm_head,network.blocks
            self.norm,self.value_head=network.norm,network.value_head;self.vision=vision;self.gradient_checkpointing=False
    reference=Reference().eval();data=SharedWordData(ROOT/'data/shared-byte-words-v1','train','optimization')
    rows=[]
    with torch.no_grad(),amp():
        photo,_=legacy.probe_batch();a=reference(photo['input_ids'],photo['pixel_values'])['logits'];b=model.base(photo['input_ids'],photo['pixel_values'])['logits']
        if not torch.allclose(a,b,atol=.25,rtol=.02):raise ValueError('Actual CUDA/BF16 vision merge exceeded predeclared sampled numerical tolerance')
        photo_error=float((a-b).abs().max())
        for p in plan['capacity_word_positions']:
            batch=data.batch([p],'cuda:0');word=batch['word_pixels']
            a=model.base.heads['word'](vision.patch_grid(word,False,word_image=True));b=model.base.word_logits(word)
            if not torch.allclose(a,b,atol=.25,rtol=.02):raise ValueError('Actual CUDA/BF16 word merge exceeded predeclared sampled tolerance')
            rows.append(dict(original_train_position=p,max_absolute_logit_error=float((a-b).abs().max()),
                source_CTC_path=a.argmax(-1).cpu().tolist(),folded_CTC_path=b.argmax(-1).cpu().tolist(),
                raw_paths_equal=torch.equal(a.argmax(-1),b.argmax(-1))))
    return dict(original_train_photo_max_logit_error=photo_error,actual_complete_word_controls=rows,CUDA_BF16_atol=.25,CUDA_BF16_rtol=.02,
        extra_source_vision_view_is_temporary_not_in_export=True,no_additional_language_reference_backbone=True,
        sampled_numerical_parity_not_held_reading_quality=True)


def evaluate_held(model,plan,arm,amp):
    from tokenizers import Tokenizer
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.dense_human_preference_evaluation_v1 import evaluate as preferences
    from slm_lab.dense_word_evaluation_v1 import evaluate as words
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation')
    preferences(model,data,REPORTS/arm/'held-preferences.json',beta=plan['training_options']['DPO_beta'],amp=amp,checked=lambda:owner.owned(plan))
    word_data=SharedWordData(ROOT/'data/shared-byte-words-v1','valid','evaluation')
    words(model,word_data,Tokenizer.from_file(str(OUTPUT/arm/'tokenizer.json')),REPORTS/arm/'held-words.json',
        batch_size=16,generation_budget=64,amp=amp,checked=lambda:owner.owned(plan),
        provenance=dict(run_plan_sha256=file_hash(REPORTS/'plan.json'),checkpoint_sha256=file_hash(OUTPUT/arm/'model.pt'),
            inference_device_type='cuda',inference_precision='bf16'))


def worker(phase,arm,gpu):
    subject='capacity-'+str(gpu) if phase=='capacity' else ('initial-policy' if phase=='initial' else 'train-'+arm)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    import random
    import numpy as np
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.dense_preference_inference_v1 import fresh,options,sources,export,load
    from slm_lab.dense_human_preference_stage_v1 import DenseHumanPreferenceStage
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    model,metadata=fresh(plan,'cuda:0')
    if phase=='initial':
        atomic_checkpoint(OUTPUT/'initial/model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),'initial',0),96*1024**2,RESERVE)
        evaluate_held(model,plan,'initial',amp);guard.check(full=True);return
    setting=options(plan,'rate_3e6_DPO' if phase=='capacity' else arm)
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
    random.seed(setting['seed']);np.random.seed(setting['seed']);torch.manual_seed(setting['seed'])
    params=[p for _,p in model.named_selected(('actor',))]
    optimizer=torch.optim.AdamW(params,lr=setting['learning_rate'],betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    kernel=DenseHumanPreferenceStage(model,optimizer,data,setting,sources(plan,file_hash(REPORTS/'plan.json')),'cuda:0',amp)
    folder=OUTPUT/('capacity-'+str(gpu) if phase=='capacity' else arm)
    if phase=='capacity':folder.mkdir()
    elif set(p.name for p in folder.iterdir())!={'tokenizer.json'}:raise ValueError('Fresh tokenizer-only policy arm required')
    base_signature=signature(model.base.parameters());critic_signature=signature(model.parts['critic'].parameters())
    initial=tensor_digest(model.parts.named_parameters());started=time.monotonic();extremes=[]
    def adam_digest():return tensor_digest((str(i)+'.'+k,state[k]) for i,state in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
    if phase=='capacity':
        kernel.update();atomic_checkpoint(folder/'resume-step1.pt',kernel.state(),MAX_STATE,RESERVE)
        def draws():return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device='cuda:0'))]
        expected_draws=draws();expected=kernel.update();learned=tensor_digest(model.parts.named_parameters());moments=adam_digest()
        saved=torch.load(folder/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
        bad={**saved,'core':{**saved['core'],'rng':{**saved['core']['rng'],'cuda':torch.zeros(4,dtype=torch.uint8)}}}
        rng=capture_rng('cuda:0');versions=signature(model.parameters())
        try:kernel.resume(bad)
        except ValueError:pass
        else:raise ValueError('Invalid private CUDA RNG must reject before any mutation')
        if (versions!=signature(model.parameters()) or not torch.equal(rng['torch'],torch.get_rng_state())
                or not torch.equal(rng['cuda'],torch.cuda.get_rng_state()) or moments!=adam_digest()):
            raise ValueError('Invalid recovery mutated live model/Adam/RNG')
        kernel.resume(saved)
        if draws()!=expected_draws or kernel.update()!=expected or learned!=tensor_digest(model.parts.named_parameters()) or moments!=adam_digest():
            raise ValueError('Actual whole production batch next policy/Adam/source metrics and all4RNG streams must recover exactly')
        atomic_checkpoint(folder/'resume-step2.pt',kernel.state(),MAX_STATE,RESERVE);del saved,bad
        from slm_lab.dense_human_preference_v1 import preference_gradients
        for p in plan['capacity_human_pair_positions']:
            owner.owned(plan);optimizer.zero_grad(set_to_none=True)
            row=preference_gradients(model,data,[p],beta=setting['DPO_beta'],chosen_CE_weight=setting['chosen_CE_weight'],device='cuda:0',amp=amp)
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params) or any(p.grad is not None for p in kernel.fixed):
                raise ValueError('Actual complete maximal chosen/rejected contexts/targets need finite actor-only gradients')
            extremes.append(row)
        optimizer.zero_grad(set_to_none=True)
        merge_parity=gpu_merge_parity(model,plan,amp);torch.cuda.empty_cache()
    else:
        with (REPORTS/(arm+'-metrics.jsonl')).open('x') as journal:
            for _ in range(setting['steps']):
                owner.owned(plan);tick=time.monotonic();record=kernel.update();journal.write(json.dumps(record)+'\n');journal.flush()
                if kernel.completed==1 or kernel.completed%100==0 or kernel.completed==setting['steps']:
                    print(json.dumps(dict(**record,step_seconds=time.monotonic()-tick,seconds=time.monotonic()-started)),flush=True)
                    atomic_json(REPORTS/(arm+'-progress.json'),dict(step=kernel.completed,planned_steps=setting['steps'],latest=record,goal_achieved=False))
                if kernel.completed%512==0 or kernel.completed==setting['steps']:
                    atomic_checkpoint(folder/'resume.pt',kernel.state(),MAX_STATE,RESERVE)
        atomic_checkpoint(folder/'model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),arm,kernel.completed),96*1024**2,RESERVE)
    if (base_signature!=signature(model.base.parameters()) or critic_signature!=signature(model.parts['critic'].parameters())
            or tensor_digest(model.base.named_parameters())!=plan['frozen_dense_sha256'] or initial==tensor_digest(model.parts.named_parameters())):
        raise ValueError('Actual policy must learn while every fused dense/vision/word/head and unused critic value/metadata stays fixed')
    common=dict(actual_counts=model.counts(),actual_actor_values=sum(p.numel() for p in params),actual_Adam_states=len(optimizer.state),
        actual_updates=kernel.completed,all_frozen_fused_dense_word_heads_and_unused_critic_values_metadata_exact=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,candidate_promoted=False,goal_achieved=False)
    if phase=='capacity':
        atomic_json(REPORTS/f'capacity-{gpu}.json',dict(**common,physical_GPU=gpu,exact_next_update_and_private_CUDA_RNG_rejection=True,
            complete_source_extremes=extremes,actual_GPU_word_folding_parity=merge_parity,logical_disposable_updates=2,physical_optimizer_calls=3,
            retained_step1_sha256=file_hash(folder/'resume-step1.pt'),retained_step2_sha256=file_hash(folder/'resume-step2.pt'),production_updates=0))
        guard.check(full=True);return
    model.eval();batch,_=legacy.probe_batch()
    with torch.no_grad(),amp():
        before_text=model(batch['input_ids'])['logits'].cpu();before_image=model(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
    learned=tensor_digest(model.parts.named_parameters());del kernel,optimizer;torch.cuda.empty_cache()
    checkpoint_sha=file_hash(folder/'model.pt')
    cold,_=load(folder/'model.pt',REPORTS/'plan.json',arm,expected_checkpoint_sha256=checkpoint_sha,expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    with torch.no_grad(),amp():
        if (not torch.equal(before_text,cold(batch['input_ids'])['logits'].cpu())
                or not torch.equal(before_image,cold(batch['input_ids'],batch['pixel_values'])['logits'].cpu())
                or learned!=tensor_digest(cold.parts.named_parameters())):
            raise ValueError('Actual full trained source-bound policy text/image cold reload must be exact')
    del cold;torch.cuda.empty_cache()
    atomic_json(REPORTS/(arm+'-training-complete.json'),dict(**common,checkpoint_sha256=checkpoint_sha,
        resume_sha256=file_hash(folder/'resume.pt'),actual_full_trained_policy_native_cold_reload_exact=True,
        actual_exposure_budget=plan['source_exposure_budget'][arm]))
    evaluate_held(model,plan,arm,amp);guard.check(full=True)


def assess(arm,suite,gpu,shard):
    subject='assess-'+arm+'-'+suite+'-'+str(shard if shard is not None else gpu)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    from slm_lab.dense_preference_inference_v1 import load
    path=OUTPUT/arm/'model.pt';model,metadata=load(path,REPORTS/'plan.json',arm,expected_checkpoint_sha256=file_hash(path),
        expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    legacy.native_suite(plan,SimpleNamespace(aligned=model.policy),metadata,arm,suite,gpu,guard,shard)


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Waiting owner must explicitly hide every GPU')
    if any(p.exists() for p in (REPORTS,OUTPUT,QUEUE)):raise ValueError('Preserve every earlier policy attempt; no automatic restart')
    tests={'reports/dense-human-preference-stage-v1-tests.log':'4 passed','reports/dense-preference-inference-v1-tests.log':'1 passed',
        'reports/dense-preference-driver-v1-tests.log':'4 passed','reports/dense-word-folding-and-evaluation-v1-tests.log':'6 passed',
        'reports/dense-human-preference-evaluation-v1-tests.log':'2 passed'}
    for name,passed in tests.items():
        if passed not in (ROOT/name).read_text():raise ValueError('Actual gradient/control/recovery/source/inference/ownership checks required')
    proof=json.loads((CPU_PROOF/'complete.json').read_text())
    if not proof['actual_next_weights_Adam_all_CPU_RNG_cursor_metrics_and_pristine_payload_exact'] or not proof['actual_trained_full_size_text_image_word_cold_reload_exact']:
        raise ValueError('Actual complete-source runner and combined under1B full-size CPU recovery/cold proof required')
    files=dict(json.loads((CPU_PROOF/'plan.json').read_text())['source_files_sha256'])
    names=['scripts/train_dense_human_preference_v1.py','src/slm_lab/dense_preference_inference_v1.py',
        'src/slm_lab/dense_preference_evidence_v1.py','src/slm_lab/dense_human_preference_evaluation_v1.py',
        'tests/test_dense_preference_inference_v1.py','tests/test_dense_preference_driver_v1.py',
        'tests/test_dense_human_preference_evaluation_v1.py','scripts/review_dense_preference_stage_CPU_v1.py',
        'reports/dense-preference-v1-proposed-exposure.json',*tests]
    for name in names:
        path=ROOT/name;sha=file_hash(path)
        if str(path) in files and files[str(path)]!=sha:raise ValueError('Never reseal a changed bound policy/source file')
        files[str(path)]=sha
    for name in ('plan.json','complete.json','independent-complete-readback.json','resume-step1.pt','resume-step2.pt','trained-disposable-policy.pt'):
        path=CPU_PROOF/name;files[str(path)]=file_hash(path)
    for name in ('preferences.train.jsonl','preferences.train.tokens.u16','preferences.valid.jsonl','preferences.valid.tokens.u16'):
        path=ROOT/'data/whole-human-alignment-v2'/name;files[str(path)]=file_hash(path)
    predecessor=json.loads(PREVIOUS_QUEUE.read_text())['owner'];me=owner.resources.identity(os.getpid());guard=FileBindings(files)
    deadline=time.monotonic()+96*3600
    while not previous_ended(predecessor):
        guard.check()
        if time.monotonic()>deadline:raise ValueError('Original full foundation owner remains live; expire without touching GPUs')
        atomic_json(QUEUE,dict(status='waiting for exact complete dense-foundation owner/workers and successful restoration',owner=me,
            predecessor=predecessor,source_files_sha256=files,observed_at=datetime.now(timezone.utc).isoformat(),production_updates=0,goal_achieved=False));time.sleep(10)
    guard.check(full=True)
    from slm_lab.dense_preference_inference_v1 import prepare_base,export_base
    from slm_lab.dense_preference_evidence_v1 import budget
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.shared_word_evaluation_v1 import wrong_permutation
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    import torch
    torch.set_num_threads(2)
    # Explicit research inputs, chosen before their new held/native scores.
    inputs=dict(foundation_arm='rate_3e6_KL',word_arm='rate_3e4',foundation_plan=str(PREVIOUS/'plan.json'),
        foundation_checkpoint=str(ROOT/'checkpoints/dense-foundation-v1/rate_3e6_KL/model.pt'),word_plan=str(WORD_REPORTS/'plan.json'),
        word_checkpoint=str(ROOT/'checkpoints/shared-words-v1/rate_3e4/model.pt'))
    for name in ('foundation_plan','foundation_checkpoint','word_plan','word_checkpoint'):inputs[name+'_sha256']=file_hash(inputs[name])
    for folder in (PREVIOUS,WORD_REPORTS):
        if (folder/'failure.json').exists() or not all((folder/n).exists() for n in ('complete.json','restored.json','independent-comparison.json')):
            raise ValueError('Both complete original source epochs and independent quality reports required')
        for name in ('plan.json','complete.json','restored.json','independent-comparison.json'):
            p=folder/name;files[str(p)]=file_hash(p)
    for name in ('foundation_plan','foundation_checkpoint','word_plan','word_checkpoint'):files[inputs[name]]=inputs[name+'_sha256']
    for path,sha in json.loads((PREVIOUS/'plan.json').read_text())['source_files_sha256'].items():
        if path in files and files[path]!=sha:raise ValueError('Never reseal a changed admitted source after the predecessor')
        files[path]=sha
    guard=FileBindings(files);base,receipt=prepare_base(inputs);model=OwnDenseActor(base).eval()
    if model.counts()['actual_unique']!=999706442:raise ValueError('Actual complete combined policy must fit the exact under1B inventory')
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
    held=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','valid','evaluation')
    words=SharedWordData(ROOT/'data/shared-byte-words-v1','valid','evaluation');wrong_permutation(words.rows)
    train_words=SharedWordData(ROOT/'data/shared-byte-words-v1','train','optimization')
    settings=dict(seed=950131,epochs=1,steps=math.ceil(len(data)/4),microbatch=1,accumulation=4,
        betas=[.9,.95],weight_decay=.01,gradient_clip=1.,DPO_beta=.1,chosen_CE_weight=.1)
    arms=dict(rate_3e6_CE=dict(learning_rate=3e-6,DPO_weight=0),rate_3e6_DPO=dict(learning_rate=3e-6,DPO_weight=1),
        rate_1e5_CE=dict(learning_rate=1e-5,DPO_weight=0),rate_1e5_DPO=dict(learning_rate=1e-5,DPO_weight=1))
    extremes=[]
    for field in ('chosen','rejected'):
        for key in ('length','targets'):
            extremes.append(max(range(len(data)),key=lambda p:data.rows[p]['native'][field]['length']-(data.rows[p]['native'][field]['target_start'] if key=='targets' else 0)))
    plan_settings=dict(base_inputs=inputs,base_folding_receipt=receipt,actual_dense_counts=base.counts(),actual_counts=model.counts(),
        actor_settings=model.settings,initial_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),
        frozen_dense_sha256=tensor_digest(base.named_parameters()),tokenizer_sha256=file_hash(ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'),
        human_manifest_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/manifest.json'),human_admission_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json'),
        training_options=settings,arms=arms,arm_GPUs={a:i for i,a in enumerate(arms)},capacity_human_pair_positions=list(dict.fromkeys(extremes)),
        capacity_word_positions=list(dict.fromkeys([max(range(len(train_words)),key=lambda p:len(train_words.rows[p]['response'].encode('utf-8'))),
            next(p for p,r in enumerate(train_words.rows) if r['byte_family']=='real_word')])),
        source_exposure_budget={a:budget(data,{**settings,**profile}) for a,profile in arms.items()})
    del model
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not previous_ended(predecessor):raise ValueError('Exact prior owner became live')
        require_idle_devices();guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir();(OUTPUT/'base').mkdir()
        atomic_checkpoint(OUTPUT/'base/model.pt',export_base(base,inputs,receipt),5*1024**3,RESERVE);del base
        files[str(OUTPUT/'base/model.pt')]=file_hash(OUTPUT/'base/model.pt');guard=FileBindings(files)
        for arm in ['initial',*arms]:(REPORTS/arm).mkdir();(OUTPUT/arm).mkdir();shutil.copyfile(ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json',OUTPUT/arm/'tokenizer.json')
        plan=dict(**plan_settings,base_checkpoint=str(OUTPUT/'base/model.pt'),base_checkpoint_sha256=file_hash(OUTPUT/'base/model.pt'),
            owner=me,waiter=owner.WAITER,source_files_sha256=files,created_at=datetime.now(timezone.utc).isoformat(),expires_at_unix=time.time()+36*3600,
            all_capacity_updates_discarded=True,automatic_retry=False,no_automatic_promotion=True,candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned full human policy run interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        driver=str(Path(__file__).resolve());watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
        def launch(args,subject,gpu):return owner.launch(plan,['-u',driver,*args],subject,str(gpu))
        def tasks(items):owner.wait_children(plan,[launch(args,subject,gpu) for args,subject,gpu in items])
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
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():
                raise ValueError('Original waiter started real work; never preempt it')
            tasks([(['--worker','capacity','--gpu',str(i)],'capacity-'+str(i),i) for i in range(4)])
            proofs=[json.loads((REPORTS/f'capacity-{i}.json').read_text()) for i in range(4)]
            if not all(p['exact_next_update_and_private_CUDA_RNG_rejection'] and p['actual_actor_values']==15305216 for p in proofs):
                raise ValueError('Every actual full-size/source-extreme configured-batch L40 recovery check required')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,all_four_actual_L40s=True,proofs=proofs,production_updates=0,goal_achieved=False))
            tasks([(['--worker','initial','--gpu','0'],'initial-policy',0)])
            old_words=json.loads((WORD_REPORTS/inputs['word_arm']/'all-held-words.json').read_text())
            new_words=json.loads((REPORTS/'initial/held-words.json').read_text())
            from tokenizers import Tokenizer
            from slm_lab.shared_word_evidence_v1 import readback_held
            tokenizer=Tokenizer.from_file(str(OUTPUT/'initial/tokenizer.json'))
            readback_held(words,tokenizer,WORD_REPORTS/inputs['word_arm']/'all-held-words.json',64)
            readback_held(words,tokenizer,REPORTS/'initial/held-words.json',64)
            comparison=word_merge_quality(old_words,new_words,[r['id'] for r in words.rows])
            atomic_json(REPORTS/'full-held-source-fused-CTC-comparison.json',comparison)
            if not comparison['passed']:
                raise ValueError('Folding loses more than1%of all held words in proper-image CTC; preserve evidence and fix the merge before preference optimization')
            native('initial')
            tasks([(['--worker','train','--arm',a,'--gpu',str(plan['arm_GPUs'][a])],'train-'+a,plan['arm_GPUs'][a]) for a in arms])
            for a in arms:native(a)
            from tokenizers import Tokenizer
            from slm_lab.dense_preference_evidence_v1 import reconcile
            reconcile(ROOT,REPORTS,OUTPUT,data,held,words,Tokenizer.from_file(str(OUTPUT/'initial/tokenizer.json')));guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(status='Complete once-only human chosen-CE/DPOepochs and all held/native controls finished; inspect measured quality',
                candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:owner.restore_owned(plan,'Complete dense human policy comparison ended; original work restored');watcher.wait(timeout=20)


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--worker',choices=['capacity','initial','train']);parser.add_argument('--assess',choices=['knowledge','code','cpp','vision','public','math'])
    parser.add_argument('--arm',default='initial');parser.add_argument('--gpu',type=int,choices=range(4),default=0);parser.add_argument('--shard',type=int,choices=range(4))
    args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker:worker(args.worker,args.arm,args.gpu)
    elif args.assess:assess(args.arm,args.assess,args.gpu,args.shard)
    else:parent()
