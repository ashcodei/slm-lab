"""Owned full-size four-L40 source-extreme/Adam/RNG capacity checks only."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
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
REPORTS=ROOT/'reports/own-alignment-four-L40-v2'
DATA=ROOT/'data/whole-human-alignment-v2'
MAX_STATE=256*1024**2
RESERVE=26*1024**3


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS


def previous_ended():
    folder=ROOT/'reports/own-onpolicy-pilot-v2'
    plan=json.loads((folder/'plan.json').read_text())
    if (owner.resources.same(plan['owner']) or any(owner.resources.same(a) for a in json.loads((folder/'children.json').read_text()))
            or not (folder/'restored.json').exists()):
        raise ValueError('Every exact previous GPU owner/worker must actually end and restore the original waiter')


def registered(shard):
    plan=json.loads((REPORTS/'plan.json').read_text());me=owner.resources.identity(os.getpid())
    while True:
        owner.owned(plan)
        if any(a['pid']==me['pid'] and a['ticks']==me['ticks'] and a['subject']=='capacity-'+str(shard)
               for a in json.loads((REPORTS/'children.json').read_text())):
            return plan
        time.sleep(.05)


def extrema(data):
    result=set()
    for field in data.fields:
        result.add(max(range(len(data)),key=lambda p:data.rows[p]['native'][field]['length']))
        result.add(max(range(len(data)),key=lambda p:data.rows[p]['native'][field]['length']-data.rows[p]['native'][field]['target_start']))
    return sorted(result)


def adam_digest(optimizer):
    import torch
    digest=hashlib.sha256()
    for index,state in enumerate(optimizer.state.values()):
        for key in ('step','exp_avg','exp_avg_sq'):
            tensor=state[key].detach().cpu().contiguous()
            digest.update(json.dumps([index,key,list(tensor.shape),str(tensor.dtype)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def worker(shard):
    plan=registered(shard)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(shard):
        raise ValueError('One exact owned physical L40 required before CUDA initialization')
    guard=FileBindings({**plan['source_files_sha256'],str(REPORTS/'plan.json'):file_hash(REPORTS/'plan.json')})
    import random
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.own_aligned_assistant_v1 import byte_ctc_loss
    from slm_lab.own_aligned_assistant_v2 import OwnAlignedAssistantV2
    from slm_lab.own_alignment_checkpoint_v1 import make_state,restore_state
    from slm_lab.own_onpolicy_inference_v1 import load,attach_original_head
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.whole_human_data_v2 import WholeHumanData
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.85,0)
    if torch.cuda.get_device_name(0)!='NVIDIA L40':
        raise ValueError('Actual original L40 required')
    old_plan_path=ROOT/'reports/own-onpolicy-pilot-v2/plan.json';old=json.loads(old_plan_path.read_text())
    receipt=json.loads((ROOT/'reports/own-onpolicy-pilot-v2/CE_control-training-complete.json').read_text())
    language,metadata=load(ROOT/'checkpoints/own-onpolicy-pilot-v2/CE_control/model.pt',old_plan_path,'CE_control',
        ROOT/'reports/own-onpolicy-pilot-v2/reference.json',expected_checkpoint_sha256=receipt['trained_components_sha256'],
        expected_plan_sha256=file_hash(old_plan_path),expected_reference_sha256=receipt['reference_scores_sha256'],device='cpu')
    original,_=attach_original_head(language,old)
    model=OwnAlignedAssistantV2(original).eval().to('cuda:0');original=model.original
    if model.counts()['total_unique']!=999860418:
        raise ValueError('Actual full original/code/grounding/new unique parameter count must remain below1B')
    dialogue=WholeHumanData(DATA,'dialogue','train','capacity');preferences=WholeHumanData(DATA,'preferences','train','capacity')
    if extrema(dialogue)!=plan['source_extremes']['dialogue'] or extrema(preferences)!=plan['source_extremes']['preferences']:
        raise ValueError('Every actual complete training-source context and response extreme required')
    word=plan['word_source'];path=Path(word['image'])
    if file_hash(path)!=word['stored_image_sha256']:
        raise ValueError('Exact actual original longest human training word image required')
    with Image.open(path) as image:
        def pixels(width,height):
            array=np.asarray(image.convert('RGB').resize((width,height),Image.Resampling.BICUBIC),dtype=np.float32)/127.5-1
            return torch.from_numpy(array.copy()).permute(2,0,1)[None].to('cuda:0')
        word_pixels=pixels(448,64);native_pixels=pixels(448,448)
    # The complete native word label is an existing admitted training row.
    from tokenizers import Tokenizer
    from slm_lab.whole_human_alignment_v2 import WholeResponseEncoder
    native_encoder=WholeResponseEncoder(Tokenizer.from_file(old['tokenizer']),model.config)
    word_ids,start=native_encoder.sequence(word,'response')
    word_batch=dict(input_ids=torch.tensor([word_ids],device='cuda:0'),
                    labels=torch.tensor([[-100]*start+word_ids[start:]],device='cuda:0'))
    groups=('language','vision','critic','reward','word');selected=model.extension_parameters(*groups)
    optimizer=torch.optim.AdamW(selected,lr=2e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
    model.aligned.gradient_checkpointing=True
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    fixed=state_digest(original);versions=[(id(p),p._version,p.data_ptr()) for p in original.parameters()]
    ids=word_batch['input_ids'][:,:min(12,word_batch['input_ids'].shape[1])]
    def retained():
        model.eval()
        with torch.no_grad(),amp():
            return {r:original(ids,native_pixels,route=r)['logits'].cpu() for r in ('general','code','grounding')}
    before=retained()
    with torch.no_grad(),amp():
        cache=original.language.code.prefill(ids)['cache']
        cached_before=original.language.code.decode(torch.tensor([[4]],device='cuda:0'),cache)['logits'].cpu()
    random.seed(947151);np.random.seed(947151);torch.manual_seed(947151)
    sources=dict(run_plan_sha256=file_hash(REPORTS/'plan.json'),source_admission_sha256=file_hash(DATA/'admission.json'),
        original_policy_sha256=old['initial_manifest_sha256'],own_code_sha256=receipt['trained_components_sha256'])
    def update():
        owner.owned(plan);optimizer.zero_grad(set_to_none=True);model.aligned.train()
        draw=random.random()+float(np.random.random())+float(torch.rand((),device='cuda:0'))
        losses=[]
        for position in plan['source_extremes']['dialogue']:
            batch=dialogue.batch([position],'response','cuda:0')
            with amp():
                out=model(**batch);loss=out['loss'];critic=(out['values']-draw).square().mean()
            (loss+.01*critic).backward()
            losses.append(dict(kind='whole_human_SFT',source_position=position,context=batch['input_ids'].shape[1],
                targets=int((batch['labels'][:,1:]!=-100).sum()),loss=float(loss.detach()),disposable_critic_loss=float(critic.detach())))
        with amp():
            native_loss=model(**word_batch,pixel_values=native_pixels)['loss']
        native_loss.backward()
        for position in plan['source_extremes']['preferences']:
            scores={}
            for field in ('chosen','rejected'):
                batch=preferences.batch([position],field,'cuda:0')
                with amp():
                    scores[field]=model.preference_score(batch['input_ids'],attention_mask=batch['attention_mask'])
            reward=F.softplus(-(scores['chosen']-scores['rejected'])).mean();reward.backward()
            losses.append(dict(kind='whole_human_ranking',source_position=position,loss=float(reward.detach())))
        with amp():
            logits=model.word_logits(word_pixels)
        # CPU CTC reduction is differentiable through the device copy and
        # avoids a nondeterministic CUDA CTC backward. The actual vision/head
        # forward and all their parameter gradients remain on this L40.
        ctc=byte_ctc_loss(logits.cpu(),[word['response']]);(.1*ctc).backward()
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected) or any(p.grad is not None for p in original.parameters()):
            raise ValueError('All600new parameters need finite gradients; every original tensor stays gradient-free')
        norm=float(torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True))
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        return dict(stochastic_draw=draw,objectives=losses,native_whole_word_SFT=float(native_loss.detach()),
            whole_word_CTC=float(ctc.detach()),actual_CTC_frames=logits.shape[1],gradient_norm_before_clip=norm)
    first=update();cursor=dict(completed_draws_each=1,original_source_extremes=plan['source_extremes'],original_word_id=word['id'])
    saved=make_state(model,optimizer,groups=groups,sources=sources,completed_steps=1,planned_steps=2,
        sampler_state=cursor,metrics=[first],device='cuda:0')
    path=REPORTS/f'capacity-{shard}-resume.pt';atomic_checkpoint(path,saved,MAX_STATE,RESERVE)
    expected=update();expected_parts=state_digest(model.parts);expected_Adam=adam_digest(optimizer)
    restore_state(model,optimizer,torch.load(path,map_location='cpu',weights_only=True,mmap=True),groups=groups,
        expected_sources=sources,planned_steps=2,expected_sampler_state=cursor,device='cuda:0')
    actual=update()
    if expected!=actual or state_digest(model.parts)!=expected_parts or adam_digest(optimizer)!=expected_Adam:
        raise ValueError('Actual CUDA next-update metrics, every new tensor and every Adam moment/RNG must reproduce exactly')
    after=retained()
    with torch.no_grad(),amp():
        cached_after=original.language.code.decode(torch.tensor([[4]],device='cuda:0'),cache)['logits'].cpu()
    if (any(not torch.equal(before[r],after[r]) for r in before) or not torch.equal(cached_before,cached_after)
            or fixed!=state_digest(original) or versions!=[(id(p),p._version,p.data_ptr()) for p in original.parameters()]):
        raise ValueError('Actual original base/code/grounding tensor bytes/versions/native probes/cache must remain exact')
    guard.check(full=True)
    proof=dict(completed_at=datetime.now(timezone.utc).isoformat(),physical_GPU=shard,GPU=torch.cuda.get_device_name(0),
        actual_counts=model.counts(),actual_selected_new_parameters=sum(p.numel() for p in selected),complete_Adam_states=len(optimizer.state),
        actual_CUDA_next_update_all_metrics_new_tensors_Adam_and_RNG_exact=True,all_original_bytes_versions_native_probes_and_code_cache_exact=True,
        original_state_digest=fixed,initial_source_extreme_update=first,next_source_extreme_update=expected,
        all600new_parameter_gradients_finite=True,no_original_gradients=True,
        critic_target_is_disposable_stochastic_capacity_scalar_not_RLHF_return=True,
        actual_original_whole_word_UTF8_bytes=len(word['response'].encode()),source_word_id=word['id'],
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
        actual_resume_bytes=path.stat().st_size,actual_resume_sha256=file_hash(path),
        original_backbones_loaded=metadata['original_backbones_loaded'],external_model_loaded=False,
        CTC_scalar_and_gradient_reduction_on_CPU=True,all_model_parameter_gradients_on_GPU=True,
        logical_disposable_updates=2,physical_optimizer_calls=3,production_updates=0,capability_gain_not_established=True,goal_achieved=False)
    atomic_json(REPORTS/f'capacity-{shard}.json',proof)


def parent():
    if REPORTS.exists():
        raise ValueError('Preserve previous GPU capacity evidence; no automatic restart')
    previous_ended();owner.idle_original()
    from slm_lab.byte_word_reader import load_rows
    from slm_lab.whole_human_data_v2 import WholeHumanData
    dialogue=WholeHumanData(DATA,'dialogue','train','capacity');preferences=WholeHumanData(DATA,'preferences','train','capacity')
    words,word_manifest=load_rows(ROOT/'data/byte-word-reader-v1')
    def need(row):
        raw=list(row['response'].encode());return len(raw)+sum(a==b for a,b in zip(raw,raw[1:]))
    word=max(words['train'],key=need)
    if need(word)>140 or len(word['response'].encode())>64:
        raise ValueError('Complete actual longest original train transcript must fit without clipping')
    if '2 passed' not in (ROOT/'reports/own-aligned-assistant-v2-tests.log').read_text() or '3 passed' not in (ROOT/'reports/whole-human-alignment-v2-tests.log').read_text():
        raise ValueError('Actual expanded-head/recovery and whole-source framing checks required')
    names=['scripts/probe_own_alignment_four_L40_v2.py','src/slm_lab/own_aligned_assistant_v1.py','src/slm_lab/own_aligned_assistant_v2.py',
        'src/slm_lab/own_alignment_checkpoint_v1.py','src/slm_lab/whole_human_alignment_v2.py','src/slm_lab/whole_human_data_v2.py',
        'scripts/train_point_grounding_joint_v1.py','scripts/assess_joint_java_comparison_v1.py',
        'src/slm_lab/own_onpolicy_inference_v1.py','src/slm_lab/own_assistant_v1.py','src/slm_lab/own_code_delta_v1.py','src/slm_lab/model.py',
        'src/slm_lab/code_residual_checkpoint_v1.py','src/slm_lab/codenet_checkpoint_v1.py','src/slm_lab/byte_word_reader.py',
        'reports/own-aligned-assistant-v2-tests.log','reports/whole-human-alignment-v2-tests.log',
        'tests/test_own_alignment_four_L40_driver_v2.py','reports/own-alignment-four-L40-v2-driver-tests.log',
        'reports/own-onpolicy-pilot-v2/plan.json','reports/own-onpolicy-pilot-v2/CE_control-training-complete.json',
        'reports/own-onpolicy-pilot-v2/reference.json','data/byte-word-reader-v1/manifest.json']
    manifest=json.loads((DATA/'manifest.json').read_text())
    files={**manifest['source_files_sha256'],**word_manifest['source_files_sha256'],
        **{str(ROOT/name):file_hash(ROOT/name) for name in names},
        **{str(DATA/name):sha for name,sha in manifest['files_sha256'].items()}}
    for path in (DATA/'manifest.json',DATA/'independent-source-readback.json',DATA/'admission.json',Path(word['image'])):
        files[str(path)]=file_hash(path)
    old=json.loads((ROOT/'reports/own-onpolicy-pilot-v2/plan.json').read_text())
    original_manifest=Path(old['initial_manifest']);original_spec=json.loads(original_manifest.read_text())
    for path in (Path(old['base_checkpoint']),Path(old['tokenizer']),original_manifest,
                 ROOT/'checkpoints/own-onpolicy-pilot-v2/CE_control/model.pt'):
        files[str(path)]=file_hash(path)
    for name in ('grounding_head','grounding_plan'):
        path=Path(original_spec[name]);path=path if path.is_absolute() else original_manifest.parent/path
        files[str(path.resolve())]=original_spec[name+'_sha256']
    guard=FileBindings(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);previous_ended();owner.idle_original()
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
        if len(memory)!=4 or any(int(v.strip())>128 for v in memory):
            raise ValueError('All four actual original L40 GPUs must be unused')
        if shutil.disk_usage(ROOT).free<RESERVE+8*MAX_STATE:
            raise ValueError('Keep persistent reserve and complete atomic disposable snapshots')
        guard.check(full=True);REPORTS.mkdir()
        plan=dict(created_at=datetime.now(timezone.utc).isoformat(),owner=owner.resources.identity(os.getpid()),waiter=owner.WAITER,
            source_files_sha256=files,source_extremes=dict(dialogue=extrema(dialogue),preferences=extrema(preferences)),word_source=word,
            actual_model_parameters=999860418,word_time_expansion=5,actual_word_required_CTC_frames=need(word),
            expires_at_unix=time.time()+30*60,maximum_seconds=30*60,automatic_retry=False,
            scope='Four independent actual full-graph source-extreme CUDA/Adam/RNG capacity probes; no production alignment or capability result',goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):
            raise KeyboardInterrupt('Owned whole-human capacity experiment interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        driver=str(Path(__file__).resolve())
        watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,
            stdout=(REPORTS/'watchdog.log').open('xb'),stderr=subprocess.STDOUT,start_new_session=True)
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);deadline=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>deadline:raise ValueError('Original sleeping childless waiter did not enter owned hold')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():
                raise ValueError('Original waiter began work; never preempt it')
            children=[owner.launch(plan,['-u',driver,'--worker',str(i)],'capacity-'+str(i),str(i)) for i in range(4)]
            owner.wait_children(plan,children)
            proofs=[json.loads((REPORTS/f'capacity-{i}.json').read_text()) for i in range(4)]
            if not all(p['actual_CUDA_next_update_all_metrics_new_tensors_Adam_and_RNG_exact'] for p in proofs):
                raise ValueError('Every actual original L40 must pass complete source/extreme/recovery checks')
            guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
                all_four_original_L40_capacity_and_exact_recovery_passed=True,proofs=proofs,production_updates=0,
                actual_model_parameters=999860418,capability_gain_not_established=True,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Owned full whole-human capacity check ended; original waiter restored')
            watcher.wait(timeout=20)


if __name__=='__main__':
    configure()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',type=int,choices=range(4));parser.add_argument('--watchdog',action='store_true')
    args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker is not None:worker(args.worker)
    else:parent()
