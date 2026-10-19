"""Owned complete-source shared image/word training after existing code work.

The language anchor is the already trained own code route, without the later
mixed language extensions. This learns reusable vision/word parts in the same
bounded single-backbone graph; final mixed-policy attachment is a later,
separately evaluated integration. No external OCR/teacher weights are loaded.
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

import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

owner=legacy.owner
ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/shared-words-v1'
OUTPUT=ROOT/'checkpoints/shared-words-v1'
QUEUE=ROOT/'reports/shared-words-v1-queue.json'
PREVIOUS=ROOT/'reports/code-initialized-alignment-v1'
PREVIOUS_QUEUE=ROOT/'reports/code-initialized-alignment-v1-queue.json'
CPU_PROOF=ROOT/'reports/shared-words-full-CPU-v1'
DATA=ROOT/'data/shared-byte-words-v1'
MAX_STATE=128*1024**2
RESERVE=26*1024**3


def configure():
    owner.REPORTS=REPORTS;owner.resources.REPORTS=REPORTS
    legacy.REPORTS=REPORTS;legacy.OUTPUT=OUTPUT


def previous_ended(identity):
    if owner.resources.same(identity):return False
    if (PREVIOUS/'children.json').exists() and any(owner.resources.same(a) for a in json.loads((PREVIOUS/'children.json').read_text())):
        return False
    if (not (PREVIOUS/'restored.json').exists() or not (PREVIOUS/'complete.json').exists()
            or not (PREVIOUS/'independent-comparison.json').exists() or (PREVIOUS/'failure.json').exists()):
        raise ValueError('Previous exact full code experiment must finish successfully and restore original work before word GPUs')
    return True


def require_idle_devices():
    owner.idle_original()
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    if len(used)!=4 or any(int(v.strip())>128 for v in used):raise ValueError('All four actual original L40s must be idle')
    if shutil.disk_usage(ROOT).free<RESERVE+16*MAX_STATE:raise ValueError('Full source, raw held deliveries and atomic state reserve required')


def fixed_digest(model):
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    selected={id(p) for p in model.extension_parameters('vision','word')}
    return tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)


def trained_digest(model):
    from slm_lab.shared_word_stage_v1 import PREFIXES
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    return tensor_digest((n,p) for n,p in model.parts.named_parameters() if n.startswith(PREFIXES))


def signature(parameters):return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in parameters]


def evaluate_words(model,metadata,plan,arm,amp):
    from tokenizers import Tokenizer
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.shared_word_evaluation_v1 import evaluate
    held=SharedWordData(DATA,'valid','evaluation');tokenizer=Tokenizer.from_file(str(OUTPUT/arm/'tokenizer.json'))
    return evaluate(model,held,tokenizer,REPORTS/arm/'all-held-words.json',amp=amp,
        batch_size=plan['evaluation']['batch_size'],generation_budget=plan['evaluation']['generation_budget'],
        checked=lambda:owner.owned(plan),provenance=dict(metadata,checkpoint_sha256=file_hash(OUTPUT/arm/'model.pt'),
            run_plan_sha256=file_hash(REPORTS/'plan.json'),inference_device='cuda',inference_precision='bf16'))


def worker(phase,arm,gpu):
    subject='capacity-'+str(gpu) if phase=='capacity' else ('initial-words' if phase=='initial' else 'train-'+arm)
    plan=legacy.registered(subject,gpu);torch,guard,amp=legacy.runtime(plan)
    import random
    import numpy as np
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.shared_word_inference_v1 import fresh,sources,options,export,load
    from slm_lab.shared_word_stage_v1 import SharedWordStage,accumulate_gradients
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    model,metadata=fresh(plan,'cuda:0');model.aligned.gradient_checkpointing=True
    if tensor_digest(model.parts.state_dict().items())!=plan['initial_parts_digest']:
        raise ValueError('Each full-word arm must start at the same fresh extension state')
    random.seed(plan['training_options']['seed']);np.random.seed(plan['training_options']['seed']);torch.manual_seed(plan['training_options']['seed'])
    if phase=='initial':
        atomic_checkpoint(OUTPUT/'initial/model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),'initial',0),64*1024**2,RESERVE)
        evaluate_words(model,metadata,plan,'initial',amp);guard.check(full=True);return
    data=SharedWordData(DATA,'train','optimization');setting=options(plan,'rate_3e4' if phase=='capacity' else arm)
    optimizer=torch.optim.AdamW(model.extension_parameters('vision','word'),lr=setting['learning_rate'],
        betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    kernel=SharedWordStage(model,optimizer,data,setting,sources(plan,file_hash(REPORTS/'plan.json')),'cuda:0',amp)
    folder=OUTPUT/('capacity-'+str(gpu) if phase=='capacity' else arm)
    if phase=='capacity':folder.mkdir()
    elif set(p.name for p in folder.iterdir())!={'tokenizer.json'}:raise ValueError('Fresh tokenizer-only full-word arm required')
    batch,photo_id=legacy.probe_batch();before=legacy.native_original(model,batch,amp)
    model.eval()
    with torch.no_grad(),amp():
        text_before=model.aligned(batch['input_ids'])['logits'].cpu()
        image_before=model.aligned(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
        cache=model.original.language.code.prefill(batch['input_ids'],batch['pixel_values'])['cache']
        cached_before=model.original.language.code.decode(torch.tensor([[4]],device='cuda:0'),cache)['logits'].cpu()
    fixed=fixed_digest(model);versions=signature(kernel.fixed);initial_trainable=trained_digest(model);started=time.monotonic()
    def Adam_digest():
        return tensor_digest((str(i)+'.'+k,state[k]) for i,state in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
    if phase=='capacity':
        kernel.update();atomic_checkpoint(folder/'resume-step1.pt',kernel.state(),MAX_STATE,RESERVE)
        def draws():return [random.random(),float(np.random.random()),float(torch.rand(())),float(torch.rand((),device='cuda:0'))]
        expected_draws=draws();expected=kernel.update();expected_weights=trained_digest(model);moments=Adam_digest()
        saved=torch.load(folder/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
        bad={**saved,'core':{**saved['core'],'rng':{**saved['core']['rng'],'cuda':torch.zeros(4,dtype=torch.uint8)}}}
        rng=capture_rng('cuda:0');all_versions=signature(model.parameters())
        try:kernel.resume(bad)
        except ValueError:pass
        else:raise ValueError('Corrupt CUDA RNG must reject before any live mutation')
        if (all_versions!=signature(model.parameters()) or not torch.equal(rng['torch'],torch.get_rng_state())
                or not torch.equal(rng['cuda'],torch.cuda.get_rng_state()) or moments!=Adam_digest()):
            raise ValueError('Invalid recovery changed live parameters/Adam/RNG')
        kernel.resume(saved)
        if draws()!=expected_draws or kernel.update()!=expected or expected_weights!=trained_digest(model) or moments!=Adam_digest():
            raise ValueError('Actual full-word next update weights/Adam/source cursor and all4RNG draws must recover exactly')
        atomic_checkpoint(folder/'resume-step2.pt',kernel.state(),MAX_STATE,RESERVE)
        extremes=list(dict.fromkeys([max(range(len(data)),key=lambda p:data.rows[p]['native']['length']),
            max(range(len(data)),key=lambda p:len(data.rows[p]['response'].encode('utf-8')))]))
        extreme_records=[]
        for position in extremes:
            owner.owned(plan);optimizer.zero_grad(set_to_none=True)
            actual=data.batch([position]*setting['microbatch'],'cuda:0')
            metric=accumulate_gradients(model,[actual],ctc_weight=setting['ctc_weight'],native_CE_weight=setting['native_CE_weight'],amp=amp)
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.parameters) or any(p.grad is not None for p in kernel.fixed):
                raise ValueError('Full configured microbatch of actual longest complete source words must have finite new-only gradients')
            extreme_records.append(dict(source_position=position,actual_repeated_capacity_microbatch=setting['microbatch'],complete_metrics=metric))
        optimizer.zero_grad(set_to_none=True)
    else:
        with (REPORTS/(arm+'-metrics.jsonl')).open('x') as stream:
            for _ in range(setting['steps']):
                owner.owned(plan);tick=time.monotonic();record=kernel.update();stream.write(json.dumps(record)+'\n');stream.flush()
                if kernel.completed==1 or kernel.completed%25==0 or kernel.completed==setting['steps']:
                    print(json.dumps(dict(**record,elapsed_seconds=time.monotonic()-started,step_seconds=time.monotonic()-tick)),flush=True)
                    atomic_json(REPORTS/(arm+'-progress.json'),dict(step=kernel.completed,planned_steps=setting['steps'],latest=record,goal_achieved=False))
                if kernel.completed%256==0 or kernel.completed==setting['steps']:
                    atomic_checkpoint(folder/'resume.pt',kernel.state(),MAX_STATE,RESERVE)
        atomic_checkpoint(folder/'model.pt',export(model,plan,file_hash(REPORTS/'plan.json'),arm,kernel.completed),64*1024**2,RESERVE)
    after=legacy.native_original(model,batch,amp)
    with torch.no_grad(),amp():
        text_after=model.aligned(batch['input_ids'])['logits'].cpu()
        image_after=model.aligned(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
        cached_after=model.original.language.code.decode(torch.tensor([[4]],device='cuda:0'),cache)['logits'].cpu()
        word_source=data.batch([plan['longest_training_word_position']],'cuda:0')
        trained_word=model.word_logits(word_source['word_pixels']).cpu()
    if (fixed!=fixed_digest(model) or versions!=signature(kernel.fixed) or initial_trainable==trained_digest(model)
            or any(not torch.equal(before[r],after[r]) for r in before) or not torch.equal(cached_before,cached_after)
            or not torch.equal(text_before,text_after) or torch.equal(image_before,image_after)):
        raise ValueError('Every fixed weight/metadata/text-only/original route/cache must stay exact while the shared image path learns')
    common=dict(actual_counts=model.counts(),actual_updates=kernel.completed,complete_Adam_states=len(optimizer.state),
        actual_optimized_values=sum(p.numel() for p in kernel.parameters),all_fixed_weights_metadata_text_only_original_routes_and_code_cache_exact=True,
        original_native_probe_photo_id=photo_id,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        elapsed_seconds=time.monotonic()-started,candidate_promoted=False,goal_achieved=False)
    if phase=='capacity':
        atomic_json(REPORTS/f'capacity-{gpu}.json',dict(**common,source_extremes=extreme_records,physical_GPU=gpu,
            exact_next_update_and_corrupt_private_RNG_rejection=True,logical_disposable_updates=2,physical_optimizer_calls=3,
            retained_step1_sha256=file_hash(folder/'resume-step1.pt'),retained_step2_sha256=file_hash(folder/'resume-step2.pt'),production_updates=0));return
    receipt=dict(**common,checkpoint_sha256=file_hash(folder/'model.pt'),resume_sha256=file_hash(folder/'resume.pt'),source_exposure_budget=plan['source_exposure_budget'])
    atomic_json(REPORTS/(arm+'-training-complete.json'),receipt)
    cold,_=load(folder/'model.pt',REPORTS/'plan.json',arm,expected_checkpoint_sha256=receipt['checkpoint_sha256'],expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    with torch.no_grad(),amp():
        cold_image=cold.aligned(batch['input_ids'],batch['pixel_values'])['logits'].cpu()
        cold_text=cold.aligned(batch['input_ids'])['logits'].cpu();cold_word=cold.word_logits(word_source['word_pixels']).cpu()
    if (not torch.equal(cold_image,image_after) or not torch.equal(cold_text,text_after) or not torch.equal(cold_word,trained_word)
            or trained_digest(model)!=trained_digest(cold) or any(not torch.equal(v,after[r]) for r,v in legacy.native_original(cold,batch,amp).items())):
        raise ValueError('Actual full trained image/word/text and all original route cold reload must be exact')
    atomic_json(REPORTS/arm/'trained-shared-reload.json',dict(actual_counts=cold.counts(),actual_trained_image_word_text_and_all_original_routes_exact=True,one_original_backbone=True,goal_achieved=False))
    del cold
    evaluate_words(model,metadata,plan,arm,amp);guard.check(full=True)


def assess(arm,gpu):
    plan=legacy.registered('assess-'+arm+'-vision-'+str(gpu),gpu);torch,guard,amp=legacy.runtime(plan)
    from slm_lab.shared_word_inference_v1 import load
    checkpoint=OUTPUT/arm/'model.pt'
    model,metadata=load(checkpoint,REPORTS/'plan.json',arm,expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=file_hash(REPORTS/'plan.json'),device='cuda:0')
    legacy.native_suite(plan,model,metadata,arm,'vision',gpu,guard)


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Queue/owner must explicitly hide GPUs')
    if any(p.exists() for p in (REPORTS,OUTPUT,QUEUE)):raise ValueError('Preserve prior word experiments; no automatic restart')
    tests={'reports/shared-word-stage-v1-tests.log':'3 passed','reports/shared-word-evaluation-v1-tests.log':'3 passed',
           'reports/shared-word-inference-v1-tests.log':'1 passed','reports/shared-word-driver-v1-tests.log':'3 passed'}
    for name,passed in tests.items():
        if passed not in (ROOT/name).read_text():raise ValueError('Actual gradient/source/recovery/evaluation/loader/resource checks required')
    proof=json.loads((CPU_PROOF/'complete.json').read_text())
    if not proof['actual_next_update_all_new_weights_Adam_RNG_and_metrics_exact'] or not proof['actual_trained_full_model_cold_reload_exact']:
        raise ValueError('Actual full-size shared-word learning/recovery/cold reload required')
    predecessor=json.loads(PREVIOUS_QUEUE.read_text())['owner'];me=owner.resources.identity(os.getpid())
    names=['scripts/train_shared_words_v1.py','src/slm_lab/shared_word_inference_v1.py','src/slm_lab/shared_word_evaluation_v1.py',
        'src/slm_lab/shared_word_evidence_v1.py','tests/test_shared_word_evaluation_v1.py','tests/test_shared_word_inference_v1.py',
        'tests/test_shared_word_driver_v1.py','reports/shared-words-v1-proposed-exposure.json',*tests]
    files={str(ROOT/name):file_hash(ROOT/name) for name in names}
    for p in [CPU_PROOF/'plan.json',CPU_PROOF/'complete.json',CPU_PROOF/'independent-complete-readback.json',
              CPU_PROOF/'resume-step1.pt',CPU_PROOF/'resume-step2.pt',CPU_PROOF/'trained-disposable-extensions.pt']:
        files[str(p)]=file_hash(p)
    for path,sha in json.loads((CPU_PROOF/'plan.json').read_text())['source_files_sha256'].items():
        if path in files and files[path]!=sha:raise ValueError('Never reseal a changed original word/model source')
        files[path]=sha
    # Bind held indexes too; source admission already independently checked
    # every held pixel/transcript. Holding this queue does not touch any GPU.
    for suffix in ('valid.jsonl','valid.tokens.u16'):
        files[str(DATA/suffix)]=file_hash(DATA/suffix)
    queued_guard=FileBindings(files);deadline=time.monotonic()+36*3600
    while not previous_ended(predecessor):
        queued_guard.check()
        if time.monotonic()>deadline:raise ValueError('Previous full code owner still live; word queue expires without touching GPUs')
        atomic_json(QUEUE,dict(status='waiting for exact full code owner/workers to finish and restore original work',owner=me,predecessor=predecessor,
            source_files_sha256=files,observed_at=datetime.now(timezone.utc).isoformat(),production_model_updates=0,goal_achieved=False));time.sleep(10)
    queued_guard.check(full=True)
    for path in [PREVIOUS/'plan.json',PREVIOUS/'complete.json',PREVIOUS/'restored.json',PREVIOUS/'independent-comparison.json']:
        files[str(path)]=file_hash(path)
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.shared_word_evidence_v1 import budget
    from slm_lab.shared_word_evaluation_v1 import wrong_permutation
    from slm_lab.shared_word_inference_v1 import fresh
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    data=SharedWordData(DATA,'train','optimization');held=SharedWordData(DATA,'valid','evaluation');wrong_permutation(held.rows)
    old=json.loads((ROOT/'reports/whole-human-SFT-v1/plan.json').read_text())
    setting=dict(seed=948131,epochs=1,steps=math.ceil(len(data)/16),microbatch=4,accumulation=4,betas=[.9,.95],weight_decay=.01,
        gradient_clip=1.,ctc_weight=1.,native_CE_weight=1.)
    settings=dict(initial_model_plan=old,actual_counts=proof['actual_counts'],original_state_sha256=proof['fixed_original_digest'],source_admission_sha256=file_hash(DATA/'admission.json'),
        source_manifest_sha256=file_hash(DATA/'manifest.json'),training_options=setting,source_exposure_budget=budget(data,setting),
        longest_training_word_position=max(range(len(data)),key=lambda p:len(data.rows[p]['response'].encode('utf-8'))),
        evaluation=dict(batch_size=16,generation_budget=64,held_examples=len(held),all_three_image_controls=True),
        arms=dict(rate_1e4=dict(learning_rate=1e-4,physical_GPU=0),rate_3e4=dict(learning_rate=3e-4,physical_GPU=1),rate_1e3=dict(learning_rate=1e-3,physical_GPU=2)))
    model,_=fresh(settings,'cpu');settings['initial_parts_digest']=tensor_digest(model.parts.state_dict().items());del model
    guard=FileBindings(files)
    with (ROOT/'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not previous_ended(predecessor):raise ValueError('Prior exact owner became live; no overlap')
        require_idle_devices();guard.check(full=True);REPORTS.mkdir();OUTPUT.mkdir()
        for arm in ['initial',*settings['arms']]:
            (REPORTS/arm).mkdir();(OUTPUT/arm).mkdir();shutil.copyfile(ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json',OUTPUT/arm/'tokenizer.json')
        plan=dict(**settings,created_at=datetime.now(timezone.utc).isoformat(),owner=me,waiter=owner.WAITER,source_files_sha256=files,
            expires_at_unix=time.time()+14*3600,automatic_retry=False,all_capacity_updates_discarded=True,
            no_held_based_automatic_promotion=True,language_anchor='original_own_code',candidate_promoted=False,goal_achieved=False)
        atomic_json(REPORTS/'plan.json',plan);atomic_json(REPORTS/'children.json',[])
        def interrupted(signum,frame):raise KeyboardInterrupt('Owned shared-word run interrupted')
        signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
        driver=str(Path(__file__).resolve())
        watcher=subprocess.Popen(['.venv/bin/python','-u',driver,'--watchdog'],cwd=ROOT,stdout=(REPORTS/'watchdog.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
        def launch(args,subject,gpu):return owner.launch(plan,['-u',driver,*args],subject,str(gpu))
        def tasks(items):owner.wait_children(plan,[launch(args,subject,gpu) for args,subject,gpu in items])
        try:
            owner.idle_original();os.kill(owner.WAITER['pid'],signal.SIGSTOP);until=time.monotonic()+5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ('T','t'):
                if time.monotonic()>until:raise ValueError('Original exact idle waiter did not enter owned hold')
                time.sleep(.05)
            if (Path('/proc')/str(owner.WAITER['pid'])/'task'/str(owner.WAITER['pid'])/'children').read_text().strip():raise ValueError('Original waiter started work; never preempt it')
            tasks([(['--worker','capacity','--gpu',str(i)],'capacity-'+str(i),i) for i in range(4)])
            proofs=[json.loads((REPORTS/f'capacity-{i}.json').read_text()) for i in range(4)]
            if not all(p['exact_next_update_and_corrupt_private_RNG_rejection'] for p in proofs):raise ValueError('Every full configured GPUword capacity/recovery proof must pass')
            atomic_json(REPORTS/'capacity.json',dict(passed=True,all_four_original_L40s=True,proofs=proofs,production_updates=0,goal_achieved=False))
            jobs=[launch(['--worker','initial','--gpu','3'],'initial-words',3)]
            for arm,spec in plan['arms'].items():jobs.append(launch(['--worker','train','--arm',arm,'--gpu',str(spec['physical_GPU'])],'train-'+arm,spec['physical_GPU']))
            owner.wait_children(plan,jobs)
            tasks([(['--assess','vision','--arm',arm,'--gpu',str(i)],'assess-'+arm+'-vision-'+str(i),i) for i,arm in enumerate(['initial',*plan['arms']])])
            from tokenizers import Tokenizer
            from slm_lab.shared_word_evidence_v1 import reconcile
            reconcile(ROOT,REPORTS,OUTPUT,data,held,Tokenizer.from_file(str(OUTPUT/'initial/tokenizer.json')));guard.check(full=True)
            atomic_json(REPORTS/'complete.json',dict(status='All full original word epochs, complete held reading controls and matched native images finished; inspect quality',candidate_promoted=False,goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise
        finally:
            owner.restore_owned(plan,'Complete shared-word experiment ended; original work restored');watcher.wait(timeout=20)


if __name__=='__main__':
    configure();parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watchdog',action='store_true')
    parser.add_argument('--worker',choices=['capacity','initial','train']);parser.add_argument('--assess',choices=['vision'])
    parser.add_argument('--arm',default='initial');parser.add_argument('--gpu',type=int,choices=range(4),default=0);args=parser.parse_args()
    if args.watchdog:owner.watchdog()
    elif args.worker:worker(args.worker,args.arm,args.gpu)
    elif args.assess:assess(args.arm,args.gpu)
    else:parent()
