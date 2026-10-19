"""Full-size fold and dense-language learning/recovery mechanism on CPU.

Uses an explicitly old, already evaluated own40-update actor to test folding,
then two full-source code examples from the new admitted corpus to test real
backbone gradients. This is not the production starting policy, a full source
epoch, a GPU admission, or a new generated capability assessment.
"""
from datetime import datetime,timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
from PIL import Image,ImageOps
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from slm_lab.code_contests_CE_view_v1 import CodeContestsCEView
from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.own_actor_dense_fold_v1 import fold,export,load
from slm_lab.own_dense_checkpoint_v1 import make_state,restore_state
from slm_lab.own_execution_contrast_policy_CPU_v1 import load as load_actor
from slm_lab.shared_word_data_v1 import SharedWordData
from slm_lab.source_jsonl_records_v1 import records
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-actor-fold-dense-update-CPU-v1'
CHECKPOINTS=ROOT/'checkpoints/full-actor-fold-dense-update-CPU-v1'
RESERVE=26*1024**3


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or CHECKPOINTS.exists():raise ValueError('Fresh full-size CPU proof required; no implicit retry')
    if shutil.disk_usage(ROOT).free<RESERVE+42*1024**3:raise ValueError('Fund complete native and two private states above original writer reserve')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    tests=ROOT/'reports/own-actor-dense-fold-v1-final-tests.log'
    if '4 passed' not in tests.read_text():raise ValueError('Nonzero-actor fold, full-gradient, cold-native and overflow tests required')
    view=CodeContestsCEView(ROOT)
    lengths=view.pack.index['length'][view.positions]
    first=int(np.argmax(lengths));first_row=view.row(first)
    second=next(i for i,r in enumerate(view.rows) if r['source_problem_id']!=first_row['source_problem_id'])
    positions=[first,second]
    if int(lengths[first])!=4096:raise ValueError('Actual longest complete native4096training example required')
    batches=[view.batch(p,'cpu') for p in positions]
    actor_path=ROOT/'checkpoints/own-execution-contrast-full-CPU-v1/anchor_contrast-actor.pt'
    actor_plan=ROOT/'reports/own-execution-contrast-full-CPU-v1/plan.json'
    reader_path=ROOT/'reports/own-execution-contrast-full-CPU-v2-outcome-readback/anchor_contrast/complete.json'
    reader=json.loads(reader_path.read_text());expected=reader['source_files_sha256'][str(actor_path)]
    if reader['actual_raw_code_deliveries_reexecuted']!=48 or reader['independent_outcomes']['native_code']['completion']['correct']!=0:
        raise ValueError('Explicitly evaluated old experimental source required; no useful-policy claim')
    photo_root=ROOT/'data/alignment-curriculum-v1';photo_manifest=json.loads((photo_root/'manifest.json').read_text())
    photo_path=photo_root/'contrastive.train.jsonl'
    if file_hash(photo_path)!=photo_manifest['files_sha256'][photo_path.name]:raise ValueError('Original train photograph inventory changed')
    photo=next(r for r in records(photo_path) if r.get('annotation_origin')=='Original human spoken image description, manually transcribed.' and r['split']=='train')
    # The original source has a concrete annotation-origin label; fail rather
    # than silently substituting an arbitrary image when it differs.
    image_path=Path(photo['image'])
    if file_hash(image_path)!=photo['stored_image_sha256']:raise ValueError('Original whole source photograph changed')
    with Image.open(image_path) as image:
        array=np.asarray(ImageOps.pad(image.convert('RGB'),(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128)),dtype=np.uint8).copy()
    if hashlib.sha256(array.tobytes()).hexdigest()!=photo['model_pixel_sha256_448']:raise ValueError('Actual original native448photo pixels differ')
    pixels=torch.from_numpy(array.astype(np.float32)/127.5-1).permute(2,0,1).unsqueeze(0)
    tokenizer=Tokenizer.from_file(str(view.tokenizer_path))
    photo_ids=torch.tensor([[1]+tokenizer.encode('User: '+photo['prompt']+'\nAssistant:').ids+tokenizer.encode(photo['response']).ids+[2]])
    words=SharedWordData(ROOT/'data/shared-byte-words-v1','train','optimization')
    word_position=next(i for i,r in enumerate(words.rows) if r['byte_family']=='real_word')
    word_batch=words.batch([word_position],'cpu')
    files={str(p):file_hash(p) for p in [Path(__file__),tests,ROOT/'tests/test_own_actor_dense_fold_v1.py',actor_path,actor_plan,reader_path,
        view.folder/'manifest.json',view.folder/'admission.json',photo_root/'manifest.json',photo_path,image_path,
        ROOT/'data/shared-byte-words-v1/manifest.json',ROOT/'data/shared-byte-words-v1/admission.json',Path(words.rows[word_position]['image'])]}
    files.update({str(p):file_hash(p) for p in python_sources(ROOT,[Path(__file__),ROOT/'tests/test_own_actor_dense_fold_v1.py'])})
    guard=FileBindings(files);OUT.mkdir();CHECKPOINTS.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    source_metadata=json.loads(actor_plan.read_text())
    sources=dict(experimental_own_actor_sha256=expected,experimental_own_actor_plan_sha256=file_hash(actor_plan),
        new_CE_view_manifest_sha256=file_hash(view.folder/'manifest.json'),new_CE_view_admission_sha256=file_hash(view.folder/'admission.json'))
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        source_is_old_step2048_plus40update_own_machine_contrast_experiment=True,
        source_is_not_current_complete_main_human_DPO_or_code_RL=True,
        required_source_values=999706442,required_folded_student_values=984399690,
        actual_complete_CE_view_positions=positions,actual_complete_context_lengths=[int(lengths[p]) for p in positions],
        original_training_photo_id=photo['id'],original_training_word_position=word_position,
        FP32_fold_calibration_max_absolute_logit_bound=.002,FP32_fold_calibration_mean_KL_bound=.000001,
        complete_original4096capacity_context_not_clipped=True,logical_updates=2,physical_optimizer_calls=3,
        CPU_threads=2,fresh_four_GPU_capacity_still_required=True,full_original_source_epoch=False,
        new_generated_code_or_general_visual_capability_not_assessed=True,candidate_promoted=False,goal_achieved=False))
    atomic_json(OUT/'progress.json',dict(owner=owner,phase='loading actual old trained own source actor',model_updates=0))
    try:
        actor,_=load_actor(actor_path,actor_plan,'anchor_contrast',expected_checkpoint_sha256=expected,expected_plan_sha256=file_hash(actor_plan))
        if actor.counts()['actual_unique']!=999706442 or torch.cuda.is_initialized():raise ValueError('Exactly one actual full-size own CPU source actor required')
        source_signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in actor.parameters()]
        rng=capture_rng('cpu');candidate,receipt=fold(actor)
        if candidate.counts()['total_unique']!=984399690 or not equal(rng,capture_rng('cpu')):
            raise ValueError('Actual full-size dense fold must remove factors without changing private RNG')
        calibrations=[]
        for name,batch in [('complete4096new_code',batches[0]),('complete_second_new_code',batches[1]),
                           ('actual_whole_train_photo448',dict(input_ids=photo_ids,pixel_values=pixels))]:
            atomic_json(OUT/'progress.json',dict(owner=owner,phase='FP32 fold calibration',current_actual_input=name,model_updates=0))
            kwargs={k:batch[k] for k in ['input_ids','attention_mask','pixel_values'] if k in batch}
            with torch.no_grad():
                original=actor(**kwargs);folded=candidate(**kwargs)
                a=original['logits'].float();b=folded['logits'].float()
                delta=float((a-b).abs().max());lp=a.log_softmax(-1);lq=b.log_softmax(-1)
                kl=float((lp.exp()*(lp-lq)).sum(-1).mean())
                observation=dict(input=name,complete_original_text_positions=batch['input_ids'].shape[1],
                    max_absolute_logit_difference=delta,mean_forward_full_vocabulary_KL=kl,
                    next_token_argmax_disagreements=int((a.argmax(-1)!=b.argmax(-1)).sum()),
                    maximum_absolute_value_difference=float((original['values']-folded['values']).abs().max()))
                if not math.isfinite(delta) or not math.isfinite(kl) or delta>.002 or kl>.000001:
                    raise ValueError('Actual full-size FP32 fold exceeds the declared native drift limits')
                calibrations.append(observation)
            del original,folded,a,b,lp,lq
        with torch.no_grad():
            word_before=actor.base.word_logits(word_batch['word_pixels']).detach().clone()
            if not torch.equal(word_before,candidate.word_logits(word_batch['word_pixels'])):
                raise ValueError('Full actual original word path must remain bit-identical through fold')
        if source_signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in actor.parameters()]:raise ValueError('Folding/calibration changed original source actor')
        atomic_json(OUT/'actual-fold-calibration.json',dict(actual_counts_before=actor.counts(),actual_fold_receipt=receipt,
            actual_complete_native_inputs=calibrations,actual_original_word_byte_logits_bit_exact=True,
            original_source_values_flags_pointers_and_versions_unchanged=True,
            no_generated_sequence_or_all_head_function_parity_claim=True))
        initial_path=CHECKPOINTS/'initial-folded-student.pt'
        atomic_checkpoint(initial_path,export(candidate,sources=sources,fold_receipt=receipt,completed_updates=0),5*1024**3,RESERVE)
        del actor;gc.collect()
        named=candidate.language_named_parameters();parameters=[p for _,p in named];chosen={id(p) for p in parameters}
        for p in parameters:p.requires_grad_(True)
        fixed=[(n,p) for n,p in candidate.named_parameters() if id(p) not in chosen]
        fixed_digest=tensor_digest(fixed);initial_language=tensor_digest(named)
        optimizer=torch.optim.AdamW(parameters,lr=1e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
        random.seed(972217);np.random.seed(972217);torch.manual_seed(972217);metrics=[]
        def adam_digest():return tensor_digest((str(i)+'.'+key,value[key]) for i,value in enumerate(optimizer.state.values()) for key in ['step','exp_avg','exp_avg_sq'])
        def step(index):
            guard.check(full=False);candidate.train();candidate.network.gradient_checkpointing=True
            optimizer.zero_grad(set_to_none=True);batch=batches[index]
            atomic_json(OUT/'progress.json',dict(owner=owner,phase='actual full-dense native code backward',next_original_view_position=positions[index],
                logical_step=index+1,complete_context_positions=batch['input_ids'].shape[1]))
            output=candidate(**batch);loss=output['loss'];loss.backward(inputs=parameters)
            if (not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)
                    or any(p.grad is not None for _,p in fixed)):
                raise ValueError('Every full dense language tensor must receive finite gradients, with fixed heads untouched')
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True));optimizer.step();optimizer.zero_grad(set_to_none=True)
            record=dict(step=index+1,original_CE_view_position=positions[index],original_source_id=view.row(positions[index])['id'],
                complete_context_positions=batch['input_ids'].shape[1],response_EOS_targets=int((batch['labels'][:,1:]!=-100).sum()),
                token_mean_CE=float(loss.detach()),gradient_norm_before_clip=norm,actual_finite_active_language_tensors=len(parameters))
            metrics.append(record);return record
        first_record=step(0)
        cursor=dict(next_update=1,original_CE_view_positions=positions,CE_view_manifest_sha256=sources['new_CE_view_manifest_sha256'])
        private_path=CHECKPOINTS/'step1-full-private.pt'
        atomic_checkpoint(private_path,make_state(candidate,optimizer,sources=sources,completed=1,planned=2,cursor=cursor,metrics=metrics,device='cpu'),14*1024**3,RESERVE)
        def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]
        expected_draws=draws();expected_update=step(1);expected_language=tensor_digest(named);expected_adam=adam_digest();expected_rng=capture_rng('cpu')
        saved=torch.load(private_path,map_location='cpu',weights_only=True,mmap=True)
        for bad in [dict(saved,cursor=dict(cursor,next_update=99)),dict(saved,rng=dict(saved['rng'],torch=torch.zeros(4,dtype=torch.uint8)))]:
            signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in candidate.parameters()];before_rng=capture_rng('cpu')
            try:restore_state(candidate,optimizer,bad,expected_sources=sources,planned=2,expected_cursor=cursor,device='cpu')
            except ValueError:pass
            else:raise ValueError('Changed source cursor or invalid private CPU RNG must reject')
            if signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in candidate.parameters()] or not equal(before_rng,capture_rng('cpu')) or adam_digest()!=expected_adam:
                raise ValueError('Invalid state changed full-size parameters, Adam or private RNG')
        completed,metrics=restore_state(candidate,optimizer,saved,expected_sources=sources,planned=2,expected_cursor=cursor,device='cpu')
        if completed!=1 or draws()!=expected_draws or step(1)!=expected_update or tensor_digest(named)!=expected_language or adam_digest()!=expected_adam or not equal(expected_rng,capture_rng('cpu')):
            raise ValueError('Every real next native update, full language weight, Adam state and private CPU RNG must replay exactly')
        if fixed_digest!=tensor_digest(fixed) or initial_language==expected_language:raise ValueError('Actual dense weights must change while every fixed value remains unchanged')
        if any(int(state['step'])!=2 for state in optimizer.state.values()) or len(optimizer.state)!=317:
            raise ValueError('Every317actual full dense language Adam clocks must reach2')
        final_private=CHECKPOINTS/'final-full-private.pt'
        atomic_checkpoint(final_private,make_state(candidate,optimizer,sources=sources,completed=2,planned=2,cursor=dict(cursor,next_update=2),metrics=metrics,device='cpu'),14*1024**3,RESERVE)
        final_path=CHECKPOINTS/'trained-native-student.pt'
        atomic_checkpoint(final_path,export(candidate,sources=sources,fold_receipt=receipt,completed_updates=2),5*1024**3,RESERVE)
        del saved,optimizer;gc.collect();candidate.eval()
        with torch.no_grad():before=candidate(**{k:v for k,v in batches[0].items() if k!='labels'})['logits'].detach().clone()
        cold,cold_metadata=load(final_path,expected_checkpoint_sha256=file_hash(final_path),expected_sources=sources)
        with torch.no_grad():
            after=cold(**{k:v for k,v in batches[0].items() if k!='labels'})['logits']
            if not torch.equal(before,after) or not torch.equal(word_before,cold.word_logits(word_batch['word_pixels'])):
                raise ValueError('Cold independently loaded final student must reproduce whole4096logits and fixed original word outputs bit-exactly')
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_fold_receipt=receipt,
            actual_FP32_full_size_fold_calibrations=calibrations,actual_language_updates=2,physical_optimizer_calls=3,
            actual_full_language_Adam_states=317,actual_native_training_records=metrics,
            actual_dense_language_values_changed=True,all_fixed_vision_critic_auxiliary_values_exact=True,
            actual_next_native_update_all_language_weights_Adam_private_CPU_RNG_bit_exact=True,
            changed_source_cursor_and_bad_CPU_RNG_rejected_before_mutation=True,
            cold_final_student_full4096logits_and_word_outputs_bit_exact=True,
            self_contained_student_parameters=984399690,original_source_not_production_start=True,
            fresh_four_GPU_capacity_and_full_source_epoch_and_generated_assessments_still_required=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',OUT/'actual-fold-calibration.json',initial_path,private_path,final_private,final_path]},
            no_frontier_or_general_functional_or_visual_gain_claim=True,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),old_actor_values=999706442,folded_student_values=984399690,
            real_full_language_updates=2,full_context4096_recovery_and_cold_native_exact=True,goal_achieved=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),candidate_promoted=False,goal_achieved=False));raise


if __name__=='__main__':main()
