"""Read the actual2048dense continuation state and matched held-loss report.

No model or optimizer is constructed. The mutable resume is observed once,
with its exact stat/hash retained; its weight/moment values are not copied.
The permanent native checkpoint, per-leaf hashes, clocks, source journal and
all four private RNG states remain as compact evidence. No loss is rerun.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/code-first-continuation-v3'
MODEL=ROOT/'checkpoints/code-first-continuation-v3'
OUT=ROOT/'reports/code-first-step2048-CPU-readback-v1'


def stat(path):
    s=path.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def live(identity):
    path=Path('/proc')/str(identity['pid'])/'stat'
    if not path.exists():return False
    fields=path.read_text().split(') ',1)[1].split()
    return fields[19]==identity['ticks'] and fields[0] not in {'Z','X'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only state reader required')
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_inference_v2 import sources
    from slm_lab.focused_code_continuation_data_v3 import FocusedCodeContinuationData
    from slm_lab.focused_continuation_run_contract_v2 import schedule_for
    from slm_lab.model import ModelConfig
    from slm_lab.phase_joint_checkpoint_v1 import validate_clocks
    from slm_lab.source_jsonl_records_v1 import records
    torch.set_num_threads(1);started=time.monotonic();plan_path=SOURCE/'plan.json';plan=json.loads(plan_path.read_text());digest=file_hash(plan_path)
    checkpoint=MODEL/'step-2048/model.pt';held_path=SOURCE/'held-packed-2048.json';deadline=time.monotonic()+24*3600
    while not checkpoint.is_file() or not held_path.is_file():
        if (SOURCE/'failure.json').exists() or not live(plan['owner']):raise ValueError('Exact original training ended before the fixed permanent checkpoint')
        if time.monotonic()>deadline:raise ValueError('Observation deadline; original source never stopped or restarted')
        time.sleep(5)
    OUT.mkdir();owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(owner=owner,exact_training_owner=plan['owner'],fixed_completed_updates=2048,
        no_model_optimizer_or_CUDA_construction=True,mutable_resume_values_not_copied=True,model_updates=0,goal_achieved=False))
    try:
        guard=FileBindings(plan['source_files_sha256']);path=MODEL/'resume.pt';before=stat(path);resume_hash=file_hash(path)
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);core=state['core']
        initial=torch.load(MODEL/'initial/model.pt',map_location='cpu',weights_only=True,mmap=True)
        native=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True);payload=native['model']
        if (state['format']!='own-complete-foundation-then-instruction-reference-checkpoint-v1'
                or core['format']!='own-fast-joint-complete-Adam-all-worker-RNG-v2' or core['mode']!='train'
                or core['completed']!=2048 or core['world_size']!=4 or core['run_plan_sha256']!=digest
                or core['training_options']!=plan['training_options'] or state['phase_bounds']!=plan['phase_bounds']
                or native['format']!='own-fast-joint-deterministic-positions-native-model-v2'
                or native['arm']!='step-2048' or native['completed_updates']!=2048
                or native['whole_source_epochs_completed']!=0 or native['run_plan_sha256']!=digest
                or payload['sources']!=sources(plan,digest) or initial['arm']!='initial' or initial['completed_updates']!=0):
            raise ValueError('Every exact partial native/private/source identity must agree')
        private=core['model']
        if ({k:v for k,v in private.items() if k not in ['model','arm']}!=
                {k:v for k,v in native.items() if k not in ['model','arm']}
                or private['arm']!='trained'
                or {k:v for k,v in private['model'].items() if k not in ['network','heads']}!=
                   {k:v for k,v in payload.items() if k not in ['network','heads']}):
            raise ValueError('The native export and observed private checkpoint must describe the same committed state')
        data=FocusedCodeContinuationData(ROOT,ModelConfig(**payload['config']),purpose='optimization')
        schedule=schedule_for(data,plan,'train');journal=[]
        for row in records(SOURCE/'training-metrics.jsonl'):
            if len(journal)==2048:break
            journal.append(row)
        if len(journal)!=2048 or journal!=core['metrics']:raise ValueError('Every actual committed external journal row must match the private state')
        validate_clocks(core,plan,'train');clocks=Counter();budgets={};options=plan['training_options']
        for step,row in enumerate(journal):
            name,positions=schedule.at(step);kind=data.kinds[name]
            active=['language'] if kind=='text' else ['vision']+(['word'] if kind=='word' else [])
            targets=sum(data.targets[name][p] for p in positions);contexts=sum(data.lengths[name][p] for p in positions)
            if (row['step']!=step+1 or row['family']!=name or row['global_original_positions']!=positions
                    or row['global_examples']!=len(positions) or row['complete_context_tokens']!=contexts
                    or row['response_EOS_targets']!=targets or row['actual_active_groups']!=active
                    or row['active_parameter_tensors']!=sum(dict(language=317,vision=112,word=16)[g] for g in active)
                    or row['reference_forward_queried']!=(kind=='text') or row['resident_training_only_reference_parameters']!=984399690
                    or row['reference_KL_weight']!=options['reference_KL_weight']):
                raise ValueError('An actual complete original source, target mask budget or active/reference decision differs')
            if (not all(math.isfinite(row[k]) for k in ['CE_sum','CTC_word_sum','full_context_reference_KL_sum','gradient_norm_before_clip'])
                    or row['token_mean_CE']!=row['CE_sum']/targets or row['words']!=(len(positions) if kind=='word' else 0)
                    or row['mean_byte_normalized_CTC']!=(row['CTC_word_sum']/len(positions) if kind=='word' else None)
                    or row['full_context_reference_KL_mean']!=(row['full_context_reference_KL_sum']/(contexts-len(positions)) if kind=='text' else None)):
                raise ValueError('Every actual finite loss, complete-token denominator and word normalization must rebuild')
            warmup=max(1,round(len(schedule)*options['warmup_fraction']))
            factor=((step+1)/warmup if step<warmup else options['cosine_floor']+(1-options['cosine_floor'])*.5*
                (1+math.cos(math.pi*(step-warmup)/max(1,len(schedule)-warmup))))
            if any(row['learning_rates'][g]!=options['learning_rates'][g]*factor for g in ['language','vision','word']):
                raise ValueError('Every actual continuation learning rate must reconstruct')
            clocks.update(active);budget=budgets.setdefault(name,Counter())
            budget.update(presentations=len(positions),complete_context_tokens=contexts,response_EOS_targets=targets)
        def tensors(model):return {group+'.'+name:value for group in ['network','heads'] for name,value in model[group].items() if name!='lm_head.weight'}
        a,b,c=tensors(initial['model']),tensors(payload),tensors(private['model'])
        if a.keys()!=b.keys() or b.keys()!=c.keys() or len(b)!=461 or sum(v.numel() for v in b.values())!=984399690:
            raise ValueError('Every original461unique tensor and984399690values must be present')
        if payload['network']['embedding.weight'].data_ptr()!=payload['network']['lm_head.weight'].data_ptr():raise ValueError('One actual tied vocabulary required')
        def leaf(value,other=None,nonnegative=False):
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or not value.is_contiguous():raise ValueError('An actual contiguous full FP32 tensor is required')
            if other is not None and (value.shape!=other.shape or value.dtype!=other.dtype):raise ValueError('Every complete native/private shape and dtype must match')
            sha=hashlib.sha256();flat=value.reshape(-1)
            for start in range(0,flat.numel(),1<<20):
                chunk=flat[start:start+(1<<20)]
                if not bool(torch.isfinite(chunk).all()) or nonnegative and bool((chunk<0).any()):raise ValueError('Every actual weight/moment must be finite and every second moment nonnegative')
                if other is not None and not torch.equal(chunk,other.reshape(-1)[start:start+len(chunk)]):raise ValueError('Actual native/private tensor differs')
                sha.update(chunk.view(torch.uint8).numpy().tobytes())
            return dict(shape=list(value.shape),dtype=str(value.dtype),values=value.numel(),sha256=sha.hexdigest())
        groups=dict(language=[],vision=[],word=[]);changed={k:[] for k in groups};fixed=[];weights={}
        for name,value in b.items():
            weights[name]=leaf(value,c[name]);group=('vision' if name.startswith('network.vision.') else
                'language' if name.startswith('network.') and not name.startswith('network.value_head.') else
                'word' if name.startswith('heads.word.') else None)
            if group:
                groups[group].append(name)
                if not torch.equal(value,a[name]):changed[group].append(name)
            else:
                if not torch.equal(value,a[name]):raise ValueError('An unused auxiliary or critic tensor changed')
                fixed.append(name)
        if {k:len(v) for k,v in groups.items()}!=dict(language=317,vision=112,word=16) or len(fixed)!=16:
            raise ValueError('Exact full317language/112vision/16word groups and16unused leaves required')
        optimizer=core['optimizer'];keys=set();moments={}
        for group,name in zip(optimizer['param_groups'],['language','vision','word']):
            if (group['name']!=name or len(group['params'])!=len(groups[name]) or group['lr']!=journal[-1]['learning_rates'][name]
                    or tuple(group['betas'])!=tuple(options['betas']) or group['weight_decay']!=options['weight_decay']):
                raise ValueError('Every actual Adam partition and hyperparameter must agree')
            for key,parameter in zip(group['params'],groups[name]):
                keys.add(key);moment=optimizer['state'][key]
                if set(moment)!= {'step','exp_avg','exp_avg_sq'} or float(moment['step'])!=clocks[name]:raise ValueError('Every actual full Adam source-active clock must match')
                if any(moment[k].shape!=b[parameter].shape for k in ['exp_avg','exp_avg_sq']):raise ValueError('Complete original-shaped full-weight moments required')
                moments[parameter]=dict(step=clocks[name],exp_avg=leaf(moment['exp_avg']),exp_avg_sq=leaf(moment['exp_avg_sq'],nonnegative=True))
        if len(optimizer['param_groups'])!=3 or len(keys)!=445 or set(optimizer['state'])!=keys:raise ValueError('Every and only445selected full Adam leaves required')
        binding=state['reference_binding']
        if binding!=plan['starting_policy_reference_binding'] or file_hash(binding['path'])!=binding['checkpoint_sha256']:
            raise ValueError('Exact complete own starting-reference binding required')
        reference=torch.load(binding['path'],map_location='cpu',weights_only=True,mmap=True);reference_sha=hashlib.sha256()
        for name,value in tensors(reference).items():
            if not torch.equal(value,a[name]):raise ValueError('The frozen own reference must equal the complete original initial student')
            reference_sha.update(json.dumps([name,str(value.dtype),list(value.shape)]).encode());reference_sha.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        if reference_sha.hexdigest()!=binding['parameter_sha256']:raise ValueError('Every own starting-reference parameter digest must match')
        if len(core['rngs'])!=4:raise ValueError('All four private training RNG states required')
        for r in core['rngs']:
            random.Random().setstate(r['python']);n=r['numpy']
            np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
            torch.Generator(device='cpu').set_state(r['torch'])
            if r['cuda'].dtype!=torch.uint8 or list(r['cuda'].shape)!=[16]:raise ValueError('Actual private CUDA byte profile required; no CUDA generator is instantiated here')
        initial_held=SOURCE/'held-packed-initial.json';before_loss,after_loss=[json.loads(p.read_text()) for p in [initial_held,held_path]]
        if {k:v for k,v in before_loss.items() if k!='token_mean_NLL'}!={k:v for k,v in after_loss.items() if k!='token_mean_NLL'}:
            raise ValueError('Every original held position, context and target denominator must be identical')
        packed=data.replay.packed;family=packed.families[0];population=packed.size(family,split='valid');positions=[i*population//128 for i in range(128)]
        held=[]
        for p in positions:
            row=packed.row(family,p,split='valid')
            if row['split']!='valid' or row['source_split']!='valid':raise ValueError('Only original held source windows required')
            held.append(dict(position=p,input_sha256=row['_replay_input_sha256'],targets=len(row['tokens'])-1))
        if (positions!=after_loss['selected_original_positions'] or after_loss['evaluated_windows']!=128
                or sum(r['targets'] for r in held)!=524160 or after_loss['complete_causal_targets']!=524160
                or after_loss['original_manifest_sha256']!=packed.manifest['packed_manifest_sha256']):
            raise ValueError('All actual128held windows and524160original targets must rebuild')
        if stat(path)!=before or torch.cuda.is_initialized():raise ValueError('Observed mutable resume changed or CUDA was initialized; evidence is not admitted')
        capture=OUT/'captured-state-evidence.pt'
        torch.save(dict(observed_mutable_resume_path=str(path),stat_at_capture=before,sha256_at_capture=resume_hash,
            completed=2048,model_leaf_hashes=weights,Adam_leaf_hashes=moments,optimizer_groups=optimizer['param_groups'],
            all_four_actual_private_RNG_states=core['rngs'],metrics=journal,no_model_or_Adam_values_retained=True),capture)
        atomic_json(OUT/'held-window-bindings.json',held);guard.check(full=False)
        retained=[Path(__file__),plan_path,checkpoint,MODEL/'initial/model.pt',Path(binding['path']),initial_held,held_path,capture,OUT/'held-window-bindings.json',OUT/'plan.json']
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_updates=2048,whole_source_epoch_completed=False,
            actual_unique_student_values=984399690,actual_model_tensors=461,actual_full_Adam_states=445,source_active_group_updates=dict(clocks),
            changed_parameter_tensors={k:len(v) for k,v in changed.items()},all16unused_tensors_exact=True,
            every_native_parameter_exactly_equal_observed_private_state=True,every_weight_moment_finite_and_source_clock_exact=True,
            actual_source_prefix_budgets={k:dict(v) for k,v in budgets.items()},every_committed_source_position_and_target_budget_rederived=True,
            all_four_private_CPU_RNG_states_parsed_CUDA_bytes_structure_only=True,actual_own_starting_reference_all_parameters_exact=True,
            held_original_windows=128,held_original_causal_targets=524160,held_input_positions_and_bytes_rederived=True,
            held_initial_NLL=before_loss['token_mean_NLL'],held_step2048_NLL=after_loss['token_mean_NLL'],
            relative_held_NLL_reduction=1-after_loss['token_mean_NLL']/before_loss['token_mean_NLL'],
            NLL_values_are_original_GPU_producer_observations_not_new_neural_forwards=True,
            next_token_gain_not_generated_program_correctness_or_all_domain_gain=True,
            original_mutable_resume_may_be_atomically_replaced_later=True,large_private_model_and_moments_not_copied=True,
            source_files_sha256={str(p):file_hash(p) for p in retained},seconds=time.monotonic()-started,
            models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_updates=2048,held_NLL=after_loss['token_mean_NLL'],source_active_clocks=dict(clocks))),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),original_GPU_jobs_not_signalled=True,goal_achieved=False));raise


if __name__=='__main__':main()
