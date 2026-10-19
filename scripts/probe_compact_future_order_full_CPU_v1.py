"""Full-size matched CPU mechanism control on two whole original train sources.

This disposable two-update/arm study is not a production epoch, held benchmark,
GPU capacity proof, or a candidate for promotion. All source tokens stay whole.
"""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())


def run(args):
    import torch
    from slm_lab.broad_anchored_execution_stage_v1 import source_order
    from slm_lab.broad_anchored_policy_v3 import fresh
    from slm_lab.codenet_checkpoint_v1 import FileBindings,capture_rng
    from slm_lab.code_residual_checkpoint_v1 import _validate_adam
    from slm_lab.compact_future_order_v1 import CompactFutureOrder,FORMAT
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=args.output.absolute();snapshot=args.snapshot.resolve();receipt_path=snapshot/'complete.json'
    receipt=json.loads(receipt_path.read_text());plan=json.loads((snapshot/'plan.json').read_text())
    if (out.exists() or out.resolve()!=out or receipt.get('complete') is not True or receipt.get('completed')!=512
            or plan.get('planned_steps')!=3721 or torch.cuda.is_initialized()):
        raise ValueError('Fresh CPU-only output and exact immutable original source snapshot required')
    files=dict(receipt['source_files_sha256']);files[str(receipt_path)]=sha(receipt_path)
    files.update({str(p):sha(p) for p in python_sources(ROOT,[Path(__file__)])});guard=FileBindings(files)
    def checked(full=False):
        available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
        if torch.cuda.is_initialized() or shutil.disk_usage(out.parent).free<42*1024**3 or available<64*1024**3:
            raise ValueError('CPU-only control requires2GiBoutput plus40GiBscratch reserve and64GiBhost headroom')
        guard.check(full=full)
    checked(True);out.mkdir();owner=dict(pid=os.getpid(),birth=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    write(out/'owner-plan.json',dict(owner=owner,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        method=FORMAT,rank=32,window=16,head_seed=973331,auxiliary_weights=[0.,.1],learning_rate=1e-5,betas=[.9,.95],
        weight_decay=.01,gradient_clip=1.,source_seed=963117,original_source_population=3721,
        predeclared_positions='first2of unchanged seeded documented-to-competition source order; no length selection',
        original_complete_sources_per_arm=2,actual_updates_per_arm=2,original_epoch_not_completed=True,
        FP32=True,CPU_threads=1,CPU_affinity=sorted(os.sched_getaffinity(0)),nice=19,scheduler='SCHED_IDLE',
        research_hypothesis_not_paper_reproduction=True,extra_training_FLOPs_not_matched=True,
        no_original_GPU_worker_or_source_changes=True,no_new_public_or_private_data_downloads=True,goal_achieved=False))
    try:
        print(json.dumps(dict(phase='load_exact_own_human_DPO_policy',owner=owner)),flush=True)
        load_plan=copy.deepcopy(plan);load_plan['starting_policy_inputs']['checkpoint']=str(snapshot/'starting-human-DPO-actor.pt')
        actor,_=fresh(load_plan,'cpu');model=CompactFutureOrder(actor,rank=32,window=16,seed=973331)
        if model.counts()['total_unique']!=999806282:raise ValueError('Actual full policy/native/critic plus99840auxiliary values must fit strictly under1B')
        corpus=ExpandedExecutionCorpus(ROOT);positions=source_order(corpus,963117)[:2]
        original=[dict(position=p,id=corpus.row(p)['id'],complete_context_positions=corpus.row(p)['complete_context_positions'],
                       response_EOS_targets=corpus.row(p)['response_EOS_targets']) for p in positions]
        initial={n:p.detach().clone() for n,p in model.named_selected(auxiliary=True)}
        selected_ids={id(p) for _,p in model.named_selected(auxiliary=True)};fixed=[p for p in model.parameters() if id(p) not in selected_ids]
        def signature():return [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
        fixed_signature=signature();base_sha=tensor_digest(model.actor.base.named_parameters());critic=model.actor.parts['critic'].weight.detach().clone()
        initial_sha=tensor_digest(initial.items());checked(True);corpus.guard.check(full=True)
        write(out/'actual-full-size-admission.json',dict(actual_counts=model.counts(),policy_counts=actor.counts(),original_sources=original,
            complete_context_positions=sum(r['complete_context_positions'] for r in original),
            complete_response_EOS_targets=sum(r['response_EOS_targets'] for r in original),initial_selected_sha256=initial_sha,
            fixed_own_native_sha256=base_sha,actual_trainable_parameters_with_auxiliary=sum(p.numel() for _,p in model.named_selected(auxiliary=True)),
            no_external_weights_or_teacher=True,original_private_training_RNG_or_Adam_not_restored=True,
            whole_epoch_or_GPU_capacity_not_proved=True,goal_achieved=False))
        print(json.dumps(dict(phase='start_actual_full_size_matched_CPU_updates',actual_counts=model.counts(),sources=original)),flush=True)
        def scores():
            model.eval();values=[]
            with torch.no_grad():
                for p in positions:
                    checked();batch=corpus.batch(p,'cpu');value=model.actor(**batch)['loss']
                    values.append(dict(position=p,original_train_source_CE=float(value),response_EOS_targets=corpus.row(p)['response_EOS_targets']))
            return values
        initial_scores=scores();arms={}
        for name,weight in [('CE_only',0.),('CE_plus_future_order',.1)]:
            with torch.no_grad():
                for n,p in model.named_selected(auxiliary=True):p.copy_(initial[n])
            if tensor_digest((n,p) for n,p in model.named_selected(auxiliary=True))!=initial_sha:
                raise ValueError('Both arms must start from every identical policy and auxiliary value')
            model.zero_grad(set_to_none=True);torch.random.default_generator.manual_seed(979331)
            named=model.named_selected(auxiliary=bool(weight));params=[p for _,p in named]
            optimizer=torch.optim.AdamW(params,lr=1e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
            records=[]
            for step,p in enumerate(positions,1):
                checked();model.train();model.actor.policy.gradient_checkpointing=True;optimizer.zero_grad(set_to_none=True);begin=time.monotonic()
                print(json.dumps(dict(phase='whole_source_backward',arm=name,step=step,position=p)),flush=True)
                loss,record=model.objective(corpus.batch(p,'cpu'),auxiliary_weight=weight,chunk_positions=32,recompute_chunks=True)
                if record['original_next_token_targets']!=corpus.row(p)['response_EOS_targets']:
                    raise ValueError('Every complete original response/EOS target must reach the objective')
                if weight and step==1 and record['original_CE']!=arms['CE_only']['records'][0]['original_CE']:
                    raise ValueError('Matched first-source native CE must agree before either arm update')
                loss.backward(inputs=params)
                if (signature()!=fixed_signature or any(p.grad is not None for p in fixed)
                        or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params)):
                    raise ValueError('All selected gradients and every original frozen value/flag must preserve the full-size contract')
                actor_gradient_nonzero=any(bool(p.grad.count_nonzero()) for n,p in named if n.startswith('actor.'))
                future_gradient_nonzero=any(bool(p.grad.count_nonzero()) for n,p in named if n.startswith('future.'))
                if not actor_gradient_nonzero or bool(weight) and not future_gradient_nonzero:raise ValueError('Actual policy/head gradients beyond Adam weight decay are required')
                norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));optimizer.step();optimizer.zero_grad(set_to_none=True)
                records.append(dict(record,step=step,position=p,source_id=corpus.row(p)['id'],complete_context_positions=corpus.row(p)['complete_context_positions'],
                    actual_actor_gradient_nonzero=actor_gradient_nonzero,actual_auxiliary_gradient_nonzero=future_gradient_nonzero,
                    gradient_norm_before_clip=norm,seconds=time.monotonic()-begin))
                checked();write(out/f'{name}-step-{step}.json',records[-1])
            _validate_adam(optimizer.state_dict(),params,2);final_scores=scores()
            if not weight and any(not torch.equal(initial[n],p) for n,p in model.named_selected(auxiliary=True) if n.startswith('future.')):
                raise ValueError('The CE-only control cannot change any unused auxiliary weight')
            saved=dict(format='disposable-two-source-compact-future-order-CPU-mechanism-control-v1',method=FORMAT,
                source_files_sha256=files,owner=owner,arm=name,settings=model.settings,completed_updates=2,original_planned_source_population=3721,
                original_sources=original,selected_names=[n for n,_ in named],selected={n:p.detach().clone() for n,p in model.named_selected(auxiliary=True)},
                optimizer=optimizer.state_dict(),own_new_CPU_rng=capture_rng('cpu'),records=records,initial_train_source_CE=initial_scores,
                final_train_source_CE=final_scores,original_source_epoch_not_completed=True,not_a_standalone_or_production_checkpoint=True,
                candidate_promoted=False,goal_achieved=False)
            path=out/f'{name}-selected-Adam-RNG-control.pt';torch.save(saved,path)
            arms[name]=dict(records=records,initial_train_source_CE=initial_scores,final_train_source_CE=final_scores,
                selected_adam_states=len(optimizer.state),actual_selected_parameters=sum(p.numel() for p in params),
                selected_actor_sha256=tensor_digest(model.actor.parts.named_parameters()),auxiliary_sha256=tensor_digest(model.future.named_parameters()),
                complete_control_checkpoint=str(path),complete_control_checkpoint_sha256=sha(path))
            del optimizer,saved
        if (signature()!=fixed_signature or tensor_digest(model.actor.base.named_parameters())!=base_sha
                or not torch.equal(critic,model.actor.parts['critic'].weight)):
            raise ValueError('Every original base/vision/word/auxiliary value and unused critic must remain exact')
        checked(True);corpus.guard.check(full=True)
        write(out/'complete.json',dict(complete=True,owner=owner,actual_counts=model.counts(),original_sources=original,
            initial_selected_sha256=initial_sha,arms=arms,source_files_sha256=files,
            every_original_frozen_native_vision_word_and_critic_value_unchanged=True,total_actual_CPU_Adam_updates=4,
            only_two_original_training_sources_per_arm=True,full_source_epoch_or_GPU_capacity_not_proved=True,
            train_source_likelihood_not_held_code_correctness=True,original_matched96program_assessment_unchanged=True,
            useful_code_general_multimodal_improvement_not_proved=True,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_CPU_updates=4,actual_counts=model.counts())),flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),partial_probe_outputs_retained=True,
            original_workers_sources_and_checkpoint_states_unchanged=True,goal_achieved=False));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,default=Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1'))
    parser.add_argument('--output',type=Path,default=Path('/tmp/slm-compact-future-order-full-CPU-v1'));args=parser.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)!=19
            or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))):
        raise ValueError('CUDA-hidden idle one-thread full-size CPU probe required')
    os.sched_setscheduler(0,os.SCHED_IDLE,os.sched_param(0));allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[-2] if len(allowed)>1 else allowed[0]});run(args)


if __name__=='__main__':main()
