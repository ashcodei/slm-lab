"""Matched CE / CE+sequence-UL pilot on existing complete original code sources."""
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.code_repetition_objective_v1 import (select_rollout_rows,
    repetition_negative_positions,sequence_unlikelihood)
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import FORMAT,make_training_state
from slm_lab.code_residual_pilot_v1 import PROFILES,learning_rate
from slm_lab.code_source_generation_diagnostic_v1 import select_controls,reference_agreement
from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint,TrainingOutputs,supervised_gradients

ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=(dict(name='CE_control',alpha=0.),dict(name='CE_plus_repetition_UL',alpha=.1))
MAX_CHECKPOINT=7*1024**3//4
RESERVE=16*1024**3
EXTRA=96*1024**2


def main():
    reports=ROOT/'reports/code-repetition-objective-pilot-v1'
    output=ROOT/'checkpoints/code-repetition-objective-pilot-v1'
    if reports.exists() or output.exists():raise ValueError('Preserve earlier pilot; no implicit restart')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            live=(proc/'stat').read_text().rsplit(')',1)[1].split()[0] not in {'Z','X'}
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if live and (b'slm_lab.training' in args or any(a.endswith(b'scripts/train_extended_curriculum.py') for a in args)):
            raise ValueError('Original main writer must be terminal before allocating separate pilot')
    free=shutil.disk_usage(ROOT).free
    if free<RESERVE+2*MAX_CHECKPOINT+EXTRA:
        raise ValueError('Fund both full1.75GiB artifacts and96MiB reports/tokenizers above16GiB reserve')
    if free>=25*1024**3+256*1024**2:
        raise ValueError('Funded original curriculum resumption has priority')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:1');torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<20*1024**3:
        raise ValueError('Initially20GiB free on BF16 experiment GPU required')
    torch.cuda.set_per_process_memory_fraction(.35,device)
    occupancy=torch.empty(256*1024**2,dtype=torch.uint8,device=device)
    pilot=ROOT/'reports/code-residual-lr-pilot-v1'
    prior=json.loads((pilot/'results.json').read_text());prior_plan=json.loads((pilot/'plan.json').read_text())
    selection_path=pilot/'selected-original-rows.json'
    if file_hash(selection_path)!=prior_plan['selection_sha256']:raise ValueError('Exact old source roles/sample required')
    selection=json.loads(selection_path.read_text());train=selection['train'];validation=selection['validation']
    rollouts=select_rollout_rows(selection,number=128,budget=128,seed=910901)
    controls=select_controls(selection,budget=1024)
    assert len(train)==512 and len(validation)==128 and len({r['language'] for r in rollouts})==54
    pack_root=ROOT/'data/codenet-durable-pack-v2'
    packs={r:CompleteCodePack(pack_root,r,purpose='training' if r=='train' else 'audit') for r in ['train','validation']}
    source=Path(prior_plan['source_checkpoint']);initial_path=Path(prior['best_complete_checkpoint'])
    token_path=initial_path.parent/'tokenizer.json'
    names=['src/slm_lab/code_repetition_objective_v1.py','src/slm_lab/code_residual_branch_v1.py',
        'src/slm_lab/code_residual_checkpoint_v1.py','src/slm_lab/code_residual_pilot_v1.py',
        'src/slm_lab/code_source_generation_diagnostic_v1.py','src/slm_lab/codenet_checkpoint_v1.py',
        'src/slm_lab/codenet_token_pack_v1.py','src/slm_lab/model.py','src/slm_lab/training.py',
        'src/slm_lab/recurrent_ablation.py','tests/test_code_repetition_objective_v1.py',
        'reports/code-repetition-objective-v1-focused-tests.log',
        'reports/code-residual-repeated-source-pilot-v1/independent-complete-readback.json',
        'reports/code-residual-source-generation-v1/results.json']
    paths=[Path(__file__),source,initial_path,token_path,selection_path,pilot/'plan.json',pilot/'results.json',
        pack_root/'manifest.json',pack_root/'admission.json']+[ROOT/n for n in names]
    bound={str(p):file_hash(p) for p in paths}
    bound.update({str(pack_root/n):h for n,h in packs['train'].manifest['files_sha256'].items()})
    if (bound[str(source)]!=prior_plan['source_checkpoint_sha256']
            or bound[str(initial_path)]!=prior['best_complete_checkpoint_sha256']
            or bound[str(pilot/'plan.json')]!=prior['run_plan_sha256']):
        raise ValueError('Exact own base and original internally selected128-update policy required')
    sequences={}
    for row in train+validation:
        pack=packs[row['role']];ids,labels=pack.sequence(row['local_position'])
        if (int(pack.positions[row['local_position']])!=row['global_position']
                or hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()!=row['full_input_ids_sha256']
                or len(ids)!=row['context_tokens'] or sum(y!=-100 for y in labels)!=row['target_tokens']):
            raise ValueError('Every original whole program/EOS/mask required without clipping')
        sequences[row['id']]=(ids,labels)
    guard=FileBindings(bound);reports.mkdir();output.mkdir()
    for profile in EXPERIMENTS:
        (output/profile['name']).mkdir();shutil.copy2(token_path,output/profile['name']/'tokenizer.json')
    atomic_json(reports/'rollout-source-schedule.json',dict(original_train_rows=rollouts))
    schedule=[[r['id'] for r in train[i:i+4]] for i in range(0,512,4)]
    atomic_json(reports/'CE-source-schedule.json',dict(update_train_ids=schedule))
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        starting_policy_checkpoint=str(initial_path),starting_policy_sha256=bound[str(initial_path)],
        original_base_checkpoint=str(source),original_base_checkpoint_sha256=bound[str(source)],
        original_selection_sha256=bound[str(selection_path)],
        CE_schedule_sha256=file_hash(reports/'CE-source-schedule.json'),
        rollout_schedule_sha256=file_hash(reports/'rollout-source-schedule.json'),profiles=list(EXPERIMENTS),
        same_exact_initial_policy_and_data_order=True,fresh_AdamW_per_profile=True,
        old_optimizer_or_rng_history_resumed=False,updates_per_profile=128,batch_size=1,gradient_accumulation=4,
        training_programmes_per_profile=512,training_languages=54,validation_programmes=128,validation_languages=53,
        train_validation_group_overlap=0,original_response_targets_per_profile=sum(r['target_tokens'] for r in train),
        rollout_train_programmes_per_profile=128,rollout_training_languages=54,max_rollout_tokens=128,
        original_prompts_or_gold_programs_clipped=False,rollout_profile_weights_before_current_update=True,
        regularizer_ngram=16,regularizer_reference_repeated_ngrams_exempt=True,EOS_always_exempt=True,
        objective='Complete original-program token-average CE plus alpha times mean sequence unlikelihood on selected own-rollout tokens',
        regularizer_formula='mean(logsumexp(all_logits)-logsumexp(logits_excluding_the_selected_negative_token))',
        regularizer_not_wrong_code_or_human_preference_labels=True,
        prior_work='Welleck et al. https://arxiv.org/abs/1908.04319 section5.2; local16-gram/reference-exemption adaptation is unproven',
        learning_rate_profile=PROFILES[1],weight_decay=.01,gradient_clip=1.,foreach=False,seed=910911,
        observed_original_sources_before_and_after_each_profile=True,final_source_generation_budget=1024,source_controls=controls,
        fixed_final128_update_checkpoints_only=True,model_selection_or_automatic_promotion=False,
        functional_or_QA_feedback_used_for_updates=False,held_rows_used_for_updates=False,
        GPU='cuda:1',maximum_GPU_memory_fraction=.35,maximum_checkpoint_bytes=MAX_CHECKPOINT,
        checkpoint_cap_basis='All48 FP32 residual weights and two Adam moments total1811939328 bytes, plus192 step bytes, full RNG/metadata/metrics. Existing full artifacts are~1812053563 bytes.1.75GiB leaves>64MiB overhead; no tensors/states omitted.',
        maximum_final_checkpoint_files=2,extra_reports_and_tokenizer_budget_bytes=EXTRA,maximum_report_bytes=64*1024**2,
        minimum_remaining_home_bytes=RESERVE,original_curriculum25GiB_preflight_unchanged=True,
        full_source_or_final_base_integration=False,candidate_promoted=False,goal_achieved=False,
        scope='Matched bounded source-training objective test on original512-source/54-language pilot. Repetition heuristics are not correctness or efficiency oracles, and may penalize valid alternatives. No new decoder heuristics, primary scoring changes, full-source promotion, broad retention or frontier claims.')
    atomic_json(reports/'plan.json',plan);plan_sha=file_hash(reports/'plan.json')

    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            candidate_promoted=False,goal_achieved=False,**fields))

    try:
        status('loading exact own policy for matched objective experiment')
        random.seed(910911);np.random.seed(910912);torch.manual_seed(910913);torch.cuda.manual_seed_all(910914)
        base,metadata=load_checkpoint(source,'cpu',training_state=False)
        old=torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True)
        if (old['format']!=FORMAT or old['stage']!='code_residual_lr_pilot_v1'
                or old['steps']!=old['planned_steps'] or old['steps']!=128
                or old['base_checkpoint_sha256']!=bound[str(source)]
                or old['tokenizer_sha256']!=bound[str(token_path)] or metadata['tokenizer_sha256']!=bound[str(token_path)]):
            raise ValueError('Exact original complete pilot/own base required')
        model=RoutedCodeResidual(base,width=2048);model.load_adapter_state(old['residuals'])
        if (model.parameter_counts()!=old['model_counts']
                or state_digest(model.base.state_dict())!=old['frozen_backbone_state_sha256']):
            raise ValueError('Original model count and every frozen base tensor required')
        initial=model.adapter_state();initial_digest=state_digest(initial)
        if initial_digest!=state_digest(old['residuals']):raise ValueError('Every warm-start tensor must match')
        del old;gc.collect();model.to(device);model.code.gradient_checkpointing=True;del occupancy
        base_digest=state_digest(model.base.state_dict())
        tokenizer=Tokenizer.from_file(str(token_path))
        encoder=packed_encoder(packs['train'],tokenizer,model.config)
        wrapped=TrainingOutputs(model.code,'sft');amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)

        @torch.no_grad()
        def assess(rows):
            model.eval();records=[]
            for row in rows:
                ids,labels=sequences[row['id']]
                with amp():loss=model.code(torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device))['loss']
                if not bool(torch.isfinite(loss)):raise ValueError('Finite full source losses required')
                records.append(dict(id=row['id'],role=row['role'],targets=row['target_tokens'],mean_NLL=float(loss)))
            return dict(token_mean_NLL=sum(r['targets']*r['mean_NLL'] for r in records)/sum(r['targets'] for r in records),rows=records)

        initial_losses=dict(train=assess(train),validation=assess(validation))
        atomic_json(reports/'initial-source-NLL.json',initial_losses)
        results=[]
        for profile_index,profile in enumerate(EXPERIMENTS):
            pending=len(EXPERIMENTS)-profile_index
            if shutil.disk_usage(ROOT).free<RESERVE+pending*MAX_CHECKPOINT+EXTRA:
                raise ValueError('Every remaining complete artifact and report must stay funded')
            model.load_adapter_state(initial)
            if state_digest(model.residuals.state_dict())!=initial_digest:raise ValueError('Both profiles need identical initial residual bytes')
            random.seed(910911);np.random.seed(910912);torch.manual_seed(910913);torch.cuda.manual_seed_all(910914)
            optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=3e-5,weight_decay=.01,foreach=False)
            metrics=[];generated_records=[];profile_reports=reports/profile['name'];profile_reports.mkdir()
            started=time.monotonic()
            for step in range(128):
                guard.check();model.train();optimizer.zero_grad(set_to_none=True)
                lr=learning_rate(PROFILES[1],step);optimizer.param_groups[0]['lr']=lr
                group_rows=train[step*4:(step+1)*4]
                ce,targets=supervised_gradients(wrapped,encoder,
                    [[{'packed_code_index':r['local_position']}] for r in group_rows],'response',device,amp)
                if targets!=sum(r['target_tokens'] for r in group_rows) or encoder.truncated:
                    raise ValueError('Complete original train-only CE targets required')
                row=rollouts[step];ids,labels=sequences[row['id']]
                start=next(i for i,y in enumerate(labels) if y!=-100);reference=ids[start:]
                model.eval()
                with torch.no_grad(),amp():
                    produced=model.code.generate(torch.tensor([ids[:start]],device=device),max_new_tokens=128,
                        eos_token_id=2,use_cache=True)[0,start:].tolist()
                negatives=repetition_negative_positions(produced,reference,n=16,eos=2)
                ul_value=0.
                if negatives:
                    full=torch.tensor([ids[:start]+produced],device=device)
                    model.train()
                    with torch.set_grad_enabled(profile['alpha']>0),amp():
                        logits=model.code(full)['logits']
                        ul=sequence_unlikelihood(logits,full,start,negatives,eos=2)
                    if not bool(torch.isfinite(ul)):raise ValueError('Finite sequence unlikelihood required')
                    ul_value=float(ul.detach())
                    if profile['alpha']>0:(profile['alpha']*ul).backward()
                    del logits,ul,full
                if any(p.grad is not None for p in model.base.parameters()):raise ValueError('Every original general/vision/critic tensor must stay frozen')
                if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.adapter_parameters()):
                    raise ValueError('All48 residual gradients must be present and finite')
                norm=torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                metrics.append(dict(step=step+1,mean_CE_before_update=float(ce),response_targets=targets,
                    mean_UL_before_update=ul_value,UL_alpha=profile['alpha'],negative_positions=len(negatives),
                    applied_UL_gradient=bool(negatives and profile['alpha']>0),learning_rate=lr,
                    gradient_norm_before_clipping=float(norm)))
                raw=tokenizer.decode(produced[:-1] if produced and produced[-1]==2 else produced,skip_special_tokens=False)
                generated_records.append(dict(step=step+1,id=row['id'],role='train',language=row['language'],
                    generation_budget=128,generated_before_current_optimizer_update=True,
                    prompt_tokens=start,prompt_input_ids_sha256=hashlib.sha256(np.asarray(ids[:start],dtype='<u2').tobytes()).hexdigest(),
                    generated_token_ids=produced,raw_text=raw,raw_text_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                    negative_generated_token_positions=negatives,mean_UL=ul_value,
                    applied_UL_gradient=metrics[-1]['applied_UL_gradient']))
                atomic_json(profile_reports/'training-rollouts.json',dict(records=generated_records))
                if step<2 or (step+1)%4==0:
                    status('training matched complete-program CE and own-rollout repetition objective',
                        profile=profile['name'],completed_steps=step+1,planned_steps=128,latest=metrics[-1])
                    print('objective_update',profile['name'],json.dumps(metrics[-1]),flush=True)
            optimizer.zero_grad(set_to_none=True);model.eval()
            trained=assess(train);validated=assess(validation)
            if state_digest(model.base.state_dict())!=base_digest:raise ValueError('Original backbone changed')
            guard.check(full=True)
            state=make_training_state(model,optimizer,base_checkpoint_sha256=bound[str(source)],
                tokenizer_sha256=bound[str(token_path)],run_plan_sha256=plan_sha,completed_steps=128,
                planned_steps=128,device=device,metrics=metrics)
            state.update(stage='code_repetition_objective_pilot_v1',objective_profile=profile,
                starting_policy_checkpoint_sha256=bound[str(initial_path)],original_selection_sha256=bound[str(selection_path)],
                source_manifest_sha256=bound[str(pack_root/'manifest.json')],fresh_optimizer=True,
                training_sources=512,training_rollout_sources=128,rollout_generation_budget=128,
                regularizer_ngram=16,reference_repeated_ngrams_exempt=True)
            checkpoint=output/profile['name']/'final.pt'
            atomic_checkpoint(checkpoint,state,maximum_bytes=MAX_CHECKPOINT,
                minimum_free_bytes=RESERVE+(pending-1)*MAX_CHECKPOINT+EXTRA)
            sha=file_hash(checkpoint)
            atomic_json(profile_reports/'frozen-final-state.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha,
                checkpoint_bytes=checkpoint.stat().st_size,updates=128,fixed_before_source_generation_or_tests=True))
            generated=[]
            with torch.no_grad():
                for control in controls:
                    guard.check();ids,labels=sequences[control['id']];start=next(i for i,y in enumerate(labels) if y!=-100)
                    with amp():produced=model.code.generate(torch.tensor([ids[:start]],device=device),max_new_tokens=1024,eos_token_id=2,use_cache=True)[0,start:].tolist()
                    raw=tokenizer.decode(produced[:-1] if produced and produced[-1]==2 else produced,skip_special_tokens=False)
                    generated.append(dict(id=control['id'],role=control['role'],language=control['language'],
                        generation_budget=1024,generated_token_ids=produced,raw_program_text=raw,
                        raw_program_sha256=hashlib.sha256(raw.encode()).hexdigest(),**reference_agreement(produced,ids[start:],2)))
                    atomic_json(profile_reports/'generations.json',dict(records=generated))
                    status('generating fixed-end raw source controls; no optimizer updates',profile=profile['name'],completed=len(generated),planned=8)
            result=dict(profile=profile,actual_updates=128,original_programme_exposures=512,
                response_target_exposures=sum(m['response_targets'] for m in metrics),own_rollouts=128,
                own_rollout_tokens=sum(len(g['generated_token_ids']) for g in generated_records),
                rollouts_with_negatives=sum(bool(g['negative_generated_token_positions']) for g in generated_records),
                actual_UL_gradient_updates=sum(m['applied_UL_gradient'] for m in metrics),
                initial_residual_state_sha256=initial_digest,final_residual_state_sha256=state_digest(model.residuals.state_dict()),
                original_base_state_sha256=base_digest,final_base_state_sha256=state_digest(model.base.state_dict()),
                train=trained,validation=validated,checkpoint=str(checkpoint),checkpoint_sha256=sha,
                checkpoint_bytes=checkpoint.stat().st_size,complete_residual_Adam_RNG_saved=True,
                training_rollouts_sha256=file_hash(profile_reports/'training-rollouts.json'),
                generation_report_sha256=file_hash(profile_reports/'generations.json'),
                generation_summaries={role:dict(sources=4,exact_reference_matches=sum(g['exact_complete_reference_token_match'] for g in generated if g['role']==role),
                    terminal_EOS=sum(g['terminal_EOS'] for g in generated if g['role']==role)) for role in ['train','validation']},
                elapsed_seconds=time.monotonic()-started,candidate_promoted=False,goal_achieved=False)
            atomic_json(profile_reports/'results.json',result);results.append(result)
            del state,optimizer;gc.collect();torch.cuda.empty_cache()
        guard.check(full=True)
        if sum(p.stat().st_size for p in reports.rglob('*') if p.is_file())>64*1024**2:raise ValueError('Declared64MiB report budget exceeded')
        atomic_json(reports/'results.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=bound,plan_sha256=plan_sha,model_counts=model.parameter_counts(),initial_source_losses=initial_losses,
            actual_Adam_updates_total=256,complete_original_train_exposures=1024,
            complete_original_response_target_exposures=2*sum(r['target_tokens'] for r in train),profiles=results,
            held_source_rows_used_for_updates=False,original_base_all_tensors_unchanged=True,
            fixed_final_states_without_selection=True,full_functional_or_knowledge_assessment_complete=False,
            full_source_or_final_base_integration=False,candidate_promoted=False,goal_achieved=False,
            scope=plan['scope']))
        status('matched objective experiment complete; inspect source behavior and unchanged capability assessments')
    except Exception as error:
        status('objective pilot stopped; preserve evidence and inspect before continuation',error=str(error))
        raise


if __name__=='__main__':main()
