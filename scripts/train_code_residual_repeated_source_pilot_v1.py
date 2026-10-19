"""Fixed four-additional-pass source-learning experiment; never production promotion."""
from datetime import datetime, timezone
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

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state, FORMAT
from slm_lab.code_residual_pilot_v1 import learning_rate, PROFILES
from slm_lab.code_source_generation_diagnostic_v1 import select_controls, reference_agreement
from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint, TrainingOutputs, supervised_gradients

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / 'reports/code-residual-repeated-source-pilot-v1'
    output = ROOT / 'checkpoints/code-residual-repeated-source-pilot-v1'
    if reports.exists() or output.exists():
        raise ValueError('Preserve previous experiment; no implicit restart')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            live = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0] not in {'Z','X'}
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if live and (b'slm_lab.training' in args or any(a.endswith(b'scripts/train_extended_curriculum.py') for a in args)):
            raise ValueError('Original main writer must be terminal before this separate experiment')
    free = shutil.disk_usage(ROOT).free
    if free < 18 * 1024**3 + 64 * 1024**2:
        raise ValueError('Fund one complete2GiB artifact and64MiB reports above16GiB remaining reserve')
    if free >= 25 * 1024**3 + 256 * 1024**2:
        raise ValueError('Original checked curriculum resumption has priority when its storage is funded')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0] < 20 * 1024**3:
        raise ValueError('Initially20GiB free on a BF16-capable experiment GPU required')
    torch.cuda.set_per_process_memory_fraction(.35, device)
    # The disk-cleanup watcher waits for idle GPUs. Occupy GPU memory before
    # CPU loading, so it cannot launch the main writer during this experiment.
    occupancy = torch.empty(256 * 1024**2, dtype=torch.uint8, device=device)
    pilot = ROOT / 'reports/code-residual-lr-pilot-v1'
    prior = json.loads((pilot / 'results.json').read_text())
    prior_plan = json.loads((pilot / 'plan.json').read_text())
    selection_path = pilot / 'selected-original-rows.json'
    if file_hash(selection_path) != prior_plan['selection_sha256']:
        raise ValueError('Exact original512 train/128 validation sample required')
    selection = json.loads(selection_path.read_text())
    train, validation = selection['train'], selection['validation']
    if (len(train) != 512 or len(validation) != 128
            or len({r['id'] for r in train}) != 512
            or any(r['role'] != 'train' for r in train)
            or any(r['role'] != 'validation' for r in validation)
            or {r['group_id'] for r in train} & {r['group_id'] for r in validation}
            or len({r['language'] for r in train}) != 54):
        raise ValueError('Whole original roles, groups and54-language training coverage required')
    controls = select_controls(selection, budget=1024)
    pack_root = ROOT / 'data/codenet-durable-pack-v2'
    packs = {role: CompleteCodePack(pack_root, role,
        purpose='training' if role == 'train' else 'audit') for role in ['train','validation']}
    source = Path(prior_plan['source_checkpoint'])
    initial_path = Path(prior['best_complete_checkpoint'])
    token_path = initial_path.parent / 'tokenizer.json'
    independent = ROOT / 'reports/source-learning-diagnostics-v1-independent-readback.json'
    readback = json.loads(independent.read_text())
    if not readback['all48_residuals_and_Adam_steps_verified'] or readback['goal_achieved']:
        raise ValueError('Completed prior learning-path evidence required')
    names = ['src/slm_lab/code_residual_branch_v1.py','src/slm_lab/code_residual_checkpoint_v1.py',
        'src/slm_lab/code_residual_pilot_v1.py','src/slm_lab/code_source_generation_diagnostic_v1.py',
        'src/slm_lab/codenet_checkpoint_v1.py','src/slm_lab/codenet_token_pack_v1.py',
        'src/slm_lab/model.py','src/slm_lab/training.py','src/slm_lab/recurrent_ablation.py']
    paths = [Path(__file__), source, initial_path, token_path, pilot/'plan.json', pilot/'results.json',
        selection_path, independent, pack_root/'manifest.json',pack_root/'admission.json'] + [ROOT/n for n in names]
    bound = {str(p):file_hash(p) for p in paths}
    bound.update({str(pack_root/n):h for n,h in packs['train'].manifest['files_sha256'].items()})
    if (bound[str(source)] != prior_plan['source_checkpoint_sha256']
            or bound[str(initial_path)] != prior['best_complete_checkpoint_sha256']
            or bound[str(pilot/'plan.json')] != prior['run_plan_sha256']):
        raise ValueError('Exact own completed-language base and frozen original pilot required')
    sequences = {}
    for row in train + validation:
        pack = packs[row['role']]
        ids, labels = pack.sequence(row['local_position'])
        if (int(pack.positions[row['local_position']]) != row['global_position']
                or hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest() != row['full_input_ids_sha256']
                or len(ids) != row['context_tokens']
                or sum(y != -100 for y in labels) != row['target_tokens']):
            raise ValueError('All640 whole original programs/EOS/masks required; never clip sources')
        sequences[row['id']] = (ids,labels)
    # Fixed original order in every pass isolates additional exposure from
    # changed sampling/order. Every train source occurs once in each pass.
    schedule = [[r['id'] for r in train[i:i+4]] for _ in range(4) for i in range(0,512,4)]
    assert len(schedule) == 512 and all(len(s)==4 for s in schedule)
    assert all(sorted(x for s in schedule[p*128:(p+1)*128] for x in s) == sorted(r['id'] for r in train) for p in range(4))
    guard = FileBindings(bound)
    reports.mkdir()
    output.mkdir()
    shutil.copy2(token_path,output/'tokenizer.json')
    atomic_json(reports/'training-source-schedule.json',dict(update_train_ids=schedule))

    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            candidate_promoted=False,goal_achieved=False,**fields))

    plan = dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        original_base_checkpoint=str(source),original_base_checkpoint_sha256=bound[str(source)],
        starting_policy_checkpoint=str(initial_path),starting_policy_sha256=bound[str(initial_path)],
        original_selection_sha256=bound[str(selection_path)],
        schedule_sha256=file_hash(reports/'training-source-schedule.json'),training_updates=512,
        original_train_programmes=512,training_languages=54,additional_complete_passes=4,
        original_validation_programmes=128,validation_languages=53,train_validation_group_overlap=0,
        original_train_response_targets_per_pass=sum(r['target_tokens'] for r in train),
        additional_training_programme_exposures=2048,
        additional_response_target_exposures=4*sum(r['target_tokens'] for r in train),
        batch_size=1,gradient_accumulation=4,profile=PROFILES[1],fresh_AdamW=True,
        old_Adam_or_rng_history_resumed=False,seed=910801,foreach=False,weight_decay=.01,gradient_clip=1.,
        objective='Unchanged token-normalized cross entropy over complete original programs and EOS',
        fixed_end_model_only=True,early_stop_or_validation_selection=False,
        source_teacher_observations=[0,128,256,512],raw_source_generation_at_final=8,
        generation_budget=1024,source_controls=controls,
        validation_or_benchmark_answers_used_for_updates=False,
        maximum_checkpoint_bytes=2*1024**3,minimum_remaining_home_bytes=16*1024**3,
        maximum_report_bytes=64*1024**2,maximum_GPU_memory_fraction=.35,GPU='cuda:1',
        original_main_writer_terminal=True,original_25GiB_preflight_unchanged=True,
        full_source_or_final_base_integration=False,candidate_promoted=False,goal_achieved=False,
        scope='Bounded additional-exposure diagnostic on the same previously declared512-program sample. Motivated by exact four-source memorization with worsened validation, not by claimed benchmark gain. Does not admit full-source scaling, change scoring, protect code-route generalization automatically, or finish multimodal/RLHF/frontier requirements.')
    atomic_json(reports/'plan.json',plan)
    plan_sha = file_hash(reports/'plan.json')
    status('loading exact own base and old pilot; fresh optimizer for fixed four additional passes')
    random.seed(910801);np.random.seed(910802);torch.manual_seed(910803);torch.cuda.manual_seed_all(910804)
    base,metadata = load_checkpoint(source,'cpu',training_state=False)
    initial = torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True)
    if (initial['format'] != FORMAT or initial['stage'] != 'code_residual_lr_pilot_v1'
            or initial['steps'] != initial['planned_steps'] or initial['steps'] != 128
            or initial['base_checkpoint_sha256'] != bound[str(source)]
            or initial['tokenizer_sha256'] != bound[str(token_path)]
            or metadata['tokenizer_sha256'] != bound[str(token_path)]):
        raise ValueError('Exact complete128-update original warm-start policy required')
    model = RoutedCodeResidual(base,width=2048)
    model.load_adapter_state(initial['residuals'])
    if (model.parameter_counts() != initial['model_counts']
            or state_digest(model.residuals.state_dict()) != state_digest(initial['residuals'])
            or state_digest(model.base.state_dict()) != initial['frozen_backbone_state_sha256']):
        raise ValueError('Every initial residual/base tensor and unique parameter count must match')
    del initial;gc.collect()
    model.to(device);model.code.gradient_checkpointing=True
    del occupancy
    optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=3e-5,weight_decay=.01,foreach=False)
    tokenizer=Tokenizer.from_file(str(token_path))
    encoders={r:packed_encoder(packs[r],tokenizer,model.config) for r in packs}
    wrapped=TrainingOutputs(model.code,'sft')
    base_before=state_digest(model.base.state_dict())
    initial_residual=state_digest(model.residuals.state_dict())
    rows_by_id={r['id']:r for r in train}
    amp=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    metrics=[];observations=[]

    @torch.no_grad()
    def assess(rows):
        model.eval();values=[]
        for row in rows:
            ids,labels=sequences[row['id']]
            with amp():
                loss=model.code(torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device))['loss']
            if not bool(torch.isfinite(loss)):
                raise ValueError('Finite full source observation required')
            values.append(dict(id=row['id'],role=row['role'],language=row['language'],
                targets=row['target_tokens'],mean_NLL=float(loss)))
        return dict(token_mean_NLL=sum(r['targets']*r['mean_NLL'] for r in values)/sum(r['targets'] for r in values),rows=values)

    def observe(step):
        value=dict(step=step,train=assess(train),validation=assess(validation))
        observations.append(value);atomic_json(reports/'observations.json',dict(observations=observations))
        status('complete original train/validation teacher observation',completed_steps=step,planned_steps=512,
            train_NLL=value['train']['token_mean_NLL'],validation_NLL=value['validation']['token_mean_NLL'])
        print('source_observation',step,value['train']['token_mean_NLL'],value['validation']['token_mean_NLL'],flush=True)

    observe(0)
    for step,ids in enumerate(schedule):
        guard.check();model.train();optimizer.zero_grad(set_to_none=True)
        lr=learning_rate(PROFILES[1],step);optimizer.param_groups[0]['lr']=lr
        started=time.monotonic()
        rows=[rows_by_id[i] for i in ids]
        loss,targets=supervised_gradients(wrapped,encoders['train'],
            [[{'packed_code_index':r['local_position']}] for r in rows],'response',device,amp)
        if targets != sum(r['target_tokens'] for r in rows) or encoders['train'].truncated:
            raise ValueError('Complete train-only target-token updates required; never clip')
        if any(p.grad is not None for p in model.base.parameters()):
            raise ValueError('Original general/vision/critic backbone must stay frozen')
        if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.adapter_parameters()):
            raise ValueError('Every residual gradient must remain present and finite')
        norm=torch.nn.utils.clip_grad_norm_(model.adapter_parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        metrics.append(dict(step=step+1,pass_number=step//128+1,response_targets=targets,
            mean_NLL_before_update=float(loss),learning_rate=lr,gradient_norm_before_clipping=float(norm),
            seconds=time.monotonic()-started))
        if step < 2 or (step+1)%8==0:
            status('training same512 whole original programs across four fixed additional passes',
                completed_steps=step+1,planned_steps=512,latest=metrics[-1])
            print('repeated_source_update',json.dumps(metrics[-1]),flush=True)
        if step+1 in [128,256,512]:
            observe(step+1)
    optimizer.zero_grad(set_to_none=True);model.eval()
    if state_digest(model.base.state_dict()) != base_before:
        raise ValueError('Every original backbone tensor must remain exact')
    guard.check(full=True)
    state=make_training_state(model,optimizer,base_checkpoint_sha256=bound[str(source)],
        tokenizer_sha256=bound[str(token_path)],run_plan_sha256=plan_sha,completed_steps=512,
        planned_steps=512,device=device,metrics=metrics)
    state.update(stage='code_residual_repeated_source_pilot_v1',starting_policy_checkpoint_sha256=bound[str(initial_path)],
        source_manifest_sha256=bound[str(pack_root/'manifest.json')],fresh_optimizer=True,
        original_selection_sha256=bound[str(selection_path)],training_sources=512,additional_complete_passes=4,
        additional_training_exposures=2048)
    checkpoint=output/'final.pt'
    atomic_checkpoint(checkpoint,state,maximum_bytes=2*1024**3,minimum_free_bytes=16*1024**3)
    checkpoint_sha=file_hash(checkpoint)
    # Freeze the fixed-end state before any raw generation or test feedback.
    atomic_json(reports/'frozen-final-state.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
        checkpoint_bytes=checkpoint.stat().st_size,plan_sha256=plan_sha,updates=512,
        selected_without_generation_or_functional_or_QA_feedback=True))
    generations=[]
    with torch.no_grad():
        for row in controls:
            guard.check();ids,labels=sequences[row['id']]
            start=next(i for i,y in enumerate(labels) if y!=-100)
            with amp():
                produced=model.code.generate(torch.tensor([ids[:start]],device=device),max_new_tokens=1024,
                    eos_token_id=2,use_cache=True)[0,start:].tolist()
            raw=tokenizer.decode(produced[:-1] if produced and produced[-1]==2 else produced,skip_special_tokens=False)
            generations.append(dict(id=row['id'],role=row['role'],language=row['language'],
                generation_budget=1024,generated_token_ids=produced,raw_program_text=raw,
                raw_program_sha256=hashlib.sha256(raw.encode()).hexdigest(),**reference_agreement(produced,ids[start:],2)))
            atomic_json(reports/'generations.json',dict(records=generations))
            status('generating frozen final source controls; no model updates',completed=len(generations),planned=8)
    guard.check(full=True)
    if sum(p.stat().st_size for p in reports.iterdir() if p.is_file()) > 64*1024**2:
        raise ValueError('Declared64MiB report budget exceeded')
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        plan_sha256=plan_sha,parameter_counts=model.parameter_counts(),actual_updates=512,
        additional_complete_training_programme_exposures=2048,original_training_sources=512,
        response_target_exposures=sum(m['response_targets'] for m in metrics),observations=observations,
        initial_residual_state_sha256=initial_residual,final_residual_state_sha256=state_digest(model.residuals.state_dict()),
        original_base_state_sha256=base_before,final_base_state_sha256=state_digest(model.base.state_dict()),
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,checkpoint_bytes=checkpoint.stat().st_size,
        complete_residual_Adam_RNG_saved=True,generation_report_sha256=file_hash(reports/'generations.json'),
        generation_summaries={role:dict(sources=4,exact_reference_matches=sum(g['exact_complete_reference_token_match'] for g in generations if g['role']==role),
            terminal_EOS=sum(g['terminal_EOS'] for g in generations if g['role']==role)) for role in ['train','validation']},
        validation_or_test_rows_used_for_updates=False,full_source_or_final_base_integration=False,
        fixed_end_model_selected_without_generation_or_test_feedback=True,
        functional_correctness_assessed=False,candidate_promoted=False,goal_achieved=False,
        scope='Four extra passes over the same original512-program sample. Source prediction/reproduction is not held functional correctness. Original full-source, broader knowledge, math, generic images, retention, RLHF and frontier capability requirements remain.')
    atomic_json(reports/'results.json',result)
    status('bounded repeated-source experiment complete; fixed candidate requires unchanged capability assessment',
        generation_summaries=result['generation_summaries'],checkpoint_sha256=checkpoint_sha)
    print('repeated_source_complete',json.dumps({k:v for k,v in result.items() if k not in ['source_files_sha256','observations']}),flush=True)


if __name__=='__main__':
    main()
