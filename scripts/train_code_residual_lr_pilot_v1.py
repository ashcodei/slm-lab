"""Controlled train/validation-only own857M learning-rate research pilot."""
from collections import Counter
from datetime import datetime,timezone
import gc
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
from tokenizers import Tokenizer
import torch

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_checkpoint_v1 import make_training_state
from slm_lab.code_residual_pilot_v1 import PROFILES,learning_rate,select_complete_rows
from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
from slm_lab.codenet_replay_v1 import FullReplayEncoder
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,packed_encoder,ROLES
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint,TrainingOutputs,supervised_gradients


ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'checkpoints/code-residual-lr-pilot-v1';reports=ROOT/'reports/code-residual-lr-pilot-v1'
    if output.exists() or reports.exists():raise ValueError('Preserve all earlier pilot artifacts; no implicit restart')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes']+256*1024**2
    maximum_checkpoint=2*1024**3
    if shutil.disk_usage(ROOT).free<reserve+2*maximum_checkpoint+64*1024**2:
        raise ValueError('Whole best-state atomic replacement and reports above original writer commitments required')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0]<12*1024**3:
        raise ValueError('Initially12GiB free on the shared BF16 GPU required')
    torch.cuda.set_per_process_memory_fraction(.20,device);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    source=ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
    source_sha='4c05d99b0c3fc775bf89e70c6cbac9c63017eb36f67e17c738b5be268fb191ca'
    if file_hash(source)!=source_sha:raise ValueError('Exact own completed-language source policy required')
    train_pack=CompleteCodePack(ROOT/'data/codenet-durable-pack-v2','train',purpose='training')
    valid_pack=CompleteCodePack(train_pack.folder,'validation',purpose='audit')
    populations={'train':[],'validation':[]}
    with gzip.open(train_pack.folder/'original-rows.jsonl.gz','rt') as stream:
        for position,line in enumerate(stream):
            row=json.loads(line);role=row['split']
            if role not in populations:continue
            item=train_pack.index[position]
            if row['token_pack_index']!=position or int(item['role'])!=ROLES[role]:
                raise ValueError('Exact original sealed role/index required')
            populations[role].append(dict(role=role,local_position=len(populations[role]),global_position=position,
                id=row['id'],group_id=row['group_id'],language=row['language'],context_tokens=int(item['length']),
                target_tokens=row['target_tokens'],full_input_ids_sha256=row['complete_input_ids_sha256']))
    if len(populations['train'])!=len(train_pack) or len(populations['validation'])!=len(valid_pack):
        raise ValueError('Entire original train/validation population required')
    selected=select_complete_rows(populations['train'],number=512,per_language=4,seed=910451,role='train')
    valid=select_complete_rows(populations['validation'],number=128,per_language=1,seed=910452,role='validation')
    if {r['group_id'] for r in selected}&{r['group_id'] for r in valid}:
        raise ValueError('Pilot train and internal validation problem groups must be disjoint')
    for pack,rows in [(train_pack,selected),(valid_pack,valid)]:
        for row in rows:
            ids,labels=pack.sequence(row['local_position'])
            if (int(pack.positions[row['local_position']])!=row['global_position'] or len(ids)!=row['context_tokens']
                    or sum(v!=-100 for v in labels)!=row['target_tokens']
                    or hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()!=row['full_input_ids_sha256']):
                raise ValueError('Whole original programme input/answer/mask must match admitted bytes')
    prior_path=ROOT/'reports/code-residual-readiness-v1/plan.json';prior=json.loads(prior_path.read_text())
    joint=ROOT/'data/extended-curriculum-v1/joint.train.jsonl';control_ids={r['id'] for r in prior['original_training_control_ids']}
    controls=[]
    for line in joint.open():
        row=json.loads(line)
        if row['id'] in control_ids:controls.append(row)
        if len(controls)==2:break
    if len(controls)!=2 or sum(bool(r.get('image')) for r in controls)!=1:
        raise ValueError('Original complete general text and actual448px image controls required')
    names=['scripts/train_code_residual_lr_pilot_v1.py','src/slm_lab/code_residual_pilot_v1.py',
        'src/slm_lab/code_residual_branch_v1.py','src/slm_lab/code_residual_checkpoint_v1.py',
        'src/slm_lab/codenet_checkpoint_v1.py','src/slm_lab/codenet_token_pack_v1.py','src/slm_lab/codenet_replay_v1.py',
        'src/slm_lab/model.py','src/slm_lab/training.py','src/slm_lab/recurrent_ablation.py',
        'scripts/prepare_codenet_native_contract_pack_v2.py','tests/test_code_residual_pilot_v1.py',
        'reports/code-residual-pilot-v1-focused-tests.log','reports/code-residual-optimizer-and-capacity-v1-readiness.json',
        'reports/language-completed-capability-baseline-v1/comparison.json',
        'reports/codenet-language-completed-baseline-v1/results.json',
        'data/codenet-algorithm-assessment-v1/manifest.json']
    bound={str(ROOT/name):file_hash(ROOT/name) for name in names}
    for p in [source,joint,prior_path,train_pack.folder/'manifest.json',train_pack.folder/'admission.json']:
        bound[str(p)]=file_hash(p)
    bound.update({str(train_pack.folder/name):h for name,h in train_pack.manifest['files_sha256'].items()})
    for row in controls:
        if row.get('image'):
            p=Path(row['image']).resolve();bound[str(p)]=file_hash(p)
            if bound[str(p)]!=row['stored_image_sha256']:raise ValueError('Original image control bytes changed')
    guard=FileBindings(bound);output.mkdir();reports.mkdir()
    shutil.copyfile(train_pack.folder/'tokenizer.json',output/'tokenizer.json')
    selection=dict(train=selected,validation=valid)
    atomic_json(reports/'selected-original-rows.json',selection)
    plan=dict(format='own-code-residual-learning-rate-pilot-v1',recorded_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound,source_checkpoint=str(source),source_checkpoint_sha256=source_sha,
        tokenizer_sha256=file_hash(output/'tokenizer.json'),selection_sha256=file_hash(reports/'selected-original-rows.json'),
        profiles=list(PROFILES),training_rows=512,training_languages=len({r['language'] for r in selected}),
        training_problem_groups=len({r['group_id'] for r in selected}),training_response_targets_per_profile=sum(r['target_tokens'] for r in selected),
        validation_rows=128,validation_languages=len({r['language'] for r in valid}),validation_problem_groups=len({r['group_id'] for r in valid}),
        validation_response_targets=sum(r['target_tokens'] for r in valid),train_validation_group_overlap=0,
        optimizer_updates_per_profile=128,batch_size=1,gradient_accumulation=4,weight_decay=.01,gradient_clip=1.,
        initial_adapter_seed=71017,profile_order_fixed=True,identical_initial_residual_weights_and_training_order=True,
        selection_criterion='Smallest complete internal-validation target-token mean NLL among the three trained profiles; fixed profile order breaks exact ties.',
        functional_tests_used_for_selection=False,benchmark_or_hidden_answers_used_for_training=False,
        frozen_general_route_preserved=True,explicit_code_route=True,automatic_routing_quality_unverified=True,
        GPU='cuda:0',maximum_GPU_process_memory_fraction=.20,maximum_checkpoint_bytes=maximum_checkpoint,
        maximum_persistent_best_checkpoint_copies_during_atomic_replace=2,remaining_original_writer_reserve_bytes=reserve,
        complete_RNG_and_Adam_saved_for_best_completed_profile=True,partial_profile_auto_resume=False,
        full_source_stage_or_final_base_integration=False,all_existing_multimodal_RLHF_guards_preserved=True,
        candidate_promoted=False,goal_achieved=False)
    atomic_json(reports/'plan.json',plan);plan_sha=file_hash(reports/'plan.json')
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            candidate_promoted=False,goal_achieved=False,**fields))
    status('loading original own base and identical initial code residuals')
    random.seed(71015);np.random.seed(71016);torch.manual_seed(71017);torch.cuda.manual_seed_all(71018)
    base,metadata=load_checkpoint(source,'cpu',training_state=False)
    if metadata['tokenizer_sha256']!=plan['tokenizer_sha256']:raise ValueError('Original exact source tokenizer required')
    base_digest=state_digest(base.state_dict());model=RoutedCodeResidual(base,width=2048).to(device);model.code.gradient_checkpointing=True;model.eval()
    if model.parameter_counts()!=dict(total_unique=857007616,frozen_base=706012672,trainable_code=150994944):
        raise ValueError('Exact own shared-backbone total unique parameter budget required')
    initial=model.adapter_state();initial_digest=state_digest(initial)
    token=Tokenizer.from_file(str(output/'tokenizer.json'))
    train_encoder=packed_encoder(train_pack,token,model.config);valid_encoder=packed_encoder(valid_pack,token,model.config)
    replay_encoder=FullReplayEncoder(token,model.config,joint.parent,448)
    control_batches=[replay_encoder.batch([row],'response',device) for row in controls]
    def amp():return torch.autocast('cuda',dtype=torch.bfloat16)
    def general_logits(batch):
        with torch.no_grad(),amp():return model(batch['input_ids'],pixel_values=batch['pixel_values'],attention_mask=batch['attention_mask'])['logits'].cpu()
    original_logits=[general_logits(batch) for batch in control_batches]
    for batch,expected in zip(control_batches,original_logits):
        with torch.no_grad(),amp():actual=model(batch['input_ids'],pixel_values=batch['pixel_values'],attention_mask=batch['attention_mask'],route='code')['logits'].cpu()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    def assess(rows,encoder,label):
        model.eval();weighted=0.;count=0;records=[]
        with torch.no_grad(),amp():
            for i,row in enumerate(rows):
                batch=encoder.batch([{'packed_code_index':row['local_position']}],'response',device)
                loss=model.code(**batch)['loss'];targets=int((batch['labels'][:,1:]!=-100).sum())
                if targets!=row['target_tokens'] or not torch.isfinite(loss):raise ValueError('Whole finite original assessment targets required')
                weighted+=float(loss)*targets;count+=targets
                records.append(dict(id=row['id'],role=row['role'],language=row['language'],targets=targets,mean_nll=float(loss)))
                if (i+1)%32==0:status('scoring complete original pilot source rows',assessment=label,scored=i+1,rows=len(rows))
        return dict(rows=len(rows),response_targets=count,token_mean_nll=weighted/count,per_row=records)
    baseline=dict(train=assess(selected,train_encoder,'initial_train'),validation=assess(valid,valid_encoder,'initial_validation'))
    atomic_json(reports/'initial-source-NLL.json',baseline)
    profile_results=[];best=None;best_path=output/'best.pt';parameters=model.adapter_parameters()
    for profile in PROFILES:
        guard.check();model.load_adapter_state(initial)
        if state_digest(model.adapter_state())!=initial_digest:raise ValueError('Every profile must start from identical original residual bytes')
        optimizer=torch.optim.AdamW(parameters,lr=profile['peak_learning_rate'],weight_decay=.01,foreach=False)
        wrapped=TrainingOutputs(model.code,'sft');metrics=[];started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
        for step in range(128):
            guard.check();model.train();optimizer.zero_grad(set_to_none=True)
            lr=learning_rate(profile,step)
            for group in optimizer.param_groups:group['lr']=lr
            group_rows=selected[step*4:(step+1)*4]
            groups=[[{'packed_code_index':r['local_position']}] for r in group_rows]
            loss,targets=supervised_gradients(wrapped,train_encoder,groups,'response',device,amp)
            if targets!=sum(r['target_tokens'] for r in group_rows) or train_encoder.truncated:
                raise ValueError('Exact complete original train-only update targets required')
            if any(p.grad is not None for p in model.base.parameters()):raise ValueError('Original base must remain frozen')
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters):raise ValueError('Every residual gradient must be finite')
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            metric=dict(step=step+1,learning_rate=lr,token_mean_nll=float(loss),response_targets=targets,gradient_norm_before_clipping=float(norm))
            metrics.append(metric)
            if step<2 or (step+1)%8==0:
                status('training fixed whole-program learning-rate profile',profile=profile['name'],completed_steps=step+1,planned_steps=128,latest=metric)
                print('pilot_update',profile['name'],metric,flush=True)
        optimizer.zero_grad(set_to_none=True);model.eval()
        trained=assess(selected,train_encoder,profile['name']+'_train');validated=assess(valid,valid_encoder,profile['name']+'_validation')
        if state_digest(model.base.state_dict())!=base_digest:raise ValueError('Any original general/vision/critic tensor changed')
        for batch,expected in zip(control_batches,original_logits):torch.testing.assert_close(general_logits(batch),expected,rtol=0,atol=0)
        current=dict(profile=profile,updates=metrics,train=trained,validation=validated,
            original_base_state_unchanged=True,sampled_general_text_and448px_image_logits_bitwise_unchanged=True,
            shared_GPU_seconds=time.monotonic()-started,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device),selected_for_held_evaluation=False)
        if best is None or validated['token_mean_nll']<best['validation']['token_mean_nll']:
            state=make_training_state(model,optimizer,base_checkpoint_sha256=source_sha,tokenizer_sha256=plan['tokenizer_sha256'],
                run_plan_sha256=plan_sha,completed_steps=128,planned_steps=128,device=device,metrics=metrics)
            state.update(pilot_profile=profile,pilot_selection_sha256=plan['selection_sha256'],
                source_manifest_sha256=file_hash(train_pack.folder/'manifest.json'),
                internal_validation_token_mean_nll=validated['token_mean_nll'],stage='code_residual_lr_pilot_v1')
            atomic_checkpoint(best_path,state,maximum_checkpoint,reserve);del state;best=current
        profile_results.append(current);atomic_json(reports/(profile['name']+'.json'),current)
        atomic_json(reports/'profile-results-so-far.json',dict(profiles=profile_results,best_profile=best['profile']['name'],functional_feedback_used=False,goal_achieved=False))
        del wrapped,optimizer;gc.collect();torch.cuda.empty_cache()
    if train_encoder.truncated or valid_encoder.truncated:raise ValueError('No pilot source context may be clipped')
    guard.check(full=True);best['selected_for_held_evaluation']=True
    saved=torch.load(best_path,map_location='cpu',weights_only=True,mmap=True)
    if saved['pilot_profile']!=best['profile'] or saved['run_plan_sha256']!=plan_sha:
        raise ValueError('Exact final internal-validation-selected complete policy state required')
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,run_plan_sha256=plan_sha,
        model_counts=model.parameter_counts(),initial_residual_state_sha256=initial_digest,original_base_state_sha256=base_digest,
        initial_train_token_mean_nll=baseline['train']['token_mean_nll'],initial_validation_token_mean_nll=baseline['validation']['token_mean_nll'],
        profiles=[dict(profile=r['profile'],train_token_mean_nll=r['train']['token_mean_nll'],validation_token_mean_nll=r['validation']['token_mean_nll'],
            shared_GPU_seconds=r['shared_GPU_seconds'],peak_GPU_allocated_bytes=r['peak_GPU_allocated_bytes'],peak_GPU_reserved_bytes=r['peak_GPU_reserved_bytes']) for r in profile_results],
        best_profile=best['profile'],best_train_token_mean_nll=best['train']['token_mean_nll'],best_validation_token_mean_nll=best['validation']['token_mean_nll'],
        best_complete_checkpoint=str(best_path),best_complete_checkpoint_sha256=file_hash(best_path),best_complete_checkpoint_bytes=best_path.stat().st_size,
        training_rows_per_profile=512,training_groups=plan['training_problem_groups'],training_language_labels=plan['training_languages'],
        validation_rows=128,validation_language_labels=plan['validation_languages'],
        actual_Adam_updates_total=384,actual_complete_training_programme_exposures=1536,
        exact_complete_response_target_exposures=3*sum(r['target_tokens'] for r in selected),
        train_validation_problem_group_overlap=0,all_original_base_tensors_unchanged=True,
        sampled_general_text_and448px_image_logits_bitwise_unchanged=True,all_source_contexts_complete=True,
        held_functional_test_feedback_used=False,functional_or_broad_capability_gain_verified=False,
        full_source_stage_or_final_base_integration=False,candidate_promoted=False,goal_achieved=False,
        scope='Controlled 512 complete train-programme/128 internal-validation programme sample, identical initialization/order, three128-update learning-rate profiles. Internal-validation teacher NLL selects one trained candidate for separate fixed held generation/execution. This is not all-source training, test-driven selection, automatic routing, broad retention or high-end parity.')
    atomic_json(reports/'results.json',result);status('all fixed pilot profiles complete; candidate frozen for separate held assessment',best_profile=best['profile'],best_checkpoint_sha256=result['best_complete_checkpoint_sha256'])
    print('code_residual_learning_rate_pilot_complete',json.dumps(result),flush=True)


if __name__=='__main__':main()
