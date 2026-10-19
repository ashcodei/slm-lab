"""Actual full dense CE/KL on admitted book/code/human/program train inputs.

CPU book windows are256tokens here; production4096token/source-extreme GPU
capacity remains separate. Every selected response is complete. Two logical
updates are disposable, never a full original corpus pass or held gain.
"""
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-foundation-full-CPU-v1'
RESERVE=26*1024**3


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Actual mixed CPU proof must explicitly hide all GPUs')
    if REPORTS.exists():raise ValueError('Preserve each actual dense foundation attempt; no automatic restart')
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.dense_foundation_data_v1 import DenseFoundationSources
    from slm_lab.dense_foundation_inference_v1 import fresh,own_reference
    from slm_lab.dense_foundation_stage_v1 import accumulate_gradients,signature
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_dense_checkpoint_v1 import make_state,restore_state
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    tests={'reports/admitted-education-v1-tests.log':'3 passed','reports/dense-foundation-stage-v1-tests.log':'3 passed',
           'reports/dense-foundation-inference-v1-tests.log':'1 passed'}
    for name,passed in tests.items():
        if passed not in (ROOT/name).read_text():raise ValueError('Actual admission/mixed-gradient/recovery/native-inference checks required')
    old_proof=ROOT/'reports/own-dense-growth-full-CPU-v1';proof=json.loads((old_proof/'complete.json').read_text())
    initializer=old_proof/'initial-model.pt';raw=torch.load(initializer,map_location='cpu',weights_only=True,mmap=True)
    old_sft=ROOT/'reports/whole-human-SFT-v1/plan.json';sft=json.loads(old_sft.read_text())
    settings=dict(initializer=str(initializer),initializer_sha256=proof['initial_model_sha256'],initializer_sources=raw['sources'],actual_counts=proof['actual_counts'],
        tokenizer_sha256=file_hash(ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'),
        own_reference_model_plan=sft,own_reference_state_sha256=proof['source_original_digest'])
    files=dict(json.loads((old_proof/'plan.json').read_text())['source_files_sha256'])
    new_names=['scripts/probe_dense_foundation_full_CPU_v1.py','src/slm_lab/admitted_education_v1.py','src/slm_lab/dense_foundation_data_v1.py',
        'src/slm_lab/dense_foundation_stage_v1.py','src/slm_lab/dense_foundation_inference_v1.py',
        'tests/test_admitted_education_v1.py','tests/test_dense_foundation_stage_v1.py','tests/test_dense_foundation_inference_v1.py',*tests,
        'data/open-education-bookpack-v1/corpus.json','data/open-education-bookpack-v1/admission.json',
        'data/open-education-bookpack-v1/admission-readback.json','data/open-education-bookpack-v1/publisher-attribution-review.json',
        'data/open-education-bookpack-v1/original-pages.jsonl','data/open-education-bookpack-v1/independent-raw-source-and-token-readback.json',
        'data/open-education-bookpack-v1/cross-corpus-hash-index-review.json',
        'reports/dense-foundation-v1-proposed-exposure.json']
    for name in new_names:files[str(ROOT/name)]=file_hash(ROOT/name)
    for path in (old_proof/'complete.json',old_proof/'independent-complete-readback.json',initializer):files[str(path)]=file_hash(path)
    guard=FileBindings(files);guard.check(full=True)
    started=time.monotonic();model,metadata=fresh(settings,'cpu');reference=own_reference(settings,'cpu');reference.eval()
    data=DenseFoundationSources(ROOT,model.config,256);positions=[0]
    for name in ('variety','human','program'):
        offset=data.ends[list(data.populations).index(name)-1];stream=data.streams[name]
        if name=='human':p=min(range(len(stream)),key=lambda p:stream.rows[p]['native']['response']['length'])
        else:p=min(range(len(stream)),key=lambda p:int(stream.pack.index[stream.pack.positions[p]]['length']))
        positions.append(offset+p)
    info=[data.information(p) for p in positions]
    REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        initial_dense_settings=settings,actual_counts=model.counts(),actual_selected_global_train_positions=positions,
        complete_selected_original_sources=info,source_populations=data.populations,CPU_FP32_threads=2,
        CPU_education_window_context=256,production_education_context4096_capacity_not_tested=True,
        coefficients=dict(education=1.,variety=1.,human=1.,program=4.),reference_KL_alpha=.1,
        own_reference_routes=dict(education='general',variety='code',human='general',program='code'),
        reference_is_own_prior_model_not_in_exported_inference_inventory=True,
        all_updates_disposable=True,planned_logical_updates=2,goal_achieved=False))
    coefficients=dict(education=1.,variety=1.,human=1.,program=4.);options=dict(reference_KL_alpha=.1,coefficients=coefficients)
    params=[p for _,p in model.language_named_parameters()];selected={id(p) for p in params}
    fixed=[p for p in model.parameters() if id(p) not in selected];fixed_signature=signature(model)
    unused_signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
    def unused_digest():return tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected)
    unused=unused_digest();ref_signature=signature(reference);ref_digest=state_digest(reference)
    optimizer=torch.optim.AdamW(params,lr=3e-6,betas=(.9,.95),weight_decay=.01,foreach=False)
    random.seed(948733);np.random.seed(948733);torch.manual_seed(948733)
    def update():
        model.train();model.network.gradient_checkpointing=True;reference.eval();optimizer.zero_grad(set_to_none=True)
        draws=[random.random(),float(np.random.random()),float(torch.rand(()))]
        record=accumulate_gradients(model,reference,data,positions,options,'cpu')
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params)
                or any(p.grad is not None for p in fixed) or any(p.grad is not None for p in reference.parameters())):
            raise ValueError('Every actual dense-language and no candidate fixed or own-reference tensor must receive finite mixed CE/KL gradients')
        record['gradient_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        record['diagnostic_CPU_python_numpy_torch_draws']=draws
        optimizer.step();optimizer.zero_grad(set_to_none=True);return record
    sources=dict(full_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),initializer_sha256=settings['initializer_sha256'],
        education_admission_sha256=file_hash(ROOT/'data/open-education-bookpack-v1/admission.json'))
    first=update();print('actual_dense_full_mixed_update',1,first,flush=True)
    cursor=dict(original_global_positions=positions,education_window_context=256,completed_updates=1)
    state=make_state(model,optimizer,sources=sources,completed=1,planned=2,cursor=cursor,metrics=[first],device='cpu')
    atomic_checkpoint(REPORTS/'resume-step1.pt',state,16*1024**3,RESERVE);del state
    expected=update();print('actual_dense_full_mixed_update',2,expected,flush=True)
    expected_weights=tensor_digest(model.network.named_parameters());expected_adam=copy.deepcopy(optimizer.state_dict());rng=capture_rng('cpu')
    saved=torch.load(REPORTS/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
    saved_adam=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))
    restore_state(model,optimizer,saved,expected_sources=sources,planned=2,expected_cursor=cursor,device='cpu')
    actual=update()
    if (actual!=expected or expected_weights!=tensor_digest(model.network.named_parameters()) or not equal(expected_adam,optimizer.state_dict())
            or not equal(rng,capture_rng('cpu')) or saved_adam!=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))):
        raise ValueError('Full actual mixed-source next weights/Adam/allCPU-RNG/cursor/metrics and immutable input payload must recover exactly')
    state=make_state(model,optimizer,sources=sources,completed=2,planned=2,cursor={**cursor,'completed_updates':2},metrics=[first,expected],device='cpu')
    atomic_checkpoint(REPORTS/'resume-step2.pt',state,16*1024**3,RESERVE);del state,saved,expected_adam
    if (unused!=unused_digest() or unused_signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
            or signature(reference)!=ref_signature or state_digest(reference)!=ref_digest):
        raise ValueError('Every candidate unused and own-reference weight/metadata must remain exact')
    model.eval();batch=data.batch(positions[0],'cpu');inputs={k:v for k,v in batch.items() if k!='labels'}
    with torch.no_grad():learned=model(**inputs)['logits'].clone()
    cold,_=fresh(settings,'cpu');final=torch.load(REPORTS/'resume-step2.pt',map_location='cpu',weights_only=True,mmap=True)
    cold.import_export(final['model'],expected_sources=sources)
    with torch.no_grad():
        if not torch.equal(learned,cold(**inputs)['logits']) or expected_weights!=tensor_digest(cold.network.named_parameters()):
            raise ValueError('Actual complete dense mixed-source trained cold reload must be exact')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(actual_counts=model.counts(),actual_dense_language_values=sum(p.numel() for p in params),
        actual_dense_language_Adam_states=len(optimizer.state),all_four_actual_source_families_in_each_disposable_update=True,
        complete_selected_original_sources=info,weighted_complete_source_metrics=[first,expected],
        actual_next_weights_Adam_all_CPU_RNG_cursor_metrics_exact=True,recovery_payload_Adam_not_mutated=True,
        all_fixed_candidate_and_own_reference_weights_metadata_exact=True,own_reference_digest=ref_digest,
        actual_trained_full_size_cold_reload_exact=True,retained_step1_sha256=file_hash(REPORTS/'resume-step1.pt'),
        retained_step2_sha256=file_hash(REPORTS/'resume-step2.pt'),logical_disposable_updates=2,physical_optimizer_calls=3,
        seconds=time.monotonic()-started,CPU_book_context=256,production_4096token_GPU_capacity_not_yet_verified=True,
        full_original_source_epoch_not_claimed=True,new_held_quality_gain_not_claimed=True,
        all_probe_updates_discarded_before_production=True,candidate_promoted=False,goal_achieved=False))


if __name__=='__main__':main()
