"""Actual own code folding, 35-layer dense growth and full-language Adam proof.

The two complete short training sources are deliberately capacity-only CPU
controls. This does not consume a full corpus, select a policy or establish
held performance. Every existing GPU job remains untouched.
"""
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/own-dense-growth-full-CPU-v1'
RESERVE=26*1024**3


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Full dense CPU proof must explicitly hide every GPU')
    if REPORTS.exists():raise ValueError('Preserve the actual dense-growth attempt; no automatic restart')
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.model import ModelConfig
    from slm_lab.own_code_training_stage_v1 import PreparedCodeStream
    from slm_lab.own_dense_growth_v1 import DenseBackbone,OwnDenseAssistant,fold_code,grow,new_backbone
    from slm_lab.own_dense_checkpoint_v1 import make_state,restore_state
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from slm_lab.whole_human_sft_inference_v1 import fresh
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if '4 passed' not in (ROOT/'reports/own-dense-growth-v1-tests.log').read_text():
        raise ValueError('Actual own fold/growth/gradient/cache/heads and immutable-payload recovery checks required')
    old_path=ROOT/'reports/whole-human-SFT-v1/plan.json';old=json.loads(old_path.read_text());files=dict(old['source_files_sha256'])
    for name in ['scripts/probe_own_dense_growth_full_CPU_v1.py','src/slm_lab/own_dense_growth_v1.py',
                 'src/slm_lab/own_dense_checkpoint_v1.py','tests/test_own_dense_growth_v1.py','reports/own-dense-growth-v1-tests.log',
                 'data/shared-byte-words-v1/manifest.json','data/shared-byte-words-v1/admission.json',
                 'data/shared-byte-words-v1/independent-source-readback.json','src/slm_lab/shared_word_data_v1.py']:
        files[str(ROOT/name)]=file_hash(ROOT/name)
    files[str(old_path)]=file_hash(old_path)
    guard=FileBindings(files);guard.check(full=True)
    human=WholeHumanData(ROOT/'data/whole-human-alignment-v2','dialogue','train','optimization')
    program=PreparedCodeStream('generation',ROOT/'data/APPS-verified-generation-v1',ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json',ModelConfig.load(ROOT/'configs/model.json'))
    hpos=min(range(len(human)),key=lambda p:human.rows[p]['native']['response']['length'])
    ppos=min(range(len(program)),key=lambda p:int(program.pack.index[program.pack.positions[p]]['length']))
    hbatch=human.batch([hpos],'response','cpu');pbatch=program.encoder.batch([dict(packed_code_index=ppos)],'response','cpu')
    batches=dict(human=hbatch,verified_program=pbatch)
    source_counts={name:dict(complete_context_tokens=int(b['attention_mask'].sum()),response_EOS_targets=int((b['labels'][:,1:]!=-100).sum())) for name,b in batches.items()}
    REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        own_code_checkpoint_sha256=old['own_code_checkpoint_sha256'],own_model_plan_sha256=file_hash(old_path),
        original_human_position=hpos,original_human_ID=human.rows[hpos]['id'],original_verified_program_position=ppos,
        complete_train_source_counts=source_counts,source_choice='shortest complete original train examples, CPU mechanics only; no held selection',
        original_layers=24,target_layers=35,initial_FP32_merge_atol=.001,initial_FP32_merge_rtol=.0001,
        heads_all_counted=True,language_updates_include_all_dense_matrices_table_and_norms=True,
        CPU_FP32_threads=2,planned_disposable_logical_updates=2,gradient_checkpointing=True,
        full_production_training=False,held_quality_gain_not_claimed=True,goal_achieved=False))
    started=time.monotonic();prior,metadata=fresh(old,'cpu');original=prior.original
    signatures=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in original.parameters()]
    original_digest=state_digest(original)
    folded=fold_code(original.language);network,layout=grow(folded,35);model=OwnDenseAssistant(network).eval()
    # Transfer only the already-owned head values; retaining a source module
    # would also retain its old general backbone and violate the model budget.
    model.heads['word'].load_state_dict(prior.parts['word'].state_dict(),strict=True)
    model.heads['reward'].load_state_dict(prior.parts['reward'].state_dict(),strict=True)
    with torch.no_grad():
        model.heads['pointing']['query'].weight.copy_(original.grounding.query_projection.weight)
        model.heads['pointing']['patch'].weight.copy_(original.grounding.patch_projection.weight)
    if model.counts()['total_unique']!=984399690 or sum(p.numel() for _,p in model.language_named_parameters())!=931244544:
        raise ValueError('Actual full35-layer dense/head inventory and fully plastic language budget must match')
    if {p.data_ptr() for p in original.parameters()}&{p.data_ptr() for p in model.parameters()}:
        raise ValueError('Folded candidate must own its weights and retain no old general/code tensor alias')
    pointer=PointerData(ROOT/'data/localized-pointer-candidate-v1','train','audit')
    photo_pos=json.loads((ROOT/'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())['original_source_position']
    photo=pointer.batch([photo_pos],'cpu');ids,pixels=photo['input_ids'],photo['pixel_values'];parity={}
    with torch.no_grad():
        source_cache=original.language.code.prefill(ids,pixels)['cache']
        source_cached_next=original.language.code.decode(torch.tensor([[4]]),source_cache)['logits'].clone()
        for name,batch in [*batches.items(),('original_native_photo',dict(input_ids=ids,pixel_values=pixels))]:
            inp={k:v for k,v in batch.items() if k!='labels'}
            own=original(**inp,route='code')['logits'];dense=folded(**inp)['logits'];grown=model(**inp)['logits']
            error=(own-dense).float()
            if not torch.allclose(own,dense,atol=.001,rtol=.0001) or not torch.equal(dense,grown):
                raise ValueError('Actual sampled full FP32 own-code fold tolerance and exact added-neutral-layer parity required')
            parity[name]=dict(merge_max_absolute_error=float(error.abs().max()),merge_mean_absolute_error=float(error.abs().mean()),
                exact_sampled_folded24_to_grown35_logits=True,logits_shape=list(own.shape))
        initial_photo=model(ids,pixels)['logits'].clone()
    print('actual_full_dense_fold_and_growth',model.counts(),'parity',parity,flush=True)
    del folded
    words=SharedWordData(ROOT/'data/shared-byte-words-v1','train','optimization');word_batch=words.batch([0],'cpu')
    with torch.no_grad():
        initial_word=model.word_logits(word_batch['word_pixels']).clone()
        reward=model.preference_score(hbatch['input_ids'],hbatch['attention_mask'])
        pointing=model.point_logits(ids,pixels);detection=model.detection_logits(pixels)
        hidden=model(ids,return_hidden=True,return_logits=False)['hidden'][:,-1];router=model.heads['router'](hidden)
        if (initial_word.shape!=(1,140,257) or detection.shape!=(1,5,28,28) or pointing.shape!=(1,196)
                or reward.shape!=(1,) or router.shape!=(1,3)
                or any(not bool(torch.isfinite(t).all()) for t in (initial_word,reward,pointing,detection,router))):
            raise ValueError('All actual counted word/reward/point/detector/router paths must execute finitely')
    sources=dict(full_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),own_code_sha256=old['own_code_checkpoint_sha256'])
    atomic_checkpoint(REPORTS/'initial-model.pt',model.export(sources),5*1024**3,RESERVE)
    parameters=[p for _,p in model.language_named_parameters()];selected_ids={id(p) for p in parameters}
    fixed=[p for p in model.parameters() if id(p) not in selected_ids]
    fixed_signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
    def fixed_digest():return tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in selected_ids)
    fixed_before=fixed_digest();optimizer=torch.optim.AdamW(parameters,lr=1e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
    def update():
        model.train();model.network.gradient_checkpointing=True;optimizer.zero_grad(set_to_none=True);metrics={}
        for name,batch in batches.items():
            loss=model(**batch)['loss']
            if not bool(torch.isfinite(loss)):raise ValueError('Every actual complete source dense-language loss must be finite')
            loss.backward(inputs=parameters);metrics[name]=dict(**source_counts[name],token_mean_CE=float(loss.detach()))
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters)
                or any(p.grad is not None for p in fixed)):
            raise ValueError('Every full dense language matrix/table/norm and no vision/critic/auxiliary tensor must receive gradients')
        metrics['gradient_norm_before_clip']=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
        optimizer.step();optimizer.zero_grad(set_to_none=True);return metrics
    first=update();print('actual_full_dense_language_update',1,first,flush=True)
    cursor=dict(human_position=hpos,verified_program_position=ppos,completed_updates=1)
    state=make_state(model,optimizer,sources=sources,completed=1,planned=2,cursor=cursor,metrics=[first],device='cpu')
    atomic_checkpoint(REPORTS/'resume-step1.pt',state,16*1024**3,RESERVE)
    # Keep the on-disk input immutable while verifying recovery. The new CPU
    # restore also privately copies moments instead of aliasing its payload.
    del state
    expected=update();print('actual_full_dense_language_update',2,expected,flush=True)
    expected_weights=tensor_digest(model.network.named_parameters());expected_Adam=copy.deepcopy(optimizer.state_dict());expected_rng=capture_rng('cpu')
    saved=torch.load(REPORTS/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
    saved_Adam=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))
    restore_state(model,optimizer,saved,expected_sources=sources,planned=2,expected_cursor=cursor,device='cpu')
    actual=update()
    if (actual!=expected or expected_weights!=tensor_digest(model.network.named_parameters()) or not equal(expected_Adam,optimizer.state_dict())
            or not equal(expected_rng,capture_rng('cpu'))
            or saved_Adam!=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))):
        raise ValueError('Actual full931M-language next weights/Adam/RNG/metrics and immutable recovery payload must be exact')
    final=make_state(model,optimizer,sources=sources,completed=2,planned=2,cursor={**cursor,'completed_updates':2},metrics=[first,expected],device='cpu')
    atomic_checkpoint(REPORTS/'resume-step2.pt',final,16*1024**3,RESERVE);del final,expected_Adam,saved
    if (fixed_before!=fixed_digest() or fixed_signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
            or original_digest!=state_digest(original) or signatures!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in original.parameters()]):
        raise ValueError('Every source original tensor and fixed candidate vision/critic/auxiliary tensor/metadata must remain exact')
    model.eval()
    with torch.no_grad():
        trained=model(ids,pixels)['logits'].clone();after_word=model.word_logits(word_batch['word_pixels']).clone()
        if torch.equal(trained,initial_photo) or not torch.equal(initial_word,after_word):raise ValueError('Actual dense language must change while standalone shared word/vision output stays exact')
        if not torch.equal(source_cached_next,original.language.code.decode(torch.tensor([[4]]),source_cache)['logits']):
            raise ValueError('Pre-existing original own code cache must remain exact')
    cold=OwnDenseAssistant(new_backbone(model.config,948531)).eval()
    final=torch.load(REPORTS/'resume-step2.pt',map_location='cpu',weights_only=True,mmap=True)
    cold.import_export(final['model'],expected_sources=sources)
    with torch.no_grad():
        if (not torch.equal(trained,cold(ids,pixels)['logits']) or not torch.equal(after_word,cold.word_logits(word_batch['word_pixels']))
                or tensor_digest(cold.network.named_parameters())!=expected_weights):
            raise ValueError('Actual full-size trained dense language and unchanged word/vision cold reload must be exact')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(actual_counts=model.counts(),actual_dense_language_values=sum(p.numel() for p in parameters),
        actual_dense_language_Adam_states=len(optimizer.state),depth_layout=layout,actual_added_identity_blocks=layout.count(-1),
        actual_initial_fold_and_growth_sampled_FP32_parity=parity,merge_arithmetic_not_bitwise_equivalent_to_separate_factors=True,
        source_model_not_retained_in_candidate_inventory=True,source_original_digest=original_digest,
        all_source_weights_metadata_and_original_code_cache_exact=True,all_fixed_candidate_vision_critic_heads_metadata_and_word_output_exact=True,
        all_full_dense_language_gradients_finite_and_no_fixed_gradients=True,
        actual_next_update_all_weights_Adam_RNG_source_cursor_and_metrics_exact=True,recovery_input_Adam_payload_not_mutated=True,
        actual_trained_full_size_cold_reload_exact=True,all_counted_heads_execute_finitely=True,
        copied_grounding_weights_do_not_preserve_old_general_grounding_function=True,
        proposed_future_actor_budget=model.future_actor_parameter_budget(),
        logical_disposable_updates=2,physical_optimizer_calls=3,complete_source_metrics=[first,expected],
        initial_model_sha256=file_hash(REPORTS/'initial-model.pt'),retained_step1_sha256=file_hash(REPORTS/'resume-step1.pt'),
        retained_step2_sha256=file_hash(REPORTS/'resume-step2.pt'),seconds=time.monotonic()-started,
        full_source_training=False,GPU_capacity_not_proven=True,new_held_coding_or_all_round_quality_gain_measured=False,
        all_probe_updates_discarded_before_any_production_training=True,candidate_promoted=False,goal_achieved=False))


if __name__=='__main__':main()
