"""Actual code-initialized under1B graph, frozen-code gradients and recovery.

Two logical mixed human/program updates are disposable mechanics checks, not a
full source pass or capability improvement. Existing GPU jobs are untouched.
"""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/own-aligned-assistant-v3-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Full CPU prototype must explicitly hide all GPUs')
    if REPORTS.exists(): raise ValueError('Preserve each actual architecture attempt; no automatic restart')
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint, capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_aligned_assistant_v3 import OwnAlignedAssistantV3
    from slm_lab.own_alignment_checkpoint_v1 import make_state, restore_state
    from slm_lab.own_code_training_stage_v1 import PreparedCodeStream
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from slm_lab.whole_human_sft_inference_v1 import fresh
    from slm_lab.whole_human_sft_stage_v1 import _LanguageRestoreView
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    if '3 passed' not in (ROOT/'reports/own-aligned-assistant-v3-tests.log').read_text():
        raise ValueError('Actual code-equivalent initialization, frozen-code gradients, recovery and format checks required')
    source_plan = ROOT/'reports/whole-human-SFT-v1/plan.json'; old = json.loads(source_plan.read_text())
    files = dict(old['source_files_sha256'])
    for name in ['scripts/probe_own_aligned_assistant_v3_full_CPU_v1.py', 'src/slm_lab/own_aligned_assistant_v3.py',
                 'tests/test_own_aligned_assistant_v3.py', 'reports/own-aligned-assistant-v3-tests.log']:
        files[str(ROOT/name)] = file_hash(ROOT/name)
    files[str(source_plan)] = file_hash(source_plan)
    guard = FileBindings(files); guard.check(full=True); REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        initial_model_settings=old['model_initialization'], expected_actual_counts=old['actual_counts'],
        original_human_train_position=0, original_verified_human_program_position=0,
        CPU_FP32_threads=2, planned_disposable_logical_updates=2, full_training_not_claimed=True, goal_achieved=False))
    started = time.monotonic(); prior, metadata = fresh(old, 'cpu'); original = prior.original; del prior
    model = OwnAlignedAssistantV3(original, **old['model_initialization']).eval()
    if model.counts() != old['actual_counts']: raise ValueError('Complete actual unchanged under1B model budget required')
    human = WholeHumanData(ROOT/'data/whole-human-alignment-v2', 'dialogue', 'train', 'optimization')
    tokenizer = ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json'
    generation = PreparedCodeStream('generation', ROOT/'data/APPS-verified-generation-v1', tokenizer, model.config)
    human_batch = human.batch([0], 'response', 'cpu')
    code_batch = generation.encoder.batch([dict(packed_code_index=0)], 'response', 'cpu')
    pointer = PointerData(ROOT/'data/localized-pointer-candidate-v1', 'train', 'audit')
    position = json.loads((ROOT/'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())['original_source_position']
    image_batch = pointer.batch([position], 'cpu'); ids, pixels = image_batch['input_ids'], image_batch['pixel_values']
    selected = model.extension_parameters('language'); selected_ids = {id(p) for p in selected}
    fixed = [p for p in model.parameters() if id(p) not in selected_ids]
    signatures = [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
    original_digest = state_digest(original)
    def fixed_parts(): return tensor_digest((n,p) for n,p in model.parts.named_parameters() if not n.startswith(('language_linears.','language_norms.')))
    unused_digest = fixed_parts()
    with torch.no_grad():
        native = {route:original(ids,pixels,route=route)['logits'].clone() for route in ('general','code','grounding')}
        aligned = model(ids,pixels)['logits']
        if not torch.equal(native['code'], aligned):
            raise ValueError('Actual full-size code-initialized image logits must exactly match the existing trained code route')
        for batch in (human_batch,code_batch):
            if not torch.equal(model(**batch)['logits'],original(**batch,route='code')['logits']):
                raise ValueError('Both actual complete human/program initial logits must exactly match own code')
        cache = original.language.code.prefill(ids,pixels)['cache']
        cached_next = original.language.code.decode(torch.tensor([[5]]),cache)['logits'].clone()
    optimizer = torch.optim.AdamW(selected,lr=1e-5,betas=(.9,.95),weight_decay=.01,foreach=False)
    def update():
        optimizer.zero_grad(set_to_none=True); metrics = {}
        for name,batch in [('human',human_batch),('verified_program',code_batch)]:
            loss = model(**batch)['loss']; loss.backward()
            metrics[name] = dict(token_mean_nll=float(loss.detach()), full_context_tokens=int(batch['input_ids'].numel()),
                response_EOS_targets=int((batch['labels'][:,1:]!=-100).sum()))
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected)
                or any(p.grad is not None for p in fixed)):
            raise ValueError('Every new language tensor and no existing original/code/unused tensor must receive gradients')
        metrics['gradient_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True))
        optimizer.step(); optimizer.zero_grad(set_to_none=True); return metrics
    sources = dict(probe_plan_sha256=file_hash(REPORTS/'plan.json'), original_source_plan_sha256=file_hash(source_plan))
    first = update(); print('actual_full_code_initialized_mixed_update',1,first,flush=True)
    cursor = dict(original_human_position=0,original_verified_program_position=0,completed_updates=1)
    state = make_state(model,optimizer,groups=('language',),sources=sources,completed_steps=1,planned_steps=2,
        sampler_state=cursor,metrics=[first],device='cpu')
    atomic_checkpoint(REPORTS/'resume.pt',state,192*1024**2,26*1024**3)
    expected = update(); expected_parts = model.extension_state(); expected_adam = copy.deepcopy(optimizer.state_dict()); expected_rng = capture_rng('cpu')
    restore_state(_LanguageRestoreView(model),optimizer,torch.load(REPORTS/'resume.pt',map_location='cpu',weights_only=True,mmap=True),
        groups=('language',),expected_sources=sources,planned_steps=2,expected_sampler_state=cursor,device='cpu')
    actual = update()
    def equal(a,b):
        if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
        if isinstance(a,(list,tuple)): return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    if (actual!=expected or not equal(expected_parts,model.extension_state()) or not equal(expected_adam,optimizer.state_dict())
            or not equal(expected_rng,capture_rng('cpu'))): raise ValueError('Every actual next-update tensor/Adam/RNG/metric must recover exactly')
    if (signatures != [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
            or original_digest!=state_digest(original) or unused_digest!=fixed_parts()):
        raise ValueError('Every original/code/grounding and unused new tensor/version/pointer/flag must remain fixed')
    with torch.no_grad():
        trained = model(ids,pixels)['logits'].clone()
        if any(not torch.equal(native[r],original(ids,pixels,route=r)['logits']) for r in native):
            raise ValueError('Every original native image route must remain exact')
        if not torch.equal(cached_next,original.language.code.decode(torch.tensor([[5]]),cache)['logits']):
            raise ValueError('Pre-existing original code cache must remain exact')
    atomic_checkpoint(REPORTS/'trained-disposable-extensions.pt',model.export(sources),64*1024**2,26*1024**3)
    cold = OwnAlignedAssistantV3(original,**old['model_initialization']).eval()
    cold.import_export(torch.load(REPORTS/'trained-disposable-extensions.pt',map_location='cpu',weights_only=True,mmap=True),expected_sources=sources)
    with torch.no_grad():
        if not torch.equal(trained,cold(ids,pixels)['logits']): raise ValueError('Actual trained full-size code-initialized cold reload must be exact')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(actual_counts=model.counts(), actual_language_parameters=sum(p.numel() for p in selected),
        actual_language_Adam_states=len(optimizer.state), initial_full_code_image_human_and_program_logits_exact=True,
        every_existing_code_parameter_detached_from_new_optimization_without_changing_metadata=True,
        actual_next_update_all_new_tensors_Adam_RNG_cursor_and_metrics_exact=True,
        original_routes_bytes_versions_pointers_flags_and_code_cache_exact=True,actual_trained_full_model_cold_reload_exact=True,
        original_native_photo_query_ID=pointer.rows[position]['id'], original_human_ID=human.rows[0]['id'],
        original_generation_population=len(generation),actual_disposable_updates=2,physical_calls=3,source_metrics=[first,expected],
        fixed_original_digest=original_digest, all_unused_new_extensions_digest=unused_digest,
        resume_sha256=file_hash(REPORTS/'resume.pt'),trained_disposable_extensions_sha256=file_hash(REPORTS/'trained-disposable-extensions.pt'),
        seconds=time.monotonic()-started,full_original_source_training=False,new_held_coding_or_broad_gain_measured=False,
        all_probe_updates_discarded_before_any_production_training=True,candidate_promoted=False,goal_achieved=False))


if __name__ == '__main__': main()
