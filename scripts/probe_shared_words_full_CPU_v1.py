"""Disposable full-size shared word learning, complete real sources and restart.

This does not run a production word epoch or change any existing GPU job. It
keeps both step1and step2Adam snapshots, actual pixel/transcript identities and
the trained disposable extension export for independent structural readback.
"""
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT/'reports/shared-words-full-CPU-v1'


def equal(a, b):
    import torch
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a,b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a, (list,tuple)): return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('Full CPU proof must explicitly hide every GPU')
    if REPORTS.exists(): raise ValueError('Preserve the original attempt; no automatic retry or overwrite')
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint, capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_aligned_assistant_v3 import OwnAlignedAssistantV3
    from slm_lab.own_alignment_checkpoint_v1 import make_state, restore_state
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.shared_word_stage_v1 import GROUPS, PREFIXES, _VisionWordRestoreView, accumulate_gradients
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from slm_lab.whole_human_sft_inference_v1 import fresh
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    test_log = ROOT/'reports/shared-word-stage-v1-tests.log'
    if '3 passed' not in test_log.read_text(): raise ValueError('Actual gradient/oracle, recovery and rejection checks must pass')
    prior_plan_path = ROOT/'reports/whole-human-SFT-v1/plan.json'
    prior_plan = json.loads(prior_plan_path.read_text()); files = dict(prior_plan['source_files_sha256'])
    for name in ['scripts/probe_shared_words_full_CPU_v1.py','src/slm_lab/shared_word_stage_v1.py',
                 'src/slm_lab/shared_word_data_v1.py','src/slm_lab/own_aligned_assistant_v3.py',
                 'tests/test_shared_word_stage_v1.py','reports/shared-word-stage-v1-tests.log',
                 'data/shared-byte-words-v1/manifest.json','data/shared-byte-words-v1/admission.json',
                 'data/shared-byte-words-v1/independent-source-readback.json',
                 'data/shared-byte-words-v1/train.jsonl','data/shared-byte-words-v1/train.tokens.u16']:
        files[str(ROOT/name)] = file_hash(ROOT/name)
    files[str(prior_plan_path)] = file_hash(prior_plan_path)
    guard = FileBindings(files); guard.check(full=True)
    data = SharedWordData(ROOT/'data/shared-byte-words-v1','train','optimization')
    longest = max(range(len(data)),key=lambda p:len(data.rows[p]['response'].encode('utf-8')))
    real = next(p for p,r in enumerate(data.rows) if r['byte_family']=='real_word')
    positions = list(dict.fromkeys([0,longest,real]))
    REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=files, full_source_population=len(data),original_word_positions=positions,
        full_transcripts=[data.rows[p]['response'] for p in positions],
        source_identifiers=[data.rows[p]['id'] for p in positions],
        original_word_families=[data.rows[p]['byte_family'] for p in positions],
        initial_model_settings=prior_plan['model_initialization'], expected_actual_counts=prior_plan['actual_counts'],
        fixed_objective=dict(CTC_word_mean_of_full_byte_normalized_losses=1.,native_response_EOS_token_mean_CE=1.),
        CPU_FP32_threads=2, gradient_checkpointing=True, planned_disposable_logical_updates=2,
        full_production_training_not_claimed=True,goal_achieved=False))
    started = time.monotonic(); prior,metadata = fresh(prior_plan,'cpu'); original = prior.original; del prior
    model = OwnAlignedAssistantV3(original,**prior_plan['model_initialization']).eval()
    model.aligned.gradient_checkpointing = True
    if model.counts()!=prior_plan['actual_counts']: raise ValueError('Actual complete under1B inventory must match')
    selected = model.extension_parameters(*GROUPS); selected_ids = {id(p) for p in selected}
    fixed = [p for p in model.parameters() if id(p) not in selected_ids]
    signatures = [(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]
    original_digest = state_digest(original)
    def unused_digest(): return tensor_digest((n,p) for n,p in model.parts.named_parameters() if not n.startswith(PREFIXES))
    unused = unused_digest()
    batches = [data.batch([p],'cpu') for p in positions]
    pointer = PointerData(ROOT/'data/localized-pointer-candidate-v1','train','audit')
    photo_position = json.loads((ROOT/'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())['original_source_position']
    photo = pointer.batch([photo_position],'cpu'); ids,pixels = photo['input_ids'],photo['pixel_values']
    with torch.no_grad():
        native = {r:original(ids,pixels,route=r)['logits'].clone() for r in ('general','code','grounding')}
        text = model(ids)['logits'].clone(); aligned_image = model(ids,pixels)['logits'].clone()
        word_logits = [model.word_logits(b['word_pixels']).clone() for b in batches]
        if any(logits.shape != (1,140,257) for logits in word_logits): raise ValueError('Actual140-step complete source word computation required')
        cache = original.language.code.prefill(ids,pixels)['cache']
        cached = original.language.code.decode(torch.tensor([[5]]),cache)['logits'].clone()
    optimizer = torch.optim.AdamW(selected,lr=3e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
    def update():
        model.aligned.train(); model.original.eval(); optimizer.zero_grad(set_to_none=True)
        metric = accumulate_gradients(model,batches,ctc_weight=1.,native_CE_weight=1.)
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected)
                or any(p.grad is not None for p in fixed)):
            raise ValueError('Every and only shared vision/word parameter must receive finite gradients')
        metric['gradient_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_(selected,1.,error_if_nonfinite=True))
        optimizer.step(); optimizer.zero_grad(set_to_none=True); return metric
    sources = dict(full_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),word_admission_sha256=file_hash(data.root/'admission.json'))
    cursor = dict(original_word_positions=positions,complete_targets=True,completed_updates=1)
    first = update(); print('actual_shared_word_update',1,first,flush=True)
    state = make_state(model,optimizer,groups=GROUPS,sources=sources,completed_steps=1,planned_steps=2,
        sampler_state=cursor,metrics=[first],device='cpu')
    atomic_checkpoint(REPORTS/'resume-step1.pt',state,128*1024**2,26*1024**3)
    expected = update(); print('actual_shared_word_update',2,expected,flush=True)
    expected_parts = model.extension_state(); expected_adam = copy.deepcopy(optimizer.state_dict()); expected_rng = capture_rng('cpu')
    restore_state(_VisionWordRestoreView(model),optimizer,torch.load(REPORTS/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True),
        groups=GROUPS,expected_sources=sources,planned_steps=2,expected_sampler_state=cursor,device='cpu')
    actual = update()
    if (actual!=expected or not equal(expected_parts,model.extension_state()) or not equal(expected_adam,optimizer.state_dict())
            or not equal(expected_rng,capture_rng('cpu'))): raise ValueError('Full next-update weights, Adam, RNG and metrics must recover exactly')
    final_state = make_state(model,optimizer,groups=GROUPS,sources=sources,completed_steps=2,planned_steps=2,
        sampler_state={**cursor,'completed_updates':2},metrics=[first,expected],device='cpu')
    atomic_checkpoint(REPORTS/'resume-step2.pt',final_state,128*1024**2,26*1024**3)
    if (original_digest!=state_digest(original) or unused!=unused_digest()
            or signatures!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in fixed]):
        raise ValueError('All original and unused language/reward/critic bytes/pointers/versions/flags must remain fixed')
    model.eval()
    with torch.no_grad():
        if not torch.equal(text,model(ids)['logits']): raise ValueError('Aligned text-only output must remain exact')
        trained_image = model(ids,pixels)['logits'].clone(); trained_words = [model.word_logits(b['word_pixels']).clone() for b in batches]
        if torch.equal(aligned_image,trained_image) or any(torch.equal(a,b) for a,b in zip(word_logits,trained_words)):
            raise ValueError('Actual learned image prefix and every supplied word head output must change')
        if any(not torch.equal(native[r],original(ids,pixels,route=r)['logits']) for r in native): raise ValueError('Original native image routes must remain exact')
        if not torch.equal(cached,original.language.code.decode(torch.tensor([[5]]),cache)['logits']): raise ValueError('Existing own code cache must remain exact')
    atomic_checkpoint(REPORTS/'trained-disposable-extensions.pt',model.export(sources),64*1024**2,26*1024**3)
    cold = OwnAlignedAssistantV3(original,**prior_plan['model_initialization']).eval()
    cold.import_export(torch.load(REPORTS/'trained-disposable-extensions.pt',map_location='cpu',weights_only=True,mmap=True),expected_sources=sources)
    with torch.no_grad():
        if (not torch.equal(trained_image,cold(ids,pixels)['logits'])
                or not torch.equal(text,cold(ids)['logits'])
                or any(not torch.equal(t,cold.word_logits(b['word_pixels'])) for t,b in zip(trained_words,batches))):
            raise ValueError('Actual trained image-prefix, whole-word and text-only cold reload must be exact')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(actual_counts=model.counts(),
        actual_vision_word_Adam_states=len(optimizer.state),actual_optimized_values=sum(p.numel() for p in selected),
        original_source_positions=positions, original_source_ids=[data.rows[p]['id'] for p in positions],
        complete_source_transcripts=[data.rows[p]['response'] for p in positions],
        source_metrics=[first,expected],fixed_original_digest=original_digest,unused_new_parts_digest=unused,
        actual_all_selected_gradients_finite_and_all_fixed_gradients_absent=True,
        actual_gradient_checkpointing_with_selected_leaf_backward=True,
        actual_next_update_all_new_weights_Adam_RNG_and_metrics_exact=True,
        all_original_and_unused_new_bytes_versions_pointers_flags_exact=True,
        all_original_native_image_routes_and_code_cache_exact=True,aligned_text_only_logits_exact=True,
        actual_trained_image_prefix_and_word_outputs_changed=True,actual_trained_full_model_cold_reload_exact=True,
        retained_step1_sha256=file_hash(REPORTS/'resume-step1.pt'),retained_step2_sha256=file_hash(REPORTS/'resume-step2.pt'),
        trained_disposable_export_sha256=file_hash(REPORTS/'trained-disposable-extensions.pt'),
        logical_disposable_updates=2,physical_optimizer_calls=3,seconds=time.monotonic()-started,
        full_production_word_training=False,new_held_image_reading_gain_measured=False,
        all_probe_weights_discarded_before_production=True,candidate_promoted=False,goal_achieved=False))


if __name__=='__main__': main()
