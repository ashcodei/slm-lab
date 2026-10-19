"""Full-size combined dense/word policy through the complete-epoch DPO runner.

The first two sources come from the declared full-source permutation, without
choosing shorter examples. CPU accumulation1is explicit; production4and GPU
source extremes remain a separate required capacity check. All updates are
disposable and this is not a complete preference epoch or held quality gain.
"""
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-human-preference-stage-full-CPU-v1'
RESERVE=26*1024**3


def equal(a,b):
    import torch
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('The full stage CPU proof must hide all GPUs')
    if REPORTS.exists():raise ValueError('Preserve prior complete or partial attempts; no automatic retry')
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.dense_foundation_inference_v1 import fresh
    from slm_lab.dense_human_preference_v1 import pair_information
    from slm_lab.dense_human_preference_stage_v1 import DenseHumanPreferenceStage,signature
    from slm_lab.diverse_images import atomic_json
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.shared_word_data_v1 import SharedWordData
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    log=ROOT/'reports/dense-human-preference-stage-v1-tests.log'
    if '4 passed' not in log.read_text():raise ValueError('Actual complete-epoch/control/gradient/recovery/source/F32 checks required')
    merged=ROOT/'reports/dense-word-folding-full-CPU-v1';old=json.loads((merged/'plan.json').read_text())
    prior=json.loads((merged/'complete.json').read_text());base_path=merged/'combined-disposable-model.pt'
    if file_hash(base_path)!=prior['combined_checkpoint_sha256']:raise ValueError('The actual source-bound combined CPU model required')
    files=dict(old['source_files_sha256'])
    for name in ['scripts/probe_dense_human_preference_stage_full_CPU_v1.py','src/slm_lab/dense_human_preference_stage_v1.py',
        'src/slm_lab/dense_human_preference_v1.py','src/slm_lab/own_dense_actor_checkpoint_v1.py',
        'tests/test_dense_human_preference_stage_v1.py','reports/dense-human-preference-stage-v1-tests.log']:
        files[str(ROOT/name)]=file_hash(ROOT/name)
    for name in ('plan.json','complete.json','independent-complete-readback.json','combined-disposable-model.pt'):
        p=merged/name;files[str(p)]=file_hash(p)
    guard=FileBindings(files);guard.check(full=True);started=time.monotonic()
    settings=old['initial_dense_settings'];base,_=fresh(settings,'cpu')
    payload=torch.load(base_path,map_location='cpu',weights_only=True,mmap=True)
    base_sources=dict(full_integration_CPU_plan_sha256=file_hash(merged/'plan.json'),
        dense_language_source_sha256=old['dense_language_sha256'],word_source_sha256=old['source_word_sha256'])
    base.import_export(payload['model'],expected_sources=base_sources);del payload
    model=OwnDenseActor(base,rank=16,seed=949331).eval()
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
    options=dict(seed=950131,epochs=1,steps=len(data),microbatch=1,accumulation=1,learning_rate=3e-6,
        betas=[.9,.95],weight_decay=.01,gradient_clip=1.,DPO_beta=.1,DPO_weight=1,chosen_CE_weight=.1)
    REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        combined_base_checkpoint=str(base_path),combined_base_sha256=file_hash(base_path),combined_base_sources=base_sources,
        initial_dense_settings=settings,actor_settings=model.settings,training_options=options,
        actual_counts=model.counts(),CPU_FP32_threads=2,CPU_accumulation=1,production_accumulation4_GPU_capacity_not_verified=True,
        original_source_permutation_no_short_example_selection=True,planned_disposable_updates=2,
        complete_source_epoch_not_claimed=True,goal_achieved=False))
    sources=dict(full_stage_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),combined_base_sha256=file_hash(base_path),
        human_admission_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json'))
    optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=options['learning_rate'],
        betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
    stage=DenseHumanPreferenceStage(model,optimizer,data,options,sources)
    selected_positions=[stage.positions(i)[0] for i in range(2)]
    original_info=[pair_information(data,p) for p in selected_positions]
    atomic_json(REPORTS/'selected-original-source-controls.json',dict(positions=selected_positions,complete_original_sources=original_info))
    pointer=PointerData(ROOT/'data/localized-pointer-candidate-v1','train','audit')
    pp=json.loads((ROOT/'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())['original_source_position']
    photo=pointer.batch([pp],'cpu');ids,pixels=photo['input_ids'],photo['pixel_values']
    words=SharedWordData(ROOT/'data/shared-byte-words-v1','train','optimization');word=words.batch([50000],'cpu')
    frozen=tensor_digest(base.named_parameters());fixed=signature(stage.fixed)
    with torch.no_grad():
        initial_image=base(ids,pixels)['logits'].clone();initial_word=base.word_logits(word['word_pixels']).clone()
        cache=base.network.prefill(ids,pixels)['cache'];cached=base.network.decode(torch.tensor([[4]]),cache)['logits'].clone()
    random.seed(950731);np.random.seed(950731);torch.manual_seed(950731)
    first=stage.update();print('actual_full_whole_epoch_kernel_DPO',first,flush=True)
    atomic_checkpoint(REPORTS/'resume-step1.pt',stage.state(),256*1024**2,RESERVE)
    def draws():return [random.random(),float(np.random.random()),float(torch.rand(()))]
    expected_draws=draws();expected=stage.update();print('actual_full_whole_epoch_kernel_DPO',expected,flush=True)
    learned=tensor_digest(model.parts.named_parameters());adam=copy.deepcopy(optimizer.state_dict());rng=capture_rng('cpu')
    saved=torch.load(REPORTS/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
    original_moments=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['core']['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))
    stage.resume(saved)
    if (draws()!=expected_draws or stage.update()!=expected or learned!=tensor_digest(model.parts.named_parameters())
            or not equal(adam,optimizer.state_dict()) or not equal(rng,capture_rng('cpu'))
            or original_moments!=tensor_digest((str(i)+'.'+k,v[k]) for i,v in saved['core']['optimizer']['state'].items() for k in ('step','exp_avg','exp_avg_sq'))):
        raise ValueError('Actual complete-stage next update/Adam/RNG/source metrics and immutable input payload must recover exactly')
    atomic_checkpoint(REPORTS/'resume-step2.pt',stage.state(),256*1024**2,RESERVE)
    atomic_checkpoint(REPORTS/'trained-disposable-policy.pt',model.export(sources),96*1024**2,RESERVE)
    if frozen!=tensor_digest(base.named_parameters()) or fixed!=signature(stage.fixed):
        raise ValueError('Every original fused dense/word/head and unused critic value/metadata must remain exact')
    model.eval()
    with torch.no_grad():
        if (not torch.equal(initial_image,base(ids,pixels)['logits']) or not torch.equal(initial_word,base.word_logits(word['word_pixels']))
                or not torch.equal(cached,base.network.decode(torch.tensor([[4]]),cache)['logits'])):
            raise ValueError('Actual fused base native image/word and pre-existing cache must stay exact')
        actual_image=model(ids,pixels)['logits'].clone()
        batch=data.batch([selected_positions[1]],'chosen','cpu');actual_text=model(**batch)['logits'].clone()
    cold_base,_=fresh(settings,'cpu')
    cold_base.import_export(torch.load(base_path,map_location='cpu',weights_only=True,mmap=True)['model'],expected_sources=base_sources)
    cold=OwnDenseActor(cold_base,**model.settings).eval()
    cold.import_export(torch.load(REPORTS/'trained-disposable-policy.pt',map_location='cpu',weights_only=True,mmap=True),expected_sources=sources)
    with torch.no_grad():
        if (not torch.equal(actual_image,cold(ids,pixels)['logits']) or not torch.equal(actual_text,cold(**batch)['logits'])
                or not torch.equal(initial_word,cold.base.word_logits(word['word_pixels'])) or learned!=tensor_digest(cold.parts.named_parameters())):
            raise ValueError('Actual full-stage learned text/image/word and every adapter value must cold-reload exactly')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(actual_counts=model.counts(),actual_actor_Adam_states=len(optimizer.state),
        actual_actor_values=sum(p.numel() for _,p in model.named_selected(('actor',))),complete_selected_original_sources=original_info,
        actual_complete_stage_metrics=stage.metrics,actual_next_weights_Adam_all_CPU_RNG_cursor_metrics_and_pristine_payload_exact=True,
        every_frozen_dense_vision_word_and_auxiliary_tensor_metadata_and_native_cache_exact=True,
        actual_trained_full_size_text_image_word_cold_reload_exact=True,
        retained_step1_sha256=file_hash(REPORTS/'resume-step1.pt'),retained_step2_sha256=file_hash(REPORTS/'resume-step2.pt'),
        trained_policy_sha256=file_hash(REPORTS/'trained-disposable-policy.pt'),frozen_combined_dense_sha256=frozen,
        seconds=time.monotonic()-started,logical_disposable_updates=2,physical_optimizer_calls=3,
        full_source_permutation_used_for_both_original_controls=True,CPU_accumulation=1,
        production_accumulation4_source_extreme_GPU_capacity_not_proven=True,
        full_preference_epoch_held_alignment_gain_PPO_and_final_capability_not_claimed=True,
        all_probe_updates_discarded=True,candidate_promoted=False,goal_achieved=False))
    print(REPORTS/'complete.json',flush=True)


if __name__=='__main__':main()
