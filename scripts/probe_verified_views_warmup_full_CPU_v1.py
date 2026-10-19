"""Full-size source-view learning/recovery; two controls, not a training epoch."""
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/verified-views-warmup-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide all GPUs for the full-size CPU control')
    if REPORTS.exists():raise ValueError('Preserve all earlier complete/partial control work')
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.verified_views_warmup_v1 import VerifiedViewsWarmup
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    random.seed(957713);np.random.seed(957713);torch.manual_seed(957713);started=time.monotonic()
    base_plan=ROOT/'reports/protected-joint-v1/plan.json';plan=json.loads(base_plan.read_text())
    files=[Path(__file__),base_plan,ROOT/'src/slm_lab/verified_views_warmup_v1.py',ROOT/'src/slm_lab/own_dense_actor_v1.py',
        ROOT/'src/slm_lab/own_dense_actor_checkpoint_v1.py',ROOT/'data/verified-code-views-v1/manifest.json',
        ROOT/'data/verified-code-views-v1/admission.json',ROOT/'reports/verified-views-warmup-four-CPU-v2/complete.json']
    bindings={str(p):file_hash(p) for p in files}
    bindings.update({p:d for p,d in plan['source_files_sha256'].items() if Path(p).suffix=='.py'})
    guard=FileBindings(bindings);REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(source_files_sha256=bindings,available_original_functions=173,available_complete_views=494,
        actual_control_function_budget=2,actual_control_presentations=6,logical_control_updates=2,physical_optimizer_calls=3,
        functions_per_update=1,maximum_context_limit=4096,short_source_controls_not_full_GPU_capacity=True,
        initializer=plan['initial_dense_settings']['initializer'],neural_production_updates=0,goal_achieved=False))
    def phase(name):atomic_json(REPORTS/'progress.json',dict(phase=name,seconds=time.monotonic()-started,production_updates=0,goal_achieved=False))
    def make():
        base,_=fresh(plan,'cpu');return OwnDenseActor(base)
    model=make();parameters=[p for _,p in model.named_selected(('actor',))]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
    stage=VerifiedViewsWarmup(model,optimizer,ROOT,method='views',epochs=1,functions_per_update=1,seed=957713)
    if model.counts()['actual_unique']!=999706442:raise ValueError('Every actual own head/base/adapter/critic must count below1B')
    frozen=tensor_digest(model.base.named_parameters());initial=tensor_digest(model.parts.named_parameters());critic=model.parts['critic'].weight.detach().clone()
    phase('first complete original/completion/repair source update');first=stage.update();snapshot=stage.state(bindings)
    atomic_checkpoint(REPORTS/'resume-step1.pt',snapshot,256*1024**2,26*1024**3);del snapshot
    phase('next full-size actual source-view update');second=stage.update();learned=tensor_digest(model.parts.named_parameters())
    def adam():return tensor_digest((str(i)+'.'+k,s[k]) for i,s in enumerate(optimizer.state.values()) for k in ('step','exp_avg','exp_avg_sq'))
    expected_adam=adam();payload=torch.load(REPORTS/'resume-step1.pt',map_location='cpu',weights_only=True,mmap=True)
    stage.resume(payload,bindings);phase('exact next full-size views/metrics/weights/Adam/CPU-RNG recovery');repeated=stage.update()
    if (repeated!=second or learned!=tensor_digest(model.parts.named_parameters()) or adam()!=expected_adam
            or initial==learned or frozen!=tensor_digest(model.base.named_parameters())
            or not torch.equal(critic,model.parts['critic'].weight) or second['token_mean_reference_KL']<=0):
        raise ValueError('Actual full-size learning, positive own KL, next-state recovery and unchanged base/critic required')
    final=model.export(bindings);atomic_checkpoint(REPORTS/'actor.pt',final,96*1024**2,26*1024**3)
    final_state=stage.state(bindings);atomic_checkpoint(REPORTS/'resume-step2.pt',final_state,256*1024**2,26*1024**3);del final_state
    ids=torch.tensor([[1,261,267,71,2]]);pixels=torch.zeros(1,3,448,448);word_pixels=torch.zeros(1,3,32,224)
    def outputs(m):
        m.eval()
        with torch.no_grad():return dict(text=m(ids)['logits'].clone(),image=m(ids,pixel_values=pixels)['logits'].clone(),word=m.base.word_logits(word_pixels).clone())
    phase('actual trained text/image/word output controls');expected=outputs(model);counts=model.counts();states=len(optimizer.state)
    del model,optimizer,stage,payload,parameters;gc.collect();phase('fresh full-size trained native cold reload')
    cold=make();cold.import_export(final,expected_sources=bindings);actual=outputs(cold)
    if any(not torch.equal(expected[k],actual[k]) for k in expected):raise ValueError('Every actual trained native cold output must match')
    guard.check(full=True)
    result=dict(at=datetime.now(timezone.utc).isoformat(),actual_counts=counts,complete_Adam_states=states,
        actual_logical_control_updates=2,actual_physical_optimizer_calls=3,actual_complete_presentations=6,
        records=[first,second],exact_next_views_metrics_weights_Adam_CPU_RNG=True,positive_trained_own_reference_KL=True,
        frozen_base_vision_heads_and_critic_exact=True,trained_native_text_image_word_cold_reload_exact=True,
        short_control_sources_not_full_GPU_capacity_or_quality=True,production_updates=0,held_quality_gain_proven=False,
        source_files_sha256=bindings,seconds=time.monotonic()-started,goal_achieved=False)
    atomic_json(REPORTS/'complete.json',result);print(json.dumps(dict(passed=True,counts=counts,seconds=result['seconds'])),flush=True)


if __name__=='__main__':main()
