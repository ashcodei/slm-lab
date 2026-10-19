"""Disposable full-size hybrid code-RL mechanics, never production quality.

Two logical updates use complete original train functions but only four own
actions per sample. This is neither maximum-horizon capacity nor a benchmark.
The current four-GPU full-weight learner is never loaded or interrupted.
"""
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/execution-group-RL-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU proof must hide every GPU')
    if REPORTS.exists():raise ValueError('Preserve previous disposable experiment')
    import numpy as np
    import torch
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng,restore_rng,FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.execution_group_RL_v1 import ExecutionGroupLearner
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    random.seed(956919);np.random.seed(956919);torch.manual_seed(956919);start=time.monotonic()
    plan_path=ROOT/'reports/fast-joint-v3/plan.json';plan=json.loads(plan_path.read_text())
    files=[Path(__file__),plan_path,ROOT/'src/slm_lab/execution_group_RL_v1.py',ROOT/'src/slm_lab/public_example_code_reward_v1.py',
           ROOT/'data/documented-doctest-verified-v1/admission.json',ROOT/'reports/execution-group-RL-v1-tests-v2.log',
           ROOT/'reports/public-example-code-reward-v1-tests.log',ROOT/'reports/execution-group-RL-small-graph-v2.json']
    bindings={str(p):file_hash(p) for p in files}
    bindings.update({p:d for p,d in plan['source_files_sha256'].items() if Path(p).suffix=='.py'})
    guard=FileBindings(bindings);REPORTS.mkdir()
    atomic_json(REPORTS/'plan.json',dict(source_files_sha256=bindings,initializer=plan['initial_dense_settings'],
        group_size=2,maximum_own_actions_per_sample=4,logical_updates=2,whole_original_train_positions=[0,1],
        production_updates=0,full_horizon_GPU_capacity_not_tested=True,goal_achieved=False))
    def phase(name):
        atomic_json(REPORTS/'progress.json',dict(phase=name,seconds=time.monotonic()-start,production_updates=0,goal_achieved=False))
    def make():
        base,_=fresh(plan,'cpu');return OwnDenseActor(base)
    def optimizer(model):return torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=1e-5,betas=(.9,.95),foreach=False)
    model=make();opt=optimizer(model)
    if model.counts()['actual_unique']!=999706442:raise ValueError('Actual full-size all-head sub1B actor required')
    stage=ExecutionGroupLearner(model,opt,ROOT,group_size=2,max_new_tokens=4)
    frozen=tensor_digest(model.base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
    initial=tensor_digest(model.parts.named_parameters());phase('first complete-source disposable hybrid update')
    first=stage.step(0)
    if initial==tensor_digest(model.parts.named_parameters()):raise ValueError('Actual actor must learn')
    payload=dict(actor=model.export(bindings),optimizer=opt.state_dict(),rng=capture_rng('cpu'),actual_completed_updates=1,records=[first])
    state_path=REPORTS/'resume-step1.pt';atomic_checkpoint(state_path,payload,256*1024**2,26*1024**3);del payload
    phase('second update from next original complete training function')
    second=stage.step(1);expected=tensor_digest(model.parts.named_parameters())
    def adam():return tensor_digest((str(i)+'.'+key,state[key]) for i,state in enumerate(opt.state.values()) for key in ('step','exp_avg','exp_avg_sq'))
    expected_adam=adam();saved=torch.load(state_path,map_location='cpu',weights_only=True)
    model.import_export(saved['actor'],expected_sources=bindings);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],'cpu')
    phase('exact next sampled actions execution rewards gradients weights and Adam recovery')
    repeated=stage.step(1)
    if second!=repeated or expected!=tensor_digest(model.parts.named_parameters()) or expected_adam!=adam():
        raise ValueError('Actual next full-size sample/reward/update/Adam/RNG record must recover exactly')
    if frozen!=tensor_digest(model.base.named_parameters()) or not torch.equal(critic,model.parts['critic'].weight):
        raise ValueError('Every frozen base/vision/head/critic value must remain exact')
    final=model.export(bindings);atomic_checkpoint(REPORTS/'disposable-actor.pt',final,96*1024**2,26*1024**3)
    ids=torch.tensor([[1,261,267,71,2]])
    pixels=torch.zeros(1,3,448,448);word_pixels=torch.zeros(1,3,32,224)
    def outputs(m):
        m.eval()
        with torch.no_grad():
            return dict(text=m(ids)['logits'].clone(),image=m(ids,pixel_values=pixels)['logits'].clone(),
                        word=m.base.word_logits(word_pixels).clone())
    expected_outputs=outputs(model);counts=model.counts();Adam_states=len(opt.state)
    del saved,stage,opt,model;gc.collect();phase('fresh full-size native text image word cold reload')
    cold=make();cold.import_export(final,expected_sources=bindings)
    actual_outputs=outputs(cold)
    if any(not torch.equal(expected_outputs[k],actual_outputs[k]) for k in expected_outputs):
        raise ValueError('Actual trained native cold outputs must match exactly')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),actual_counts=counts,
        actual_logical_disposable_updates=2,actual_physical_optimizer_calls=3,actual_Adam_states=Adam_states,
        complete_original_train_positions=[0,1],actual_records=[first,second],
        exact_actual_next_actions_execution_rewards_records_weights_Adam_CPU_RNG_recovered=True,
        frozen_base_vision_heads_and_critic_exact=True,actual_trained_text_image_word_cold_reload_exact=True,
        full_horizon_GPU_capacity_and_controlled_quality_gain_not_tested=True,genuine_human_preference_labels=0,
        production_updates=0,seconds=time.monotonic()-start,source_files_sha256=bindings,goal_achieved=False))
    phase('complete disposable mechanics; production and quality remain unproved')
    print(json.dumps(dict(actual_counts=counts,seconds=time.monotonic()-start,production_updates=0)),flush=True)


if __name__=='__main__':main()
