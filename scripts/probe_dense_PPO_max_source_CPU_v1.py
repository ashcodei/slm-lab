"""Full-size complete maximum-train-prompt plus128fixed-action capacity.

This is a synthetic backward capacity control, not a sampled PPO rollout or
human judgment. No Adam step is performed. Actual GPU/optimizer capacity must
still be checked by every owned production worker.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-PPO-max-source-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('The full CPU capacity control must explicitly hide every GPU')
    if REPORTS.exists():raise ValueError('Preserve prior complete or partial controls')
    import torch
    from slm_lab.codenet_checkpoint_v1 import FileBindings,capture_rng
    from slm_lab.corpus import file_hash
    from slm_lab.dense_ppo_reward_v1 import load as load_reward,signature
    from slm_lab.dense_ppo_control_v1 import DenseControlledPPOStage
    from slm_lab.dense_ppo_capacity_v1 import stress
    from slm_lab.diverse_images import atomic_json
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_data_v2 import WholeHumanData
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    from train_dense_PPO_pilot_v1 import equal
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    prior=ROOT/'reports/dense-ppo-cached-full-CPU-v2';old=json.loads((prior/'plan.json').read_text())
    files=dict(old['source_files_sha256'])
    for name in ['scripts/probe_dense_PPO_max_source_CPU_v1.py','scripts/train_dense_PPO_pilot_v1.py','src/slm_lab/dense_ppo_capacity_v1.py',
        'src/slm_lab/dense_ppo_control_v1.py','tests/test_dense_ppo_capacity_v1.py','reports/dense-ppo-capacity-v1-tests.log']:
        p=ROOT/name;files[str(p)]=file_hash(p)
    for name in ['plan.json','complete.json','independent-complete-neural-readback.json']:
        p=prior/name;files[str(p)]=file_hash(p)
    if '2 passed' not in (ROOT/'reports/dense-ppo-capacity-v1-tests.log').read_text():raise ValueError('Actual complete-horizon no-optimizer capacity tests required')
    guard=FileBindings(files);guard.check(full=True);started=time.monotonic()
    raw=torch.load(old['combined_base_checkpoint'],map_location='cpu',weights_only=True,mmap=True)['model']
    base=OwnDenseAssistant(new_backbone(ModelConfig(**raw['config']),948531),**raw['settings']).eval()
    base.import_export(raw,expected_sources=old['combined_base_sources']);del raw
    model=OwnDenseActor(base,**old['actor_settings']).eval()
    model.import_export(torch.load(old['initial_disposable_actor'],map_location='cpu',weights_only=True,mmap=True),expected_sources=old['initial_disposable_actor_sources'])
    scorer=load_reward(old['reward_inputs']);data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
    setting=old['training_options'];position=max(range(len(data)),key=lambda p:data.rows[p]['native']['chosen']['target_start'])
    REPORTS.mkdir();atomic_json(REPORTS/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        original_source_position=position,complete_prompt_tokens=data.rows[position]['native']['chosen']['target_start'],fixed_non_EOS_action_tokens=128,
        actual_counts=model.counts(),separate_training_only_reward=scorer.counts(),all_source_model_artifacts_disposable=True,
        Adam_memory_and_actual_GPU_capacity_not_established=True,synthetic_control_not_training_rollout=True,goal_achieved=False))
    sources=dict(maximum_source_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),initial_own_actor_sha256=file_hash(old['initial_disposable_actor']),
        reward_checkpoint_sha256=old['reward_inputs']['checkpoint_sha256'])
    optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor','critic'))],lr=setting['learning_rate'],
        betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    kernel=DenseControlledPPOStage(model,optimizer,data,scorer,setting,sources,human_reward_scale=1)
    before=signature(model.parameters());reward_before=signature(scorer.parameters());weights=tensor_digest(model.named_parameters());reward_weights=tensor_digest(scorer.named_parameters());rng=capture_rng('cpu')
    print('full_maximum_source_capacity_started',position,'whole_prompt_tokens',data.rows[position]['native']['chosen']['target_start'],flush=True)
    result=stress(kernel,position)
    if (before!=signature(model.parameters()) or reward_before!=signature(scorer.parameters()) or weights!=tensor_digest(model.named_parameters())
            or reward_weights!=tensor_digest(scorer.named_parameters()) or not equal(rng,capture_rng('cpu')) or optimizer.state or kernel.completed):
        raise ValueError('Every real counted model/reward value, Adam/source cursor and CPU RNG must remain unchanged')
    guard.check(full=True)
    atomic_json(REPORTS/'complete.json',dict(actual_counts=model.counts(),separate_training_only_reward=scorer.counts(),
        complete_original_maximum_source_and_full_action_stress=result,all_actual_562_actor_critic_parameter_gradients_finite=len(kernel.parameters)==562,
        every_model_reward_value_metadata_Adam_source_cursor_and_CPU_RNG_unchanged=True,physical_optimizer_calls=0,production_updates=0,
        peak_process_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,seconds=time.monotonic()-started,
        actual_configured_GPU_Adam_and_next_update_capacity_not_claimed=True,source_model_artifacts_disposable=True,candidate_promoted=False,goal_achieved=False))
    print(REPORTS/'complete.json',flush=True)


if __name__=='__main__':main()
