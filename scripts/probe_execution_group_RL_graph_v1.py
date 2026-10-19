"""Small random graph, real public train source, own sampling and fallback.

This is integration evidence for the proposed hybrid, not training/evaluation
of the full-size student and not a claimed code-quality or RLHF improvement.
"""
from datetime import datetime,timezone
import os
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/execution-group-RL-small-graph-v1.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Every GPU must be hidden')
    if OUTPUT.exists():raise ValueError('Preserve prior own-output integration evidence')
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.execution_group_RL_v1 import ExecutionGroupLearner
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(956811);started=time.monotonic()
    config=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=4096,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    base=OwnDenseAssistant(new_backbone(config,956811),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4)
    model=OwnDenseActor(base,rank=2);parameters=[p for _,p in model.named_selected(('actor',))]
    optimizer=torch.optim.AdamW(parameters,lr=.001,foreach=False)
    stage=ExecutionGroupLearner(model,optimizer,ROOT,group_size=2,max_new_tokens=2)
    fixed=tensor_digest(base.named_parameters());before=tensor_digest(model.parts.named_parameters());critic=model.parts['critic'].weight.detach().clone()
    record=stage.step(0)
    if (not record['verified_public_target_fallback_required'] or record['public_fallback_CE']<=0
            or fixed!=tensor_digest(base.named_parameters()) or before==tensor_digest(model.parts.named_parameters())
            or not torch.equal(critic,model.parts['critic'].weight)):
        raise ValueError('Flat own execution rewards must lead to real public-target actor gradients with unchanged base/critic')
    files=[Path(__file__),ROOT/'src/slm_lab/execution_group_RL_v1.py',ROOT/'src/slm_lab/public_example_code_reward_v1.py',
        ROOT/'data/documented-doctest-verified-v1/admission.json',ROOT/'reports/documented-doctest-verified-v1/independent-complete-source-and-runtime-readback-v1.json']
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),small_random_graph=True,actual_counts=model.counts(),
        actual_original_train_position=0,actual_own_action_tokens=sum(len(t['response_ids']) for t in record['trajectories']),
        actual_complete_public_target_fallback_used=True,actual_actor_values_changed=True,all_base_and_critic_values_exact=True,
        actual_record=record,source_files_sha256={str(p.resolve()):file_hash(p) for p in files},seconds=time.monotonic()-started,
        full_size_GPU_capacity_controller_and_controlled_quality_comparison_pending=True,production_model_updates=0,
        genuine_human_preference_labels=0,goal_achieved=False))
    print(json.dumps(dict(actual_own_action_tokens=sum(len(t['response_ids']) for t in record['trajectories']),
        complete_public_target_CE=record['public_fallback_CE'],all_base_and_critic_values_exact=True,seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
