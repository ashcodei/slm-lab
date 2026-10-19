import copy
import json
import pytest
import torch

from slm_lab import dense_preference_inference_v1 as initial
from slm_lab import dense_ppo_inference_v1 as inference
from slm_lab.corpus import file_hash
from slm_lab.dense_human_preference_stage_v1 import DenseHumanPreferenceStage
from slm_lab.dense_ppo_control_v1 import DenseControlledPPOStage
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_stage_v1 import pairs,options as old_options
from test_dense_ppo_v1 import reward,stage as old_ppo
from test_own_dense_actor_v1 import actor


def initial_production_graph(tmp_path):
    model=actor();data=pairs();setting=old_options(data);setting.update(epochs=1,steps=1,accumulation=5)
    rate=setting.pop('learning_rate');dpo=setting.pop('DPO_weight')
    inputs={k:chr(97+i)*64 for i,k in enumerate(('foundation_checkpoint_sha256','foundation_plan_sha256','word_checkpoint_sha256','word_plan_sha256'))}
    receipt=dict(actual_counts=model.base.counts(),synthetic_small_test_graph=True)
    base=tmp_path/'base.pt';torch.save(initial.export_base(model.base,inputs,receipt),base)
    plan=dict(base_checkpoint=str(base),base_checkpoint_sha256=file_hash(base),base_inputs=inputs,base_folding_receipt=receipt,
        frozen_dense_sha256=tensor_digest(model.base.named_parameters()),actual_dense_counts=model.base.counts(),actor_settings=model.settings,
        actual_counts=model.counts(),initial_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),tokenizer_sha256='e'*64,
        human_admission_sha256='f'*64,human_manifest_sha256='1'*64,training_options=setting,
        arms=dict(DPO=dict(learning_rate=rate,DPO_weight=dpo)),source_exposure_budget=dict(DPO=dict(original_pairs=5)))
    path=tmp_path/'DPO-plan.json';path.write_text(json.dumps(plan));sha=file_hash(path)
    options={**setting,'learning_rate':rate,'DPO_weight':dpo}
    opt=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=rate,betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
    learner=DenseHumanPreferenceStage(model,opt,data,options,initial.sources(plan,sha));learner.update();model.eval()
    checkpoint=tmp_path/'DPO.pt';torch.save(initial.export(model,plan,sha,'DPO',1),checkpoint)
    return model,data,dict(checkpoint=str(checkpoint),plan=str(path),arm='DPO',checkpoint_sha256=file_hash(checkpoint),plan_sha256=sha)


def test_actual_PPO_trained_loader_accepts_learned_critic_and_rejects_extra_encoder_changed_profile_sources_or_budget(tmp_path):
    model,data,inputs=initial_production_graph(tmp_path);setting=old_ppo().options;setting.update(steps=1);rate=setting.pop('learning_rate')
    plan=dict(initial_policy=inputs,actual_counts=model.counts(),frozen_dense_sha256=tensor_digest(model.base.named_parameters()),
        initial_parts_sha256=tensor_digest(model.parts.named_parameters()),training_options=setting,
        reward_inputs=dict(checkpoint_sha256='a'*64),human_admission_sha256='b'*64,
        arms=dict(human=dict(learning_rate=rate,human_reward_scale=1)),source_exposures=dict(human=dict(whole_prompts=2,complete_epoch=False)))
    path=tmp_path/'PPO-plan.json';path.write_text(json.dumps(plan));sha=file_hash(path);checkpoint=tmp_path/'PPO.pt'
    torch.save(inference.export(model,plan,sha,'initial',0),checkpoint)
    cold,metadata=inference.load(checkpoint,path,'initial',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
    assert metadata['separate_training_only_reward_not_loaded']
    settings=inference.options(plan,'human');opt=torch.optim.AdamW([p for _,p in cold.named_selected(('actor','critic'))],
        lr=rate,betas=tuple(settings['betas']),weight_decay=settings['weight_decay'],foreach=False)
    learner=DenseControlledPPOStage(cold,opt,data,reward(),settings,inference.sources(plan,sha),human_reward_scale=1)
    learner.update();cold.eval();assert cold.parts['critic'].weight.count_nonzero()
    trained=inference.export(cold,plan,sha,'human',1);torch.save(trained,checkpoint)
    restored,metadata=inference.load(checkpoint,path,'human',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
    ids=torch.tensor([[1,7,9,2]]);pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16)
    with torch.no_grad():
        for image in (None,pixels):
            a,b=cold(ids,image),restored(ids,image)
            assert torch.equal(a['logits'],b['logits']) and torch.equal(a['values'],b['values'])
    assert metadata['separate_training_only_reward_not_loaded'] and sum(p.numel() for p in restored.parameters())==plan['actual_counts']['actual_unique']
    for mutate in [lambda s:s.update(reward_encoder={'extra_unwanted_weights':torch.ones(3)}),
        lambda s:s.update(terminal_human_reward_scale=0),lambda s:s.update(actual_own_response_PPO_updates=0),
        lambda s:s['policy'].update(frozen_dense_sha256='0'*64),lambda s:s['source_exposures'].update(whole_prompts=1)]:
        bad=copy.deepcopy(trained);mutate(bad);torch.save(bad,checkpoint)
        with pytest.raises(ValueError):inference.load(checkpoint,path,'human',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
