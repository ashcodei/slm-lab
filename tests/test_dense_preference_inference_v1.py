import copy
import json

import pytest
import torch

from slm_lab import dense_preference_inference_v1 as inference
from slm_lab.corpus import file_hash
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_stage_v1 import pairs,options as initial_options
from test_own_dense_actor_v1 import actor


def test_actual_trained_policy_loader_keeps_same_fused_base_and_rejects_changed_sources_critic_and_exposure(tmp_path):
    model=actor();data=pairs();setting=initial_options(data);setting.update(epochs=1,steps=1,accumulation=5)
    rate=setting.pop('learning_rate');dpo=setting.pop('DPO_weight')
    inputs={k:chr(97+i)*64 for i,k in enumerate(('foundation_checkpoint_sha256','foundation_plan_sha256','word_checkpoint_sha256','word_plan_sha256'))}
    receipt=dict(actual_counts=model.base.counts(),synthetic_small_test_graph=True)
    base_path=tmp_path/'base.pt';torch.save(inference.export_base(model.base,inputs,receipt),base_path)
    plan=dict(base_checkpoint=str(base_path),base_checkpoint_sha256=file_hash(base_path),base_inputs=inputs,
        base_folding_receipt=receipt,frozen_dense_sha256=tensor_digest(model.base.named_parameters()),actual_dense_counts=model.base.counts(),
        actor_settings=model.settings,actual_counts=model.counts(),initial_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),
        tokenizer_sha256='e'*64,human_admission_sha256='f'*64,human_manifest_sha256='1'*64,training_options=setting,
        arms=dict(DPO=dict(learning_rate=rate,DPO_weight=dpo)),source_exposure_budget=dict(DPO=dict(original_pairs=5,rounded_repeats=0)))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));sha=file_hash(path)
    initial=inference.export(model,plan,sha,'initial',0);checkpoint=tmp_path/'model.pt';torch.save(initial,checkpoint)
    cold,_=inference.load(checkpoint,path,'initial',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
    assert tensor_digest(cold.parts.named_parameters())==plan['initial_actor_parts_sha256']
    from slm_lab.dense_human_preference_stage_v1 import DenseHumanPreferenceStage
    options={**setting,'learning_rate':rate,'DPO_weight':dpo}
    opt=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=rate,betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
    learner=DenseHumanPreferenceStage(model,opt,data,options,inference.sources(plan,sha));learner.update();model.eval()
    trained=inference.export(model,plan,sha,'DPO',1);torch.save(trained,checkpoint)
    cold,_=inference.load(checkpoint,path,'DPO',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
    ids=torch.tensor([[1,4,7,2]])
    with torch.no_grad():assert torch.equal(model(ids)['logits'],cold(ids)['logits'])
    for mutate in [lambda s:s['policy']['parts']['critic.weight'].add_(1),
        lambda s:s['policy'].update(frozen_dense_sha256='0'*64),lambda s:s['source_exposure_budget'].update(original_pairs=4),
        lambda s:s.update(actual_human_preference_updates=0)]:
        bad=copy.deepcopy(trained);mutate(bad);torch.save(bad,checkpoint)
        with pytest.raises(ValueError):inference.load(checkpoint,path,'DPO',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
