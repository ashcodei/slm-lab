from dataclasses import asdict
import json

import pytest
import torch
from tokenizers import Tokenizer,models

from slm_lab.corpus import file_hash
from slm_lab.own_code_delta_inference_v1 import load
from slm_lab.own_code_delta_v1 import OwnCodeDelta
from test_own_code_delta_v1 import own_models,components


def test_reference_bound_reload_uses_one_own_base_and_exact_both_route_logits(tmp_path):
    general,target=own_models()
    tokenizer=Tokenizer(models.WordLevel({'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,**{str(i):i for i in range(4,32)}},unk_token='<unk>'))
    token=tmp_path/'tokenizer.json';tokenizer.save(str(token));base=tmp_path/'base.pt';target_path=tmp_path/'own-target.pt'
    for path,model in [(base,general),(target_path,target)]:
        torch.save(dict(model=model.state_dict(),config=asdict(model.config),tokenizer_sha256=file_hash(token)),path)
    plan=tmp_path/'source-plan.json';plan.write_text(json.dumps(dict(own_base_sha256=file_hash(base),own_target_sha256=file_hash(target_path))))
    parts=components(general,target,2);original=OwnCodeDelta(general,parts,reserved_parameters=256);original.eval();counts=original.counts()
    payload=dict(format='own-conditional-code-delta-v1',components=parts,config=asdict(general.config),
        base_checkpoint_sha256=file_hash(base),own_code_source_checkpoint_sha256=file_hash(target_path),
        tokenizer_sha256=file_hash(token),source_plan_sha256=file_hash(plan),reserved_point_head_parameters=256,
        base_parameters=counts['frozen_general'],additional_code_parameters=counts['trainable_code'],
        total_unique_parameters=counts['total_unique'],total_with_reserved_point_head=counts['total_with_reserved_head'],
        candidate_promoted=False,goal_achieved=False)
    checkpoint=tmp_path/'code-components.pt';torch.save(payload,checkpoint)
    args=dict(expected_checkpoint_sha256=file_hash(checkpoint),expected_base_sha256=file_hash(base),
        expected_target_sha256=file_hash(target_path),expected_plan_sha256=file_hash(plan))
    loaded,metadata=load(checkpoint,base,token,plan,**args)
    ids=torch.tensor([[1,4,5,2]]);pixels=torch.randn(1,3,16,16)
    with torch.no_grad():
        for route in ['general','code']:
            assert torch.equal(original(ids,pixels,route=route)['logits'],loaded(ids,pixels,route=route)['logits'])
            assert torch.equal(original.generate(ids,pixels,max_new_tokens=4,route=route),loaded.generate(ids,pixels,max_new_tokens=4,route=route))
    assert metadata['model_counts']==counts and not metadata['full_code_source_loaded_at_inference']
    assert sum(p.numel() for p in loaded.code.parameters())==counts['total_unique']
    with pytest.raises(ValueError,match='original provenance'):
        load(checkpoint,base,token,plan,**{**args,'expected_target_sha256':'0'*64})
