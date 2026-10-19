import copy
import json

import pytest
import torch

from slm_lab import shared_word_inference_v1 as inference
from slm_lab.corpus import file_hash
from slm_lab.shared_word_stage_v1 import SharedWordStage
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_shared_word_stage_v1 import model,options,synthetic_words


def test_actual_trained_word_route_roundtrip_and_unused_language_source_budget_corruption_reject(monkeypatch,tmp_path):
    initial=model();data=synthetic_words();setting=options(data,epochs=1,microbatch=1,accumulation=5)
    rate=setting.pop('learning_rate')
    plan=dict(initial_model_plan=dict(own_code_checkpoint_sha256='a'*64),actual_counts=initial.counts(),
        source_admission_sha256='b'*64,source_manifest_sha256='c'*64,initial_parts_digest=tensor_digest(initial.parts.state_dict().items()),
        training_options=setting,arms=dict(rate=dict(learning_rate=rate)),source_exposure_budget=dict(words=5,repeated_rounding_sources=0))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));sha=file_hash(path)
    monkeypatch.setattr(inference,'fresh',lambda p,device='cpu':(model(),dict(synthetic_small_fixture=True)))
    zero=inference.export(initial,plan,sha,'initial',0);zero_path=tmp_path/'initial.pt';torch.save(zero,zero_path)
    restored,_=inference.load(zero_path,path,'initial',expected_checkpoint_sha256=file_hash(zero_path),expected_plan_sha256=sha)
    assert tensor_digest(restored.parts.state_dict().items())==plan['initial_parts_digest']
    optimizer=torch.optim.AdamW(initial.extension_parameters('vision','word'),lr=rate,betas=tuple(setting['betas']),weight_decay=setting['weight_decay'],foreach=False)
    learner=SharedWordStage(initial,optimizer,data,{**setting,'learning_rate':rate},inference.sources(plan,sha));learner.update();initial.eval()
    trained=inference.export(initial,plan,sha,'rate',1);checkpoint=tmp_path/'trained.pt';torch.save(trained,checkpoint)
    cold,_=inference.load(checkpoint,path,'rate',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
    batch=data.batch([0]);ids=batch['input_ids'];pixels=batch['pixel_values']
    with torch.no_grad():
        for route in ('aligned','general','code','grounding'):
            assert torch.equal(initial(ids,pixels,route=route)['logits'],cold(ids,pixels,route=route)['logits'])
        assert torch.equal(initial.word_logits(batch['word_pixels']),cold.word_logits(batch['word_pixels']))
    def unused(s):
        name=next(n for n in s['aligned_model']['extensions'] if n.startswith('language_linears.'))
        s['aligned_model']['extensions'][name].add_(1.)
    for mutate in [unused,lambda s:s.update(actual_shared_word_updates=0),
                   lambda s:s['actual_source_exposure_budget'].update(words=4),
                   lambda s:s['aligned_model']['sources'].update(word_admission_sha256='d'*64)]:
        bad=copy.deepcopy(trained);mutate(bad);torch.save(bad,checkpoint)
        with pytest.raises(ValueError):inference.load(checkpoint,path,'rate',expected_checkpoint_sha256=file_hash(checkpoint),expected_plan_sha256=sha)
