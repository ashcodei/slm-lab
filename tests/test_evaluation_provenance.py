import pytest
import torch
import json
from dataclasses import asdict
from tokenizers import Tokenizer,models,pre_tokenizers
from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.evaluation import evaluate
from slm_lab.corpus import file_hash
from slm_lab.evaluation_provenance import capture,pixels_sha256,seal


def test_changed_evaluator_or_dataset_prevents_sealing_a_result(tmp_path):
    source=tmp_path/'evaluator.py';source.write_text('original')
    data=tmp_path/'data.jsonl';data.write_text('original data')
    provenance=capture([source]);digest=file_hash(data)
    assert seal({},provenance,files=[source],data=data,digest=digest)['evaluation_provenance']==provenance
    data.write_text('changed data')
    with pytest.raises(ValueError,match='data changed'):seal({},provenance,files=[source],data=data,digest=digest)
    source.write_text('changed implementation')
    with pytest.raises(ValueError,match='implementation/runtime changed'):seal({},provenance,files=[source])


def test_image_fingerprint_tracks_actual_model_input_pixels():
    pixels=torch.zeros(3,4,4);same=pixels.clone();changed=pixels.clone();changed[0,0,0]=1.
    assert pixels_sha256(pixels)==pixels_sha256(same)
    assert pixels_sha256(pixels)!=pixels_sha256(changed)


def test_policy_preference_reports_evaluation_feedback_independently_of_sft_training(tmp_path):
    vocabulary={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,'User':4,':':5,'Assistant':6,'Explain':7,'good':8,'bad':9}
    tokenizer=Tokenizer(models.WordLevel(vocabulary,unk_token='<unk>'));tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    config=ModelConfig(vocab_size=10,hidden_size=32,intermediate_size=64,num_layers=1,num_heads=4,num_kv_heads=2,
                       vision_hidden_size=16,vision_intermediate_size=32,vision_layers=1,vision_heads=2,
                       image_size=16,patch_size=4,pool_size=2)
    model=TinyMultimodalLM(config);paths=[]
    for name in ['candidate','reference']:
        root=tmp_path/name;root.mkdir();tokenizer.save(str(root/'tokenizer.json'))
        path=root/'model.pt';torch.save({'config':asdict(config),'model':model.state_dict(),
            'tokenizer_sha256':file_hash(root/'tokenizer.json'),'feedback_sources':[]},path);paths.append(path)
    data=tmp_path/'preferences.jsonl';data.write_text(json.dumps({'id':'human-pair','prompt':'Explain',
                'chosen':'good','rejected':'bad','feedback_source':'human'})+'\n')
    report=evaluate(paths[0],data,'policy_preference',tmp_path/'result.json',device='cpu',threads=1,reference_checkpoint=paths[1])
    assert report['feedback_sources']==['human'] and report['checkpoint_feedback_sources']==[]
    assert report['examples']==1 and report['ties']==1 and report['mean_reference_adjusted_margin']==0
