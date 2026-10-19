import shutil
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from slm_lab.corpus import file_hash
from slm_lab.deterministic_positions_v1 import install
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.own_dense_standalone_v1 import export,load


def fixture(tmp_path):
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
    token_path=root/'data/tokenizer-corpus/tokenizer.json';t=Tokenizer.from_file(str(token_path))
    c=ModelConfig(vocab_size=t.get_vocab_size(),hidden_size=32,intermediate_size=64,num_layers=2,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    model=install(OwnDenseAssistant(new_backbone(c,975119),reward_hidden=8,word_temporal=4,pointing_rank=8,router_hidden=8,detector_hidden=4))
    ids=torch.tensor([[1,5,7,8,2]]);optimizer=torch.optim.AdamW([p for _,p in model.language_named_parameters()],lr=.001)
    model(ids,labels=ids)['loss'].backward();optimizer.step();optimizer.zero_grad(set_to_none=True);model.eval()
    shutil.copyfile(token_path,tmp_path/'tokenizer.json');sources=dict(own_complete_source_sha256='a'*64)
    path=tmp_path/'model.pt';torch.save(export(model,sources=sources,provenance=dict(kind='plain_dense_no_actor_fold'),
        completed_updates=1,tokenizer_sha256=file_hash(token_path)),path)
    return model,ids,path,sources


def test_actual_learned_complete_text_image_word_and_every_parameter_cold_load_without_ancestor_or_RNG_change(tmp_path):
    model,ids,path,sources=fixture(tmp_path);rng=torch.get_rng_state().clone()
    cold,metadata=load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources)
    assert torch.equal(rng,torch.get_rng_state()) and metadata['no_old_base_actor_optimizer_or_training_reference_loaded']
    assert cold.network.embedding.weight is cold.network.lm_head.weight
    assert not ({p.data_ptr() for p in cold.parameters()}&{p.data_ptr() for p in model.parameters()})
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),cold.parameters()))
    pixels=torch.linspace(-1,1,3*16*16).reshape(1,3,16,16);word=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32)
    with torch.no_grad():
        for image in [None,pixels]:assert torch.equal(model(ids,image)['logits'],cold(ids,image)['logits'])
        assert torch.equal(model.word_logits(word),cold.word_logits(word))


def test_wrong_complete_source_or_packaged_tokenizer_rejected(tmp_path):
    model,ids,path,sources=fixture(tmp_path)
    with pytest.raises(ValueError):load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=dict(own_complete_source_sha256='b'*64))
    (tmp_path/'tokenizer.json').write_text('{}')
    with pytest.raises(ValueError,match='tokenizer'):load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=sources)
