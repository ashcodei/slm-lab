import copy
from contextlib import nullcontext
import torch
from tokenizers import Tokenizer,models,pre_tokenizers,trainers
from slm_lab.model import ModelConfig,TinyMultimodalLM
from slm_lab.training import Encoder,supervised_gradients


def test_token_weighted_accumulation_matches_large_variable_length_batch(tmp_path):
    tok=Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(['Short and longer responses'],trainers.BpeTrainer(vocab_size=260,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['<pad>','<bos>','<eos>','<unk>']))
    c=ModelConfig(vocab_size=260,hidden_size=32,intermediate_size=64,num_layers=1,
        num_heads=4,num_kv_heads=2,max_text_length=96,vision_hidden_size=16,
        vision_intermediate_size=32,vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    encoder=Encoder(tok,c,tmp_path)
    rows=[{'prompt':'Hi','response':text} for text in ['A','Much longer answer','Yes','Another response here']]
    torch.manual_seed(4);full=TinyMultimodalLM(c);accum=copy.deepcopy(full)
    batch=encoder.batch(rows,'response','cpu')
    big_loss=full(**batch)['loss'];big_loss.backward()
    small_loss,count=supervised_gradients(accum,encoder,[rows[:2],rows[2:]],'response','cpu',nullcontext)
    torch.testing.assert_close(big_loss.detach(),small_loss,rtol=1e-6,atol=1e-6)
    assert count==int((batch['labels'][:,1:]!=-100).sum())
    for p,q in zip(full.parameters(),accum.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6)
