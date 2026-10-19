from argparse import Namespace
from dataclasses import asdict
import json
import torch
from tokenizers import Tokenizer,models,pre_tokenizers,trainers
from slm_lab.model import ModelConfig
from slm_lab.training import train,load_checkpoint


def test_resume_restores_optimizer_and_matches_uninterrupted_training(tmp_path):
    tok=Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(['This is a small test.'],trainers.BpeTrainer(vocab_size=260,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['<pad>','<bos>','<eos>','<unk>']))
    tp=tmp_path/'tokenizer.json';tok.save(str(tp))
    config=ModelConfig(vocab_size=260,hidden_size=32,intermediate_size=64,num_layers=1,
        num_heads=4,num_kv_heads=2,max_text_length=32,vision_hidden_size=16,
        vision_intermediate_size=32,vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    cp=tmp_path/'config.json';cp.write_text(json.dumps(asdict(config)))
    data=tmp_path/'data.jsonl';data.write_text(''.join(json.dumps({'text':s})+'\n' for s in ['A short text.','Another document.']))
    args=Namespace(stage='pretrain',config=str(cp),tokenizer=str(tp),data=str(data),
        output=str(tmp_path/'full'),checkpoint=None,reward_checkpoint=None,device='cpu',steps=4,
        batch_size=1,learning_rate=0.001,seed=42,threads=2,beta=0.1,kl_coef=0.05,
        ppo_epochs=2,max_new_tokens=8,log_every=100,gradient_checkpointing=False,
        checkpoint_every=2,keep_checkpoints=2,resume=False)
    train(args)
    final,_=load_checkpoint(tmp_path/'full'/'model.pt')
    args.checkpoint=str(tmp_path/'full'/'step-00000002.pt')
    args.output=str(tmp_path/'resumed');args.resume=True
    train(args)
    resumed,_=load_checkpoint(tmp_path/'resumed'/'model.pt')
    for key,value in final.state_dict().items():
        torch.testing.assert_close(value,resumed.state_dict()[key],rtol=0,atol=0)
