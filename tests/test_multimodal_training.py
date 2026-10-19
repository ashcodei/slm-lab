from argparse import Namespace
from dataclasses import asdict
import json
from PIL import Image
import numpy as np
import torch
from tokenizers import Tokenizer,models,pre_tokenizers,trainers
from slm_lab.corpus import file_hash
from slm_lab.training import train
from test_model import tiny


def setup(tmp_path):
    tok=Tokenizer(models.BPE(unk_token='<unk>'));tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.train_from_iterator(['Describe a square image. A red square.'],trainers.BpeTrainer(vocab_size=260,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=['<pad>','<bos>','<eos>','<unk>']))
    tp=tmp_path/'tokenizer.json';tok.save(str(tp))
    model=tiny();cp=tmp_path/'config.json';cp.write_text(json.dumps(asdict(model.config)))
    checkpoint=tmp_path/'initial.pt';torch.save({'model':model.state_dict(),'config':asdict(model.config),
        'tokenizer_sha256':file_hash(tp),'stage':'pretrain'},checkpoint)
    Image.fromarray(np.full((16,16,3),[200,50,50],dtype=np.uint8)).save(tmp_path/'image.png')
    args=Namespace(stage='sft',config=str(cp),tokenizer=str(tp),data=str(tmp_path/'data.jsonl'),
        output=str(tmp_path/'trained'),checkpoint=str(checkpoint),reward_checkpoint=None,device='cpu',steps=3,
        batch_size=2,learning_rate=0.001,seed=42,threads=2,beta=0.1,kl_coef=0.05,
        ppo_epochs=2,max_new_tokens=8,log_every=100,gradient_checkpointing=True,
        checkpoint_every=0,keep_checkpoints=1,resume=False,trainable='all')
    return model,args


def test_vision_warmup_changes_only_the_vision_encoder(tmp_path):
    original,args=setup(tmp_path);args.trainable='vision'
    rows=[{'prompt':'Describe the image.','response':'A red square.','image':'image.png'}]*4
    (tmp_path/'data.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));train(args)
    trained=torch.load(tmp_path/'trained/model.pt',weights_only=True)['model']
    for name,value in original.state_dict().items():
        if not name.startswith('vision.'):torch.testing.assert_close(value,trained[name],rtol=0,atol=0)
    assert not torch.equal(trained['vision.patch.weight'],original.vision.patch.weight)
    report=json.loads((tmp_path/'trained/training.json').read_text())
    assert report['trainable_parameters']==sum(p.numel() for p in original.vision.parameters())


def test_joint_sft_uses_homogeneous_minibatches_for_mixed_modalities(tmp_path):
    _,args=setup(tmp_path);args.steps=6
    rows=[{'prompt':'Describe the image.','response':'A red square.','image':'image.png'}]*4
    rows+=[{'prompt':'Add two and three.','response':'Five.'}]*4
    (tmp_path/'data.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));train(args)
    report=json.loads((tmp_path/'trained/training.json').read_text())
    assert report['steps']==6 and all(m['global_target_tokens']>0 for m in report['metrics'])


def test_contrastive_vision_warmup_preserves_language_parameters(tmp_path):
    original,args=setup(tmp_path);args.stage='contrastive';args.trainable='vision';args.batch_size=4
    Image.fromarray(np.full((16,16,3),[50,50,200],dtype=np.uint8)).save(tmp_path/'blue.png')
    rows=[{'response':'A red image.','image':'image.png'}]*2
    rows+=[{'response':'A blue image.','image':'blue.png'}]*2
    (tmp_path/'data.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));train(args)
    trained=torch.load(tmp_path/'trained/model.pt',weights_only=True)['model']
    for name,value in original.state_dict().items():
        if not name.startswith('vision.'):torch.testing.assert_close(value,trained[name],rtol=0,atol=0)
    assert not torch.equal(trained['vision.patch.weight'],original.vision.patch.weight)
    report=json.loads((tmp_path/'trained/training.json').read_text())
    assert report['stage']=='contrastive' and report['metrics'][-1]['global_images']==4
