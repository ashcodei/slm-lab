"""Own-encoder CTC with learned height pooling and local sequence readout."""
import argparse
from dataclasses import asdict
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .character_ctc import ALPHABET,train
from .corpus import file_hash
from .model import ModelConfig,TinyMultimodalLM,VisionEncoder


class LearnedCharacterSequenceModel(nn.Module):
    def __init__(self,vision,frames=56):
        super().__init__();self.vision=vision;self.frames=frames
        self.readout_kind='learned height attention, temporal convolutions and character classifier'
        self.freeze_projector=True;self.vision.project.requires_grad_(False)
        width=vision.config.vision_hidden_size;channels=min(256,width)
        self.height_attention=nn.Linear(width,1)
        self.sequence=nn.Sequential(nn.Conv1d(width,channels,3,padding=1),nn.GELU(),
                                   nn.ConvTranspose1d(channels,channels,4,stride=2,padding=1),nn.GELU(),
                                   nn.Conv1d(channels,channels,3,padding=1),nn.GELU())
        self.classifier=nn.Linear(channels,len(ALPHABET)+1)
        nn.init.zeros_(self.height_attention.weight);nn.init.zeros_(self.height_attention.bias)

    def readout(self,grid):
        if grid.ndim!=4:raise ValueError('Readout requires batch/height/width/features')
        weights=self.height_attention(grid).float().softmax(1).to(grid.dtype)
        columns=(grid*weights).sum(1).transpose(1,2)
        sequence=self.sequence(columns)
        if sequence.shape[-1]!=self.frames:
            sequence=F.interpolate(sequence,size=self.frames,mode='linear',align_corners=False)
        return self.classifier(sequence.transpose(1,2)).float()

    def forward(self,pixels):
        saved=[]
        handle=self.vision.norm.register_forward_hook(lambda module,args,out:saved.append(out))
        try:self.vision(pixels,gradient_checkpointing=self.training and any(p.requires_grad for p in self.vision.parameters()))
        finally:handle.remove()
        if len(saved)!=1:raise ValueError('Expected one unpooled own-vision feature grid')
        values=saved[0];side=math.isqrt(values.shape[1])
        if side*side!=values.shape[1]:raise ValueError('Vision feature grid differs')
        return self.readout(values.reshape(len(values),side,side,values.shape[-1]))


def from_base(checkpoint,frames=56):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    config=ModelConfig(**state['config']);vision=VisionEncoder(config)
    vision.load_state_dict({name.removeprefix('vision.'):value for name,value in state['model'].items() if name.startswith('vision.')})
    model=LearnedCharacterSequenceModel(vision,frames)
    added=sum(p.numel() for name,p in model.named_parameters() if not name.startswith('vision.'))
    with torch.device('meta'):full=TinyMultimodalLM(config)
    total=full.parameter_counts()['total']+added
    if total>=1_000_000_000:raise ValueError('Combined deployment exceeds user parameter limit')
    return model,{'config':asdict(config),'base_checkpoint':str(Path(checkpoint).resolve()),
                  'base_checkpoint_sha256':file_hash(checkpoint),'tokenizer_sha256':state['tokenizer_sha256'],
                  'frames':frames,'readout_kind':model.readout_kind,'new_trainable_parameters':added,
                  'proposed_deployed_parameters':total,'readout_implementation_sha256':file_hash(__file__)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--data',required=True);parser.add_argument('--output',required=True);parser.add_argument('--reports',required=True)
    parser.add_argument('--device',default='cuda:3');parser.add_argument('--passes',type=int,default=3);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.passes<1:parser.error('Positive passes required')
    train(args.checkpoint,args.data,args.output,args.reports,args.device,args.passes,
          learning_rate=3e-4,resume=args.resume,model_factory=from_base,head_warmup_steps=200,encoder_rate_scale=.1)


if __name__=='__main__':main()
