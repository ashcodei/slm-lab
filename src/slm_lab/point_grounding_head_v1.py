"""Own frozen-language queries with a trainable vision/attention alignment head.

Adds exactly2*hidden_size*rank parameters. Human pointer targets describe
annotator attention, not exact object boxes. No external encoder or teacher.
This module does not load data, choose a policy or start a training run.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .model import TinyMultimodalLM


class PointGroundingHead(nn.Module):
    def __init__(self,base,rank=256):
        super().__init__()
        if type(base) is not TinyMultimodalLM or type(rank) is not int or not 1<=rank<=base.config.hidden_size:
            raise ValueError('Own exact multimodal architecture and a bounded explicit head rank required')
        self.base=base;self.rank=rank
        for parameter in base.parameters():parameter.requires_grad_(False)
        for parameter in base.vision.parameters():parameter.requires_grad_(True)
        self.query_projection=nn.Linear(base.config.hidden_size,rank,bias=False)
        self.patch_projection=nn.Linear(base.config.hidden_size,rank,bias=False)
        self.base.gradient_checkpointing=True
        self.extra_parameters=2*base.config.hidden_size*rank

    def query_features(self,input_ids,attention_mask=None):
        if input_ids.ndim!=2 or input_ids.shape[1]<3 or input_ids.shape[1]>self.base.config.max_text_length:
            raise ValueError('Whole bounded original text-query tokens required')
        mask=torch.ones_like(input_ids,dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if mask.shape!=input_ids.shape:raise ValueError('Exact query token mask shape required')
        # Native own BPE: pad0, BOS1, EOS2. The content pool excludes framing
        # tokens, keeping all actual phrase tokens including punctuation.
        content=mask&(input_ids!=0)&(input_ids!=1)&(input_ids!=2)
        if not bool(content.any(1).all()):raise ValueError('Every original query needs actual content tokens')
        mode=self.base.training
        try:
            self.base.eval()
            with torch.no_grad():
                hidden=self.base(input_ids,attention_mask=mask,return_logits=False,return_hidden=True)['hidden']
                result=(hidden.float()*content.unsqueeze(-1)).sum(1)/content.sum(1,keepdim=True)
        finally:self.base.train(mode)
        return result.detach()

    def forward(self,input_ids,pixel_values,attention_mask=None,targets=None,valid_cells=None):
        if (pixel_values.ndim!=4 or pixel_values.shape[0]!=input_ids.shape[0] or pixel_values.shape[1]!=3
                or pixel_values.shape[2]!=pixel_values.shape[3]):
            raise ValueError('One exact square RGB image per complete original query required')
        cells=self.base.config.image_token_count(pixel_values.shape[-1])
        if valid_cells is None:
            valid_cells=torch.ones(input_ids.shape[0],cells,dtype=torch.bool,device=input_ids.device)
        if (valid_cells.shape!=(input_ids.shape[0],cells) or valid_cells.dtype!=torch.bool
                or not bool(valid_cells.any(1).all())):
            raise ValueError('Every original image needs its explicit nonempty valid patch mask')
        probabilities=None
        if targets is not None:
            if (targets.shape!=valid_cells.shape or not bool(torch.isfinite(targets).all())
                    or bool((targets<0).any()) or not bool((targets.sum(1)>0).all())
                    or bool((targets.masked_select(~valid_cells)!=0).any())):
                raise ValueError('Complete finite original attention counts on valid image cells required')
            probabilities=targets.float()/targets.float().sum(1,keepdim=True)
        query=self.query_features(input_ids,attention_mask)
        patches=self.base.vision(pixel_values,gradient_checkpointing=self.training)
        if patches.shape[1]!=cells:raise ValueError('Actual own vision token layout differs from the original label grid')
        q=self.query_projection(query.to(self.query_projection.weight.dtype))
        k=self.patch_projection(patches)
        logits=(k.float()*q.float().unsqueeze(1)).sum(-1)/math.sqrt(self.rank)
        logits=logits.masked_fill(~valid_cells,torch.finfo(logits.dtype).min)
        result=dict(logits=logits,predicted_cell=logits.argmax(1),valid_cells=valid_cells)
        if probabilities is not None:
            result['loss']=-(probabilities*F.log_softmax(logits,dim=-1)).sum(1).mean()
        return result

    def trainable_inventory(self):
        names=[name for name,p in self.named_parameters() if p.requires_grad]
        if any(not name.startswith(('base.vision.','query_projection.','patch_projection.')) for name in names):
            raise ValueError('Original language/critic must remain frozen')
        return dict(total_unique_parameters=sum(p.numel() for p in self.parameters()),
            actual_trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
            actual_additional_head_parameters=sum(p.numel() for module in [self.query_projection,self.patch_projection] for p in module.parameters()),
            trainable_parameter_names=names)
