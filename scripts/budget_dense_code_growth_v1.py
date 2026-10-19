"""Reproduce prospective dense-growth counts on meta tensors only.

This is a parameter/design audit. It does not fold weights, instantiate a real
trained network, establish image equivalence or prove any capability gain.
"""
import json
from pathlib import Path

import torch
from torch import nn

from slm_lab.model import ModelConfig, TinyMultimodalLM, RMSNorm
from slm_lab.own_aligned_assistant_v1 import ByteWordHead
from slm_lab.own_aligned_assistant_v2 import ExpandedByteWordHead

ROOT=Path(__file__).resolve().parents[1]


def counts():
    base=json.loads((ROOT/'configs/model.json').read_text());result={}
    for patch in (16,8):
        config=ModelConfig(**{**base,'num_layers':35,'patch_size':patch})
        with torch.device('meta'):
            modules=dict(dense_text_image_critic=TinyMultimodalLM(config),
                reward=nn.Sequential(RMSNorm(1536,1e-6),nn.Linear(1536,128,bias=False),nn.SiLU(),nn.Linear(128,1,bias=False)),
                word=ExpandedByteWordHead(ByteWordHead(512,1e-6),5),
                pointing=nn.ModuleList([nn.Linear(1536,256,bias=False) for _ in range(2)]),
                router=nn.Sequential(nn.Linear(1536,64),nn.SiLU(),nn.Linear(64,3)),
                detector=nn.Sequential(nn.Conv2d(512,512,3,padding=1,groups=512),nn.Conv2d(512,64,1),nn.SiLU(),nn.Conv2d(64,5,1)))
        inventory={name:sum(p.numel() for p in module.parameters()) for name,module in modules.items()}
        total=sum(inventory.values())
        if total>=10**9:raise ValueError('Every distinct proposed neural head must fit under1B')
        result[str(patch)]=dict(meta_device_actual_parameter_inventory=inventory,total_unique_parameters=total,
            unallocated_parameter_budget=10**9-total,native448_image_prefix_tokens=config.image_token_count(448),
            word448_CTC_frames=448//patch*5)
    return result


if __name__=='__main__':
    actual=counts();report=json.loads((ROOT/'reports/dense-code-fold-and-depth-growth-parameter-budget-v1.json').read_text())
    if actual!=report['variant_parameter_budgets']:raise ValueError('Prospective complete parameter inventory no longer reproduces')
    print(json.dumps(actual,indent=2))
