import json
import subprocess
import sys
import torch


def test_distributed_contrastive_gradient_matches_the_global_batch(tmp_path):
    script=tmp_path/'worker.py'
    script.write_text('''import os,sys,torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.nn.functional import all_gather
from slm_lab.objectives import contrastive_image_text_loss

torch.set_num_threads(1);torch.manual_seed(80)
dist.init_process_group('gloo')
rank=dist.get_rank();layer=torch.nn.Linear(3,2,bias=False);model=DistributedDataParallel(layer)
inputs=torch.tensor([[1.,.2,.3],[1.2,.1,.3],[.1,1.1,.2],[.3,1.2,.1]])
target=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
ids=torch.tensor([1,1,2,2]);local=model(inputs[rank*2:(rank+1)*2])
features=torch.cat(all_gather(local))
loss=contrastive_image_text_loss(features,target,ids,.15);loss.backward()
if rank==0:
 torch.save({'gradient':layer.weight.grad,'weight':layer.weight.detach()},sys.argv[1])
dist.destroy_process_group()
''')
    output=tmp_path/'gradient.pt'
    completed=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node','2',str(script),str(output)],
                   capture_output=True,text=True,timeout=40)
    assert completed.returncode==0,completed.stdout+'\n'+completed.stderr
    from slm_lab.objectives import contrastive_image_text_loss
    saved=torch.load(output,weights_only=True);layer=torch.nn.Linear(3,2,bias=False)
    with torch.no_grad():layer.weight.copy_(saved['weight'])
    inputs=torch.tensor([[1.,.2,.3],[1.2,.1,.3],[.1,1.1,.2],[.3,1.2,.1]])
    targets=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
    contrastive_image_text_loss(layer(inputs),targets,torch.tensor([1,1,2,2]),.15).backward()
    torch.testing.assert_close(layer.weight.grad,saved['gradient'],rtol=2e-5,atol=2e-6)
