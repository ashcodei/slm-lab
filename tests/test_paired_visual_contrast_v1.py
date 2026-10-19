"""Independent autograd, actual vision-only updates and four-worker normalizers."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
from torch import distributed as dist,nn
from torch.nn import functional as F
import torch.multiprocessing as mp

from slm_lab.codenet_runner_v1 import average_gradients
from slm_lab.finalized_visual_pairs_v1 import FinalizedVisualPairs
from slm_lab.model import TinyMultimodalLM
from slm_lab.paired_visual_contrast_v1 import gradients,pixel_digest
from slm_lab.recurrent_ablation import state_digest
from test_codenet_runner_v1 import small_config

CONTRACT='a'*64


def examples(n=5,offset=0):
    out=[]
    for i in range(offset,offset+n):
        images=[torch.full((3,16,16),.2+i*.03,dtype=torch.float64),torch.full((3,16,16),-.3-i*.01,dtype=torch.float64)]
        bodies=[[11]*(1+i%4)+[2],[12]*(2+i%3)+[2]]
        sequences=[([1,5]+b,[-100,-100]+b) for b in bodies]
        out.append(dict(id='fixture:'+str(i),split='train',source='original_paired_VQA_human_answers',fully_finalized_source=True,
            source_contract_sha256=CONTRACT,human_preferences=False,external_model_teacher_used=False,question='same?',
            answers=['yes','no'],answer_types=['yes/no','yes/no'],human_votes=[10,10] if i%3==0 else [6,8],
            photo_ids=[f'fixture:{i}:a',f'fixture:{i}:b'],pixel_sha256=[pixel_digest(x) for x in images],images=images,sequences=sequences))
    return out


class PicturePolicy(nn.Module):
    def __init__(self,blind=False):
        super().__init__();self.config=replace(small_config(),vocab_size=32);self.blind=blind
        self.bias=nn.Parameter(torch.randn(32,dtype=torch.float64)*.02)
        self.visual=nn.Parameter(torch.randn(32,dtype=torch.float64)*.03)
        self.previous=nn.Parameter(torch.randn(32,dtype=torch.float64)*.01)
    def forward(self,input_ids,labels,pixel_values):
        value=0. if self.blind else pixel_values.mean()
        logits=self.bias[None,None,:]+self.previous[None,None,:]*input_ids[:,:,None]/32
        if not self.blind:logits=logits+self.visual[None,None,:]*value
        return dict(loss=F.cross_entropy(logits[:,:-1].reshape(-1,32),labels[:,1:].reshape(-1),ignore_index=-100))


def direct(model,rows,alpha=.3,beta=2.,margin=.1):
    # Independently retain all four graphs and construct the scalar definition.
    terms=[];denominator=0;contrasts=[]
    for row in rows:
        losses={}
        for answer in range(2):
            ids,labels=row['sequences'][answer];n=sum(y!=-100 for y in labels[1:]);denominator+=n
            for image in range(2):
                losses[answer,image]=model(torch.tensor([ids]),torch.tensor([labels]),row['images'][image].unsqueeze(0))['loss']
            terms.append(n*losses[answer,answer])
        if row['human_votes'][0]>=9 and row['human_votes'][1]>=9:
            contrasts.extend(F.softplus(margin+beta*(losses[a,a]-losses[a,1-a])) for a in range(2))
    return sum(terms)/denominator+(alpha*sum(contrasts)/len(contrasts) if contrasts else 0.)


def call(model,rows,world=1,alpha=.3):
    return gradients(model,rows,contract_sha256=CONTRACT,alpha=alpha,beta=2.,margin=.1,world=world,device='cpu',amp=nullcontext)


def test_streamed_full_answer_gradients_match_independent_FP64_four_graph_objective():
    torch.manual_seed(783);model=PicturePolicy();oracle=deepcopy(model);rows=examples()
    value=direct(oracle,rows);value.backward();actual=call(model,rows)
    assert actual['summed_objective']==pytest.approx(float(value.detach()),abs=1e-12)
    assert actual['global_pair_exposures']==5 and actual['global_eligible_pair_exposures']==2
    assert actual['global_proper_target_tokens']==sum(sum(v!=-100 for v in labels[1:]) for r in rows for _,labels in r['sequences'])
    for a,b in zip(model.parameters(),oracle.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-12,rtol=1e-12)


def test_image_blind_policy_gets_no_contrast_only_language_gradient():
    torch.manual_seed(788);model=PicturePolicy(blind=True);control=deepcopy(model);rows=examples()
    value=call(model,rows);call(control,rows,alpha=0.)
    assert value['weighted_image_contrast']==pytest.approx(.3*float(F.softplus(torch.tensor(.1,dtype=torch.float64))),abs=1e-12)
    for a,b in zip(model.parameters(),control.parameters()):
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,atol=1e-12,rtol=1e-12)


def test_vision_only_update_preserves_every_language_tensor_and_text_only_logits():
    torch.set_num_threads(1);torch.manual_seed(990);model=TinyMultimodalLM(small_config())
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('vision.'))
    before={n:t.clone() for n,t in model.state_dict().items() if not n.startswith('vision.')}
    vision=state_digest(model.vision.state_dict());model.eval();ids=torch.tensor([[1,5,11,2]])
    with torch.no_grad():expected=model(ids)['logits'].clone()
    rows=examples(2)
    for r in rows:
        r['images']=[x.float() for x in r['images']];r['pixel_sha256']=[pixel_digest(x) for x in r['images']]
    model.train();model.gradient_checkpointing=True
    optimizer=torch.optim.SGD(model.vision.parameters(),lr=.01)
    call(model,rows);assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('vision.'))
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.vision.parameters())
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert state_digest(model.vision.state_dict())!=vision
    assert all(torch.equal(t,model.state_dict()[n]) for n,t in before.items())
    model.eval()
    with torch.no_grad():torch.testing.assert_close(model(ids)['logits'],expected,atol=0,rtol=0)


def test_held_partial_wrong_pixels_and_changed_question_reject_before_gradients():
    for mutation in ['held','partial','pixels','question','labels']:
        model=PicturePolicy();rows=examples(1)
        if mutation=='held':rows[0]['split']='valid'
        if mutation=='partial':rows[0]['fully_finalized_source']=False
        if mutation=='pixels':rows[0]['images'][0][0,0,0]+=.1
        if mutation=='question':rows[0]['sequences'][1][0][1]=6
        if mutation=='labels':rows[0]['sequences'][0][1][0]=-100.
        with pytest.raises(ValueError):call(model,rows)
        assert all(p.grad is None for p in model.parameters())
    with pytest.raises(ValueError,match='Explicit standalone'):
        call(PicturePolicy(),examples(1),world=True)


def test_zero_eligible_pairs_preserve_all_original_CE_and_unfinalized_source_never_reads_partial_rows(tmp_path):
    rows=examples(2,offset=1);model=PicturePolicy();actual=call(model,rows)
    assert actual['global_eligible_pair_exposures']==0 and actual['weighted_image_contrast']==0.
    assert actual['global_pair_exposures']==2
    data=tmp_path/'data/vqa-real-pairs-v1';data.mkdir(parents=True)
    (data/'images.collected.jsonl').write_text('not even parsed as a training row')
    with pytest.raises(ValueError,match='Finalized original real-photo pair manifest'):
        FinalizedVisualPairs(tmp_path,None,None)
    assert not (data/'train.jsonl').exists()


def worker(rank,rendezvous,folder):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,timeout=timedelta(seconds=120))
    try:
        torch.manual_seed(772);model=PicturePolicy();oracle=deepcopy(model)
        populations=[examples(r+1,offset=r*(r+1)//2) for r in range(4)]
        rows=[x for group in populations for x in group];value=direct(oracle,rows);value.backward()
        actual=call(model,populations[rank],world=4);average_gradients(model,'cpu',4)
        assert actual['global_pair_exposures']==10 and actual['global_eligible_pair_exposures']==4
        assert actual['summed_objective']==pytest.approx(float(value.detach()),abs=1e-12)
        for a,b in zip(model.parameters(),oracle.parameters()):torch.testing.assert_close(a.grad,b.grad,atol=1e-12,rtol=1e-12)
        model.zero_grad(set_to_none=True)
        if rank==2:populations[rank][0]['split']='valid'
        with pytest.raises(ValueError,match='Collective preflight failed'):call(model,populations[rank],world=4)
        assert all(p.grad is None for p in model.parameters())
        Path(folder,f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,actual_four_CPU_workers=True,
            unequal_pair_populations=[1,2,3,4],global_pairs=10,global_eligible_pairs=4,
            global_CE_and_eligible_contrast_gradients_match_independent_FP64_autograd=True,
            one_rank_held_source_rejects_before_any_gradient=True,scope='Artificial image/source fixtures and own small policies. Not finalized real source admission, full706M GPU training or capability.')))
    finally:dist.destroy_process_group()


def test_four_actual_workers_match_global_normalizers_and_reject_one_rank_held_role(tmp_path):
    mp.spawn(worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=4,join=True)
    for rank in range(4):assert json.loads((tmp_path/f'rank-{rank}.json').read_text())['global_CE_and_eligible_contrast_gradients_match_independent_FP64_autograd']
