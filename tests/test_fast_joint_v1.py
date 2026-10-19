import copy
from types import SimpleNamespace
import math

import pytest
import torch
from torch.nn import functional as F

from slm_lab.fast_joint_data_v1 import CompleteBucketSchedule,collate
from slm_lab.fast_joint_stage_v1 import parameter_groups,make_optimizer,backward_batch,update
from slm_lab.shared_word_stage_v1 import ctc_word_sum
from test_own_dense_growth_v1 import model as make_model


def options():
    return dict(learning_rates=dict(language=.001,vision=.002,word=.003),betas=[.9,.95],weight_decay=.01,
        gradient_clip=1.,warmup_fraction=.1,cosine_floor=.1,maximum_microbatch=2,maximum_microbatch_tokens=128,
        ctc_weight=1.,word_native_CE_weight=1.,gradient_bucket_values=128,image_size=32)


def batch(image=False,word=False):
    b=dict(input_ids=torch.tensor([[1,4,5,6,2],[1,5,7,2,0]]),labels=torch.tensor([[-100,-100,5,6,2],[-100,5,7,2,-100]]),
        attention_mask=torch.tensor([[1,1,1,1,1],[1,1,1,1,0]],dtype=torch.bool))
    if image:b['pixel_values']=torch.linspace(-1,1,2*3*32*32).reshape(2,3,32,32)
    if word:b.update(word_pixels=b['pixel_values'],texts=['ab','xy'])
    return b


def test_length_bucket_curriculum_covers_every_original_once_and_distributed_last_partial_never_repeats():
    lengths={'code':[3+i*7 for i in range(67)],'words':[4+i for i in range(19)]};widths=dict(code=8,words=12)
    s=CompleteBucketSchedule(lengths,widths,51881,bucket_updates=8);again=CompleteBucketSchedule(lengths,widths,51881,bucket_updates=8)
    assert s.entries==again.entries
    seen={n:[] for n in lengths}
    for step in range(len(s)):
        name,positions=s.at(step);seen[name]+=positions
        distributed=sum([s.rank_positions(step,rank,4)[1] for rank in range(4)],[])
        assert sorted(distributed)==sorted(positions) and len(distributed)==len(set(distributed))
    assert all(sorted(seen[n])==list(range(len(lengths[n]))) for n in lengths)
    assert len(s)==math.ceil(67/8)+math.ceil(19/12)
    with pytest.raises(IndexError):s.at(len(s))


@pytest.mark.parametrize('word',[False,True])
def test_global_target_and_word_normalization_match_independent_full_loss_gradients(word):
    m,_=make_model();oracle=copy.deepcopy(m);setting=options();make_optimizer(m,setting);make_optimizer(oracle,setting)
    b=batch(image=word,word=word);parameters=[p for g in parameter_groups(m).values() for _,p in g]
    actual=backward_batch(m,b,global_targets=6,global_words=2 if word else 0,word=word)
    inputs={k:v for k,v in b.items() if k in ('input_ids','attention_mask','pixel_values')}
    logits=oracle(**inputs)['logits'][:,:-1].float();target=b['labels'][:,1:];mask=target!=-100
    loss=F.cross_entropy(logits[mask],target[mask],reduction='sum')/6
    if word:loss=loss+ctc_word_sum(oracle.word_logits(b['word_pixels']),b['texts'])[0]/2
    loss.backward(inputs=[p for g in parameter_groups(oracle).values() for _,p in g])
    assert actual['response_EOS_targets']==6
    for (name,p),(_,q) in zip(m.named_parameters(),oracle.named_parameters()):
        if p.grad is None:assert q.grad is None
        else:assert torch.allclose(p.grad,q.grad,atol=3e-7,rtol=3e-5),name


def test_full_dense_text_update_changes_language_while_preserving_all_vision_word_and_other_heads():
    m,_=make_model();setting=options();opt=make_optimizer(m,setting);b=batch()
    d=SimpleNamespace(role='train',purpose='optimization',kinds={'text':'text'},lengths={'text':[5,4]},targets={'text':[3,3]})
    d.batch=lambda name,positions,device:{k:v[positions].to(device) for k,v in b.items()}
    s=CompleteBucketSchedule(d.lengths,dict(text=2),3)
    before={n:p.detach().clone() for n,p in m.named_parameters()}
    record=update(m,opt,d,s,0,setting)
    assert record['global_examples']==2 and record['response_EOS_targets']==6
    assert any(not torch.equal(p,before[n]) for n,p in parameter_groups(m)['language'])
    assert all(torch.equal(p,before[n]) for n,p in m.named_parameters() if not n.startswith('network.') or n.startswith(('network.vision.','network.value_head.')))
    d.role='valid'
    with pytest.raises(ValueError,match='Held'):update(m,opt,d,s,0,setting)


def test_collation_keeps_complete_masks_and_does_not_mistake_none_pixels_for_an_image():
    b=batch();single=[{k:v[i:i+1,:4+i] for k,v in b.items()}|{'pixel_values':None} for i in range(2)]
    joined=collate(single)
    assert joined['input_ids'].shape==(2,5) and 'pixel_values' not in joined
    assert sum(int((x['labels']!=-100).sum()) for x in single)==int((joined['labels']!=-100).sum())
    with pytest.raises(ValueError,match='homogeneous'):collate([single[0],batch(image=True)])
