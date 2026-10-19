"""Actual own-neural source/cursor/Adam/RNG recovery and rejection controls."""
import copy
from contextlib import nullcontext

import pytest
import torch

from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
from slm_lab.public_reasoning_stage_v1 import PublicReasoningStage
from slm_lab.public_reasoning_train_data_v1 import PublicReasoningTrainData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest


@pytest.fixture(scope='module')
def data():
    return PublicReasoningTrainData('data/public-reasoning-curriculum-v1',
        'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')


def model():
    config=ModelConfig(vocab_size=32768,hidden_size=8,intermediate_size=16,num_layers=1,num_heads=2,num_kv_heads=1,
        max_text_length=4096,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    return OwnDenseActor(OwnDenseAssistant(new_backbone(config,983113),pointing_rank=4,reward_hidden=4,word_temporal=4,
        router_hidden=4,detector_hidden=4,word_expansion=1),rank=2).eval()


def setup(data,profile='whole_answer_warmup',precision='CPU_FP32',entries=None,mode='capacity'):
    student=model();optimizer=torch.optim.AdamW([p for _,p in student.named_selected(('actor',))],lr=1e-4,
        betas=(.9,.95),weight_decay=.01,foreach=False)
    options=dict(precision=precision,learning_rate=1e-4,gradient_clip=1.,epochs=3,width=16,seed=983113)
    if profile!='whole_answer_warmup':options.update(epochs=1,width=1,group_size=2,KL_weight=.01,anchor_weight=1.,
        cache_absolute_bound=.001,CPU_mechanics_horizon=2 if mode=='capacity' else None)
    cache=CPUFrozenCastCache() if precision=='CPU_BF16_cached' else None
    if cache:cache.attach(student.policy)
    if entries is None:
        positions=sorted(data.slots,key=lambda p:data.information(p)['complete_context_tokens'])[:2]
        entries=[[position] for position in positions]
    stage=PublicReasoningStage(student,optimizer,data,entries,profile,options,dict(experiment_sha256='d'*64),mode=mode,cache=cache)
    return student,optimizer,stage,cache


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a) is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


@pytest.mark.parametrize(('profile','precision'),[('whole_answer_warmup','CPU_FP32'),('whole_answer_warmup','CPU_BF16_cached'),
    ('anchor_CE','CPU_FP32'),('anchor_RL','CPU_FP32')])
def test_actual_neural_stage_private_recovery_and_cold_outputs_and_bad_source_rejections(data,profile,precision):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(983217)
    student,optimizer,stage,cache=setup(data,profile,precision)
    fixed=tensor_digest(student.base.named_parameters());stage.update();saved=copy.deepcopy(stage.state())
    wanted=stage.update();expected=tensor_digest(student.parts.named_parameters());adam=copy.deepcopy(optimizer.state_dict());rng=capture_rng('cpu')
    for corruption in ['source_mask','cursor','options','RNG','Adam']:
        bad=copy.deepcopy(saved)
        if corruption=='source_mask':
            record=bad['core']['metrics'][0]
            (record if profile=='whole_answer_warmup' else record['anchor'])['complete_sources'][0]['response_EOS_targets']-=1
        elif corruption=='cursor':bad['core']['cursor']['actual_complete_source_batches'][0][0]=-1
        elif corruption=='options':bad['stage_binding']['options']['gradient_clip']=.5
        elif corruption=='RNG':bad['core']['rng']['torch']=torch.zeros(1,dtype=torch.uint8)
        else:bad['core']['optimizer']['state'][0]['exp_avg'].view(-1)[0]=float('nan')
        with pytest.raises(ValueError):stage.resume(bad)
        assert tensor_digest(student.parts.named_parameters())==expected and equal(optimizer.state_dict(),adam) and equal(capture_rng('cpu'),rng)
    if profile!='whole_answer_warmup':
        for corruption in ['reward','action','behavior']:
            bad=copy.deepcopy(saved);trajectory=bad['core']['metrics'][0]['trajectories'][0]
            if corruption=='reward':trajectory['actual_verifier']['reward']=1.-trajectory['actual_verifier']['reward']
            elif corruption=='action':trajectory['response_ids']=[2,2]
            else:trajectory['actual_full_behavior_logps'][0]-=1.
            with pytest.raises(ValueError):stage.resume(bad)
            assert tensor_digest(student.parts.named_parameters())==expected and equal(optimizer.state_dict(),adam) and equal(capture_rng('cpu'),rng)
    stage.resume(saved);assert stage.completed==1
    assert stage.update()==wanted and tensor_digest(student.parts.named_parameters())==expected
    assert equal(optimizer.state_dict(),adam) and equal(capture_rng('cpu'),rng)
    final=stage.state()
    with pytest.raises(ValueError):stage.resume(final)
    stage.resume(final,allow_completed=True)
    with pytest.raises(ValueError):stage.update()
    cold=OwnDenseActor(student.base,**student.settings).eval();cold.import_export(student.export(stage.sources),expected_sources=stage.sources)
    if cache:cache.attach(cold.policy)
    batch=data.batch(stage.entries[0][0]);batch.pop('labels');student.eval()
    amp=torch.autocast('cpu',dtype=torch.bfloat16) if cache else nullcontext()
    with torch.no_grad(),amp:assert torch.equal(student(**batch)['logits'],cold(**batch)['logits'])
    assert tensor_digest(student.base.named_parameters())==fixed and all(p.grad is None for p in student.parameters())
    if cache:cache.close()


def test_production_rejects_small_graph_and_reduced_source_schedules_and_live_option_mutation(data):
    for entries in [[[0]],data.complete_schedule()]:
        with pytest.raises(ValueError,match='Production requires'):setup(data,entries=entries,mode='train')
    student,optimizer,stage,_=setup(data)
    before=tensor_digest(student.parts.named_parameters());rng=capture_rng('cpu')
    stage.options['gradient_clip']=.5
    with pytest.raises(ValueError,match='unchanged'):stage.update()
    assert tensor_digest(student.parts.named_parameters())==before and equal(capture_rng('cpu'),rng) and not optimizer.state


def test_failed_partial_source_update_is_terminal_without_committing_parameters_or_export(data,monkeypatch):
    student,optimizer,stage,_=setup(data,entries=[[0,1],[2]])
    before=tensor_digest(student.parts.named_parameters());original=data.batch
    def fail_later(position,device='cpu'):
        if position==1:raise RuntimeError('injected complete-source failure')
        return original(position,device)
    with monkeypatch.context() as context:
        context.setattr(data,'batch',fail_later)
        with pytest.raises(RuntimeError,match='injected'):stage.update()
    assert stage.failed and stage.completed==0 and not optimizer.state and stage.metrics==[]
    assert tensor_digest(student.parts.named_parameters())==before and all(p.grad is None for p in student.parameters())
    with pytest.raises(ValueError):stage.update()
    with pytest.raises(ValueError):stage.state()
