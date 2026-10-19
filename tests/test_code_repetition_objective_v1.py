"""Candidate exclusions, causal alignment and real objective gradient tests."""
import math

import pytest
import torch

from slm_lab.code_repetition_objective_v1 import (repetition_negative_positions,
    sequence_unlikelihood,select_rollout_rows)


def test_only_later_repeated_response_windows_are_negative():
    assert repetition_negative_positions([3,4,5,6,3,4,5,6,2],[7,8,2],n=4)==[7]


def test_exact_original_code_with_repeated_blocks_is_always_exempt():
    code=[3,4,5,6]*5+[2]
    assert repetition_negative_positions(code,code,n=4)==[]


def test_repeated_reference_windows_and_EOS_are_exempt():
    assert repetition_negative_positions([3,4]*8+[2],[3,4]*4+[2],n=4)==[]
    assert repetition_negative_positions([3,4,2],[8,9,2],n=2)==[]


def test_incomplete_reference_or_tokens_after_true_EOS_reject():
    with pytest.raises(ValueError):repetition_negative_positions([3,4],[3,4],n=4)
    with pytest.raises(ValueError):repetition_negative_positions([3,2,4],[3,4,2],n=4)


def test_causal_shift_and_only_selected_row_gets_gradients():
    logits=torch.zeros(1,5,5,requires_grad=True)
    ids=torch.tensor([[1,3,4,3,2]])
    loss=sequence_unlikelihood(logits,ids,2,[1])
    assert math.isclose(float(loss.detach()),-math.log(.8),rel_tol=1e-6)
    loss.backward()
    assert float(logits.grad[0,2,3])>0
    assert bool((logits.grad[0,2,[0,1,2,4]]<0).all())
    assert torch.count_nonzero(logits.grad[0,[0,1,3,4]])==0


def test_near_unit_probability_is_finite_and_has_nonzero_gradient():
    logits=torch.tensor([[[0.,0.,0.],[0.,10000.,0.],[0.,0.,0.]]],requires_grad=True)
    loss=sequence_unlikelihood(logits,torch.tensor([[1,0,1]]),2,[0])
    assert bool(torch.isfinite(loss)) and float(loss.detach())>9000
    loss.backward()
    assert float(logits.grad[0,1,1])>.9 and bool(torch.isfinite(logits.grad).all())


def test_actual_gradient_step_reduces_selected_negative_probability():
    parameter=torch.nn.Parameter(torch.tensor([[[0.,0.,0.],[0.,2.,0.],[0.,0.,0.]]]))
    opt=torch.optim.SGD([parameter],lr=.2)
    before=float(parameter[0,1].softmax(-1)[1].detach())
    sequence_unlikelihood(parameter,torch.tensor([[1,0,1]]),2,[0]).backward();opt.step()
    assert float(parameter[0,1].softmax(-1)[1].detach())<before


@pytest.mark.parametrize('positions',[[0,0],[1,0],[3],[-1]])
def test_invalid_response_positions_reject(positions):
    with pytest.raises(ValueError):sequence_unlikelihood(torch.zeros(1,5,5),torch.tensor([[1,3,4,3,2]]),2,positions)


def test_EOS_loss_negative_rejects_and_empty_mask_has_zero_gradient():
    logits=torch.zeros(1,5,5,requires_grad=True);ids=torch.tensor([[1,3,4,3,2]])
    with pytest.raises(ValueError):sequence_unlikelihood(logits,ids,2,[2])
    loss=sequence_unlikelihood(logits,ids,2,[]);loss.backward()
    assert float(loss)==0 and torch.count_nonzero(logits.grad)==0


def selection():
    train=[dict(id=f't{i}',role='train',language=['C++','Python','D'][i%3],
        group_id=f'g{i}',target_tokens=16,context_tokens=40) for i in range(9)]
    return dict(train=train,validation=[dict(id='v',role='validation',group_id='held')])


def test_rollouts_cover_every_language_deterministically_from_train_only():
    s=selection();a=select_rollout_rows(s,number=6,budget=128)
    assert a==select_rollout_rows(s,number=6,budget=128)
    assert len({r['id'] for r in a})==6 and {r['language'] for r in a}=={'C++','Python','D'}
    assert all(r['role']=='train' for r in a)


@pytest.mark.parametrize('fault',['held_role','overlap','clipped_prompt','budget'])
def test_held_or_incomplete_rollout_schedules_reject(fault):
    s=selection()
    if fault=='held_role':s['train'][0]['role']='validation'
    if fault=='overlap':s['validation'][0]['group_id']='g0'
    if fault=='clipped_prompt':
        for r in s['train']:r['context_tokens']=4096
    if fault=='budget':
        with pytest.raises(ValueError):select_rollout_rows(s,number=2,budget=128)
        return
    with pytest.raises(ValueError):select_rollout_rows(s,number=6,budget=128)
