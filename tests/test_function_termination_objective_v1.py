import copy

import pytest
import torch
from torch.nn import functional as F

from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
from slm_lab.function_termination_objective_v1 import backward
from test_fast_joint_v1 import batch,options
from test_own_dense_growth_v1 import model as make_model


@pytest.mark.parametrize('weight',[0.,.05])
def test_whole_response_and_true_EOS_gradients_match_independent_scalar_objective(weight):
    model,_ = make_model();oracle = copy.deepcopy(model)
    make_optimizer(model,options());make_optimizer(oracle,options());b = batch()
    actual = backward(model,b,global_targets=6,global_examples=2,EOS_weight=weight)
    logits = oracle(input_ids=b['input_ids'],attention_mask=b['attention_mask'])['logits'][:,:-1].float()
    target = b['labels'][:,1:];mask = target != -100
    last = b['attention_mask'].sum(1)-2;rows = torch.arange(2)
    objective = F.cross_entropy(logits[mask],target[mask],reduction='sum')/6
    objective = objective+weight*F.cross_entropy(logits[rows,last],target[rows,last],reduction='sum')/2
    objective.backward(inputs=[p for _,p in parameter_groups(oracle)['language']])
    assert actual['complete_response_EOS_targets'] == 6 and actual['complete_examples'] == 2
    for (name,p),(_,q) in zip(model.named_parameters(),oracle.named_parameters()):
        if p.grad is None:assert q.grad is None
        else:assert torch.allclose(p.grad,q.grad,atol=3e-7,rtol=3e-5),name


def test_dropped_or_replaced_original_EOS_is_rejected_before_any_model_gradient():
    model,_ = make_model();make_optimizer(model,options())
    for edit in ['mask','id','duplicate','nonEOS_target']:
        b = batch()
        if edit == 'mask':b['labels'][0,-1] = -100
        elif edit == 'id':b['input_ids'][0,-1] = 4
        elif edit == 'duplicate':b['labels'][0,2] = 2
        else:b['labels'][0,2] = 11
        with pytest.raises(ValueError,match='exactly its original final EOS'):
            backward(model,b,global_targets=6,global_examples=2,EOS_weight=.05)
        assert all(p.grad is None for p in model.parameters())
