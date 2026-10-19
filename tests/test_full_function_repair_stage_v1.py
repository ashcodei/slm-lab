"""Small neural fixtures test mathematics; production capacity is separate."""
import copy
import numpy as np
import pytest
import torch

from slm_lab.code_target_regions_v1 import weighted_CE
from slm_lab.function_repair_curriculum_v1 import FunctionRepairCurriculum,collate
from slm_lab.full_function_repair_stage_v1 import CodeWeights,FunctionRepairStage,gradients
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_actor_v1 import actor
from test_own_alignment_checkpoint_v1 import assert_nested_equal


def fixture():
    torch.set_num_threads(1);model=actor()
    data=FunctionRepairCurriculum.__new__(FunctionRepairCurriculum)
    data.role='train';data.purpose='optimization';data.source_files_sha256={'unit':'a'*64}
    data.plain=[([1,5,7,8,2],[-100,-100,7,8,2]),([1,6,9,2],[-100,-100,9,2])]
    data.alternate={0:([1,4,5,7,8,2],[-100,-100,-100,7,8,2])}
    data.slots=[(0,False),(1,False),(0,True)];data.lengths=[5,4,6];data.targets=[3,2,3]
    data.original_rows=[{'id':'unit-0'},{'id':'unit-1'}]
    weights=CodeWeights.__new__(CodeWeights);weights.rows=[np.array([4,2,1],dtype='<f4'),np.array([3,1],dtype='<f4')]
    options=dict(maximum_microbatch=1,maximum_microbatch_tokens=16,gradient_clip=1.,learning_rate=.0001)
    return model,data,weights,options


@pytest.mark.parametrize('profile', ['plain_repeat','semantic_repair','weighted_semantic_repair'])
def test_complete_microbatch_gradients_match_explicit_global_original_target_objective(profile):
    model,data,weights,options=fixture();other=copy.deepcopy(model);positions=[0,1,2]
    before=tensor_digest(model.base.named_parameters())
    record=gradients(model,data,positions,profile,options,weights)
    arm='plain_repeat' if profile=='plain_repeat' else 'semantic_repair'
    batch=data.batch(arm,positions,for_optimization=True);values=torch.zeros_like(batch['labels'],dtype=torch.float32)
    for i,p in enumerate(positions):
        mask=batch['labels'][i]!=-100
        values[i,mask]=torch.from_numpy(weights.for_slot(data,p).copy()) if profile.startswith('weighted') else 1.
    other.train();other.policy.gradient_checkpointing=True
    logits=other(input_ids=batch['input_ids'],attention_mask=batch['attention_mask'])['logits']
    numerator,denominator=weighted_CE(logits,batch['labels'],values)
    (numerator/denominator).backward(inputs=[p for _,p in other.named_selected(('actor',))])
    assert record['response_EOS_targets']==8 and record['actual_microbatches']==3
    assert record['actual_global_weighted_targets']==float(denominator)
    assert record['weighted_token_mean_NLL']==pytest.approx(float((numerator/denominator).detach()),abs=2e-6)
    for (_,a),(_,b) in zip(model.named_selected(('actor',)),other.named_selected(('actor',))):
        assert a.grad is not None and torch.allclose(a.grad,b.grad,rtol=3e-5,atol=2e-6)
    assert before==tensor_digest(model.base.named_parameters())
    assert all(p.grad is None for p in model.base.parameters())


def test_preparation_and_validation_roles_never_supply_gradients():
    model,data,weights,options=fixture();data.purpose='source_preparation'
    with pytest.raises(ValueError):gradients(model,data,[0],'semantic_repair',options,weights)
    data.purpose='optimization';data.role='validation'
    with pytest.raises(ValueError):gradients(model,data,[0],'semantic_repair',options,weights)
    assert all(p.grad is None for p in model.parameters())


def test_exact_next_update_recovery_and_invalid_source_RNG_or_metric_rejected_before_mutation():
    model,data,weights,options=fixture()
    optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=options['learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False)
    stage=FunctionRepairStage(model,optimizer,data,[[0,1],[2],[0]],'weighted_semantic_repair',options,{'unit':'a'*64},weights)
    stage.update();saved=stage.state();wanted=stage.update();parts=copy.deepcopy(model.parts.state_dict());adam=copy.deepcopy(optimizer.state_dict());rng=capture_rng('cpu')
    stage.resume(saved);assert stage.update()==wanted
    assert_nested_equal(parts,model.parts.state_dict());assert_nested_equal(adam,optimizer.state_dict());assert_nested_equal(rng,capture_rng('cpu'))
    for corruption in ('source','RNG','metric'):
        bad=copy.deepcopy(saved)
        if corruption=='source':bad['model']['sources']['unit']='b'*64
        elif corruption=='RNG':bad['rng']['torch']=torch.zeros(4,dtype=torch.uint8)
        else:bad['metrics'][0]['original_presentation_slots']=[2,1]
        signature=stage.signature();before=tensor_digest(model.parts.named_parameters())
        with pytest.raises(ValueError):stage.resume(bad)
        assert before==tensor_digest(model.parts.named_parameters()) and signature==stage.signature()
        assert_nested_equal(adam,optimizer.state_dict());assert_nested_equal(rng,capture_rng('cpu'))
