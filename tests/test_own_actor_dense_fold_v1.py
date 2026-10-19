import copy

import pytest
import torch

from slm_lab.deterministic_positions_v1 import install
from slm_lab.own_actor_dense_fold_v1 import export,fold,load
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.corpus import file_hash
from test_own_dense_growth_v1 import model as make_model
from test_own_aligned_assistant_v1 import original_signature

SOURCES=dict(own_source='a'*64)


def nonzero_actor():
    base,_=make_model();actor=OwnDenseActor(install(base),rank=2).eval()
    # Nonzero learned-shaped updates, policy norms and critic prevent a neutral
    # adapter from making this a trivial identity or zero-value-head test.
    with torch.no_grad():
        for i,parts in enumerate(actor.parts['linears']):parts.up.fill_(.0005*(1+i%3))
        for i,norm in enumerate(actor.parts['norms']):norm.weight.mul_(1+.001*(1+i%3))
        actor.parts['critic'].weight.fill_(.003)
    return actor


def test_nonzero_policy_fold_preserves_small_native_text_image_values_word_heads_and_source_RNG_without_aliases():
    actor=nonzero_actor();signature=original_signature(actor);rng=torch.get_rng_state().clone()
    candidate,receipt=fold(actor)
    assert signature==original_signature(actor) and torch.equal(rng,torch.get_rng_state())
    assert receipt['actual_neural_values_removed']==actor.counts()['actor']+actor.config.hidden_size
    assert candidate.counts()['total_unique']==sum(p.numel() for p in candidate.parameters())
    ids=torch.tensor([[1,4,7,8,2]]);pixels=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32)
    with torch.no_grad():
        for image in (None,pixels):
            a=actor(ids,pixel_values=image);b=candidate(ids,pixel_values=image)
            for key in ['logits','values']:assert torch.allclose(a[key],b[key],atol=2e-6,rtol=2e-5)
        assert torch.equal(actor.base.word_logits(pixels),candidate.word_logits(pixels))
    assert not {p.data_ptr() for p in actor.parameters()}&{p.data_ptr() for p in candidate.parameters()}


def test_folded_full_language_backprop_updates_dense_tables_and_matrices_while_image_and_auxiliary_weights_stay_fixed():
    candidate,_=fold(nonzero_actor());named=candidate.language_named_parameters();params=[p for _,p in named]
    chosen={id(p) for p in params};fixed=[(p,p.detach().clone(),p._version) for p in candidate.parameters() if id(p) not in chosen]
    before={n:p.detach().clone() for n,p in named};optimizer=torch.optim.AdamW(params,lr=.001,foreach=False)
    for p in params:p.requires_grad_(True)
    candidate.train();candidate.network.gradient_checkpointing=True
    ids=torch.tensor([[1,4,7,8,2]]);candidate(ids,labels=ids)['loss'].backward(inputs=params)
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params)
    assert all(p.grad is None for p,_,_ in fixed)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert all(not torch.equal(before[n],p) for n,p in named)
    assert all(torch.equal(saved,p) and p._version==version for p,saved,version in fixed)


def test_self_contained_folded_student_native_reload_needs_no_source_actor_or_old_base(tmp_path):
    candidate,receipt=fold(nonzero_actor());path=tmp_path/'student.pt'
    torch.save(export(candidate,sources=SOURCES,fold_receipt=receipt,completed_updates=0),path)
    cold,metadata=load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=SOURCES)
    ids=torch.tensor([[1,4,7,2]])
    with torch.no_grad():assert torch.equal(candidate(ids)['logits'],cold(ids)['logits'])
    assert metadata['no_old_model_or_actor_or_training_reference_loaded']
    with pytest.raises(ValueError):load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=dict(own_source='b'*64))


def test_finite_source_factors_whose_product_overflows_are_rejected_without_source_mutation():
    actor=nonzero_actor()
    with torch.no_grad():
        actor.parts['linears'][0].up.fill_(3e38);actor.parts['linears'][0].down.fill_(3e38)
    signature=original_signature(actor)
    with pytest.raises(ValueError,match='finite'):fold(actor)
    assert signature==original_signature(actor)
