import copy

import pytest
import torch

from slm_lab.completed_public_coding_policy_v1 import install_completed_policy
from slm_lab.own_dense_actor_checkpoint_v1 import make_state
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from slm_lab.serial_public_code_epoch_v1 import train
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_parallel_public_reasoned_code_SFT_stage_v1 import own_model, SOURCES
from test_public_reasoned_code_train_data_v1 import packet

OPTIONS = dict(lr=.001,betas=(.9,.95),eps=1e-8,weight_decay=.01,foreach=False,fused=False)


def completed(root, profile):
    torch.set_num_threads(1)
    packet(root)
    data = PublicReasonedCodeSFTData(root)
    model = own_model()
    initial = tensor_digest(model.parts.named_parameters())
    fixed = tensor_digest(model.base.named_parameters())
    adam = torch.optim.AdamW([p for _,p in model.named_selected(('actor',))], **OPTIONS)
    snapshots = []
    def snapshot(step, entries, records):
        snapshots.append(make_state(model,adam,groups=('actor',),sources=SOURCES,completed=step,planned=len(entries),
            cursor={'next_global_batch':step},metrics=records,device='cpu'))
    proof = train(model,adam,data,profile,checked=lambda:None,record=lambda *a:None,checkpoint=snapshot,allow_mechanics=True)
    proof.update(initial_selected_policy_sha256=initial,frozen_native_sha256=fixed,
        final_selected_policy_sha256=tensor_digest(model.parts.named_parameters()),
        actual_registered_parameters=model.counts()['actual_unique'],actual_selected_Adam_states=len(adam.state))
    return model,data,snapshots[-1],proof


@pytest.mark.parametrize('profile',['plain_tokens','eos_weight4'])
def test_actual_whole_epoch_materializes_exact_final_values_without_Adam_or_caller_RNG_resume(tmp_path,profile):
    trained,data,state,proof = completed(tmp_path,profile)
    fresh = own_model().eval()
    rng = torch.get_rng_state().clone()
    original = value_digest(state)
    result = install_completed_policy(fresh,data,state,proof,profile=profile,expected_sources=SOURCES,
        optimizer_options=OPTIONS,allow_mechanics=True)
    assert result['complete'] and result['completed_updates']==1
    assert tensor_digest(fresh.named_parameters()) == tensor_digest(trained.named_parameters())
    assert torch.equal(rng,torch.get_rng_state()) and value_digest(state)==original
    assert not torch.cuda.is_initialized()


def test_partial_or_corrupt_source_clock_objective_unused_critic_and_optimizer_reject_before_mutation(tmp_path):
    model,data,state,proof = completed(tmp_path,'plain_tokens')
    changes = [lambda s:s.__setitem__('completed',0),
        lambda s:s['metrics'][0]['actual_positions'].pop(),
        lambda s:s['metrics'][0].__setitem__('weighted_token_mean_CE',999.),
        lambda s:s['optimizer']['state'][0]['step'].fill_(2),
        lambda s:s['model']['parts']['critic.weight'].fill_(1),
        lambda s:s['optimizer']['param_groups'][0].__setitem__('lr',.01)]
    for change in changes:
        broken = copy.deepcopy(state)
        change(broken)
        fresh = own_model().eval()
        before = tensor_digest(fresh.named_parameters())
        rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError):
            install_completed_policy(fresh,data,broken,proof,profile='plain_tokens',expected_sources=SOURCES,
                optimizer_options=OPTIONS,allow_mechanics=True)
        assert tensor_digest(fresh.named_parameters())==before and torch.equal(rng,torch.get_rng_state())
