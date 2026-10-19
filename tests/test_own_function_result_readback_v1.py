from copy import deepcopy
import hashlib
import struct

import pytest

from slm_lab.own_function_result_readback_v1 import reconstruct
from slm_lab.own_function_trajectory_preferences_v1 import native_pair


def fixture():
    row = dict(id='unit-source',split='train',group_id='unit-file',source_candidate=dict(id='unit-source',role='train'))
    pair = native_pair(row,[1,5],[7,2],[9,10,11],stopped=False,budget=3)
    event = dict(case_position=0,original_input_token_ids=[1,5],actual_generated_token_ids=[9,10,11],
        complete_input_ids_sha256=hashlib.sha256(struct.pack('<HH',1,5)).hexdigest(),requested_native_action_budget=3,
        actual_native_action_count=3,actual_stopped_on_EOS=False,greedy_native_output_tensor_returned_unchanged=True,
        selected_action_log_probabilities_not_retained=True)
    program='def increment(x):\n    return x+1'
    observed=dict(status='all_original_examples_passed',syntax_valid=True,all_original_examples_observed=True,
        all_original_examples_passed=True,original_public_example_count=2,passed_examples=2,entry_body_calls=2,
        actual_generated_program_sha256=hashlib.sha256(program.encode()).hexdigest(),original_source_id=pair['id'],
        held_benchmark=False,generated_program_not_repaired=True)
    delivery=dict(original_source_id=pair['id'],optimization_role=pair['optimization_role'],actual_generated_token_ids=[9,10,11],
        raw_decoded_response=program,actual_generated_program=program,actual_public_example_observations=observed)
    return pair,event,delivery,program


def test_actual_censored_actions_reconstruct_but_fake_eos_shortening_or_internal_eos_reject():
    pair,event,delivery,program=fixture()
    result=reconstruct([pair],[event],[delivery],lambda actions:program,lambda text:text)
    assert result[0]['actual_generated_tokens']==3 and not result[0]['actual_stopped_on_EOS']
    for actions in [[9,10],[9,2,11],[9,10,11,2]]:
        bad=deepcopy(event);bad['actual_generated_token_ids']=actions;bad['actual_native_action_count']=len(actions)
        changed=deepcopy(delivery);changed['actual_generated_token_ids']=actions
        with pytest.raises(ValueError):reconstruct([pair],[bad],[changed],lambda a:program,lambda t:t)


def test_source_role_prompt_code_substitution_and_duplicate_delivery_reject():
    pair,event,delivery,program=fixture()
    for field,value in [('optimization_role','invented_held_benchmark'),('actual_generated_program',program+'\n#repair'),('original_source_id','other-source')]:
        bad=deepcopy(delivery);bad[field]=value
        with pytest.raises(ValueError):reconstruct([pair],[event],[bad],lambda a:program,lambda t:t)
    bad=deepcopy(event);bad['original_input_token_ids']=[1,6]
    with pytest.raises(ValueError):reconstruct([pair],[bad],[delivery],lambda a:program,lambda t:t)
    with pytest.raises(ValueError):reconstruct([pair,pair],[event,event],[delivery,delivery],lambda a:program,lambda t:t)


def test_valid_actual_eos_boundary_retains_exact_short_action_count():
    pair,event,delivery,program=fixture()
    event.update(actual_generated_token_ids=[9,2],actual_native_action_count=2,actual_stopped_on_EOS=True)
    delivery['actual_generated_token_ids']=[9,2]
    actual=reconstruct([pair],[event],[delivery],lambda a:program,lambda t:t)
    assert actual[0]['actual_generated_tokens']==2 and actual[0]['actual_stopped_on_EOS']


def test_real_producer_metric_masks_validate_but_control_gradient_or_changed_denominator_reject():
    from slm_lab.own_function_result_readback_v1 import validate_training
    from slm_lab.own_function_trajectory_preference_stage_v1 import information
    from slm_lab.own_function_trajectory_preferences_v1 import control_group
    pairs=[]
    for wanted_control in [False,True]:
        group=next('unit-group-'+str(i) for i in range(100) if control_group('unit-group-'+str(i))==wanted_control)
        row=dict(id='unit-'+str(wanted_control),split='train',group_id=group,
                 source_candidate=dict(id='unit-'+str(wanted_control),role='train'))
        pairs.append(native_pair(row,[1,5],[7,2],[9,10,11],stopped=False,budget=3))
    options=dict(steps=1,learning_rate=.0001,beta=.1,chosen_CE_weight=1.)
    record=dict(step=1,method='chosen_CE',original_source_positions=[0],original_trajectory_pairs=[information(pairs[0],0)],
        learning_rate=.0001,beta=.1,chosen_CE_weight=1.,own_starting_actor_reference_queried=False,
        unoptimized_train_controls_not_used=True,failed_program_actions_never_positive_CE_targets=True,
        no_unobserved_rejected_EOS_or_other_actions=True,human_preferences=False,
        chosen_response_EOS_targets=2,rejected_actual_action_targets=3,normalization_chosen_targets=2,normalization_pairs=1,
        chosen_token_mean_NLL=1.,pair_mean_preference=0.,summed_objective=1.,gradient_norm_before_clip=.5)
    validate_training([record],[[0]],pairs,'chosen_CE',options)
    bad=deepcopy(record);bad['normalization_chosen_targets']=1
    with pytest.raises(ValueError):validate_training([bad],[[0]],pairs,'chosen_CE',options)
    bad=deepcopy(record);bad['original_source_positions']=[1];bad['original_trajectory_pairs']=[information(pairs[1],1)]
    with pytest.raises(ValueError):validate_training([bad],[[1]],pairs,'chosen_CE',options)
