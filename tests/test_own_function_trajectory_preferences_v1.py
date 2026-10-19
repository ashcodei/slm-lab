from copy import deepcopy
import pytest

from slm_lab.own_function_trajectory_preferences_v1 import native_pair,validate,batch


def source():return dict(id='unit-train',split='train',group_id='unit-source-file',source_candidate=dict(id='unit-train',role='train'))


def test_budget_censored_negative_uses_only_actual_actions_without_fake_EOS_or_prompt_loss():
    pair=native_pair(source(),[1,5,6],[7,8,2],[9,10,11],stopped=False,budget=3)
    validate(pair)
    actual=batch(pair,'rejected')
    assert actual['input_ids'].tolist()==[[1,5,6,9,10,11]]
    assert actual['labels'].tolist()==[[-100,-100,-100,9,10,11]]
    assert 2 not in actual['labels'].tolist()[0]
    chosen=batch(pair,'chosen');assert chosen['labels'].tolist()==[[-100,-100,-100,7,8,2]]
    ended=native_pair(source(),[1,5],[7,2],[9,2],stopped=True,budget=3)
    assert batch(ended,'rejected')['labels'].tolist()==[[-100,-100,9,2]]


def test_held_roles_altered_termination_dropped_actions_and_control_role_substitution_are_rejected():
    pair=native_pair(source(),[1,5],[7,2],[9,10,11],stopped=False,budget=3)
    for field,value in [('original_source_role','valid'),('rejected_actual_stopped_on_EOS',True),
        ('rejected_actual_sampled_action_ids',[9,10]),('optimization_role','invented-held-test'),
        ('rejected_actual_sampled_action_ids',[9,2,11])]:
        bad=deepcopy(pair);bad[field]=value
        with pytest.raises(ValueError):validate(bad)
    row=source();row['split']='validation'
    with pytest.raises(ValueError):native_pair(row,[1,5],[7,2],[9,10,11],stopped=False,budget=3)


def test_admitted_dataset_enforces_whole_group_controls_and_exact_independent_completion(tmp_path):
    import json
    from slm_lab.corpus import file_hash
    from slm_lab.own_function_trajectory_preferences_v1 import OwnFunctionTrajectoryPreferences,FORMAT
    rows=[]
    for i in range(14):
        row=source();row.update(id='synthetic-unit-'+str(i),group_id='synthetic-group-'+str(i));row['source_candidate']['id']=row['id']
        rows.append(native_pair(row,[1,5],[7,2],[9,10,11],stopped=False,budget=3))
    pairs=tmp_path/'all-original-function-trajectory-pairs.jsonl';pairs.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    proof=tmp_path/'independent-unit-proof.json';proof.write_text(json.dumps(dict(complete=True,
        all14_original_sources_native_actions_and_fresh_executions_rebuilt=True,actual_original_source_train_trajectories=14,synthetic_unit_fixture_not_runtime_proof=True)))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(format=FORMAT,pairs_sha256=file_hash(pairs),source_files_sha256={})))
    admission=dict(actual_trajectory_machine_preference_training_admitted=True,human_preferences=False,source_valid_or_test_records=0,
        manifest_sha256=file_hash(manifest),independent_readback=str(proof),independent_readback_sha256=file_hash(proof))
    (tmp_path/'admission.json').write_text(json.dumps(admission));data=OwnFunctionTrajectoryPreferences(tmp_path,'optimization')
    assert data.optimization_positions and data.unoptimized_control_positions
    data.batch(data.optimization_positions[0],'chosen',for_optimization=True)
    with pytest.raises(ValueError,match='cannot provide gradients'):data.batch(data.unoptimized_control_positions[0],'chosen',for_optimization=True)
    audit=OwnFunctionTrajectoryPreferences(tmp_path,'audit')
    with pytest.raises(ValueError,match='cannot provide gradients'):audit.batch(audit.optimization_positions[0],'chosen',for_optimization=True)
    proof.write_text(json.dumps(dict(complete=True,actual_original_source_train_trajectories=14)))
    admission['independent_readback_sha256']=file_hash(proof);(tmp_path/'admission.json').write_text(json.dumps(admission))
    with pytest.raises(ValueError,match='full independent'):OwnFunctionTrajectoryPreferences(tmp_path,'optimization')
