import copy

import pytest

from slm_lab.dense_foundation_evidence_v1 import validate_envelope
from slm_lab.dense_foundation_inference_v1 import sources,export
from test_dense_foundation_stage_v1 import data,options,stage
from test_own_dense_growth_v1 import model as make_model


def payloads():
    model,_=make_model();d=data();setting=options(d,0.)
    setting.update(epochs=1,steps=1,accumulation=7)
    rate=setting.pop('learning_rate');setting.pop('reference_KL_alpha')
    plan=dict(initializer_sha256='a'*64,education_admission_sha256='b'*64,code_admission_sha256='c'*64,
        human_admission_sha256='d'*64,program_admission_sha256='e'*64,actual_counts=model.counts(),
        source_exposure_budget=d.budget(),training_options=setting,
        arms=dict(control=dict(learning_rate=rate,reference_KL_alpha=0.)))
    sha='f'*64;bound=sources(plan,sha);initial=model.export(bound)
    learner=stage(model,None,d,{**setting,'learning_rate':rate,'reference_KL_alpha':0.});learner.sources=bound
    learner.update();state=learner.state();saved=export(model,plan,sha,'control',1)
    return plan,initial,saved,state,d,sha


def test_actual_complete_dense_update_envelope_and_full_source_cursor_are_accepted():
    plan,initial,saved,state,d,sha=payloads()
    validate_envelope(plan,'control',initial,saved,state,d,sha)
    assert state['core']['cursor']['actual_draws']==7
    assert len(state['core']['optimizer']['state'])==len(model_names(state))


def model_names(state):return state['core']['optimized_names']


def test_changed_admission_source_cursor_optimizer_options_and_false_completion_reject():
    plan,initial,saved,state,d,sha=payloads()
    for mutate in [lambda s:s['core']['model']['sources'].update(education_admission_sha256='0'*64),
                   lambda s:s['core']['cursor'].update(actual_draws=6),
                   lambda s:s['core']['optimizer']['param_groups'][0].update(lr=.5),
                   lambda s:s.update(goal_achieved=True)]:
        bad=copy.deepcopy(state);mutate(bad)
        with pytest.raises(ValueError,match='envelope'):validate_envelope(plan,'control',initial,saved,bad,d,sha)
