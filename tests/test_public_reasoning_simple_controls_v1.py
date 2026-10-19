"""Prediction has no reserved-gold interface or hidden algorithmic solver."""
import copy

import pytest

from slm_lab.public_reasoning_simple_controls_v1 import copy_input,fit,predict


def row(identity,answer,question='grid:\n0 1\n1 0',task='rotate_matrix',split='train'):
    return dict(id=identity,split=split,task=task,response=answer,original_entry=dict(question=question,answer=answer,metadata=dict(solution=answer)))


def test_majority_tie_break_and_predictors_never_consult_reserved_gold():
    training=[row('a','True',task='isomorphic_strings'),row('b','False',task='isomorphic_strings')]
    fitted=fit(training)
    assert predict(fitted,'isomorphic_strings','different input')=='False'
    assert fitted['isomorphic_strings']['train_distinct_canonical_answers']==2
    changed=row('held','True',task='isomorphic_strings',split='validation')
    first=predict(fitted,changed['task'],changed['original_entry']['question'])
    changed['response']='different';changed['original_entry'].update(answer='different',metadata=dict(solution='different'))
    assert predict(fitted,changed['task'],changed['original_entry']['question'])==first
    with pytest.raises(ValueError):fit([changed])


def test_literal_copy_reads_the_whole_actual_matrix_and_ignores_rotation_instructions():
    question='Rotate this matrix by 90 degrees clockwise:\n1 2\n3 4'
    assert copy_input('rotate_matrix',question)=='1 2\n3 4'
    assert copy_input('binary_matrix','Input:\n0 1\n1 0')=='0 1\n1 0'
    assert copy_input('gcd','Find the gcd of 12 and 24') is None
    with pytest.raises(ValueError):copy_input('binary_matrix','Input:\n1 2 3\n4 5 6')


def test_control_selection_is_fitted_only_on_training_and_ties_choose_majority():
    training=[row('a','0 1\n1 0'),row('b','1 0\n0 1',question='grid:\n1 0\n0 1')]
    fitted=fit(training)
    assert fitted['rotate_matrix']['selected_control']=='copy_input'
    assert predict(fitted,'rotate_matrix','rotate by 270 degrees:\n3 4\n5 6')=='3 4\n5 6'
    tied=fit([row('a','0 1\n1 0')])
    assert tied['rotate_matrix']['selected_control']=='training_majority'
