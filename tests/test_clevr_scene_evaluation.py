import pytest
from slm_lab.clevr_scene_evaluation import parse,score


TARGET='Objects from left to right:\n1. small red metal cube.\n2. large blue rubber sphere.\n3. small green metal cube.\nShape counts: cubes 2, cylinders 0, spheres 1.'


def test_all_source_objects_attributes_and_counts_must_match_for_full_scene_credit():
    result=score(TARGET,TARGET)
    assert result['full_scene_exact'] and result['object_attribute_multiset_f1']==1 and result['declared_counts_consistent']


def test_swapped_objects_can_match_multiset_without_matching_projected_order():
    swapped=TARGET.replace('small red metal cube','temporary').replace('large blue rubber sphere','small red metal cube').replace('temporary','large blue rubber sphere')
    result=score(swapped,TARGET)
    assert result['object_attribute_multiset_f1']==1 and not result['full_scene_exact'] and result['ordered_attributes_accuracy']<1


def test_inconsistent_or_wrong_declared_counts_do_not_receive_exact_scene_credit():
    result=score(TARGET.replace('cubes 2','cubes 3'),TARGET)
    assert result['object_count_exact'] and not result['shape_counts_exact'] and not result['declared_counts_consistent']


@pytest.mark.parametrize('value',['','Objects from left to right:\n1. unknown object.','Objects from left to right:\n1. small red metal cube.\n1. large blue rubber sphere.\n3. small green metal cube.\nShape counts: cubes 2, cylinders 0, spheres 1.'])
def test_missing_malformed_or_repeated_object_indices_cannot_score_as_valid_description(value):
    assert parse(value) is None and not score(value,TARGET)['valid_format']
