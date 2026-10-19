import pytest

from slm_lab.codenet_problem_groups_v1 import selected_clone_groups
from slm_lab.codenet_publisher_qualification_v1 import QualifiedPublisherLines


FRAGMENT = '/Volume1/AI4CODE/CodeNe'
MEMBER = 'Project_CodeNet/data/p00001/Python/s000000001.py'


def test_exact_inspected_terminal_fragment_is_reported_with_incomplete_coverage():
    lines = QualifiedPublisherLines([MEMBER + ':\n', '\n', FRAGMENT], FRAGMENT)
    assert list(selected_clone_groups(lines, {MEMBER}, 'Python')) == [(MEMBER,)]
    assert lines.incomplete_terminal_record == {'line': 3, 'fragment': FRAGMENT, 'source_coverage_complete': False}


def test_fragment_in_middle_cannot_silently_discard_remaining_relations():
    lines = QualifiedPublisherLines([FRAGMENT, MEMBER + ':\n'], FRAGMENT)
    with pytest.raises(ValueError, match='not the terminal'):
        list(selected_clone_groups(lines, {MEMBER}, 'Python'))


def test_other_malformed_tail_still_fails_strict_publisher_parser():
    lines = QualifiedPublisherLines([MEMBER + ':\n', '/other/incomplete'], FRAGMENT)
    with pytest.raises(ValueError, match='delimiter missing'):
        list(selected_clone_groups(lines, {MEMBER}, 'Python'))
