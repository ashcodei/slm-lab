import pytest

from slm_lab.codenet_problem_groups_v1 import ProblemGroups, canonical_member, identical_groups, selected_clone_groups


def test_transitive_groups_are_stable_under_edge_order_and_keep_unrelated_problems():
    problems = [f'p{n:05d}' for n in range(5)]
    edges = [('p00001', 'p00003'), ('p00000', 'p00001'), ('p00003', 'p00004')]
    first, second = ProblemGroups(problems), ProblemGroups(reversed(problems))
    for edge in edges:
        first.merge(edge)
    for edge in reversed(edges):
        second.merge(reversed(edge))
    assert first.mapping() == second.mapping()
    assert first.root('p00004') == 'p00000' and first.root('p00002') == 'p00002'
    with pytest.raises(ValueError, match='Unknown'):
        first.merge(['p00000', 'p99999'])


def test_original_csv_aliases_and_legacy_source_directory_are_explicit():
    assert identical_groups('p00001,p00002\n\np00003,p00004,p00005\n') == [
        ('p00001', 'p00002'), ('p00003', 'p00004', 'p00005')]
    assert canonical_member('/Volume1/AI4CODE/CodeNet/data/p00001/C++/s000000001.cpp') == 'Project_CodeNet/data/p00001/C++/s000000001.cpp'
    with pytest.raises(ValueError):
        identical_groups('p00001,p0002')


def test_selected_clones_keep_boundaries_even_when_other_members_are_not_selected():
    a = 'Project_CodeNet/data/p00001/C++/s000000001.cpp'
    b = 'Project_CodeNet/data/p00002/C++/s000000002.cpp'
    c = 'Project_CodeNet/data/p00003/C++/s000000003.cpp'
    lines = ['/v/CodeNet/data/p00001/C++/s000000001.cpp:\n',
             '/v/CodeNet/data/p00009/C++/s000000009.cpp: 0.95, 0.91\n',
             '/v/CodeNet/data/p00002/C++/s000000002.cpp: 0.94, 0.92\n',
             '\n', '/v/CodeNet/data/p00003/C++/s000000003.cpp:\n']
    assert list(selected_clone_groups(lines, {a, b, c}, 'C++')) == [(a, b), (c,)]


def test_new_representative_without_blank_line_starts_another_cluster():
    a = 'Project_CodeNet/data/p00001/Python/s000000001.py'
    b = 'Project_CodeNet/data/p00002/Python/s000000002.py'
    lines = [a + ':\n', b + ':\n']
    assert list(selected_clone_groups(lines, {a, b}, 'Python')) == [(a,), (b,)]


@pytest.mark.parametrize('lines', [
    ['Project_CodeNet/data/p00001/Python/s000000001.py: 0.99, 0.98\n'],
    ['Project_CodeNet/data/p00001/Python/s000000001.py\n'],
    ['Project_CodeNet/data/p00001/C++/s000000001.cpp:\n'],
])
def test_malformed_or_wrong_language_clusters_are_not_silently_ignored(lines):
    with pytest.raises(ValueError):
        list(selected_clone_groups(lines, set(), 'Python'))
