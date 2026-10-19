import copy

import pytest
import torch

from slm_lab.code_priority_gradient_blend_v1 import combine


@pytest.mark.parametrize('project', [False, True])
def test_conflicting_partial_gradients_match_independent_flat_FP64_projection_oracle_without_source_mutation(project):
    primary = {'actor.a': torch.tensor([2., 1.], dtype=torch.float64), 'actor.b': torch.tensor([3.], dtype=torch.float64),
        'vision': None, 'unused': None}
    secondary = {'actor.a': torch.tensor([-5., 2.], dtype=torch.float64), 'actor.b': None,
        'vision': torch.tensor([7., 8.], dtype=torch.float64), 'unused': None}
    before = copy.deepcopy((primary, secondary))
    result, info = combine(primary, secondary, primary_names=['actor.a', 'actor.b'], secondary_weight=.7, project_conflicts=project)
    p = torch.tensor([2., 1., 3.], dtype=torch.float64); s = torch.tensor([-5., 2., 0.], dtype=torch.float64)
    if project: s = s - (p.dot(s) / p.dot(p)) * p
    assert torch.allclose(torch.cat([result['actor.a'], result['actor.b']]), p + .7 * s, atol=1e-14, rtol=1e-14)
    assert torch.equal(result['vision'], .7 * secondary['vision']) and result['unused'] is None
    assert info['actual_projection_applied'] == project
    for actual, original in zip([primary, secondary], before):
        assert all(v is None and original[n] is None or torch.equal(v, original[n]) for n, v in actual.items())
    if project: assert abs(info['actual_primary_projected_secondary_dot']) < 1e-12


def test_aligned_or_missing_primary_gradients_preserve_complete_secondary_updates_and_sparse_None():
    for value in [None, torch.zeros(2), torch.tensor([1., 2.])]:
        result, info = combine({'actor': value, 'vision': None, 'unused': None},
            {'actor': torch.tensor([1., 2.]), 'vision': torch.tensor([3.]), 'unused': None},
            primary_names=['actor'], secondary_weight=2., project_conflicts=True)
        assert not info['actual_projection_applied'] and result['unused'] is None
        assert torch.equal(result['vision'], torch.tensor([6.]))


@pytest.mark.parametrize('kind', ['nonfinite', 'undeclared_primary', 'shape', 'weight', 'order'])
def test_invalid_complete_gradient_inputs_are_rejected_before_source_mutation(kind):
    p = {'actor': torch.tensor([1., 2.]), 'vision': None}; s = {'actor': torch.tensor([3., 4.]), 'vision': None}
    weight = 1.
    if kind == 'nonfinite': s['actor'][0] = float('inf')
    elif kind == 'undeclared_primary': p['vision'] = torch.tensor([1.]); s['vision'] = torch.tensor([2.])
    elif kind == 'shape': s['actor'] = torch.tensor([1.])
    elif kind == 'weight': weight = -1.
    else: s = dict(reversed(list(s.items())))
    versions = [v._version for values in [p, s] for v in values.values() if v is not None]
    with pytest.raises(ValueError): combine(p, s, primary_names=['actor'], secondary_weight=weight, project_conflicts=True)
    assert versions == [v._version for values in [p, s] for v in values.values() if v is not None]
