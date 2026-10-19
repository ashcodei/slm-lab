from itertools import product

import pytest

from slm_lab.codenet_sample_oracles_v1 import check_sample, maximum_mean_median_gap, mean_median_gap, stagecoach_times
from fractions import Fraction


def test_original_probability_tolerance_accepts_more_digits_but_rejects_real_error():
    assert check_sample('p01411', '', '0.571428571', '0.5714285714')['passed']
    assert check_sample('p01411', '', '0.375661376', '0.3756613757')['passed']
    assert check_sample('p01411', '', '0.5', '0.500001')['passed']
    assert not check_sample('p01411', '', '0.5', '0.500002')['passed']
    for bad in ['NaN', 'Infinity', '0.5 0.6', '1e999999999', '']:
        assert not check_sample('p01411', '', '0.5', bad)['passed']


def test_original_nonunique_score_distributions_and_constraints():
    assert check_sample('p01853', '1 100', '50', '0')['passed']
    assert check_sample('p01853', '3 100', '0 0 100', '100 100 0')['passed']
    assert check_sample('p01853', '4 100', '', '0 0 0 100')['passed']
    for bad in ['50 50 50', '-1 0 100', '0 0 101', '0 100', '0 0 100 extra']:
        assert not check_sample('p01853', '3 100', '', bad)['passed']


def test_optimum_formula_matches_exhaustive_small_distributions():
    for n in range(1, 7):
        for maximum in range(1, 4):
            actual = max(mean_median_gap(values) for values in product(range(maximum + 1), repeat=n))
            assert maximum_mean_median_gap(n, maximum) == actual


def test_unknown_or_invalid_input_never_gets_a_default_checker():
    with pytest.raises(ValueError, match='No source-qualified'):
        check_sample('p00001', '', '', '')
    with pytest.raises(ValueError, match='bounds'):
        check_sample('p01853', '0 100', '', '')


def test_stagecoach_ticket_identity_disconnected_graph_and_exact_tolerance():
    data = '2 3 2 1 3\n1 3\n1 2 10\n2 3 20\n1 2 0 1 2\n1\n0 0 0 0 0\n'
    assert stagecoach_times(data) == [Fraction(50, 3), None]
    assert check_sample('p00719', data, '', '16.666667\nImpossible\n')['passed']
    assert not check_sample('p00719', data, '', '10\nImpossible\n')['passed']
    simple = '1 2 1 1 2\n3\n1 2 10\n0 0 0 0 0\n'
    assert check_sample('p00719', simple, '3.333', '3.3342')['passed']
    assert not check_sample('p00719', simple, '3.333', '3.3344')['passed']
    assert not check_sample('p00719', simple, '', 'Impossible')['passed']
    assert not check_sample('p00719', simple, '', '3.333\nextra')['passed']


def test_sugar_water_original_alternatives_and_important_invalid_equal_density():
    data = '1 2 1 2 100 1000'
    assert check_sample('p03601', data, '200 100', '1000 500')['passed']
    assert check_sample('p03601', data, '200 100', '400 200')['passed']
    assert not check_sample('p03601', data, '200 100', '300 150')['passed']
    assert not check_sample('p03601', data, '200 100', '1000 600')['passed']
    assert not check_sample('p03601', data, '200 100', '1200 600')['passed']
    assert not check_sample('p03601', data, '200 100', '100 0')['passed']
    assert check_sample('p03601', '1 2 10 20 15 200', '', '110 10')['passed']
    assert check_sample('p03601', '17 19 22 26 55 2802', '', '2634 934')['passed']
