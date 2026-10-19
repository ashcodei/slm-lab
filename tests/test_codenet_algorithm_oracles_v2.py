import pytest

from slm_lab.codenet_algorithm_oracles_v2 import check_output


ONE = '1 2 1 1 2\n1\n1 2 1\n0 0 0 0 0\n'
THIRD = '1 2 1 1 2\n3\n1 2 1\n0 0 0 0 0\n'


def test_original_any_decimal_precision_includes_the_actual99999_digit_style():
    value = format(1 / 3, '.99999f')
    assert check_output('p00719', THIRD, '0.3333333333', value)
    assert check_output('p00719', ONE, '1', '1.' + '0' * 99999)


def test_inclusive_absolute_tolerance_remains_exact_beyond_float_precision():
    for value in ['1.001', '.999', '1e0', '+0001.000']:
        assert check_output('p00719', ONE, '1', value)
    for value in ['1.001000000000000000000000000000000001',
                  '.998999999999999999999999999999999999']:
        assert not check_output('p00719', ONE, '1', value)
    assert check_output('p00719', THIRD, '', '.334333333333333333333333333333333333')
    assert not check_output('p00719', THIRD, '', '.334333333333333333333333333333333334')


@pytest.mark.parametrize('actual', ['NaN','Infinity','1e999999999999999999999',
                                  '1e-999999999999999999999','-1','0','Impossible','1\n1','not a number'])
def test_invalid_remote_or_extra_outputs_are_rejected(actual):
    assert not check_output('p00719', ONE, '1', actual)


def test_integer_and_property_outputs_can_have_leading_zeroes_within_output_cap():
    assert check_output('p03095','1\na\n','1','0'*10000+'1')
    assert check_output('p03601','1 2 1 2 100 1000','200 100','000400 000200')
    assert check_output('p03746','2 1\n1 2\n','','02\n001 002')
    assert not check_output('p03095','1\na\n','1','9'*10000)


def test_output_limit_remains_an_explicit_bounded_profile_qualification():
    assert not check_output('p00719', ONE, '1', '1.' + '0' * 1024**2)
