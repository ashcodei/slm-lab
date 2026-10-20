import pyarrow as pa

from slm_lab.public_reasoned_code_metadata_v1 import normalized_pass_rate, reported_all_pass


def test_publisher_arrow_string_schema_preserves_reported_positives():
    values = pa.array(['1.0', '-1', '0.98', '0.0', '1.0'], type=pa.string()).to_pylist()
    assert [reported_all_pass(value) for value in values] == [True, False, False, False, True]
    assert [normalized_pass_rate(value) for value in values] == [1., -1., .98, 0., 1.]


def test_numeric_equivalence_does_not_coerce_invalid_or_unknown_values():
    assert all(reported_all_pass(value) for value in ['1', '1e0', 1, 1.])
    for value in [True, False, None, [], {}, '', 'unknown', 'nan', 'inf', '-inf', '-2', '1.1', float('nan')]:
        assert normalized_pass_rate(value) is None
        assert not reported_all_pass(value)
    assert normalized_pass_rate('-1') == -1.
    assert not reported_all_pass('-1')
