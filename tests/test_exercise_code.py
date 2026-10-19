from slm_lab.exercise_code import exercise_split, fingerprint_functions, literal_tests, target_for
import pytest


def test_reference_aliases_ignore_local_names_annotations_and_docs_but_keep_behavior():
    first = 'def reverse(s):\n    """Original source"""\n    return s[::-1]'
    second = 'def other(value: str) -> str:\n    return value[::-1]'
    wrong = 'def other(value):\n    return value[::1]'
    assert fingerprint_functions(first) == fingerprint_functions(second)
    assert not fingerprint_functions(first) & fingerprint_functions(wrong)


def test_literal_test_reader_does_not_execute_test_code_or_invent_error_results():
    source = '''raise RuntimeError("not executed")
def test():
    self.assertEqual(square(5), 25)
    self.assertTrue(positive(3))
    self.assertFalse(positive(-3))
    self.assertEqual(square(value=5), 25)
    self.assertRaises(ValueError, square, -1)
'''
    groups = literal_tests(source)
    assert groups == {'square': [{'args_literal': '(5,)', 'expected_literal': '25'}],
                      'positive': [{'args_literal': '(3,)', 'expected_literal': 'True'},
                                   {'args_literal': '(-3,)', 'expected_literal': 'False'}]}


def test_cpp_reference_pair_maps_to_named_solution_files_and_group_split_is_language_independent():
    assert target_for('.meta/example.cpp', ['task.cpp', 'task.h']) == 'task.cpp'
    assert target_for('.meta/example.h', ['task.cpp', 'task.h']) == 'task.h'
    assert exercise_split('difference-of-squares') in {'train', 'valid'}
    with pytest.raises(ValueError, match='Ambiguous'):
        target_for('.meta/example.cpp', ['first.cpp', 'second.cpp'])
