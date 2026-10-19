from copy import deepcopy
import ast

import pytest

from slm_lab.public_function_contracts_v1 import from_function,execute_candidate,verified_training_row,signature
from slm_lab.public_pretraining_intake_v2 import candidate as source_candidate
from test_public_pretraining_intake_v1 import fixture

PROGRAM = '''def move_last_first(values):
    """Move the final item to the beginning, preserving all other items.

    >>> move_last_first([4, 7, 9])
    [9, 4, 7]
    >>> move_last_first([])
    []
    """
    return values[-1:] + values[:-1]
'''


def example(program=PROGRAM):
    t,q,s,r = fixture();r = dict(r,code=program)
    original,reason = source_candidate(r,s,t,q)
    assert original is not None,reason
    node = next(n for n in ast.parse(program).body if isinstance(n,ast.FunctionDef))
    value,reason = from_function(original,node,t,q)
    return value,reason,t,q,original


def test_whole_literal_function_docstring_native_mask_EOS_and_original_role_are_preserved():
    c,reason,t,q,original = example()
    assert reason is None and c['source_original_file'] == original
    assert c['response'] == ast.get_source_segment(PROGRAM,ast.parse(PROGRAM).body[0])
    assert c['role'] == original['role'] and c['whole_original_example_count'] == 2
    start = c['native_target_start'];ids = c['whole_native_ids']
    assert c['whole_native_labels'] == [-100]*start+ids[start:]
    assert ids[-1] == 2 and t.decode(ids[start:-1],skip_special_tokens=False) == c['response']
    assert c['optimization_admitted'] is False and c['model_generated_target'] is False
    assert signature('def source(\n x: dict[str, int],\n y=(1, 2),\n) -> int:\n return 1') == 'def source(\n x: dict[str, int],\n y=(1, 2),\n) -> int:'


def test_unknown_surrounding_context_decorators_and_skipped_original_examples_cannot_be_labels():
    for code,expected in [(PROGRAM.replace('values[-1:] + values[:-1]','missing_helper(values)'), 'unresolved_function_context'),
        ('@cache\n'+PROGRAM,'decorated_async'),
        (PROGRAM.replace('move_last_first([])','move_last_first([]) # doctest: +SKIP'),'requires_all_original_examples')]:
        c,reason,*_ = example(code)
        assert c is None and reason.startswith(expected)


def test_two_actual_confined_original_example_runs_and_full_native_rebuild_allow_a_candidate_pack_row():
    c,_,t,q,_ = example()
    observations = [execute_candidate(c,seconds=3),execute_candidate(c,seconds=3)]
    assert all(r['all_original_examples_passed'] for r in observations)
    assert all(r['expected_text_not_written_to_worker'] is False for r in observations)
    assert all(r['original_program_docstring_includes_public_example_answers'] for r in observations)
    assert all(r['expected_outputs_not_sent_as_separate_assertions_or_comparison_artifacts'] for r in observations)
    row,ids,labels = verified_training_row(c,observations,t,q)
    assert row['actual_fresh_complete_example_repetitions'] == 2 and row['dataset_admitted_to_training'] is False
    assert ids == c['whole_native_ids'] and labels == c['whole_native_labels']
    bad = deepcopy(c);bad['whole_native_labels'][-1] = -100
    with pytest.raises(ValueError,match='must rebuild'):verified_training_row(bad,observations,t,q)
    bad = deepcopy(observations);bad[1]['raw_results'][0]['stdout'] = 'wrong\n'
    with pytest.raises(ValueError,match='must pass'):verified_training_row(c,bad,t,q)


def test_actual_complete_native_input_is_screened_again_after_prompt_and_context_composition():
    c,_,t,_,original = example()
    class RejectActual:
        def literal(self,text):return ['declared held text']
        def lexical_copies(self,text):return []
    value,reason = from_function(original,ast.parse(PROGRAM).body[0],t,RejectActual())
    assert value is None and reason == 'actual_function_native_input_or_response_has_declared_held_overlap'
