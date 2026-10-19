import ast
import pytest
from tokenizers import Tokenizer
from pathlib import Path

from slm_lab.code_training_views_v1 import completion_view,repair_candidates,encode_view,sha,verify_mutation


def source(program):
    response='```python\n'+program+'\n```'
    return dict(id='original:1',group_id='github:publisher/file.py',split='train',publisher_split='train',language='Python',
        prompt='Return the documented result. The complete surrounding standard-library context is supplied.',response=response,
        parent_source=dict(id='0123456789abcdef',original_partition='train',language='Python'),original_source_segment_row=17,
        executable_program_sha256=sha(program),unchanged_retained_target_sha256=sha(response))


def test_completion_preserves_signature_indentation_unicode_and_every_original_byte():
    program='@staticmethod\ndef café(entrée):\n    """The original description."""\n    return entrée + "é"'
    row=completion_view(source(program))
    assert row['original_prefix']+row['original_continuation']==program
    assert row['original_continuation']=='    return entrée + "é"'
    assert row['complete_original_function_bytes_preserved']


def test_inline_completion_and_multiline_statement_repair_reconstruct_exact_utf8_source():
    program='def f(é):\n    label = "雪"\n    return (\n        é + label\n    )'
    rows=repair_candidates(source(program));row=rows[0]
    assert row['original_middle']=='return (\n        é + label\n    )'
    assert row['original_prefix']+row['original_middle']+row['original_suffix']==program
    ast.parse(row['mutant_program'])
    inline=completion_view(source('def f(x): return x + 1'))
    assert inline['original_prefix']=='def f(x): ' and inline['original_continuation']=='return x + 1'


def test_held_source_roles_are_rejected_before_any_derived_training_view():
    row=source('def f(x):\n    return x + 1');row['parent_source']['original_partition']='test'
    with pytest.raises(ValueError,match='publisher-train'):completion_view(row)
    with pytest.raises(ValueError,match='publisher-train'):repair_candidates(row)


def test_complete_target_mask_and_strict_no_truncation_with_own_tokenizer():
    root=Path(__file__).resolve().parents[1];tokenizer=Tokenizer.from_file(str(root/'data/documented-doctest-verified-v1/tokenizer.json'))
    row=repair_candidates(source('def f(x):\n    return x + 1'))[0];ids,labels=encode_view(row,tokenizer)
    first=next(i for i,x in enumerate(labels) if x!=-100)
    assert ids[0]==1 and ids[-1]==2 and labels[:first]==[-100]*first and labels[first:]==ids[first:]
    assert tokenizer.decode(ids[first:-1])==row['original_middle']
    with pytest.raises(ValueError,match='no truncation'):encode_view(row,tokenizer,maximum=len(ids)-1)


def test_mutation_runs_all_original_examples_and_never_rewrites_publisher_or_calls_it_policy_output():
    import copy
    from slm_lab.documented_doctest_v1 import execute
    program='def add_one(x):\n    return x + 1';text='>>> n = 2\n>>> add_one(n)\n3\n>>> add_one(7)\n8\n'
    formatted='```python\n'+program+'\n```'
    candidate=dict(source=dict(id='fixture',original_partition='train',language='Python',prompt=text,response=formatted),
        program=program,program_sha256=sha(program),doctest_text=text,doctest_text_sha256=sha(text),function='add_one',
        existing_formatted_source_target_sha256=sha(formatted),whole_original_example_count=3)
    reference=execute(candidate);before=copy.deepcopy(candidate)
    result=verify_mutation(candidate,reference,'def add_one(x):\n    pass')
    assert result['all_original_examples_observed'] and result['qualifies_as_observed_failure']
    assert result['scored_original_examples']==2 and result['fraction_passed']==0
    assert candidate==before and result['own_policy_generation'] is False
    candidate['source']['original_partition']='test'
    with pytest.raises(ValueError,match='original train'):verify_mutation(candidate,reference,'def add_one(x):\n    pass')
