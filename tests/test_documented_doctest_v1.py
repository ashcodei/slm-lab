import copy
import pytest

from slm_lab.documented_doctest_v1 import contract,execute,sha,training_example,PRELUDE
from slm_lab.verified_code import doctest_cases


def candidate(program,text):
    return dict(source=dict(id='fixture',language='Python',original_partition='train',prompt=text,response='```python\n'+program+'\n```'),
        function='transform',program=program,doctest_text=text,program_sha256=sha(program),doctest_text_sha256=sha(text),
        existing_formatted_source_target_sha256=sha('```python\n'+program+'\n```'),whole_original_example_count=len(__import__('doctest').DocTestParser().get_examples(text)))


def test_complete_multistep_published_example_state_executes_while_old_literal_parser_cannot_cover_it():
    text=""">>> items = ['a', 'b']
>>> transform(items)
['A', 'B']
>>> items.append('c')
>>> transform(items)
['A', 'B', 'C']
"""
    c=candidate('def transform(items):\n    return [s.upper() for s in items]',text);before=copy.deepcopy(c)
    assert not doctest_cases(text,'transform')
    result=execute(c)
    assert result['all_original_examples_passed'] and result['passed_examples']==4 and result['all_original_examples_observed']
    assert result['expected_text_not_written_to_worker'] and c==before


def test_actual_source_flags_and_expected_exception_match_and_wrong_result_is_never_admitted():
    text=""">>> transform(0)
Traceback (most recent call last):
...
ValueError: original message
>>> transform(4)
2.0
"""
    c=candidate("def transform(x):\n    if not x: raise ValueError('original message')\n    return x / 2",text)
    assert execute(c)['all_original_examples_passed']
    c=candidate("def transform(x):\n    if not x: raise ValueError('wrong message')\n    return 99",text)
    result=execute(c);assert not result['all_original_examples_passed'] and result['all_original_examples_observed']
    assert result['example_passes']==[False,False]


def test_source_roles_skips_changed_bytes_and_missing_full_example_count_reject_before_execution():
    c=candidate('def transform(x):\n    return x',">>> transform(3)\n3\n>>> transform(7)\n7\n")
    for mutate in [lambda c:c['source'].update(original_partition='test'),lambda c:c.update(program=c['program']+'\n#change'),
        lambda c:c.update(program=c['program']+'\n#changed target',program_sha256=sha(c['program']+'\n#changed target')),
        lambda c:c.update(whole_original_example_count=1)]:
        bad=copy.deepcopy(c);mutate(bad)
        with pytest.raises(ValueError):contract(bad)
    skipped=candidate('def transform(x):\n    return x',">>> transform(3) # doctest: +SKIP\n3\n>>> transform(7)\n7\n")
    with pytest.raises(ValueError,match='skipped'):contract(skipped)


def test_worker_settings_contain_commands_only_and_cannot_read_a_parent_private_file(tmp_path):
    secret=tmp_path/'outside-work.txt';secret.write_text('not-a-credential')
    code="def transform(x):\n    import json\n    if x: return sorted(json.load(open('settings.json')).keys())\n    return open("+repr(str(secret))+").read()"
    text=""">>> transform(True)
['examples', 'function', 'prelude', 'program']
>>> transform(False)
Traceback (most recent call last):
...
PermissionError: expected # doctest: +IGNORE_EXCEPTION_DETAIL
"""
    # Put the standard directive on the actual source command, not output.
    text=text.replace('>>> transform(False)','>>> transform(False) # doctest: +IGNORE_EXCEPTION_DETAIL').replace('expected # doctest: +IGNORE_EXCEPTION_DETAIL','expected')
    result=execute(candidate(code,text));assert result['all_original_examples_passed']


def test_unexercised_missing_helpers_are_qualified_by_static_scope_instead_of_becoming_verified_targets():
    code='def transform(x):\n    if x: return missing_helper(x)\n    return 0'
    c=candidate(code,">>> transform(0)\n0\n>>> transform(False)\n0\n")
    with pytest.raises(ValueError,match='undeclared global'):contract(c)


def test_setup_only_examples_cannot_qualify_an_unexecuted_reference_function():
    c=candidate("def transform(x):\n    raise RuntimeError('not exercised')",">>> values = [1, 2]\n>>> total = sum(values)\n")
    result=execute(c)
    assert result['all_original_examples_observed'] and result['example_passes']==[True,True]
    assert result['entry_body_calls']==0 and not result['all_original_examples_passed']


def test_training_preserves_whole_original_target_and_checks_actual_results_before_complete_masks():
    from types import SimpleNamespace
    text=(">>> transform(3)\n6\n>>> transform(7)\n14\n\n"
        'The surrounding module or class supplies its existing imports, types and helpers. Return only the implementation.')
    c=candidate('def transform(x):\n    return x * 2',text)
    c['source']['group_id']='fixture-repository:path';c['original_source_segment_row']=0
    observed=execute(c);seen=[]
    class Tokens:
        def encode(self,text):
            seen.append(text);return SimpleNamespace(ids=[4]*len(text))
    row,ids,labels=training_example(c,observed,Tokens())
    assert row['response']==c['source']['response'] and PRELUDE in row['prompt']
    assert '\n\nProvided standard-library context:\n```python\n' in row['prompt']
    assert seen==['User: '+row['prompt']+'\nAssistant:',c['source']['response']]
    start=next(i for i,x in enumerate(labels) if x!=-100)
    assert ids[0]==1 and ids[-1]==2 and labels[:start]==[-100]*start and labels[start:]==ids[start:]
    assert not row['dataset_admitted_to_training'] and row['machine_execution_feedback'] and not row['human_preferences']
    wrong=copy.deepcopy(observed);wrong['raw_results'][0]['stdout']='99\n';seen.clear()
    with pytest.raises(ValueError,match='Every actual whole'):training_example(c,wrong,Tokens())
    assert not seen


def test_complete_source_training_target_is_never_truncated_to_fit():
    from types import SimpleNamespace
    text=(">>> transform(3)\n3\n>>> transform(7)\n7\n\n"
        'The surrounding module or class supplies its existing imports, types and helpers. Return only the implementation.')
    c=candidate('def transform(x):\n    return x',text);c['source']['group_id']='fixture';c['original_source_segment_row']=0
    class TooManyTokens:
        def encode(self,text):return SimpleNamespace(ids=[4]*4096)
    with pytest.raises(ValueError,match='exceed'):training_example(c,execute(c),TooManyTokens())
