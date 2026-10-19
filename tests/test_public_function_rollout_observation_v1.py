from copy import deepcopy
import pytest

from slm_lab.public_function_rollout_observation_v1 import observe


def source():
    return dict(id='unit-source-only',split='train',source_candidate=dict(id='unit-source-only',role='train',function='increment',
        original_literal_docstring='Return x plus one.\n>>> increment(0)\n1\n>>> increment(-2)\n-1\n',whole_original_example_count=2))


def test_real_generated_correct_wrong_syntax_and_missing_entrypoint_keep_every_public_case():
    row=source();good=observe(row,'def increment(x):\n    return x + 1')
    assert good['all_original_examples_passed'] and good['passed_examples']==2 and good['entry_body_calls']==2
    bad=observe(row,'def increment(x):\n    return x - 1')
    assert bad['syntax_valid'] and not bad['all_original_examples_passed'] and len(bad['raw_results'])==2
    assert observe(row,'def increment(:')['status']=='generated_syntax_error'
    absent=observe(row,'def another(x):\n    return x + 1')
    assert absent['syntax_valid'] and absent['entry_body_calls']==0 and not absent['all_original_examples_passed']


def test_held_roles_changed_source_identity_or_dropped_public_examples_reject():
    row=source()
    for modify in [lambda r:r.update(split='validation'),lambda r:r['source_candidate'].update(role='valid'),
        lambda r:r['source_candidate'].update(id='other'),lambda r:r['source_candidate'].update(whole_original_example_count=1)]:
        bad=deepcopy(row);modify(bad)
        with pytest.raises(ValueError):observe(bad,'def increment(x):\n    return x + 1')
