import hashlib

import pytest

from slm_lab.public_token_suffix_curriculum_v1 import LEVELS, view, descriptor, nontrivial, tokens_sha
from test_public_reference_prefix_curriculum_v1 import fixture


def test_all_four_levels_keep_every_original_CE_target_and_exact_token_reconstruction():
    tokenizer, row, ids, labels = fixture()
    values = [view(tokenizer, row, ids, labels, n) for n in LEVELS]
    assert values[0]['retained_reference_tokens'] > values[1]['retained_reference_tokens']
    for n, value in zip(LEVELS, values):
        assert value['native_rollout_prefix_ids'] + value['original_missing_suffix_and_EOS_ids'] == ids
        assert value['complete_original_anchor_labels'] == labels and value['complete_original_anchor_ids'] == ids
        assert value['actual_missing_response_EOS_targets'] <= value['full_native_action_budget']
        assert not value['source_training_admitted'] and not value['production_RL_admitted']
        if n and value['retained_reference_tokens']:
            assert value['actual_missing_response_EOS_targets'] >= n + 1
    assert values[-1]['retained_reference_tokens'] == 0 and values[-1]['exact_unhinted_original_source']
    assert values[-1]['native_rollout_prefix_ids'] == ids[:values[-1]['original_prompt_tokens']]


@pytest.mark.parametrize('case', ['held', 'publisher_held', 'synthetic', 'changed_source', 'lost_EOS', 'changed_mask'])
def test_changed_roles_or_original_targets_cannot_supply_a_training_view(case):
    tokenizer, row, ids, labels = fixture()
    row, ids, labels = dict(row), list(ids), list(labels)
    if case == 'held':
        row['split'] = 'validation'
    elif case == 'publisher_held':
        row['publisher_split'] = 'test'
    elif case == 'synthetic':
        row['model_generated_examples'] = True
    elif case == 'changed_source':
        row['response'] += '\n'
    elif case == 'lost_EOS':
        ids[-1] = 5
    else:
        labels[-2] = -100
    with pytest.raises(ValueError):
        view(tokenizer, row, ids, labels, 4)


def test_trivial_prefix_and_failed_reference_never_qualify_and_metadata_never_claims_admission():
    tokenizer, row, ids, labels = fixture()
    value = view(tokenizer, row, ids, labels, 4)
    assert nontrivial(value, dict(reward=1.), dict(reward=0.))
    assert not nontrivial(value, dict(reward=1.), dict(reward=1.))
    assert not nontrivial(value, dict(reward=.9), dict(reward=0.))
    assert not nontrivial(dict(value, original_has_embedded_boundary_tokens=True), dict(reward=1.), dict(reward=0.))
    summary = descriptor(value)
    assert 'public_reference_prefix_text' not in summary and 'complete_original_anchor_ids' not in summary
    assert summary['original_response_EOS_targets'] == sum(x != -100 for x in labels)
    assert not summary['source_training_admitted'] and not summary['production_RL_admitted']


def test_unicode_prefixes_preserve_original_characters_at_every_short_suffix_boundary():
    tokenizer, _, _, _ = fixture()
    response = 'def label():\n    return "café 🧬 漢字"\n'
    prompt = [1] + tokenizer.encode('User: Implement label.\nAssistant:', add_special_tokens=False).ids
    target = tokenizer.encode(response, add_special_tokens=False).ids + [2]
    ids, labels = prompt + target, [-100] * len(prompt) + target
    row = dict(id='unit-only-unicode', split='train', publisher_split='train', response=response,
        external_model_teacher_used=False, model_generated_examples=False, complete_input_ids_sha256=tokens_sha(ids))
    for level in LEVELS:
        value = view(tokenizer, row, ids, labels, level)
        assert response.startswith(value['public_reference_prefix_text'])
        assert tokenizer.decode(value['public_reference_prefix_ids'] + value['original_missing_suffix_and_EOS_ids'],
            skip_special_tokens=True) == response


def test_longest_whole_native_context_keeps_complete_suffix_inside_4096_positions():
    tokenizer, row, ids, labels = fixture()
    original_start = next(i for i, x in enumerate(labels) if x != -100)
    prompt = [1] + [7] * (4096 - (len(ids) - original_start) - 1)
    ids = prompt + ids[original_start:]
    labels = [-100] * len(prompt) + ids[len(prompt):]
    row = dict(row, complete_input_ids_sha256=tokens_sha(ids))
    assert len(ids) == 4096
    for level in LEVELS:
        value = view(tokenizer, row, ids, labels, level)
        assert value['conditional_rollout_prefix_tokens'] + value['full_native_action_budget'] <= 4096
        assert value['full_native_action_budget'] >= value['actual_missing_response_EOS_targets']
