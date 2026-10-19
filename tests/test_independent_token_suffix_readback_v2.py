import copy

import pytest

from slm_lab.public_token_suffix_curriculum_v1 import LEVELS, view, descriptor
from slm_lab.independent_token_suffix_readback_v1 import rebuild, eligible
from test_public_reference_prefix_curriculum_v1 import fixture


def test_all_original_native_targets_and_all4views_rebuild_without_calling_producer_in_reader():
    tokenizer, row, ids, labels = fixture()
    for level in LEVELS:
        original = view(tokenizer, row, ids, labels, level)
        actual = rebuild(tokenizer, row, ids, labels, level, descriptor(original))
        assert actual['original_prefix_text'] == original['public_reference_prefix_text']
        assert actual['original_reference_prefix_ids'] + actual['original_missing_suffix_and_EOS_ids'] == ids[actual['metadata']['original_prompt_tokens']:]
        # Short references cannot supply a real64-token hint. Such duplicate
        # unhinted contexts stay in CE and do not gain conditional RL admission.
        assert eligible(actual['metadata'], dict(reward=1.), dict(reward=.9)) == (
            level == 0 or actual['metadata']['retained_reference_tokens'] > 0)
        assert not eligible(actual['metadata'], dict(reward=1.), dict(reward=1.))


@pytest.mark.parametrize('field,value', [('original_response_EOS_targets', 1), ('original_unconditional_labels_sha256', 'a' * 64),
    ('original_reference_prefix_sha256', 'b' * 64), ('full_native_action_budget', 1),
    ('source_training_admitted', True), ('unassisted_capability_score', True)])
def test_changed_targets_prefix_budget_or_admission_cannot_pass_independent_reconstruction(field, value):
    tokenizer, row, ids, labels = fixture()
    metadata = descriptor(view(tokenizer, row, ids, labels, 4))
    metadata[field] = value
    with pytest.raises(ValueError):
        rebuild(tokenizer, row, ids, labels, 4, metadata)


def test_actual_execution_abstention_or_nonpassing_reference_never_qualifies():
    tokenizer, row, ids, labels = fixture()
    metadata = descriptor(view(tokenizer, row, ids, labels, 4))
    for ref, prefix in [(None, 0.), (1., None), (.9, 0.), (1., 1.)]:
        assert not eligible(metadata, dict(reward=ref), dict(reward=prefix))
