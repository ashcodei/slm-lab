import copy

import pytest

from slm_lab.complete_joint_objective_bridge_v3 import checkpoint_entry


def original_metadata():
    row = dict(source_step=1, family='replay:image_reading', original_positions=[7],
        complete_context_tokens=21, response_EOS_targets=5,
        complete_original_negative_response_EOS_targets=20, complete_original_correction_indices=[0], whole_word_UTF8_bytes=0)
    return dict(step=1, source_entry_sha256='a' * 64,
        components={'visual': dict(reference=dict(stream='visual', source_step=1, original_entry_sha256='b' * 64), entry=row)})


def test_longer_original_rejected_response_preserves_all_positive_targets_and_independent_normalizers():
    # Actual public document correction geometry: 21 positive input tokens,
    # five positive EOS targets, 36 negative input tokens and20negative targets.
    # A longer negative must not make the positive CE denominator negative.
    metadata = original_metadata(); before = copy.deepcopy(metadata)
    objective = checkpoint_entry(metadata, source_identity_sha256='c' * 64, negative_context_tokens=36)['objectives'][0]
    assert objective['complete_context_tokens'] == 57
    assert objective['response_EOS_targets'] == 25
    assert objective['normalization_denominators'] == dict(positive_native_EOS=5, corrective_pairs=1, negative_EOS=20)
    assert metadata == before


def test_uncorrected_native_ce_keeps_original_counts_without_invented_negative_targets():
    metadata = original_metadata(); row = metadata['components']['visual']['entry']
    row.update(complete_original_negative_response_EOS_targets=0, complete_original_correction_indices=[])
    objective = checkpoint_entry(metadata, source_identity_sha256='c' * 64)['objectives'][0]
    assert objective['complete_context_tokens'] == 21 and objective['response_EOS_targets'] == 5
    assert objective['normalization_denominators'] == dict(positive_native_EOS=5)


@pytest.mark.parametrize('contexts', [0, 19, True, -1])
def test_missing_or_incomplete_original_negative_context_is_rejected(contexts):
    with pytest.raises(ValueError):
        checkpoint_entry(original_metadata(), source_identity_sha256='c' * 64, negative_context_tokens=contexts)
