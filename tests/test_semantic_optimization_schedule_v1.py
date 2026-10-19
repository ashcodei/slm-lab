from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest

from slm_lab.semantic_optimization_schedule_v1 import LANGUAGES, schedule, validate_targets


def sources():
    pairs = [{'id': str(i), 'original_program_sha256': str(i), 'prompt': 'p',
              'chosen': 'yes', 'rejected': 'no', 'chosen_target_tokens': 4,
              'rejected_target_tokens': 3, 'feedback_source': 'synthetic'} for i in range(83)]
    pools = {name: [{'id': name + str(i), 'prompt': 'p', 'response': 'text'} for i in range(120)]
             for name in [*LANGUAGES, 'human_instruction']}
    return pairs, pools


class Encoder:
    config = SimpleNamespace(max_text_length=64)
    bos = 1; eos = 2; truncated = 0
    def prompt(self, row): return row['prompt']
    def tokens(self, text): return list(text.encode())
    def sequence(self, row, field):
        prompt, response = self.tokens(row['prompt']), self.tokens(row[field])
        return [1] + prompt + response + [2], [-100] * (len(prompt) + 1) + response + [2]


def test_both_controls_can_replay_all_eight_complete_pair_passes_without_source_mutation():
    pairs, pools = sources(); original = deepcopy((pairs, pools))
    allocation = schedule(pairs, pools)
    assert len(allocation) == 664
    assert Counter(row['pair_index'] for row in allocation) == {i: 8 for i in range(83)}
    counts = Counter(row['module_pool'] for row in allocation)
    assert set(counts) == set(LANGUAGES) and max(counts.values()) - min(counts.values()) == 1
    assert allocation == schedule(pairs, pools)
    assert (pairs, pools) == original


def test_actual_exposure_budgets_and_masks_are_reconstructed():
    pairs, pools = sources(); allocation = schedule(pairs, pools)
    result = validate_targets(pairs, pools, allocation, Encoder())
    assert result['planned_target_exposures'] == {'chosen': 2656, 'rejected': 1992, 'module': 3320, 'human': 3320}
    assert all(row['sequence_sha256'] for row in result['source_records'])
    assert len(result['exposures']) == 664


def test_forged_target_counts_or_missing_rehearsal_language_are_rejected():
    pairs, pools = sources(); pairs[0]['chosen_target_tokens'] = 9
    with pytest.raises(ValueError, match='response targets'):
        validate_targets(pairs, pools, schedule(pairs, pools), Encoder())
    pools['Go'] = []
    with pytest.raises(ValueError, match='six languages'): schedule(pairs, pools)


def test_duplicate_algorithms_or_human_label_claims_are_rejected():
    pairs, pools = sources(); pairs[1]['original_program_sha256'] = pairs[0]['original_program_sha256']
    with pytest.raises(ValueError, match='Repeated'): schedule(pairs, pools)
    pairs, pools = sources(); pairs[0]['feedback_source'] = 'human'
    with pytest.raises(ValueError, match='feedback origin'):
        validate_targets(pairs, pools, schedule(pairs, pools), Encoder())
