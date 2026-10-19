from copy import deepcopy

import pytest

from test_code_contests_contrast_candidates_v1 import observed, Alphabet
from slm_lab.code_contests_contrast_candidates_v1 import select, counterexample
from slm_lab.code_contests_contrast_readback_v1 import reconstruct_pair, reconstruct_witness


def test_separate_reader_reconstructs_full_original_pair_EOS_and_actual_counterexample(observed):
    row, problem, reference, cases, actual = observed
    pair, why = reconstruct_pair(row, problem, reference, Alphabet())
    assert (pair, why) == select(row, problem, reference, Alphabet())
    witness, why = reconstruct_witness(pair['rejected'], problem, actual)
    assert (witness, why) == counterexample(pair['rejected'], cases, actual)
    assert witness['original_case_ordinal'] == 1
    assert pair['chosen_ids'][-1] == pair['rejected_ids'][-1] == 2
    assert pair['optimization_admitted'] is False


def test_separate_reader_rejects_source_native_or_actual_observation_corruption(observed):
    row, problem, reference, cases, actual = observed
    forged = deepcopy(actual)
    forged['first']['original_case_verdicts'][0]['actual_stdout_b64'] = 'OQo='
    with pytest.raises(ValueError):
        reconstruct_witness(problem.incorrect_solutions[1].solution, problem, forged)
    with pytest.raises(ValueError):
        reconstruct_pair(dict(row, publisher_split='test'), problem, reference, Alphabet())
    forged = deepcopy(actual); forged['repeat']['original_program_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        reconstruct_witness(problem.incorrect_solutions[1].solution, problem, forged)
    changed = deepcopy(problem); changed.incorrect_solutions[1].solution += '# changed original\n'
    pair, _ = reconstruct_pair(row, changed, reference, Alphabet())
    with pytest.raises(ValueError):
        reconstruct_witness(pair['rejected'], changed, actual)
