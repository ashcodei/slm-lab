"""Native masks, source rights and exact contract compatibility for new data."""
import copy
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from slm_lab.contract_source_contrasts_v2 import direct,contrast,alternative
from slm_lab.public_function_contract_census_v2 import identity,observe
from test_public_function_contract_census_v2 import rows

ROOT=Path(__file__).resolve().parents[1]


def test_complete_public_originals_and_new_cross_implementation_contrast_both_roundtrip():
    tok=Tokenizer.from_file(str(ROOT/'data/direct-algorithm-function-contracts-admitted-v1/tokenizer.json'));by={identity(r):r for r in rows()}
    donor=by['maths/is_int_palindrome.py:is_int_palindrome'];row=by['project_euler/problem_125/sol1.py:is_palindrome']
    direct_row,ids,labels=direct(donor,tok);assert direct_row['response']==donor['response'] and direct_row['prompt']==donor['prompt']
    assert direct_row['rights']==donor['rights'] and ids[-1]==labels[-1]==2
    bad=observe(row);candidate,ids,labels,pair=contrast(row,donor,bad,tok)
    assert candidate['response'].startswith(donor['response']) and candidate['rights']==row['rights'] and candidate['donor_rights']==donor['rights']
    start=len(pair['native_prompt_ids']);assert ids==pair['native_prompt_ids']+pair['chosen_complete_response_EOS_ids']
    assert labels==[-100]*start+ids[start:] and pair['rejected_complete_response_EOS_ids'][-1]==2
    assert tok.decode(pair['rejected_complete_response_EOS_ids'][:-1],skip_special_tokens=False)==row['response']
    assert observe(row,candidate['response'])['all_cases_passed']


def test_equal_names_with_different_contracts_and_falsified_outcomes_or_held_roles_reject():
    tok=Tokenizer.from_file(str(ROOT/'data/direct-algorithm-function-contracts-admitted-v1/tokenizer.json'));by={identity(r):r for r in rows()}
    index=by['searches/binary_search.py:binary_search'];boolean=by['searches/simple_binary_search.py:binary_search']
    with pytest.raises(ValueError):alternative(index,boolean)
    invalid=copy.deepcopy(index);invalid['split']='validation'
    with pytest.raises(ValueError):direct(invalid,tok)
    row=by['project_euler/problem_125/sol1.py:is_palindrome'];donor=by['maths/is_int_palindrome.py:is_int_palindrome'];bad=observe(row)
    invalid=copy.deepcopy(bad);invalid['case_passes']=[True]*len(invalid['case_passes'])
    with pytest.raises(ValueError):contrast(row,donor,invalid,tok)
