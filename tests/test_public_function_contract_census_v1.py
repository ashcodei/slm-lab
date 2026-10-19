"""Finite source qualification; no model quality or global correctness claim."""
import copy
import gzip
import json
from pathlib import Path

import pytest

from slm_lab.public_function_contract_census_v1 import cases,equal,family,identity,observe,passes,registry

ROOT=Path(__file__).resolve().parents[1]


def rows():return list(map(json.loads,gzip.open(ROOT/'data/direct-algorithm-function-contracts-admitted-v1/original-rows.jsonl.gz','rt')))


def test_exact_source_registry_covers_original_train_interfaces_and_never_equates_names_or_held_roles():
    source=rows();train=[r for r in source if r['split']=='train'];registered=registry()
    assert len(source)==841 and len(train)==826 and len(registered)>50
    selected=[r for r in train if identity(r) in registered]
    assert {identity(r) for r in selected}==set(registered)
    for row in selected:
        chosen=cases(row);assert chosen==cases(row) and 2<=len(chosen)<=64
        assert len({c['case_id'] for c in chosen})==len(chosen)
        wrong=copy.deepcopy(row);wrong['source_candidate']['role']='valid'
        with pytest.raises(ValueError):cases(wrong)
    by={identity(r):r for r in train}
    assert family(by['searches/binary_search.py:binary_search'])=='search_first'
    assert family(by['searches/simple_binary_search.py:binary_search'])=='search_bool'
    assert family(by['backtracking/n_queens.py:is_safe']) is None
    with pytest.raises(ValueError):cases(by['backtracking/n_queens.py:is_safe'])


def test_return_shape_none_bool_integer_set_and_mutation_contracts_are_distinct():
    assert not equal(False,0) and not equal(True,1) and not equal([1,2],(1,2))
    assert equal({3,2},{'set':[2,3]}) and not equal([2,3],{'set':[2,3]})
    row=next(r for r in rows() if identity(r)=='sorts/pigeonhole_sort.py:pigeonhole_sort');selected=cases(row)
    case=next(c for c in selected if c['args']==[[1,0]])
    raw=lambda text:dict(stdout=text,exception_only=None,entry_body_calls=1)
    assert passes(case,raw('(None, [[0, 1]])\n'))
    assert not passes(case,raw('(None, [[1, 0]])\n'))
    assert not passes(case,raw('([0, 1], [[0, 1]])\n'))


def test_real_confined_network_search_sort_string_and_set_reference_contracts():
    selected=['conversions/ipv4_conversion.py:ipv4_to_decimal','conversions/ipv4_conversion.py:decimal_to_ipv4',
        'searches/binary_search.py:binary_search','searches/simple_binary_search.py:binary_search',
        'sorts/pigeonhole_sort.py:pigeonhole_sort','sorts/counting_sort.py:counting_sort',
        'strings/naive_string_search.py:naive_pattern_search','strings/hamming_distance.py:hamming_distance',
        'project_euler/problem_047/sol1.py:unique_prime_factors']
    by={identity(r):r for r in rows()}
    for key in selected:
        result=observe(by[key]);assert result['actual_program_launched'] and result['all_cases_observed'] and result['all_cases_passed'],(key,result)
        assert result['expected_values_not_sent_to_worker'] and result['generated_program_not_repaired']


def test_actual_confined_mathematical_counterexamples_and_bad_interfaces_never_hide_cases():
    by={identity(r):r for r in rows()}
    row=by['project_euler/problem_125/sol1.py:is_palindrome'];actual=observe(row)
    assert actual['actual_program_launched'] and actual['all_cases_observed'] and not actual['all_cases_passed']
    selected=cases(row);position=next(i for i,c in enumerate(selected) if c['args']==[0]);assert not actual['case_passes'][position]
    row=by['searches/simple_binary_search.py:binary_search']
    actual=observe(row,'def binary_search(a_list,item):\n    return 1 if item in a_list else 0')
    assert len(actual['case_passes'])==64 and not any(actual['case_passes'])
    wrong=copy.deepcopy(row);wrong['response']='def binary_search(x):\n    return True'
    with pytest.raises(ValueError):cases(wrong)
