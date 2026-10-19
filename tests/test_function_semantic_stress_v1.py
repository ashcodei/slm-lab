"""Source qualification, independent expected values, real isolated executions."""
import copy
import gzip
import json
from pathlib import Path

import pytest

from slm_lab.function_semantic_stress_v1 import NAMES,DONORS,cases,expected,matches,observe,public_alternative

ROOT=Path(__file__).resolve().parents[1]


def sources():return [s['row'] for s in json.loads((ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/all14-original-source-train-records.json').read_text())]


def donors():
    rows=list(map(json.loads,gzip.open(ROOT/'data/direct-algorithm-function-contracts-admitted-v1/original-rows.jsonl.gz','rt')))
    return {r['source_candidate']['original_source_git_path']+':'+r['source_candidate']['function']:r for r in rows}


def test_all_fourteen_case_domains_keep_exact_source_roles_and_mark_exhausted_binary_domain():
    rows=sources();assert [r['source_candidate']['function'] for r in rows]==NAMES
    for i,row in enumerate(rows):
        selected=cases(row);assert selected==cases(row) and len({c['case_id'] for c in selected})==len(selected)
        assert len(selected)==(2 if i==0 else 30 if i==7 else 64)
        assert all(c['arguments_not_in_original_public_examples']==bool(i) for c in selected)
        changed=copy.deepcopy(row);changed['split']='validation'
        with pytest.raises(ValueError):cases(changed)


def test_independent_integer_oracles_cover_known_weak_source_boundaries_and_output_types():
    assert [expected('is_prime',[n]) for n in [0,1,2,3,4,49,97]]==[False,False,True,True,False,False,True]
    assert expected('perfect_cube',[64]) is True and expected('perfect_cube',[65]) is False
    assert expected('is_sq',[(2**53+1)**2]) is True
    raw=lambda x:dict(stdout=x,exception_only=None,entry_body_calls=1)
    assert matches(False,raw('False\n')) and not matches(False,raw('0\n'))
    assert not matches(1,raw('True\n')) and not matches(1,raw('extra\n1\n'))
    assert not matches(1.,raw('float("nan")')) and not matches(1,dict(raw('1'),entry_body_calls=0))


def test_actual_isolated_reference_failures_and_public_alternative_repairs():
    selected_donors=donors()
    for row in sources():
        name=row['source_candidate']['function']
        if name not in DONORS:continue
        all_cases=cases(row);bad=observe(row['response'],name,all_cases)
        assert bad['actual_program_launched'] and bad['all_cases_observed'] and not bad['all_cases_passed']
        donor=selected_donors[DONORS[name]];program=public_alternative(row,donor)
        assert program.startswith(donor['response'])
        good=observe(program,name,all_cases)
        assert good['actual_program_launched'] and good['all_cases_passed'] and good['expected_outputs_not_sent_to_worker']
        invalid=copy.deepcopy(donor);invalid['split']='validation'
        with pytest.raises(ValueError):public_alternative(row,invalid)


def test_actual_syntax_semantic_exception_and_corrupt_expected_values_never_drop_cases():
    row=next(r for r in sources() if r['source_candidate']['function']=='abs_val');selected=cases(row)
    for program in ['def abs_val(x) return x','def abs_val(x):\n    return x','def abs_val(x):\n    raise ValueError("bad")']:
        actual=observe(program,'abs_val',selected)
        assert actual['cases']==64 and len(actual['case_passes'])==64 and not actual['all_cases_passed']
    bad=copy.deepcopy(selected);bad[0]['expected']+=1
    with pytest.raises(ValueError):observe('def abs_val(x):\n    return abs(x)','abs_val',bad)
