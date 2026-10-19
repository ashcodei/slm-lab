"""Real native parent/subtest preservation and the four original qualified tasks."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from slm_lab.python_execution_feedback_v2 import PublishedPythonSuitesV2,classify,inventory,PREFIX
from test_python_execution_feedback_v1 import suite

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def runner():
    r=PublishedPythonSuitesV2(ROOT/'data/exercism-source-cache-v1/python')
    yield r
    r.close()


@pytest.fixture(scope='module')
def measured(runner):
    tests='''import unittest
from answer import f
class Cases(unittest.TestCase):
    def test_subcases(self):
        for value in [1,2,3]:
            with self.subTest(value=value):self.assertEqual(f(value),value+1)
def test_plain():assert f(4)==5
'''
    positive=runner.run({'answer.py':'def f(x):return x+1\n'},suite(tests))
    negative=runner.run({'answer.py':'def f(x):return x-1\n'},suite(tests))
    return positive,negative


def test_all_original_subtests_remain_distinct_and_assertion_failures_are_not_lost(measured):
    positive,negative=measured
    assert classify(positive)=='all_finite_published_cases_pass'
    assert len(positive['inventory']['parent_cases'])==2
    assert len(positive['inventory']['subtest_contexts'])==3
    assert len(positive['inventory']['cases'])==5
    assert classify(negative,positive['inventory'])=='completed_published_assertion_counterexample'
    assert len(negative['inventory']['assertion_failed_cases'])==4


def test_changed_subtest_context_never_silently_matches_reference(measured):
    positive,negative=measured;reference=deepcopy(positive['inventory'])
    next(iter(reference['subtest_contexts'].values()))['kwargs']['value']='DIFFERENT'
    assert classify(negative,reference)=='changed_subtest_context_qualification'


def test_duplicate_parent_and_missing_subtest_ordinal_reject(measured):
    raw=measured[0]['raw_inventory_output'].encode()
    value=json.loads(next(x[len(PREFIX):] for x in raw.splitlines() if x.startswith(PREFIX)))
    parent=next(e for e in value['events'] if e['kind']=='parent');value['events'].append(parent)
    with pytest.raises(ValueError,match='parent phases'):inventory(PREFIX+json.dumps(value).encode())
    value['events'].pop();next(e for e in value['events'] if e['kind']=='subtest')['subtest_index']=5
    with pytest.raises(ValueError,match='ordered original subtest'):inventory(PREFIX+json.dumps(value).encode())


def test_all_four_original_qualified_programs_pass_without_changing_solutions_or_tests(runner):
    targets={'electric-bill','meltdown-mitigation','diffie-hellman','error-handling'}
    rows=[json.loads(line) for line in (ROOT/'data/exercism-support-context-v3/train.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['language']=='Python' and r['exercise'] in targets]
    assert len(rows)==4
    for row in rows:
        spec=runner.suite(row);proof=runner.run(spec['source_files'],spec)
        assert classify(proof)=='all_finite_published_cases_pass',row['id']+str(proof)
        if row['exercise']=='error-handling':
            assert 'exercises/practice/error-handling/test_utils.py' in spec['original_provided_helpers']
        else:assert proof['inventory']['subtest_contexts']
