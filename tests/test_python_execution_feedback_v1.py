"""Real confined publisher-style unittest/pytest cases, not model capability."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from slm_lab.python_execution_feedback_v1 import (PublishedPythonSuites,PREFIX,
    classify,extract_files,inventory,mutations)

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def suites():
    runner=PublishedPythonSuites(ROOT/'data/exercism-source-cache-v1/python')
    yield runner
    runner.close()


def suite(test):
    return dict(expected_solution_files=['answer.py'],test_files={'answer_test.py':test.encode()},
        test_modules=['answer_test.py'])


TESTS='''import unittest
from answer import f
class Cases(unittest.TestCase):
    def test_one(self):self.assertEqual(f(1),2)
    def test_three(self):self.assertEqual(f(3),4)
def test_negative():assert f(-1)==0
'''


@pytest.fixture(scope='module')
def measured(suites):
    spec=suite(TESTS)
    passed=suites.run({'answer.py':'def f(x):return x+1\n'},spec)
    failed=suites.run({'answer.py':'def f(x):return x-1\n'},spec)
    error=suites.run({'answer.py':'def f(x):\n if x==3:raise ValueError("runtime")\n return x-1\n'},spec)
    return passed,failed,error


def test_actual_complete_unittest_and_pytest_inventory_and_assertion_counterexample(measured):
    positive,negative,_=measured
    assert len(positive['inventory']['cases'])==3
    assert positive['inventory']['complete_case_phases'] and positive['inventory']['all_passed']
    assert classify(positive)=='all_finite_published_cases_pass'
    assert classify(negative,positive['inventory'])=='completed_published_assertion_counterexample'
    assert len(negative['inventory']['assertion_failed_cases'])==3
    assert not negative['inventory']['nonassertion_call_failures']


def test_nonassertion_exceptions_and_case_population_changes_are_qualified(measured):
    positive,negative,error=measured
    assert classify(error,positive['inventory'])=='nonassertion_test_exception_qualification'
    reference=deepcopy(positive['inventory']);reference['cases'].append('invented::case')
    assert classify(negative,reference)=='changed_case_population_qualification'


def test_skipped_publisher_case_is_never_silently_dropped_or_negative(suites):
    spec=suite(TESTS+'\nimport pytest\n@pytest.mark.skip(reason="original skip")\ndef test_skipped():assert False\n')
    result=suites.run({'answer.py':'def f(x):return x-1\n'},spec)
    assert len(result['inventory']['cases'])==4
    assert result['inventory']['skipped_or_expected_failures']
    assert classify(result,dict(cases=result['inventory']['cases']))=='incomplete_test_inventory_qualification'


def test_changed_raw_evidence_and_duplicate_phase_reject(measured):
    positive,negative,_=measured
    forged=deepcopy(negative);forged['report_sha256']='0'*64
    with pytest.raises(ValueError,match='Raw complete'):classify(forged,positive['inventory'])
    raw=negative['raw_inventory_output'].encode()
    payload=json.loads(next(line[len(PREFIX):] for line in raw.splitlines() if line.startswith(PREFIX)))
    payload['events'].append(payload['events'][0])
    with pytest.raises(ValueError,match='Repeated Python'):inventory(PREFIX+json.dumps(payload).encode())
    with pytest.raises(ValueError,match='Exactly one'):inventory(raw+b'\n'+raw)


def test_code_token_mutations_preserve_comments_strings_unicode_and_complete_files():
    source='''File: answer.py
```python
# True 0 + != and
text = "True 0 + != and"
other = f"False {1+2}"
\u03c0 = 3.14
\u00e9 = 1
def f(x):
    return x + 1 if x >= 0 and True else 0
```
'''
    changed=mutations(source,'fixture:unicode')
    assert changed==mutations(source,'fixture:unicode') and len(changed)==4
    for item in changed:
        patch=item['patch']
        assert source[patch['start']:patch['end']]==patch['old']
        assert item['response']==source[:patch['start']]+patch['new']+source[patch['end']:]
        assert '# True 0 + != and' in item['response']
        assert 'text = "True 0 + != and"' in item['response']
        assert 'other = f"False {1+2}"' in item['response'] and '\u03c0 = 3.14' in item['response']
        assert hashlib.sha256(item['response'].encode()).hexdigest()==item['response_sha256']
        assert set(extract_files(item['response']))=={'answer.py'}


def test_original_training_role_and_whole_file_layout_are_mandatory(suites):
    rows=list(map(json.loads,(ROOT/'data/exercism-support-context-v3/train.jsonl').read_text().splitlines()))
    row=next(r for r in rows if r['language']=='Python');spec=suites.suite(row)
    assert extract_files(row['response'],spec['expected_solution_files'])==spec['source_files']
    with pytest.raises(ValueError,match='training-only'):suites.suite({**row,'split':'validation'})
    with pytest.raises(ValueError):extract_files(row['response']+'\nextra prose')
    with pytest.raises(ValueError):extract_files(row['response'].replace('File: electric_bill.py','File: ../escape.py'))


def test_actual_preexec_sandbox_denies_outside_file_network_and_fork(suites,tmp_path):
    private=tmp_path/'owned-private-fixture.txt';private.write_text('must not be read by candidate')
    source='''import os,socket
from pathlib import Path
def read_file():return Path(%r).read_text()
def network():return socket.socket()
def fork():return os.fork()
'''%str(private)
    test='''import pytest
from answer import read_file,network,fork
@pytest.mark.parametrize("fn",[read_file,network,fork])
def test_confined(fn):
    with pytest.raises(PermissionError):fn()
'''
    observed=suites.run({'answer.py':source},suite(test))
    assert len(observed['inventory']['cases'])==3
    assert classify(observed)=='all_finite_published_cases_pass'


def test_actual_canonical_artifact_uses_its_original_separate_repository(suites):
    rows=list(map(json.loads,(ROOT/'data/exercism-support-context-v3/train.jsonl').read_text().splitlines()))
    row=next(r for r in rows if r['language']=='Python' and any(i.get('repository')=='exercism/problem-specifications'
        for i in r['source_files']+r['published_test_artifacts']))
    item=next(i for i in row['source_files']+row['published_test_artifacts'] if i.get('repository')=='exercism/problem-specifications')
    assert hashlib.sha256(suites.read_item(item)).hexdigest()==item['sha256']
    suites.suite(row)
    with pytest.raises(ValueError,match='repository/revision'):suites.read_item({**item,'repository':'unknown'})
