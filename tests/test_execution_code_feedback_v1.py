"""Mutation boundaries and conservative labels on actual native test outcomes."""
import hashlib
import json

import pytest

from slm_lab.execution_code_feedback_v1 import mutations,classify,LEX
from slm_lab.exercise_java_runner import inventory


def test_mutation_changes_one_code_token_never_strings_comments_or_file_framing():
    response='File: src/main/java/A.java\n```java\nclass A {\n // true == false 0+1\n String s="true == false 0+1";\n char c=\'0\';\n boolean a(boolean b) { return b && true; }\n int f(int x) { if(x <= 1) return x + 0; return 1; }\n}\n```\n'
    values=mutations(response,'fixture-source',count=8)
    assert values==mutations(response,'fixture-source',count=8) and len(values)==7
    for value in values:
        p=value['patch']
        assert response[p['start']:p['end']]==p['old']
        assert value['response']==response[:p['start']]+p['new']+response[p['end']:]
        assert '// true == false 0+1' in value['response'] and '"true == false 0+1"' in value['response']
        assert "char c='0';" in value['response']


def test_float_exponent_signs_and_hexadecimal_digits_are_not_standalone_operators_or_integers():
    code='1e-3 + 0x1.fp-1 + .5f + 1.0 + 0b01 + 10L'
    assert [(m.lastgroup,m.group()) for m in LEX.finditer(code) if m.lastgroup=='number']==[
        ('number','1e-3'),('number','0x1.fp-1'),('number','.5f'),('number','1.0'),('number','0b01'),('number','10L')]
    response='File: src/main/java/A.java\n```java\nclass A { double x='+code+'; }\n```\n'
    assert all(v['patch']['old']=='+' for v in mutations(response,'fixture-numerals',count=12))


def test_java_prelexical_unicode_escape_files_are_qualified_out_of_mutation():
    response='File: src/main/java/A.java\n```java\nclass A { String x="\\u0022"; int y=1+0; }\n```\n'
    assert mutations(response,'fixture-unicode')==[]


def observed(failed=False):
    ids=['case1','case2'];raw=dict(mode='jupiter',junit4_version='4.13',registered=ids,started=ids,
        passed=['case1'] if failed else ids,failed=['case2'] if failed else [],skipped=[],container_errors=[])
    output='SLM_JAVA_INVENTORY\t'+json.dumps(raw)+'\n';measured=inventory(output.encode())
    return dict(compilation_passed=True,published_tests_passed=not failed,inventory=measured,
        execution=dict(returncode=0,limit=None,diagnostic='TEST_FAILURE: org.opentest4j.AssertionFailedError: expected1actual0' if failed else ''),
        raw_inventory_output=output,report_sha256=hashlib.sha256(output.encode()).hexdigest())


def test_same_complete_publisher_assertion_failure_is_a_finite_counterexample():
    reference=observed();bad=observed(True)
    assert classify(reference)=='all_finite_published_cases_pass'
    assert classify(bad,reference['inventory'])=='completed_published_assertion_counterexample'
    assert classify(bad)=='original_reference_test_failure_qualification'


@pytest.mark.parametrize('change,expected',[
    ('compile','compiler_or_environment_qualification'),('runtime','runtime_or_environment_qualification'),
    ('assertion','test_failure_without_retained_assertion_qualification'),('population','changed_case_population_qualification')])
def test_compiler_runtime_unproven_failure_and_changed_population_never_become_preferences(change,expected):
    reference=observed();bad=observed(True)
    if change=='compile':bad['compilation_passed']=False
    if change=='runtime':bad['execution']['returncode']=1
    if change=='assertion':bad['execution']['diagnostic']='TEST_FAILURE: java.lang.SecurityException: operation denied'
    if change=='population':reference['inventory']['cases'].append('anothercase')
    assert classify(bad,reference['inventory'])==expected


def test_altered_parsed_inventory_rejects_against_retained_raw_output():
    bad=observed(True);bad['inventory']['failed']=[]
    with pytest.raises(ValueError,match='raw Java inventory/hash'):classify(bad,observed()['inventory'])
