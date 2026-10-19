"""Original full Java interfaces and independently recomputed promotion guards."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer

import slm_lab.exercise_java_evaluation as evaluation
from slm_lab.corpus import file_hash
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

from test_exercise_java_runner import suites, fixture_suite, program


def test_every_real_java_prompt_support_and_full_reference_fit_without_gold_in_model_input(suites, tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = root / 'data/exercism-support-context-v3/valid.jsonl'
    token = Tokenizer.from_file(str(root / 'data/codenet-durable-pack-v2/tokenizer.json'))
    encoder = Encoder(token, ModelConfig.load(root / 'data/codenet-durable-pack-v2/model-config.json'), data.parent)
    rows = [r for r in map(json.loads,data.open()) if r['language']=='Java']
    checked=[]
    for row in rows:
        suite = suites.suite(row)
        prompt,tokens,budget = evaluation.case_layout(row,suite,encoder)
        assert row['response'] not in prompt
        assert evaluation.question(row,suite)['response']==row['response']
        assert budget >= len(token.encode(row['response']).ids)+1
        assert len(tokens)+budget <= 4096
        for raw in suite['editors'].values():
            assert raw.decode() in prompt
        checked.append(dict(id=row['id'],prompt_tokens=len(tokens),response_tokens=len(token.encode(row['response']).ids)+1,
            generation_budget=budget,provided_files=list(suite['editors']),test_classes=suite['test_classes']))
    assert len(checked)==35
    satellite=next(r for r in checked if r['id'].endswith(':satellite'))
    assert satellite['test_classes']==['SatelliteTest','TreeTest']
    assert not encoder.truncated
    (tmp_path / 'actual35-Java-interfaces.json').write_text(json.dumps(checked,indent=2))


@pytest.fixture(scope='module')
def executed_case(suites):
    suite=fixture_suite()
    good=suites.run(program('public static int answer() { return 42; }'),suite)
    bad=suites.run(program('public static int answer() { return 0; }'),suite,good['inventory'])
    assert good['published_tests_passed'] and not bad['published_tests_passed']
    def row(i, passed):
        return dict(id='fixture-java-'+str(i),group_id='fixture-group-'+str(i),prompt_sha256='a'*64,
            generation_budget=2048,original_test_sha256=suite['test_sha256'],original_editor_sha256={},
            expected_solution_files=suite['expected_solution_files'],expected_registered_cases=good['inventory']['cases'],
            raw_response='File: src/main/java/Probe.java\n```java\npublic class Probe { public static int answer() { return '
                + ('42' if passed else '0')+'; } }\n```',full_file_delivery=True,compilation_passed=True,
            published_tests_passed=passed,execution=good if passed else bad)
    return row


def reports(make_row):
    value={key:'same-fixture' for key in ['evaluation_provenance','data_sha256','manifest_sha256','tokenizer_sha256',
        'source_archive_sha256','Java_toolchain_manifest_sha256','test_framework_manifest_sha256','reference_report_sha256',
        'declared_generation_budget','protocol','inference_precision','inference_device_type']}
    value.update(checkpoint_sha256='a'*64,results=[make_row(i,True) for i in range(35)],full_file_delivery_rate=1.,
        compilation_success_rate=1.,published_module_pass_rate=1.)
    return value


def test_real_jvm_outcomes_drive_matched_guard_and_loss_of_one_previously_correct_module_fails(executed_case,tmp_path):
    before=reports(executed_case)
    after=deepcopy(before)
    after['checkpoint_sha256']='b'*64
    assert evaluation.compare_reports(before,after)['passed']
    after['results'][11]=executed_case(11,False)
    after['published_module_pass_rate']=34/35
    comparison=evaluation.compare_reports(before,after)
    assert not comparison['passed'] and comparison['previously_correct_modules_lost']==['fixture-java-11']
    (tmp_path/'fixture-guard-comparison.json').write_text(json.dumps(comparison))


@pytest.mark.parametrize('change',['pass_flag','aggregate','raw_inventory','reference_cases'])
def test_invented_passes_or_changed_evidence_cannot_satisfy_java_guard(executed_case,change):
    before=reports(executed_case)
    after=deepcopy(before)
    if change=='pass_flag':
        after['results'][0]=executed_case(0,False)
        after['results'][0]['published_tests_passed']=True
    elif change=='aggregate':
        after['published_module_pass_rate']=0.
    elif change=='raw_inventory':
        after['results'][0]['execution']['raw_inventory_output']='forged inventory'
    else:
        after['results'][0]['expected_registered_cases']=['unregistered']
    with pytest.raises(ValueError):
        evaluation.compare_reports(before,after)


def test_incomplete_original_reference_gate_runs_before_any_model_load(tmp_path,monkeypatch):
    data=tmp_path/'valid.jsonl'
    data.write_text('')
    (tmp_path/'manifest.json').write_text(json.dumps(dict(files_sha256={'valid.jsonl':file_hash(data)})))
    ref=tmp_path/evaluation.REFERENCE
    ref.parent.mkdir(parents=True)
    ref.write_text(json.dumps(dict(complete_all35_reference_suites_passed=False)))
    monkeypatch.setattr(evaluation,'ROOT',tmp_path)
    monkeypatch.setattr(evaluation.shutil,'disk_usage',lambda p:SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(evaluation,'load_checkpoint',lambda *a,**k:pytest.fail('Incomplete references reached model/GPU load'))
    with pytest.raises(ValueError,match='All35 original Java references'):
        evaluation.evaluate(tmp_path/'unloaded-model.pt',data,tmp_path/'new-output.json')
    assert not (tmp_path/'new-output-evidence').exists()
