import importlib
import json
from pathlib import Path
import shutil

import pytest

from slm_lab.corpus import file_hash


ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def original_evidence_pair(tmp_path,monkeypatch):
    # Real completed original baseline, copied only as a comparison-test fixture.
    # These copies are never labelled as trained-candidate capability evidence.
    monkeypatch.syspath_prepend(str(ROOT/'scripts'))
    module=importlib.import_module('assess_code_residual_lr_pilot_v1')
    original=ROOT/'reports/codenet-language-completed-baseline-v1';candidate=tmp_path/'copied-original-code';candidate.mkdir()
    for name in ['plan.json','results.json','generation-seal.json','generations.jsonl','execution.jsonl']:
        shutil.copyfile(original/name,candidate/name)
    qa=tmp_path/'copied-original-knowledge.json'
    shutil.copyfile(ROOT/'reports/language-completed-knowledge-baseline-v1/knowledge.json',qa)
    return module,candidate,qa,tmp_path/'comparison.json'


def write_json(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def test_known_original_evidence_recomputes_every_code_and_knowledge_denominator(original_evidence_pair):
    module,folder,qa,output=original_evidence_pair
    r=module.compare_fixed_results(folder,qa,output)
    assert r['baseline_code']==r['candidate_code']
    assert r['candidate_code']['questions']==28 and r['candidate_code']['functional_correct']==0
    assert r['candidate_code']['stress_eligible_questions']==26
    assert r['baseline_raw_knowledge']==r['candidate_code_route_raw_knowledge']
    assert r['baseline_raw_knowledge']['questions']==559
    assert r['baseline_raw_knowledge']['generated_correct']==22


def test_self_consistently_resealed_wrong_prompt_is_rejected(original_evidence_pair):
    module,folder,qa,output=original_evidence_pair
    p=folder/'generations.jsonl';rows=[json.loads(line) for line in p.open()];rows[0]['prompt_input_ids_sha256']='f'*64
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    seal=json.loads((folder/'generation-seal.json').read_text());seal['generations_sha256']=file_hash(p)
    write_json(folder/'generation-seal.json',seal)
    report=json.loads((folder/'results.json').read_text());report['generations_sha256']=file_hash(p)
    report['generation_seal_sha256']=file_hash(folder/'generation-seal.json');write_json(folder/'results.json',report)
    with pytest.raises(ValueError,match='prompt IDs/budgets'):module.compare_fixed_results(folder,qa,output)


def test_relabelled_functional_success_does_not_survive_execution_readback(original_evidence_pair):
    module,folder,qa,output=original_evidence_pair
    p=folder/'execution.jsonl';rows=[json.loads(line) for line in p.open()];rows[0]['functional_passed']=True
    p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=json.loads((folder/'results.json').read_text());report['execution_records_sha256']=file_hash(p)
    write_json(folder/'results.json',report)
    with pytest.raises(ValueError,match='rederive'):module.compare_fixed_results(folder,qa,output)


def test_changed_QA_gold_is_rejected(original_evidence_pair):
    module,folder,qa,output=original_evidence_pair
    r=json.loads(qa.read_text());r['predictions'][0]['answer_index']=(r['predictions'][0]['answer_index']+1)%4
    write_json(qa,r)
    with pytest.raises(ValueError,match='gold'):module.compare_fixed_results(folder,qa,output)
