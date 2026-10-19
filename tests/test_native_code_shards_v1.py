import json
from pathlib import Path

import pytest

from slm_lab.code_tasks import TASKS
from slm_lab.corpus import file_hash
from slm_lab import native_code_shards_v1 as sharding


def reports(tmp_path):
    envelopes=[]
    for rank in range(4):
        cases=[dict(language=lang,task=TASKS[i][0],candidate=dict(syntax_valid=True,functional_passed=i%3==0),
            efficiency_stress_passed=False) for lang in sharding.LANGUAGES for i in sharding.positions(rank,4)]
        raw=tmp_path/f'raw-{rank}.json';raw.write_text(json.dumps(dict(checkpoint_sha256='a'*64,generation_budget=512,
            decoding='greedy, cached FP32',prompt_mode='completion',suite_sha256=file_hash(Path(sharding.__file__).with_name('code_tasks.py')),tokenizer_sha256='c'*64,
            inference_device_type='cpu',inference_precision='fp32',evaluation_provenance={},limitations=[],
            task_count=len(cases),results=cases,syntax_pass_rate=1.,functional_pass_at_1=sum(r['candidate']['functional_passed'] for r in cases)/len(cases),
            efficiency_stress_pass_rate=0.)))
        envelope=tmp_path/f'shard-{rank}.json';envelope.write_text(json.dumps(dict(format=sharding.FORMAT,rank=rank,world=4,
            original_task_positions=sharding.positions(rank,4),prompt_mode='completion',generation_budget=512,raw_report=str(raw),
            raw_report_sha256=file_hash(raw),sharding_implementation_sha256=file_hash(sharding.__file__))))
        envelopes.append(envelope)
    return envelopes


def test_complete_reconciliation_keeps_every_case_and_rederives_metrics(tmp_path):
    inputs=reports(tmp_path);merged=sharding.reconcile(inputs,tmp_path/'complete.json')
    assert merged['task_count']==24 and len(merged['results'])==24 and merged['functional_pass_at_1']==pytest.approx(1/3)
    assert merged['by_language']['JavaScript']['tasks']==12
    assert merged['exact_all_original_cases_reconciled']


def test_omission_or_changed_weights_cannot_be_reported_as_complete(tmp_path):
    inputs=reports(tmp_path)
    with pytest.raises(ValueError):sharding.reconcile(inputs[:-1],tmp_path/'missing.json')
    raw=tmp_path/'raw-1.json';d=json.loads(raw.read_text());d['checkpoint_sha256']='z'*64;raw.write_text(json.dumps(d))
    env=json.loads(inputs[1].read_text());env['raw_report_sha256']=file_hash(raw);inputs[1].write_text(json.dumps(env))
    with pytest.raises(ValueError,match='must match'):sharding.reconcile(inputs,tmp_path/'changed.json')
