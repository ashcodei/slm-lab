from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer,models,trainers,pre_tokenizers,decoders

from slm_lab.corpus import file_hash
from slm_lab.full_math_shards_v1 import evaluate_shard,reconcile
from slm_lab.math_corpus import prompt
from slm_lab.math_evaluation import evaluate
from slm_lab.model import ModelConfig,TinyMultimodalLM


def test_actual_native_CPU_protocol_matches_complete_shards_and_rejects_missing_questions(tmp_path):
    data=tmp_path/'test.jsonl'
    rows=[dict(id='fixture-'+str(i),question=f'A box has {i+1} balls. How many balls are in the box?',
        answer_numerator=i+1,answer_denominator=1) for i in range(8)]
    data.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(official_test_examples=8,
        files_sha256={'test.jsonl':file_hash(data)},limitations=['Own small fixture, not published math data.'])))
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'));tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.train_from_iterator(['User: '+prompt(r['question'])+'\nAssistant:' for r in rows],
        trainers.BpeTrainer(vocab_size=128,special_tokens=['<pad>','<bos>','<eos>','<unk>'],show_progress=False))
    tokenizer.save(str(tmp_path/'tokenizer.json'))
    torch.manual_seed(31);torch.set_num_threads(1)
    config=ModelConfig(vocab_size=tokenizer.get_vocab_size(),hidden_size=16,intermediate_size=32,num_layers=2,
        num_heads=4,num_kv_heads=2,max_text_length=256,vision_hidden_size=8,vision_intermediate_size=16,
        vision_layers=1,vision_heads=2,image_size=16,patch_size=4,pool_size=2)
    model=TinyMultimodalLM(config);checkpoint=tmp_path/'model.pt'
    torch.save(dict(model=model.state_dict(),config=asdict(config),tokenizer_sha256=file_hash(tmp_path/'tokenizer.json')),checkpoint)
    native=evaluate(checkpoint,data,tmp_path/'native.json',device='cpu',max_new_tokens=8)
    paths=[]
    for i in range(4):
        path=tmp_path/f'shard-{i}.json';paths.append(path)
        evaluate_shard(checkpoint,data,path,shard=i,shards=4,device='cpu',max_new_tokens=8,expected_count=8)
    joined=reconcile(paths,data,tmp_path/'tokenizer.json',tmp_path/'complete.json',expected_count=8)
    assert joined['full_official_test'] and joined['evaluated_examples']==8
    for actual,expected in zip(joined['results'],native['results']):
        assert all(actual[k]==v for k,v in expected.items())
    assert joined['correct']==native['correct'] and joined['format_compliance']==native['format_compliance']
    journal=paths[0].with_suffix('.jsonl');journal.write_text(journal.read_text().splitlines()[0]+'\n')
    certificate=json.loads(paths[0].read_text());certificate.update(journal_sha256=file_hash(journal),evaluated_examples=1)
    paths[0].write_text(json.dumps(certificate))
    with pytest.raises(ValueError,match='Every assigned original question'):
        reconcile(paths,data,tmp_path/'tokenizer.json',tmp_path/'must-not-complete.json',expected_count=8)


def test_active_original_training_rejected_before_GPU_query_signals_or_output(tmp_path,monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/assess_apps_full_math_gpu_v1.py'
    monkeypatch.syspath_prepend(str(script.parent))
    spec=importlib.util.spec_from_file_location('own_full_math_driver_fixture',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'REPORTS',tmp_path/'reports')
    monkeypatch.setattr(module.resources,'same',lambda actor:dict(state='R'))
    def forbidden(*args,**kwargs):raise AssertionError('Active original training must be rejected first')
    monkeypatch.setattr(module.subprocess,'run',forbidden);monkeypatch.setattr(module.os,'kill',forbidden)
    with pytest.raises(ValueError,match='Never preempt'):
        module.parent({})
    assert not module.REPORTS.exists()
