"""Any change outside the declared resource-only edits must reject."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]


def verified(spec,derived):
    original=Path(spec['original']).read_text()
    assert hashlib.sha256(original.encode()).hexdigest()==spec['original_sha256']
    for change in spec['allowed_resource_and_import_changes']:
        assert original.count(change['before'])==1
        original=original.replace(change['before'],change['after'])
    if original!=derived:raise ValueError('Unreviewed primary/runtime change in derived assessment')
    return True


def specs():return json.loads((ROOT/'reports/exclusive-inference-driver-derivation-v1.json').read_text())['drivers']


@pytest.mark.parametrize('spec',specs())
def test_full_driver_is_exactly_original_plus_declared_resource_edits(spec):
    assert verified(spec,Path(spec['derived']).read_text())


@pytest.mark.parametrize('fault',['code_budget','code_generation','code_scoring','knowledge_budget','knowledge_gold'])
def test_any_primary_change_rejects(fault):
    code,knowledge=specs();spec=knowledge if fault.startswith('knowledge') else code
    text=Path(spec['derived']).read_text()
    pairs={'code_budget':("budget=manifest['generation_budget']","budget=128"),
        'code_generation':('record=generate_record(model,tokenizer,encoder,question,budget,device,amp)','record=generate_record(model,tokenizer,encoder,question,128,device,amp)'),
        'code_scoring':('tools,assess_program)','tools,lambda *args: True)'),
        'knowledge_budget':('max_new_tokens=32','max_new_tokens=1'),
        'knowledge_gold':("gold=int(row['answer_index'])","gold=0")}
    before,after=pairs[fault];assert before in text
    with pytest.raises(ValueError):verified(spec,text.replace(before,after))


def test_answer_parsing_and_interval_functions_are_unchanged():
    old=ast.parse((ROOT/'src/slm_lab/knowledge_evaluation.py').read_text())
    new=ast.parse((ROOT/'scripts/assess_knowledge_exclusive_v1.py').read_text())
    for name in ['letter','interval']:
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b)
