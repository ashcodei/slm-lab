import copy
import json
from pathlib import Path

import pytest

from slm_lab.apps_native_metrics_v1 import NAMES,matched,summarize

ROOT=Path(__file__).resolve().parents[1]


def reports():
    folder=ROOT/'reports/apps-full-language-v1/baseline'
    return {name:json.loads((folder/name).read_text()) for name in NAMES}


def test_all_actual_native_baseline_schemas_recount_and_self_match():
    original=reports();questions=list(map(json.loads,(ROOT/'data/knowledge-eval-v1/questions.jsonl').read_text().splitlines()))
    sizes={'code-completion.json':24,'code-instruction.json':24,'public-Python.json':175,'C++.json':12,
        'knowledge.json':559,'ocr.json':170,'objects.json':164,'vision.json':128}
    for name,report in original.items():
        assert matched(report,report,name)
        value=summarize(report,name,questions if name=='knowledge.json' else None)
        assert value.get('tasks',value.get('questions',value.get('examples',value.get('images'))))==sizes[name]


def test_false_pass_wrong_raw_answer_and_changed_protocol_or_pixels_reject():
    original=reports()
    changed=copy.deepcopy(original['public-Python.json']);changed['results'][0]['execution']['functional_passed']=True
    with pytest.raises(ValueError,match='every actual finite native case'):summarize(changed,'public-Python.json')
    questions=list(map(json.loads,(ROOT/'data/knowledge-eval-v1/questions.jsonl').read_text().splitlines()))
    changed=copy.deepcopy(original['knowledge.json']);changed['predictions'][0]['generated_correct']=not changed['predictions'][0]['generated_correct']
    with pytest.raises(ValueError,match='reproduce every domain outcome'):summarize(changed,'knowledge.json',questions)
    changed=copy.deepcopy(original['ocr.json']);changed['results'][0]['character_edits']+=1
    with pytest.raises(ValueError,match='OCR transcription'):summarize(changed,'ocr.json')
    changed=copy.deepcopy(original['objects.json']);changed['results'][0]['input_pixels_sha256']='0'*64
    with pytest.raises(ValueError,match='image intervention'):matched(original['objects.json'],changed,'objects.json')
    changed=copy.deepcopy(original['code-completion.json']);changed['generation_budget']=256
    with pytest.raises(ValueError,match='protocol/input mismatch'):matched(original['code-completion.json'],changed,'code-completion.json')
