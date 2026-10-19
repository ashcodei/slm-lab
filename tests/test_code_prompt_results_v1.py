"""Independently reparse current actual full model development reports."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from slm_lab.code_prompt_results_v1 import code_metrics,knowledge_metrics,source_metrics

ROOT=Path(__file__).resolve().parents[1]


def test_actual_current_joint_all559_raw_answers_and_likelihood_decisions_recompute():
    report=json.loads((ROOT/'reports/extended-curriculum-v1/knowledge.json').read_text())
    questions=list(map(json.loads,(ROOT/'data/knowledge-eval-v1/questions.jsonl').read_text().splitlines()))
    result=knowledge_metrics(report,report,questions,report['checkpoint_sha256'])
    assert (result['questions'],result['generated'],result['likelihood'])==(559,13,74)
    changed=deepcopy(report);changed['predictions'][0]['generated_correct']=not changed['predictions'][0]['generated_correct']
    with pytest.raises(ValueError,match='outcomes must recompute'):
        knowledge_metrics(changed,report,questions,report['checkpoint_sha256'])


@pytest.mark.parametrize('filename,syntax',[('code-quality.json',18),('code-quality-instruction.json',20)])
def test_actual_current_joint_all24_code_scores_recompute_and_aggregate_forgery_rejects(filename,syntax):
    report=json.loads((ROOT/'reports/extended-curriculum-v1'/filename).read_text())
    result=code_metrics(report,report,report['checkpoint_sha256'])
    assert result['syntax']==syntax and result['functional']==result['efficiency']==0
    changed=deepcopy(report);changed['functional_pass_at_1']=1.
    with pytest.raises(ValueError,match='aggregate'):
        code_metrics(changed,report,report['checkpoint_sha256'])


def test_unequal_length_source_metrics_keep_all_targets_and_are_not_correctness_labels():
    selected={};report={}
    for role in ['train','validation']:
        selected[role]=[dict(id='a',language='A',target_tokens=2),dict(id='b',language='B',target_tokens=6)]
        rows=[dict(id='a',language='A',role=role,targets=2,matched=2.,mismatched=3.),
            dict(id='b',language='B',role=role,targets=6,matched=4.,mismatched=3.)]
        report[role]=dict(rows=rows,examples=2,target_tokens=8,matched_token_mean_NLL=3.5,
            matching_gap_per_token=-.5,source_association_accuracy=.5,correctness_metric=False)
    result=source_metrics(report,selected)
    assert result['validation']['matching_gap_per_token']==-.5
    report['validation']['matched_token_mean_NLL']=3.
    with pytest.raises(ValueError,match='aggregate'):source_metrics(report,selected)
