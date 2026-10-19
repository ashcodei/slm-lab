from copy import deepcopy
import json
from pathlib import Path

import pytest

from slm_lab.public_billion_candidate_v1 import QUOTAS
from slm_lab.public_billion_metadata_v1 import manifest,completion,validate_summary


def actual():
    root=Path(__file__).resolve().parents[1]
    failure=json.loads((root/'reports/public-billion-causal-candidate-CPU-v2/failure.json').read_text())
    progress=json.loads((root/'reports/public-billion-causal-candidate-CPU-v2/progress.json').read_text())
    return failure['partial_candidate'],progress


def test_actual_complete_summary_with_existing_optimization_field_builds_both_metadata_records():
    summary,progress=actual()
    result=manifest(summary,QUOTAS,'a'*64,{'original-native':'b'*64})
    assert result['optimization_admitted'] is False and result['actual_train_targets']==1000085544
    done=completion(summary,QUOTAS,rows=progress['original_rows_observed'],source_files=progress['actual_source_files'],
        dispositions=progress['actual_prefix_dispositions'],bindings={'manifest':'c'*64})
    assert done['complete'] and done['optimization_admitted'] is False and done['neural_training_updates']==0
    assert result['native_context_token_counts']==summary['native_context_token_counts']


def test_missing_quota_or_changed_native_target_denominator_cannot_be_finalized():
    summary,_=actual()
    wrong=deepcopy(summary);wrong['train_target_tokens_by_quota']['code:Rust']=0
    with pytest.raises(ValueError):validate_summary(wrong,QUOTAS)
    wrong=deepcopy(summary);wrong['native_context_token_counts']['valid']-=1
    with pytest.raises(ValueError):validate_summary(wrong,QUOTAS)
