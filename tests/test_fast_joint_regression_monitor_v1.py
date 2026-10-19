import importlib.util
from pathlib import Path
import sys
import pytest


def test_only_matched_broad_nonimprovement_with_large_syntax_loss_triggers_retirement():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('regression_monitor_test',root/'scripts/monitor_fast_joint_regression_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    b=dict(syntax=20,correct=1,efficient=0,knowledge_generated=0.,math_correct=1,OCR_exact=0.,OCR_CER=.95,protocol=dict(budget=128))
    p=dict(b,syntax=9,correct=0,math_correct=0,OCR_CER=2.)
    assert m.regression(b,p)
    assert not m.regression(b,None)
    assert not m.regression(b,dict(p,correct=2))
    assert not m.regression(b,dict(p,syntax=16))
    assert not m.regression(b,dict(p,math_correct=2))
    with pytest.raises(ValueError):m.regression(b,dict(p,protocol=dict(budget=512)))
