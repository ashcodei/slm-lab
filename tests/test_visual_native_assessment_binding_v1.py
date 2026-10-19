"""Dispatch-only controls; no fake worker is presented as a neural assessment."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from slm_lab.visual_native_assessment_binding_v1 import bind, SUITES


def fixture():
    model_loader = lambda *args: 'old'
    register = lambda *args: dict(owner='old')
    evaluator = SimpleNamespace(REPORTS=Path('original'), load_student=model_loader,
        legacy=SimpleNamespace(registered=register), reconcile_suite=lambda *args: 'same-reconciler')
    def worker(suite, gpu):
        return evaluator.legacy.registered(suite, gpu), evaluator.load_student(), evaluator.REPORTS
    evaluator.worker = worker
    plan = dict(suites=list(SUITES), expected_student_values=984399690, owner='exact-new-owner')
    return evaluator, plan


def test_only_owned_binding_changes_and_original_numerical_function_objects_restore():
    evaluator, plan = fixture()
    before = evaluator.REPORTS, evaluator.load_student, evaluator.legacy.registered
    worker, reconcile = evaluator.worker, evaluator.reconcile_suite
    with bind(evaluator, reports=Path('new'), plan=plan, load_student=lambda: 'new-native',
            register=lambda *args: dict(owner='exact-new-owner')) as functions:
        assert functions == (worker, reconcile)
        assert evaluator.worker is worker and evaluator.reconcile_suite is reconcile
        assert worker('QA', 2) == (plan, 'new-native', Path('new'))
    assert before == (evaluator.REPORTS, evaluator.load_student, evaluator.legacy.registered)
    assert evaluator.worker is worker and evaluator.reconcile_suite is reconcile


def test_worker_failure_restores_all_new_process_only_bindings():
    evaluator, plan = fixture()
    before = evaluator.REPORTS, evaluator.load_student, evaluator.legacy.registered
    with pytest.raises(RuntimeError, match='deliberate control'):
        with bind(evaluator, reports=Path('new'), plan=plan, load_student=lambda: 'new-native',
                register=lambda *args: dict(owner='exact-new-owner')):
            raise RuntimeError('deliberate control')
    assert before == (evaluator.REPORTS, evaluator.load_student, evaluator.legacy.registered)


@pytest.mark.parametrize('case', ['wrong_owner', 'short_suite_list', 'tiny_model', 'wrong_GPU'])
def test_scope_and_owned_registration_cannot_be_reduced_or_reassigned(case):
    evaluator, plan = fixture()
    if case == 'short_suite_list':
        plan['suites'].remove('words')
    elif case == 'tiny_model':
        plan['expected_student_values'] = 68098
    with pytest.raises(ValueError):
        with bind(evaluator, reports=Path('new'), plan=plan, load_student=lambda: 'new-native',
                register=lambda *args: dict(owner='other' if case == 'wrong_owner' else 'exact-new-owner')):
            evaluator.worker('QA', 4 if case == 'wrong_GPU' else 0)
