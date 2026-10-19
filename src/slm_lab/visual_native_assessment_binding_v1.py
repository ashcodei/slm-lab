"""Bind an owned new native candidate to the unchanged full assessment worker.

Only output location, exact candidate loading and owned registration change.
The existing worker, scorers, generators, populations, budgets and reconciler
function objects remain unchanged. Bindings are local to a new process and
are always restored, including when its evaluation raises.
"""
from contextlib import contextmanager

SUITES = ('code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words')


@contextmanager
def bind(evaluator, *, reports, plan, load_student, register):
    if (tuple(plan.get('suites', ())) != SUITES or plan.get('expected_student_values') != 984399690
            or not callable(load_student) or not callable(register)):
        raise ValueError('Every original complete native assessment suite and actual full-size student must be retained')
    previous = evaluator.REPORTS, evaluator.load_student, evaluator.legacy.registered
    worker, reconcile = evaluator.worker, evaluator.reconcile_suite

    def registered(subject, gpu):
        if type(gpu) is not int or not 0 <= gpu < 4:
            raise ValueError('One actual original physical GPU is required')
        owned = register(subject, gpu)
        if owned['owner'] != plan['owner']:
            raise ValueError('The assessment candidate must belong to the exact live owned root run')
        return plan

    evaluator.REPORTS = reports
    evaluator.load_student = load_student
    evaluator.legacy.registered = registered
    try:
        yield worker, reconcile
        if evaluator.worker is not worker or evaluator.reconcile_suite is not reconcile:
            raise ValueError('Numerical full assessment and reconciliation functions must remain unchanged')
    finally:
        evaluator.REPORTS, evaluator.load_student, evaluator.legacy.registered = previous
