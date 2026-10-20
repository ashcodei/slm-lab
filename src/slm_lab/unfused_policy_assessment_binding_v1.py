"""Bind the unchanged full native workers to an unfused counted policy.

The separately typed human-preference worker is dispatched by the caller.
Other numerical generators, scoring functions and original populations remain
the original worker objects. No adapter is folded into reward features.
"""
from contextlib import contextmanager

SUITES = ('code','broad','preferences','QA','AI2D','MBPP','words')


@contextmanager
def bind(evaluator, *, reports, plan, load_student, register):
    if (tuple(plan.get('suites',())) != SUITES or plan.get('expected_student_values') != 999706442
            or plan.get('expected_native_base_values') != 984399690
            or plan.get('policy_factors_not_merged') is not True
            or not callable(load_student) or not callable(register)):
        raise ValueError('All original suites and the complete shared-base unfused sub1Bpolicy must remain explicit')
    previous = evaluator.REPORTS,evaluator.load_student,evaluator.legacy.registered
    worker,reconcile = evaluator.worker,evaluator.reconcile_suite
    def registered(subject,gpu):
        if type(gpu) is not int or not 0 <= gpu < 4:
            raise ValueError('One actual original physical GPU is required')
        owned = register(subject,gpu)
        if owned['owner'] != plan['owner']:
            raise ValueError('The assessment policy must belong to the exact live owned parent')
        return plan
    evaluator.REPORTS,evaluator.load_student,evaluator.legacy.registered = reports,load_student,registered
    try:
        yield worker,reconcile
        if evaluator.worker is not worker or evaluator.reconcile_suite is not reconcile:
            raise ValueError('Original full numerical worker and reconciliation functions must remain unchanged')
    finally:
        evaluator.REPORTS,evaluator.load_student,evaluator.legacy.registered = previous
