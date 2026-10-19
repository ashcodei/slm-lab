"""Output formatting refinement for the unchanged held algorithm case set.

The original stagecoach statement permits any decimal precision, bounded here
only by the executor's output cap. Decimal arithmetic checks its exact rational
error; no float conversion, arbitrary32-fractional-digit limit or generic reward.
"""
from decimal import Decimal, DecimalException, localcontext
import re

from .codenet_algorithm_assessment_v1 import check_output as legacy_check
from .codenet_sample_oracles_v1 import stagecoach_times

MAXIMUM_OUTPUT_BYTES = 1024**2
REAL = re.compile(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?')
INTEGER = re.compile(r'([+-]?)([0-9]+)')


def check_output(problem, text, expected, actual):
    if not isinstance(actual, str) or len(actual.encode()) > MAXIMUM_OUTPUT_BYTES:
        return False
    if problem != 'p00719':
        # Leading zeroes do not change integer outputs or property witnesses.
        normalized = []
        for token in actual.split():
            match = INTEGER.fullmatch(token)
            if match is None:
                return False
            sign, digits = match.groups()
            digits = digits.lstrip('0') or '0'
            if len(digits) > 30:
                return False
            normalized.append(sign + digits)
        return legacy_check(problem, text, expected, ' '.join(normalized))
    truth = stagecoach_times(text)
    lines = actual.strip().splitlines()
    if len(lines) != len(truth):
        return False
    for value, raw in zip(truth, lines):
        raw = raw.strip()
        if value is None:
            if raw != 'Impossible':
                return False
            continue
        if not REAL.fullmatch(raw):
            return False
        try:
            observed = Decimal(raw)
            # Any feasible route uses at most8 tickets, roads<=100 and horses>=1.
            # Positive feasible optima are>=0.1; reject remote exponents before arithmetic.
            if not observed.is_finite() or observed < Decimal('.099') or observed > Decimal('800.001'):
                return False
            with localcontext() as context:
                context.prec = len(raw) + 32
                delta = abs(observed * value.denominator - value.numerator)
                if delta * 1000 > value.denominator:
                    return False
        except (DecimalException, ValueError, OverflowError):
            return False
    return True
