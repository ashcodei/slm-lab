"""Parse publisher pass-rate metadata without converting it into a reward."""
import math


def normalized_pass_rate(value):
    # The pinned NVIDIA Parquet schema stores this numeric field as text.
    # Booleans must not acquire a perfect-score label through float(True).
    if type(value) not in (str, int, float):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(number) or not (number == -1. or 0. <= number <= 1.):
        return None
    return number


def reported_all_pass(value):
    return normalized_pass_rate(value) == 1.
