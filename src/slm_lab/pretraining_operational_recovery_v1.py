"""Keep runtime ownership separate from the immutable training mathematics."""
import json
from pathlib import Path

from .corpus import file_hash


def math_for_runtime(runtime):
    path = Path(runtime['original_neural_plan'])
    if (file_hash(path) != runtime['original_neural_plan_sha256']
            or runtime.get('neural_options_unchanged') is not True):
        raise ValueError('Original immutable neural plan required for operational recovery')
    return json.loads(path.read_text())


def validate_interruption(plan, children, receipt, same):
    required = [plan['owner'], *children]
    recorded = {(row['pid'], row['ticks']) for row in receipt.get('old_registered_births', [])}
    if (receipt.get('complete') is not True or receipt.get('all_old_registered_births_terminal') is not True
            or any((row['pid'], row['ticks']) not in recorded for row in required)
            or any(same(row) for row in required)):
        raise ValueError('Every exact old owner and registered worker must be inventoried and terminal')
