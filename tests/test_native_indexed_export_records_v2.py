"""Stream actual closed multi-attempt observations; reject lost source updates."""
import copy
from pathlib import Path

import pytest

from slm_lab.native_history_summary_v1 import summarize
from slm_lab.native_suffix_standalone_export_v2 import indexed_records
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from test_native_suffix_indexed_stage_v4 import BUDGETS

ROOT = Path(__file__).resolve().parents[1]
ACTUAL = ROOT / 'reports/native-indexed-neural-v4-proof-v1/pytest/test_actual_four_replicas_both0'


@pytest.mark.parametrize('profile', ['anchor_CE', 'anchor_execution'])
def test_actual_closed_multi_attempt_records_retain_all_original_CE_and_empty_tail(profile):
    saved, _, _ = read_latest(ACTUAL / profile / 'durable')
    assert len({row['directory'] for row in saved['metrics']}) == 2
    checks = []
    rows = indexed_records(saved['metrics'], reference=saved['reference_binding'], budgets=BUDGETS,
        guard=lambda: checks.append(True))
    summary = summarize(rows, group_size=8)
    assert summary['actual_complete_updates'] == 3
    assert summary['actual_source_presentations'] == 9
    assert summary['actual_unconditional_original_CE_targets'] == 612
    assert summary['actual_complete_CE_only_presentations'] == 1
    assert summary['actual_single_source_three_empty_rank_steps'] == 1
    assert summary['actual_committed_prefix_trajectory_count'] == 64
    assert checks


@pytest.mark.parametrize('kind', ['omit', 'reorder', 'sequence', 'previous', 'bounds', 'reference'])
def test_actual_closed_record_chain_rejects_missing_reordered_or_replaced_sources(kind):
    saved, _, _ = read_latest(ACTUAL / 'anchor_execution/durable')
    receipts, reference, budgets = copy.deepcopy(saved['metrics']), copy.deepcopy(saved['reference_binding']), dict(BUDGETS)
    if kind == 'omit': del receipts[0]
    elif kind == 'reorder': receipts[0], receipts[1] = receipts[1], receipts[0]
    elif kind == 'sequence': receipts[-1]['record']['sequence'] = 3
    elif kind == 'previous': receipts[-1]['record']['previous_receipt_sha256'] = 'e' * 64
    elif kind == 'bounds': budgets['maximum_records'] += 1
    else: reference['sources'][str(Path(__file__).resolve())] = 'e' * 64
    with pytest.raises(ValueError):
        list(indexed_records(receipts, reference=reference, budgets=budgets, guard=lambda: None))
