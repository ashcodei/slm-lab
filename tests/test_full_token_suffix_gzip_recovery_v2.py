"""Actual retained prefix snapshots; no binding to an actively written journal."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from slm_lab.bounded_gzip_observation_stream_v1 import RawBudget, Writer, lines
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from qualify_full_public_token_suffixes_gzip_recovery_CPU_v2 import validate_record


@pytest.fixture(scope='module')
def corpus(): return ExpandedExecutionCorpus(Path(__file__).resolve().parents[1])


def snapshot(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = root / 'reports/full-public-token-suffix-qualification-CPU-v1'
    # Copy only two already complete byte-LF observations. The immutable copy,
    # not the producer's still-live journal or progress, is the test input.
    with (source / 'all3721-reference-and14884-prefix-controls.jsonl').open('rb') as stream:
        raw = stream.readline() + stream.readline()
    path = tmp_path / 'immutable-two-original-control-rows.jsonl'
    path.write_bytes(raw)
    rows = [r for _, r in lines(path)]
    selected = {}
    for _, r in lines(source / 'all14884-native-view-descriptors.jsonl'):
        if r['source_position'] < 2: selected[(r['source_position'], r['nominal_missing_non_EOS_tokens'])] = r
        else: break
    assert len(rows) == 2 and len(selected) == 8
    return path, rows, selected


def test_two_actual_original_sources_all8views_and_all10controls_survive_lossless_recovery(corpus, tmp_path):
    original, rows, metadata = snapshot(tmp_path)
    counts = None
    budget = RawBudget(1024**3)
    with Writer(tmp_path / 'retained.jsonl.gz', root=tmp_path, budget=budget, reserve_bytes=0,
            guard=lambda: None, reserve=lambda *a, **k: None) as output:
        for position, (raw, row) in enumerate(lines(original)):
            actual = validate_record(corpus, position, row, metadata)
            counts = actual if counts is None else counts + actual
            output.append_raw(raw)
    assert counts['full_reference_runs'] == 2 and counts['prefix_control_runs'] == 8
    assert [r for _, r in lines(output.path)] == rows
    assert b''.join(raw for raw, _ in lines(output.path)) == original.read_bytes()
    (tmp_path / 'actual-closed-prefix-byte-readback.json').write_text(json.dumps(dict(
        actual_original_sources=2, actual_native_views=8, actual_original_runtime_controls=10,
        exact_immutable_snapshot_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        all_prior_bytes_roles_masks_and_execution_dispositions_preserved=True,
        live_original_journal_and_progress_not_bound=True, no_new_execution_or_neural_update=True,
        original_full3721source_scope_not_completed_by_this_test=True), indent=2))


def test_changed_source_target_level_reward_disposition_and_reference_prefix_reject(corpus, tmp_path):
    _, rows, metadata = snapshot(tmp_path)
    mutations = []
    bad = copy.deepcopy(rows[0]); bad['source_position'] = 2; mutations.append(bad)
    bad = copy.deepcopy(rows[0]); bad['original_CE_response_EOS_targets'] -= 1; mutations.append(bad)
    bad = copy.deepcopy(rows[0]); bad['views'][0]['nominal_missing_non_EOS_tokens'] = 16; mutations.append(bad)
    bad = copy.deepcopy(rows[0]); bad['views'][0]['proposed_nontrivial_bounded_execution_view'] = not bad['views'][0]['proposed_nontrivial_bounded_execution_view']; mutations.append(bad)
    bad = copy.deepcopy(rows[0]); bad['views'][0]['original_prefix_text'] += 'pass'; mutations.append(bad)
    bad = copy.deepcopy(rows[0]); bad['source_role'] = 'reserved'; mutations.append(bad)
    for bad in mutations:
        with pytest.raises(ValueError): validate_record(corpus, 0, bad, metadata)


def test_changed_closed_metadata_and_missing_runtime_view_cannot_be_reused(corpus, tmp_path):
    _, rows, metadata = snapshot(tmp_path)
    changed = copy.deepcopy(metadata)
    changed[(0, 4)]['original_response_EOS_targets'] -= 1
    with pytest.raises(ValueError): validate_record(corpus, 0, rows[0], changed)
    bad = copy.deepcopy(rows[0]); bad['views'].pop()
    with pytest.raises(ValueError): validate_record(corpus, 0, bad, metadata)
