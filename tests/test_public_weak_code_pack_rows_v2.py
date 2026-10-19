"""Actual closed public source observations; no new program/model execution."""
import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer

from slm_lab.public_code_instruction_screen_v1 import parse_whole
from slm_lab.public_synthetic_code_screen_policy_v1 import body_fingerprint
from slm_lab.public_token_suffix_curriculum_v1 import view
from slm_lab.public_weak_code_pack_rows_v2 import build

ROOT = Path(__file__).resolve().parents[1]
FRESH = ROOT / 'reports/full-public-code-shard-readback-CPU-v1'
PRIOR = ROOT / 'reports/full-public-code-shard-qualification-CPU-v1'
DATA = ROOT / 'data/public-code-instruction-candidate-v1'


def rows(path):
    with Path(path).open('rb') as stream:
        for raw in stream:
            assert raw.endswith(b'\n')
            yield json.loads(raw)


@pytest.fixture(scope='module')
def sources():
    roles = {r['original_shard_position']: r for r in rows(FRESH / 'all-original-candidate-roles.jsonl')}
    chosen = {}
    for observed in rows(FRESH / 'all100000-fresh-readback-observations.jsonl'):
        if observed['original_candidate']:
            status = observed['fresh_whole_program_execution']['status']
            key = observed['source_role'] if status == 'all_published_tests_pass' else status
            if key not in chosen: chosen[key] = observed
        elif observed['original_shard_position'] == 30: chosen['veto'] = observed
    positions = {r['original_shard_position'] for r in chosen.values()}
    prior = {r['original_shard_position']: r for r in rows(PRIOR / 'all100000-original-dispositions.jsonl') if r['original_shard_position'] in positions}
    original = {}
    position = 0
    for batch in pq.ParquetFile(DATA / 'train-00000-of-00050.parquet').iter_batches(batch_size=256, use_threads=False):
        for row in batch.to_pylist():
            if position in positions: original[position] = row
            position += 1
    veto = json.loads((ROOT / 'reports/public-code-instruction-precision-counterexample-CPU-v1/semantic-veto.json').read_text())
    bad = body_fingerprint(parse_whole(original[30])['program'])
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/public-reasoning-curriculum-v1/tokenizer.json'))
    def call(key, *, changed_observation=None, changed_role=None):
        observed = chosen[key] if changed_observation is None else changed_observation
        p = observed['original_shard_position']
        return build(original[p], prior[p], observed, roles.get(p) if changed_role is None else changed_role, tokenizer,
            veto_sha256=veto['original_entry_sha256'], veto_AST_sha256=bad)
    return chosen, call


@pytest.mark.parametrize('role', ['train', 'reserved'])
def test_actual_whole_source_two_closed_executions_roles_full_native_masks_and_teacher_provenance(sources, role, tmp_path):
    chosen, call = sources
    value, status = call(role)
    row, ids, labels = value
    assert status == 'weak_SFT_positive:' + role
    assert row['split'] == ('train' if role == 'train' else 'validation')
    assert ids[0] == 1 and ids[-1] == labels[-1] == 2 and len(ids) <= 4096
    assert sum(t != -100 for t in labels) == chosen[role]['complete_native']['response_EOS_targets']
    assert row['response'] == row['original_entry']['output'] and row['prompt'] == row['original_entry']['input']
    assert row['model_generated_examples'] and row['external_model_teacher_used']
    assert not row['human_preferences'] and not row['own_model_on_policy']
    assert row['use_as_RL_reward_oracle_forbidden'] and row['admitted_use'] == 'weak_supervised_coding_only'
    (tmp_path / ('actual-public-weak-' + role + '-row.json')).write_text(json.dumps(dict(
        original_source_position=row['original_shard_position'], original_source_id=row['original_source_id'],
        sealed_role=role, split=row['split'], complete_native_positions=len(ids), response_EOS_targets=sum(t != -100 for t in labels),
        original_whole_response_and_true_EOS_retained=True, two_actual_closed_test_agreement_observations_preserved=True,
        response_origin=row['response_origin'], tests_are_public_synthetic_not_independent_oracle=True,
        use_as_RL_reward_oracle_forbidden=True, models_constructed=0, new_program_executions=0, production_updates=0), indent=2))


@pytest.mark.parametrize('status', ['runtime_resource_or_confinement_abstention', 'published_test_mismatch'])
def test_actual_fresh_failures_remain_excluded_without_reassigning_original_train_roles(sources, status):
    chosen, call = sources
    before = copy.deepcopy(chosen[status])
    value, disposition = call(status)
    assert value is None and disposition == 'preserved_fresh_' + status
    assert chosen[status] == before and chosen[status]['source_role'] == 'train'


def test_actual_known_semantically_wrong_source_stays_excluded_despite_provider_pass_metadata(sources):
    _, call = sources
    value, status = call('veto')
    assert value is None and status == 'original_exclusion:known_independent_semantic_veto'


def test_changed_actual_returned_literal_cannot_keep_an_all_pass_label(sources):
    chosen, call = sources
    observed = copy.deepcopy(chosen['train'])
    actual = observed['fresh_whole_program_execution']['actual_value_literals']
    assert actual[0] == 'True'
    actual[0] = 'False'
    with pytest.raises(ValueError): call('train', changed_observation=observed)


def test_reserved_role_cannot_silently_become_training(sources):
    chosen, call = sources
    changed = {k: v for k, v in chosen['reserved'].items() if k in ['original_shard_position', 'source_role',
        'declared_duplicate_component_sha256', 'declared_duplicate_component_size', 'role_fixed_before_fresh_replay', 'optimization_forbidden']}
    changed.update(source_role='train', optimization_forbidden=False)
    with pytest.raises(ValueError): call('reserved', changed_role=changed)


def test_public_synthetic_positive_cannot_enter_original_human_code_RL_curriculum(sources):
    _, call = sources
    (row, ids, labels), _ = call('train')
    with pytest.raises(ValueError): view(None, row, ids, labels, 4)
