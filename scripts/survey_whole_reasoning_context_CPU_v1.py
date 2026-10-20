"""Measure whole-response context geometry without admitting more training data.

Uses only closed token-length and screening ledgers. Longer responses have
not received the complete target overlap/inclusion checks applied at 4K.
No model context, source role, token sequence or live worker changes.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
JOIN = Path('/tmp/slm-public-reasoned-code-question-join-v2')
SCREEN = Path('/tmp/slm-public-reasoned-code-screen-v2')
OUT = ROOT / 'reports/whole-reasoning-context-survey-v1'
WIDTHS = (4096, 8192, 16384, 32768)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('Fresh CUDA-hidden source-only geometry survey required')
    receipts = {p: json.loads((p / 'complete.json').read_text()) for p in (JOIN, SCREEN)}
    files = {str(Path(__file__).resolve()): sha(__file__)}
    for root, ledger in [(JOIN, 'every-candidate-disposition.jsonl'),
                         (SCREEN, 'every-original-row-screen.jsonl')]:
        receipt = receipts[root]
        if receipt.get('complete') is not True or sha(root / ledger) != receipt['ledger_sha256']:
            raise ValueError('Both exact closed complete source ledgers are required')
        for path in (root / 'complete.json', root / ledger):
            files[str(path)] = sha(path)
    selected = {}
    with (SCREEN / 'every-original-row-screen.jsonl').open() as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            assert row['original_shard_position'] == index and row['training_admitted'] is False
            if not row['fresh_execution_pending']:
                continue
            assert row['status'] == 'screened_metadata_positive_pending_fresh_execution'
            assert row['split'] == 'train' and row['source_role'] in ('train', 'reserved')
            assert row['fixed_pre_execution_role']['source_role'] == row['source_role']
            assert row['optimization_forbidden'] is (row['source_role'] == 'reserved')
            selected[index] = row
    assert index + 1 == 20000 and len(selected) == 6116
    roles = Counter(row['source_role'] for row in selected.values())
    assert roles == {'train': 5509, 'reserved': 607}
    totals = {str(width): {role: Counter() for role in roles} for width in WIDTHS}
    screened = {role: Counter() for role in roles}
    rows = []
    with (JOIN / 'every-candidate-disposition.jsonl').open() as stream:
        for index, line in enumerate(stream):
            source = json.loads(line)
            assert source['position'] == index and source['data_admitted_to_training'] is False
            if index not in selected:
                continue
            screen = selected[index]
            assert source['id'] == screen['id'] and source['question_id'] == screen['question_id']
            assert source['dataset'] == screen['dataset'] and source['split'] == screen['split']
            geometry = {}
            for view_name, view in source['views'].items():
                prompt = view['prompt_tokens_including_BOS']
                response = view['entire_response_tokens_including_EOS']
                context = view['entire_context_tokens']
                assert all(type(n) is int and n > 0 for n in (prompt, response, context))
                assert context == prompt + response
                admissible_geometry = response > 1 and not view['contains_reserved_native_source_token']
                assert view['complete_context_fits_4096'] is (admissible_geometry and context <= 4096)
                if view_name in screen['fitting_views']:
                    assert view['complete_context_fits_4096']
                    screened[screen['source_role']][view_name] += 1
                geometry[view_name] = dict(prompt_tokens=prompt, response_EOS_tokens=response,
                    whole_context_tokens=context, native_source_tokens_clean=admissible_geometry)
                for width in WIDTHS:
                    counter = totals[str(width)][screen['source_role']]
                    if admissible_geometry and context <= width:
                        counter[view_name + '_whole_geometric_fits'] += 1
                        counter[view_name + '_response_EOS_targets'] += response
                        counter[view_name + '_whole_context_tokens'] += context
            rows.append(dict(original_shard_position=index, source_role=screen['source_role'],
                original_id=source['id'], dataset=source['dataset'], geometry=geometry,
                longer_view_target_screening_and_execution_admission_not_established=True))
    assert index + 1 == 20000 and len(rows) == len(selected)
    assert screened['train'] == {'code_only': 5509, 'reasoning': 1784}
    assert screened['reserved'] == {'code_only': 607, 'reasoning': 219}
    assert totals['4096']['train']['code_only_whole_geometric_fits'] == 5509
    assert totals['4096']['reserved']['code_only_whole_geometric_fits'] == 607
    if any(sha(path) != value for path, value in files.items()):
        raise ValueError('Every closed source byte must remain unchanged')
    OUT.mkdir()
    ledger = OUT / 'every-fixed-candidate-context.jsonl'
    with ledger.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    result = dict(complete=True, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        original_rows_read=20000, fixed_selected_positions=len(selected), fixed_roles=dict(roles),
        existing_4K_screened_whole_views=screened, whole_context_fit_upper_bounds=totals,
        every_candidate_geometry_sha256=sha(ledger),
        original_native_model_context=4096, model_context_changed=False, model_updates=0,
        public_long_reasoning_views_not_target_screened_or_training_admitted=True,
        fixed_4K_candidate_subset_not_all_long_context_eligible_public_rows=True,
        fresh_complete_execution_and_readback_still_required=True,
        larger_context_GPU_capacity_and_neural_usefulness_unproved=True,
        data_admitted_to_training=False, model_quality_gain_established=False, goal_achieved=False)
    (OUT / 'complete.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('fixed_roles', 'whole_context_fit_upper_bounds')}))


if __name__ == '__main__':
    main()
