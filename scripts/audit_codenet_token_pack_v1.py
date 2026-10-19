"""Reconstruct every durable row against the earlier independently sealed audit."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from slm_lab.codenet_complete_context_v1 import problem_program_row
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, ROLES
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/codenet-durable-pack-v1'
    report_path = ROOT / 'reports/codenet-durable-pack-v1/reconstruction-check.json'
    if report_path.exists():
        raise ValueError('Preserve earlier reconstruction evidence')
    status = json.loads(report_path.with_name('status.json').read_text())
    if status['status'] != 'durable complete-code candidate pack finished; independent reconstruction and admission pending':
        raise ValueError('Durable preparation must be completed before assessment')
    if file_hash(folder / 'manifest.json') != status['manifest_sha256']:
        raise ValueError('Completed pack manifest differs')
    source_path = ROOT / 'data/codenet-internal-splits-v1/members.jsonl'
    source_manifest_path = source_path.with_name('manifest.json')
    source_manifest = json.loads(source_manifest_path.read_text())
    if file_hash(source_path) != source_manifest['members_sha256']:
        raise ValueError('Original sealed source-role bytes differ')
    selected = {row['archive_member']: row for row in map(json.loads, source_path.open())
                if row['context_qualification'] == 'complete_original_pair_fits' and not row['duplicate_complete_example']}
    metadata_path = ROOT / 'data/codenet-candidate-selection-v1/selected-metadata.jsonl'
    metadata = {row['archive_member']: row['metadata'] for row in map(json.loads, metadata_path.open())}
    description_path = ROOT / 'reports/codenet-description-format-v1/records.jsonl'
    descriptions = {row['problem_id']: row for row in map(json.loads, description_path.open())}
    source_body = json.loads((ROOT / 'reports/codenet-candidate-bodies-v1/bodies.json').read_text())
    source_index_path = Path(source_body['candidate_cache']) / 'source-index.jsonl'
    if file_hash(source_index_path) != source_body['source_index_sha256'] or file_hash(metadata_path) != source_body['selected_metadata_sha256']:
        raise ValueError('Original program/metadata index differs')
    indices = {row['archive_member']: row for row in map(json.loads, source_index_path.open())}
    pack = CompleteCodePack(folder, 'train', purpose='audit')
    plan_path = ROOT / 'reports/codenet-durable-pack-v1/plan.json'
    plan = json.loads(plan_path.read_text())
    if file_hash(description_path) != plan['input_files_sha256'][str(description_path)]:
        raise ValueError('Original description export differs')
    paths = [Path(__file__), ROOT / 'src/slm_lab/codenet_token_pack_v1.py',
             ROOT / 'src/slm_lab/codenet_complete_context_v1.py', folder / 'manifest.json',
             source_path, source_manifest_path, metadata_path, description_path, source_index_path, plan_path]
    bound = {p: file_hash(p) for p in paths}
    seen, counts, component_roles, pair_roles = set(), Counter(), defaultdict(set), defaultdict(set)
    reverse_roles = {code: role for role, code in ROLES.items()}
    full_target_longer_than_half = 0
    with gzip.open(folder / 'original-rows.jsonl.gz', 'rt', encoding='utf-8') as source:
        for position, line in enumerate(source):
            row = json.loads(line)
            member = row['original_archive_member']
            if member in seen or member not in selected or row['token_pack_index'] != position:
                raise ValueError('Durable row identity/order/population differs')
            seen.add(member)
            expected, item = selected[member], pack.index[position]
            role = reverse_roles[int(item['role'])]
            if role != row['split'] or role != expected['split'] or row['group_id'] != expected['group_id']:
                raise ValueError('Durable row crosses original sealed roles')
            ids = pack.ids[int(item['offset']):int(item['offset']) + int(item['length'])]
            digest = hashlib.sha256(ids.tobytes()).hexdigest()
            targets = int(item['length']) - int(item['target_start'])
            if (digest != expected['complete_input_ids_sha256'] or digest != row['complete_input_ids_sha256']
                or len(ids) != expected['context_tokens'] or targets != expected['target_tokens'] or targets != row['target_tokens']
                or ids[0] != 1 or ids[-1] != 2 or np.any(ids >= 32768)):
                raise ValueError('Original full token/mask identity differs')
            m = metadata[member]
            desc = descriptions[m['problem_id']]
            source_digest = hashlib.sha256(row['response'].encode()).hexdigest()
            if (source_digest != indices[member]['body_sha256'] or source_digest != row['original_source_body_sha256']
                or row['original_submission_metadata'] != m or row['original_html_sha256'] != desc['original_html_sha256']):
                raise ValueError('Original programme/metadata/HTML identity differs')
            rebuilt = problem_program_row(desc, m, row['response'])
            if any(row[key] != value for key, value in rebuilt.items() if key not in {'group_id', 'split'}):
                raise ValueError('Original complete prompt/programme framing differs')
            for key, value in [('examples', 1), ('context_tokens', len(ids)), ('target_tokens', targets)]:
                counts[(role, key)] += value
            component_roles[row['group_id']].add(role)
            pair_roles[(digest, targets)].add(role)
            full_target_longer_than_half += targets - 1 > 2047
    if seen != set(selected) or any(len(v) != 1 for v in component_roles.values()) or any(len(v) != 1 for v in pair_roles.values()):
        raise ValueError('Durable population incomplete or copied sources cross roles')
    if len(pair_roles) != len(seen):
        raise ValueError('Previously deduplicated complete examples repeated')
    actual_counts = {role: {key: counts[(role, key)] for key in ['examples', 'context_tokens', 'target_tokens']} for role in ROLES}
    if actual_counts != pack.manifest['counts']:
        raise ValueError('Durable manifest role counts differ from actual contents')
    for path, digest in bound.items():
        if file_hash(path) != digest:
            raise ValueError('Bound reconstruction source changed')
    for name, digest in pack.manifest['files_sha256'].items():
        if file_hash(folder / name) != digest:
            raise ValueError('Durable source bytes changed during reconstruction')
    atomic_json(report_path, {'completed_at': datetime.now(timezone.utc).isoformat(),
        'source_files_sha256': {str(p): digest for p, digest in bound.items()},
        'original_complete_examples_reconstructed': len(seen), 'actual_counts': actual_counts,
        'all_original_programme_bytes_metadata_prompt_framing_and_full_token_masks_verified': True,
        'all_source_roles_preserved_without_rehash_splitting': True,
        'problem_components_and_complete_token_mask_pairs_cannot_cross_roles': True,
        'complete_targets_longer_than_legacy_half_window': full_target_longer_than_half,
        'dataset_admitted_to_training': False, 'new_model_training_started': False, 'goal_achieved': False,
        'scope': 'Exact durable source/input/mask reconstruction only. Historical accepted status is not an independently verified universal judge. Broader execution admission, model learning and capability gains remain pending.'})
    print('reconstructed_complete_original_examples', len(seen), 'without target clipping or split reassignment')


if __name__ == '__main__':
    main()
