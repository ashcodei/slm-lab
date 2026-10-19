"""Select original judge-accepted metadata, without assigning splits or training."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from slm_lab.codenet_candidate_selection_v1 import select
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/codenet-candidate-selection-v1'
    if folder.exists():
        raise ValueError('Preserve the earlier candidate selection')
    if shutil.disk_usage(ROOT).free < 26.25 * 1024 ** 3:
        raise ValueError('Preserve persistent model/collector reserve')
    inspection_path = ROOT / 'reports/codenet-original-inspection-v1/inspection.json'
    inspection = json.loads(inspection_path.read_text())
    if not inspection['whole_gzip_stream_drained_crc_checked'] or not inspection['code_header_and_metadata_row_counts_match']:
        raise ValueError('Complete original archive and metadata inspection required')
    cache = Path(inspection['metadata_cache'])
    index_path = cache / 'retained-members.jsonl'
    if file_hash(index_path) != inspection['retained_member_index_sha256']:
        raise ValueError('Original metadata member index changed')
    members = [json.loads(x) for x in index_path.read_text().splitlines()]
    metadata = sorted((r for r in members if re.fullmatch(r'metadata/p[0-9]{5}\.csv', r['cache_relative_path'])),
                      key=lambda r: r['cache_relative_path'])
    if len(metadata) != 4053:
        raise ValueError('Original complete problem metadata population differs')
    bound = {path: file_hash(path) for path in [Path(__file__), inspection_path, index_path,
        ROOT / 'src/slm_lab/codenet_candidate_selection_v1.py']}
    folder.mkdir()
    atomic_json(folder / 'plan.json', {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'inspection_sha256': file_hash(inspection_path), 'source_archive_sha256': inspection['archive_sha256'],
        'bound_files_sha256': {str(p): d for p, d in bound.items()},
        'per_problem_language_distinct_author_cap': 4, 'maximum_publisher_declared_code_bytes': 32768,
        'selection': 'Minimum SHA256 ranks of original anonymized authors within each problem/language; lexically first eligible submission per selected author.',
        'byte_bound_is_not_complete_model_context_fit': True,
        'split_assignment_deferred_until_problem_family_and_code_duplicate_checks': True,
        'original_judge_label_is_not_human_preference_or_universal_correctness_proof': True,
        'source_code_executed': False, 'model_training_started': False, 'goal_achieved': False})
    totals = Counter()
    languages = Counter()
    problems = set()
    declared_bytes = 0
    output = folder / 'selected-metadata.jsonl'
    with output.open('x') as stream:
        for number, member in enumerate(metadata, 1):
            path = cache / member['cache_relative_path']
            raw = path.read_bytes()
            if len(raw) != member['bytes'] or hashlib.sha256(raw).hexdigest() != member['sha256']:
                raise ValueError('Original problem metadata changed')
            chosen, counts = select(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))), path.stem)
            totals.update(counts)
            for candidate in chosen:
                row = candidate['metadata']
                if not re.fullmatch(r's[0-9]{9}', row['submission_id']) or not re.fullmatch(r'[A-Za-z0-9_+#-]+', row['filename_ext']):
                    raise ValueError('Original submission/extension format differs')
                if '/' in row['language'] or row['language'] in {'.', '..'}:
                    raise ValueError('Unexpected original language path')
                relative = PurePosixPath('Project_CodeNet', 'data', row['problem_id'], row['language'], row['submission_id'] + '.' + row['filename_ext'])
                stream.write(json.dumps({**candidate, 'archive_member': str(relative),
                    'source_metadata_sha256': member['sha256'], 'split': None,
                    'code_bytes_and_context_not_yet_verified': True}, ensure_ascii=False) + '\n')
                totals['selected_candidates'] += 1
                languages[row['language']] += 1
                problems.add(row['problem_id'])
                declared_bytes += int(row['code_size'])
            if stream.tell() > 256 * 1024 ** 2:
                raise ValueError('Bounded candidate metadata output exceeds 256MiB')
            if number % 100 == 0:
                atomic_json(folder / 'progress.json', {
                    'updated_at': datetime.now(timezone.utc).isoformat(),
                    'status': 'selecting original accepted metadata; no splits or training',
                    'problem_metadata_files': number, 'planned_problem_metadata_files': 4053,
                    'actual_counts': dict(totals), 'model_training_started': False, 'goal_achieved': False})
    if totals['rows'] != inspection['metadata_submission_rows'] or totals['accepted'] != inspection['metadata_status_counts']['Accepted']:
        raise ValueError('Candidate selection did not cover all original metadata rows')
    if any(file_hash(path) != digest for path, digest in bound.items()):
        raise ValueError('Bound candidate selection implementation/source changed')
    atomic_json(folder / 'manifest.json', {
        'completed_at': datetime.now(timezone.utc).isoformat(), 'counts': dict(totals),
        'selected_original_problems': len(problems), 'selected_by_original_language_label': dict(languages),
        'publisher_declared_selected_bytes': declared_bytes, 'selected_metadata_sha256': file_hash(output),
        'plan_sha256': file_hash(folder / 'plan.json'), 'inspection_sha256': file_hash(inspection_path),
        'selector_sha256': file_hash(ROOT / 'src/slm_lab/codenet_candidate_selection_v1.py'),
        'script_sha256': file_hash(Path(__file__)),
        'split_assignment_completed': False, 'candidate_source_bodies_copied': False,
        'model_training_started': False, 'goal_achieved': False,
        'scope': 'Bounded original judge-accepted metadata candidate pool only. Includes original language labels such as Text, whose task/target eligibility remains for review. No runnable-code, efficiency, complete-context, duplicate-clean split or capability claim.'})
    atomic_json(folder / 'progress.json', {'updated_at': datetime.now(timezone.utc).isoformat(),
        'status': 'candidate metadata selected; source bodies, duplicates, contexts and splits pending',
        'actual_counts': dict(totals), 'model_training_started': False, 'goal_achieved': False})
    print(json.dumps({'counts': dict(totals), 'selected_original_problems': len(problems),
                      'original_language_labels': len(languages), 'declared_code_bytes': declared_bytes}, indent=2))


if __name__ == '__main__':
    main()
