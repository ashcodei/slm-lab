"""Prepare public human algorithm articles for a future own-model experiment.

Original prose, equations and code remain together. No problem/solution labels,
preference labels, model-generated teaching text or executable imports are added.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json, jsonl
from slm_lab.public_sources import collect_repository

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/algorithm-explanations-v1'
    if folder.exists():
        raise ValueError('Preserve earlier source preparation')
    if shutil.disk_usage(ROOT).free < 26.25 * 1024 ** 3:
        raise ValueError('Preserve the existing collector/model storage reserve')
    folder.mkdir()
    licenses = folder / 'licenses'
    licenses.mkdir()
    atomic_json(folder / 'progress.json', {
        'status': 'collecting pinned original algorithm articles',
        'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False})
    item = {'repo': 'cp-algorithms/cp-algorithms',
            'expected_license': 'CC-BY-SA-4.0', 'domain': 'software_engineering'}
    source = collect_repository(item, folder / 'repository.jsonl', licenses)
    rows = []
    excluded = Counter()
    for row in jsonl(folder / 'repository.jsonl'):
        relative = row['source_url'].split('/blob/' + source['revision'] + '/', 1)[1]
        if not relative.startswith('src/') or not relative.endswith('.md'):
            excluded['outside_original_markdown_articles'] += 1
            continue
        if relative in {'src/index.md', 'src/navigation.md', 'src/tags.md', 'src/contrib.md'}:
            excluded['site_navigation_or_contribution_page'] += 1
            continue
        rows.append({**row,
            'annotation_origin': 'Original public human article: prose, mathematics and code; no synthetic answer or preference labels.',
            'extraction': 'Existing bounded collector: original UTF-8 file text with path header, split on line boundaries; file groups retained.',
            'attribution_history_url': 'https://github.com/' + item['repo'] + '/commits/' + source['revision'] + '/' + relative})
    if not rows:
        raise ValueError('No original algorithm articles admitted')
    documents = folder / 'documents.jsonl'
    with documents.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    group_counts = Counter(row['group_id'] for row in rows)
    atomic_json(folder / 'manifest.json', {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'source': source, 'license': item['expected_license'],
        'license_text_sha256': file_hash(licenses / 'cp-algorithms--cp-algorithms.txt'),
        'documents_sha256': file_hash(documents), 'chunks': len(rows),
        'original_file_groups': len(group_counts), 'characters': sum(len(r['text']) for r in rows),
        'chunks_containing_fenced_code': sum('```' in r['text'] or '~~~' in r['text'] for r in rows),
        'excluded_chunks': dict(excluded), 'preparer_sha256': file_hash(Path(__file__)),
        'collector_sha256': file_hash(ROOT / 'src/slm_lab/public_sources.py'),
        'model_training_started': False, 'existing_frozen_training_plan_changed': False,
        'goal_achieved': False,
        'limitations': [
            'A modest human explanatory supplement, not enough data to establish frontier capability.',
            'Original algorithms and snippets are not independently proved correct or standalone-compilable.',
            'Article chunks may contain partial code; they are pretraining documents, not verified instruction targets.',
            'Generic algorithm overlap and earlier pretraining contamination are not certified absent.',
            'Preserve original attribution and CC-BY-SA-4.0 terms; no corpus or model publication is performed.']})
    atomic_json(folder / 'progress.json', {
        'status': 'original algorithm articles prepared; not trained',
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'original_file_groups': len(group_counts), 'chunks': len(rows), 'goal_achieved': False})
    print(json.dumps({'original_file_groups': len(group_counts), 'chunks': len(rows),
                      'revision': source['revision'], 'model_training_started': False}, indent=2))


if __name__ == '__main__':
    main()
