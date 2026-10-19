"""Scan the complete pinned educational blob; no download or training admission.

The first collector retained and verified the blob, then its schema-counter
used mapping values instead of keys. Preserve that failure and audit the same
complete bytes with explicit key counting. No network retry is needed.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import urllib.parse

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/open-education-source-v1'
REPORTS = ROOT / 'reports/open-education-source-v2'


def main():
    if REPORTS.exists():
        raise ValueError('Preserve any earlier educational-source audit')
    prior = ROOT / 'reports/open-education-source-v1'
    plan = json.loads((prior / 'plan.json').read_text()); blob = SOURCE / plan['filename']
    if blob.stat().st_size != plan['expected_bytes'] or file_hash(blob) != plan['expected_sha256']:
        raise ValueError('All actual pinned publisher educational-source bytes required')
    metadata = json.loads((SOURCE / 'publisher-metadata.json').read_text())
    original = next(r for r in metadata['siblings'] if r['rfilename'] == plan['filename'])
    if metadata['sha'] != plan['revision'] or original['lfs']['sha256'] != plan['expected_sha256']:
        raise ValueError('Exact immutable publisher revision and original blob receipt required')
    REPORTS.mkdir()
    atomic_json(REPORTS / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        actual_pinned_source=plan, original_schema_failure_preserved=str((prior / 'failure.json').relative_to(ROOT)),
        source_script_sha256=file_hash(Path(__file__).resolve()), no_new_network_request=True,
        dataset_admitted_to_training=False, model_updates=0, goal_achieved=False))
    rows = 0; text_bytes = 0; total_bytes = 0
    top = Counter(); fields = Counter(); licenses = Counter(); domains = Counter(); books = set(); identities = set(); texts = Counter(); samples = []
    with gzip.open(blob, 'rt', encoding='utf-8') as stream:
        for line in stream:
            total_bytes += len(line.encode())
            if total_bytes > 4 * 1024**3 or len(line.encode()) > 32 * 1024**2:
                raise ValueError('Full source decompression/document-size budget requires review')
            row = json.loads(line); rows += 1; top.update(row.keys())
            if not isinstance(row['text'], str) or not isinstance(row['metadata'], dict):
                raise ValueError('Actual complete text and explicit metadata dictionary required')
            meta = row['metadata']; fields.update(meta.keys())
            body = row['text'].encode(); text_bytes += len(body)
            texts[hashlib.sha256(body).hexdigest()] += 1
            licenses[json.dumps(meta.get('license'), sort_keys=True)] += 1
            host = urllib.parse.urlsplit(meta.get('url', '')).hostname
            domains[host or 'missing_source_host'] += 1
            if meta.get('book_url'):
                books.add(meta['book_url'])
            if row['id'] in identities:
                raise ValueError('Repeated original publisher source identity requires explicit review')
            identities.add(row['id'])
            if len(samples) < 3:
                samples.append(dict(id=row['id'], source=row['source'], metadata=meta,
                    text_bytes=len(body), text_sha256=hashlib.sha256(body).hexdigest()))
    if any(count != rows for count in top.values()) or not rows or sum(domains.values()) != rows or sum(licenses.values()) != rows:
        raise ValueError('Every whole original row must have complete source/schema/license statistics')
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), source_sha256=file_hash(blob), stored_bytes=blob.stat().st_size,
        complete_original_rows=rows, complete_original_text_bytes=text_bytes, actual_unique_original_books=len(books),
        complete_gzip_CRC_and_JSON_scan=True, original_full_row_key_counts=dict(top), original_metadata_key_counts=dict(fields),
        original_license_values=dict(licenses), original_source_host_counts=dict(domains),
        full_text_sha256_duplicates=sum(n-1 for n in texts.values()), original_source_metadata_examples=samples,
        primary_data_paper='https://arxiv.org/abs/2506.05209v1', publisher_repository=plan['repository'], revision=plan['revision'],
        further_book_group_split_dedup_attribution_license_and_held_screening_required=True,
        dataset_admitted_to_training=False, downloaded_code_executed=False, new_model_training_started=False,
        model_updates=0, candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'complete-source.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'original_source_metadata_examples'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
