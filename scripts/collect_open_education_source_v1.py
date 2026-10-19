"""Retain one complete immutable CommonPile LibreTexts source for later audit.

No training admission, token packing, downloaded code execution or neural
updates. The whole publisher blob must match its declared size/SHA and gzip
CRC before any content/schema statistics are considered complete.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/open-education-source-v1'
REPORTS = ROOT / 'reports/open-education-source-v1'
REPO = 'common-pile/libretexts_filtered'
REVISION = '70388bca52b4a93515e14b1d56618fd7944988fd'
NAME = 'libretexts-0000.json.gz'
SIZE = 115037937
SHA = '71851a2b5acfccf80659729a0eccd07ab6ed737f0edc659a662aa9c1cd2194db'


def main():
    if OUTPUT.exists() or REPORTS.exists():
        raise ValueError('Keep prior source/download observations; inspect rather than overwrite')
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + SIZE:
        raise ValueError('Keep original26GiB persistent reserve above this bounded source blob')
    OUTPUT.mkdir(); REPORTS.mkdir()
    url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{NAME}?download=true'
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), repository=REPO, revision=REVISION,
        original_content_source='LibreTexts openly licensed educational text; row licenses/attribution require separate audit',
        source_paper='https://arxiv.org/abs/2506.05209', collection_code='https://github.com/r-three/common-pile',
        url=url, filename=NAME, expected_bytes=SIZE, expected_sha256=SHA,
        source_script_sha256=file_hash(Path(__file__).resolve()), storage_reserve_GiB=26,
        dataset_admitted_to_training=False, downloaded_code_executed=False, model_updates=0, goal_achieved=False)
    atomic_json(REPORTS / 'plan.json', plan)
    try:
        api = f'https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true'
        with urllib.request.urlopen(api, timeout=45) as response:
            metadata = response.read(2 * 1024**2 + 1)
        if len(metadata) > 2 * 1024**2:
            raise ValueError('Publisher metadata exceeds bound')
        info = json.loads(metadata)
        blob = next(r for r in info['siblings'] if r['rfilename'] == NAME)
        if info['sha'] != REVISION or blob['size'] != SIZE or blob['lfs']['sha256'] != SHA:
            raise ValueError('Pinned publisher source revision/blob receipt must match')
        (OUTPUT / 'publisher-metadata.json').write_bytes(metadata)
        pending = OUTPUT / (NAME + '.partial'); digest = hashlib.sha256(); count = 0
        request = urllib.request.Request(url, headers={'User-Agent': 'own-scratch-SLM-public-data-research/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response, pending.open('xb') as stream:
            while True:
                chunk = response.read(1024**2)
                if not chunk:
                    break
                count += len(chunk)
                if count > SIZE or shutil.disk_usage(ROOT).free < 26 * 1024**3 + len(chunk):
                    raise ValueError('Publisher byte bound or original persistent reserve exceeded')
                stream.write(chunk); digest.update(chunk)
                if count // (16*1024**2) != (count-len(chunk)) // (16*1024**2):
                    atomic_json(REPORTS / 'progress.json', dict(status='retaining complete pinned educational source',
                        observed_at=datetime.now(timezone.utc).isoformat(), downloaded_bytes=count, expected_bytes=SIZE,
                        admitted_to_training=False, goal_achieved=False))
        if count != SIZE or digest.hexdigest() != SHA:
            raise ValueError('Whole actual source bytes/SHA must match before renaming')
        complete = OUTPUT / NAME; pending.rename(complete)
        rows = 0; text_bytes = 0; uncompressed = 0; top_keys = Counter(); metadata_keys = Counter(); license_values = Counter(); samples = []
        with gzip.open(complete, 'rt', encoding='utf-8') as stream:
            for line in stream:
                uncompressed += len(line.encode())
                if uncompressed > 4 * 1024**3 or len(line.encode()) > 32 * 1024**2:
                    raise ValueError('Complete-source decompression/individual document requires review')
                row = json.loads(line); rows += 1; top_keys.update(row)
                if not isinstance(row.get('text'), str):
                    raise ValueError('Complete original educational text requires explicit string schema')
                text_bytes += len(row['text'].encode()); fields = row.get('metadata', {})
                if isinstance(fields, dict):
                    metadata_keys.update(fields)
                    for key, value in fields.items():
                        if 'licen' in key.lower():
                            license_values[json.dumps(value, sort_keys=True)] += 1
                if len(samples) < 3:
                    samples.append(dict(id=row.get('id'), source=row.get('source'), metadata=fields,
                        text_bytes=len(row['text'].encode()), text_sha256=hashlib.sha256(row['text'].encode()).hexdigest()))
        receipt = dict(completed_at=datetime.now(timezone.utc).isoformat(), source_sha256=file_hash(complete),
            stored_bytes=count, complete_original_rows=rows, complete_text_bytes=text_bytes,
            complete_gzip_CRC_and_JSON_scan=True, actual_top_level_keys=dict(top_keys), actual_metadata_keys=dict(metadata_keys),
            observed_license_values=dict(license_values), original_source_metadata_examples=samples,
            publisher_metadata_sha256=file_hash(OUTPUT / 'publisher-metadata.json'),
            full_document_splits_license_attribution_duplicate_and_held_screening_pending=True,
            dataset_admitted_to_training=False, new_model_training_started=False, model_updates=0, goal_achieved=False)
        atomic_json(REPORTS / 'complete-source.json', receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k != 'original_source_metadata_examples'}, indent=2), flush=True)
    except BaseException as error:
        atomic_json(REPORTS / 'failure.json', dict(error=str(error), preserve_partial=True, automatic_retry=False,
            dataset_admitted_to_training=False, model_updates=0, goal_achieved=False)); raise


if __name__ == '__main__':
    main()
