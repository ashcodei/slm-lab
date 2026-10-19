"""Retain the original public APPS archive and publisher declarations.

Collection only: no archive contents are executed or admitted to training.
Publisher train/test roles must be preserved by a later preparation/audit.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen

from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/public-apps-publisher-source-v1'
URL = 'https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz'
MAX_BYTES = 2 * 1024**3
MIN_FREE = 26 * 1024**3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-declarations', action='store_true')
    args = parser.parse_args()
    continued = OUTPUT.exists()
    if continued:
        if not args.resume_declarations or (OUTPUT / 'APPS.tar.gz').exists() or (OUTPUT / 'APPS.tar.gz.partial').exists():
            raise ValueError('Preserve original archive/partial transfer; explicit declaration-only continuation required')
        previous = json.loads((OUTPUT / 'collection.json').read_text())
        if previous['status'] != 'collection stopped; preserve partial original bytes':
            raise ValueError('Only the stopped original declaration-only attempt may continue')
        for name, info in previous['collected_files'].items():
            if hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() != info['sha256']:
                raise ValueError('Every previously collected original declaration must remain unchanged')
        with (OUTPUT / 'stopped-declaration-only-attempt.json').open('x') as stream:
            json.dump(previous, stream, indent=2)
    if shutil.disk_usage(ROOT).free < MIN_FREE + MAX_BYTES:
        raise ValueError('Fund bounded original archive above existing persistent reserve')
    if not continued: OUTPUT.mkdir()
    def fetch(url):
        return urlopen(Request(url, headers={'User-Agent': 'slm-lab-public-research/1.0', 'Accept-Encoding': 'identity'}), timeout=45)
    metadata = dict(started_at=datetime.now(timezone.utc).isoformat(), publisher_repository='https://github.com/hendrycks/apps',
        dataset_url=URL, maximum_archive_bytes=MAX_BYTES, collected_files={}, dataset_admitted_to_training=False,
        pretrained_or_teacher_model_downloads=False, model_updates=0, goal_achieved=False)
    try:
        if continued:
            metadata['publisher_revision'] = previous['publisher_revision']
            metadata['collected_files'] = previous['collected_files']
            metadata['continues_preserved_declaration_only_attempt'] = True
        else:
            with fetch('https://api.github.com/repos/hendrycks/apps/commits/main') as response:
                commit = json.loads(response.read(1024**2))['sha']
            if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
                raise ValueError('Exact original publisher repository revision required')
            metadata['publisher_revision'] = commit
            for name in ['README.md', 'LICENSE']:
                source = f'https://raw.githubusercontent.com/hendrycks/apps/{commit}/{name}'
                with fetch(source) as response:
                    raw = response.read(1024**2 + 1)
                if len(raw) > 1024**2: raise ValueError('Publisher declaration unexpectedly large')
                (OUTPUT / ('publisher-' + name)).write_bytes(raw)
                metadata['collected_files']['publisher-' + name] = dict(url=source, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        atomic_json(OUTPUT / 'collection.json', dict(**metadata, status='fetching original public archive'))
        tick = time.monotonic(); total = 0; digest = hashlib.sha256(); last = 0
        with fetch(URL) as response:
            expected = response.headers.get('Content-Length')
            if expected and int(expected) > MAX_BYTES: raise ValueError('Original archive exceeds funded bounded collection')
            # urllib does not decode HTTP content encodings. The publisher
            # labels this original .tar.gz as Content-Encoding:gzip; retain its
            # actual raw gzip bytes and header rather than decompressing it.
            if response.headers.get('Content-Encoding', 'identity') not in ['identity', 'gzip', 'x-gzip']:
                raise ValueError('Unsupported transfer encoding for original raw archive')
            metadata['archive_response'] = dict(final_url=response.geturl(), status_code=response.status,
                HTTP_content_automatically_decoded=False,
                headers={k: v for k, v in response.headers.items() if k.lower() in ['content-length', 'content-type', 'content-encoding', 'etag', 'last-modified']})
            temporary = OUTPUT / 'APPS.tar.gz.partial'
            with temporary.open('xb') as stream:
                while True:
                    chunk = response.read(1024**2)
                    if not chunk: break
                    total += len(chunk)
                    if total > MAX_BYTES or time.monotonic() - tick > 3600 or shutil.disk_usage(ROOT).free < MIN_FREE:
                        raise ValueError('Bounded original collection size/time/storage allowance exceeded')
                    stream.write(chunk); digest.update(chunk)
                    if total - last >= 64 * 1024**2:
                        print('original_APPS_download_bytes', total, flush=True); last = total
            if expected and total != int(expected): raise ValueError('Incomplete original archive transfer')
            with temporary.open('rb') as stream:
                magic = stream.read(2)
            if total < 1024**2 or magic != b'\x1f\x8b':
                raise ValueError('Expected original nontrivial gzip archive bytes')
            target = OUTPUT / 'APPS.tar.gz'; temporary.rename(target)
        metadata['collected_files']['APPS.tar.gz'] = dict(url=URL, bytes=total, sha256=digest.hexdigest())
        metadata.update(completed_at=datetime.now(timezone.utc).isoformat(), status='original public archive retained; not prepared or admitted',
            publisher_train_test_partitions_preserved_in_original_archive=True,
            pending=['Inspect all original archive roles, declarations, problem/solution/test schemas.',
                'Keep publisher test solutions outside all training; split train problems and check prior held sources.',
                'Review dataset/third-party provenance; repository MIT declaration alone is not a blanket assertion about every scraped source.',
                'Encode whole source examples with own BPE, quarantine over-window cases rather than clip, verify finite execution separately.'])
        atomic_json(OUTPUT / 'collection.json', metadata)
        print(json.dumps(dict(status=metadata['status'], archive_bytes=total, sha256=digest.hexdigest(), training_admitted=False)), flush=True)
    except BaseException as error:
        metadata.update(status='collection stopped; preserve partial original bytes', error=str(error), automatic_retry=False)
        atomic_json(OUTPUT / 'collection.json', metadata)
        raise


if __name__ == '__main__': main()
