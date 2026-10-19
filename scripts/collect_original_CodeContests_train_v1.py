"""Collect every official training shard directly from the public GCS bucket.

Only publisher train objects are fetched. Immutable generation IDs, original
object sizes/MD5 and local SHA256 bind every binary. This is a source cache,
not parsed training admission, model training or a claim about unseen tests.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import urllib.parse
import urllib.request

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/code-contests-original-train-cache-v1'
REPORTS = ROOT / 'reports/code-contests-original-train-cache-v1'
CATALOG = 'https://storage.googleapis.com/storage/v1/b/dm-code_contests/o?prefix=code_contests_train&maxResults=1000'
REPOSITORY = 'https://github.com/google-deepmind/code_contests'


def get(url, maximum):
    request = urllib.request.Request(url, headers={'User-Agent': 'SLM-Lab-public-source-collector/1.0', 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(maximum + 1)
    if len(raw) > maximum: raise ValueError('Original metadata exceeds declared source limit')
    return raw


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or REPORTS.exists(): raise ValueError('Fresh CPU-hidden original-source collection required')
    OUT.mkdir(); REPORTS.mkdir(); started = time.monotonic()
    owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(REPORTS / 'owner.json', dict(owner=owner, CUDA_VISIBLE_DEVICES='', automatic_network_retry=False, model_updates=0))
    try:
        raw_catalog = get(CATALOG, 4 * 1024**2); (OUT / 'original-GCS-training-object-catalog.json').write_bytes(raw_catalog)
        catalog = json.loads(raw_catalog)
        if catalog.get('nextPageToken') or len(catalog.get('items', [])) != 128: raise ValueError('Complete original128-training-shard catalog required')
        objects = sorted(catalog['items'], key=lambda r: r['name'])
        expected_names = [f'code_contests_train.riegeli-{index:05d}-of-00128' for index in range(128)]
        if [r['name'] for r in objects] != expected_names: raise ValueError('Only every exact original publisher train object may be collected')
        if any(r['bucket'] != 'dm-code_contests' or not r['generation'].isdigit() or int(r['size']) <= 0 or not r['md5Hash'] for r in objects):
            raise ValueError('Original bucket/generation/size/MD5 metadata required for every shard')
        total = sum(int(r['size']) for r in objects)
        if total > 4 * 1024**3 or shutil.disk_usage(ROOT).free < total + 27 * 1024**3: raise ValueError('Bound original data volume and preserve existing writer reserve')
        commit_raw = get('https://api.github.com/repos/google-deepmind/code_contests/commits/main', 2 * 1024**2)
        commit = json.loads(commit_raw)['sha']; (OUT / 'publisher-repository-commit.json').write_bytes(commit_raw)
        if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit): raise ValueError('Exact publisher source commit required')
        source_files = {}
        for name in ['README.md', 'LICENSE', 'contest_problem.proto', 'WORKSPACE', 'print_names_and_sources.py']:
            url = f'https://raw.githubusercontent.com/google-deepmind/code_contests/{commit}/{name}'
            path = OUT / ('publisher-' + name); path.write_bytes(get(url, 2 * 1024**2))
            source_files[name] = dict(url=url, path=str(path), sha256=file_hash(path))
        readme = (OUT / 'publisher-README.md').read_text()
        if 'CC BY 4.0' not in readme or 'Apache 2.0' not in readme: raise ValueError('Preserve the exact original publisher licensing text before caching data')
        plan = dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, source_repository=REPOSITORY,
            publisher_source_commit=commit, official_train_shards=128, original_train_bytes=total,
            catalog_url=CATALOG, catalog_sha256=file_hash(OUT / 'original-GCS-training-object-catalog.json'),
            original_source_metadata=source_files, direct_publisher_storage_not_a_hub_mirror=True,
            validation_and_test_objects_not_downloaded=True, original_object_generations=[r['generation'] for r in objects],
            independent_HTTP_streams=4, automatic_network_retry=False,
            original_license_text_and_third_party_source_caveats_retained=True,
            parsed_records=0, training_admitted=False, neural_weights_downloaded=False, model_updates=0, goal_achieved=False)
        atomic_json(REPORTS / 'plan.json', plan)
        def download(item):
            name = item['name']; size = int(item['size']); generation = item['generation']
            url = 'https://storage.googleapis.com/dm-code_contests/' + urllib.parse.quote(name, safe='') + '?generation=' + generation
            target = OUT / name; pending = OUT / (name + '.part'); sha = hashlib.sha256(); md5 = hashlib.md5(); written = 0
            request = urllib.request.Request(url, headers={'User-Agent': 'SLM-Lab-public-source-collector/1.0', 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(request, timeout=45) as response, pending.open('xb') as stream:
                if response.headers.get('x-goog-generation') != generation or int(response.headers.get('Content-Length', '-1')) != size:
                    raise ValueError('Every response must be the exact immutable publisher object generation and byte length')
                while True:
                    block = response.read(2 * 1024**2)
                    if not block: break
                    written += len(block)
                    if written > size: raise ValueError('Original shard exceeded its declared immutable byte budget')
                    stream.write(block); sha.update(block); md5.update(block)
                stream.flush(); os.fsync(stream.fileno())
            digest = base64.b64encode(md5.digest()).decode()
            if written != size or digest != item['md5Hash']: raise ValueError('Every actual downloaded byte must match the original object size and MD5')
            pending.rename(target)
            return dict(name=name, path=str(target), generation=generation, original_size=size, original_MD5_base64=digest,
                actual_SHA256=sha.hexdigest(), source_url=url, original_split='train', parsed_or_admitted_for_training=False)
        records = []; journal = REPORTS / 'actual-original-shard-downloads.jsonl'
        with journal.open('x') as stream, ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(download, item) for item in objects]
            for future in as_completed(futures):
                row = future.result(); records.append(row); stream.write(json.dumps(row, separators=(',', ':')) + '\n'); stream.flush()
                value = dict(owner=owner, actual_original_train_shards=len(records), original_train_shards=128,
                    actual_original_train_bytes=sum(r['original_size'] for r in records), expected_original_train_bytes=total,
                    elapsed_seconds=time.monotonic() - started, current_shard=row['name'], parsed_records=0, model_updates=0, goal_achieved=False)
                atomic_json(REPORTS / 'progress.json', value)
                if len(records) % 8 == 0: print(json.dumps(value), flush=True)
            os.fsync(stream.fileno())
        records.sort(key=lambda r: r['name'])
        if [r['name'] for r in records] != expected_names or sum(r['original_size'] for r in records) != total:
            raise ValueError('Every original training shard and all bytes must be collected once')
        for row in records:
            if file_hash(row['path']) != row['actual_SHA256']: raise ValueError('Original shard changed after collection')
        manifest = dict(format='own-direct-original-CodeContests-publisher-train-source-cache-v1', at=datetime.now(timezone.utc).isoformat(),
            source_repository=REPOSITORY, publisher_source_commit=commit, original_split='train', original_train_shards=128,
            original_train_bytes=total, complete_original_shards=records, source_metadata=source_files,
            catalog_url=CATALOG, original_catalog_sha256=file_hash(OUT / 'original-GCS-training-object-catalog.json'),
            direct_publisher_storage_not_a_hub_mirror=True, no_validation_or_test_content_collected=True,
            source_code_and_noncode_license_notices_and_third_party_caveats_preserved=True,
            original_format='Riegeli records of original ContestProblem protobufs; not yet decoded',
            parsed_records=0, data_quality_execution_and_context_and_benchmark_overlap_checks_pending=True,
            training_admitted=False, neural_model_weights_downloaded=False, model_updates=0, goal_achieved=False)
        atomic_json(OUT / 'manifest.json', manifest)
        atomic_json(REPORTS / 'complete.json', dict(at=manifest['at'], complete=True, original_train_shards=128,
            original_train_bytes=total, all_original_generations_sizes_MD5_and_SHA256_checked=True,
            source_manifest_sha256=file_hash(OUT / 'manifest.json'),
            source_files_sha256={str(p): file_hash(p) for p in [Path(__file__), REPORTS / 'plan.json', journal,
                OUT / 'manifest.json', OUT / 'original-GCS-training-object-catalog.json', OUT / 'publisher-repository-commit.json',
                *(Path(r['path']) for r in source_files.values())]},
            actual_record_decoding_and_training_admission_pending=True, model_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=str(REPORTS / 'complete.json'), original_train_shards=128, bytes=total,
            actual_record_decoding_and_training_admission_pending=True)), flush=True)
    except BaseException as error:
        atomic_json(REPORTS / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            automatic_network_retry=False, training_admitted=False, model_updates=0, goal_achieved=False)); raise


if __name__ == '__main__': main()
