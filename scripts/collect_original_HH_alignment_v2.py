"""Retain four original public human-ranking blobs pinned by prior receipts."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/original-HH-alignment-v2'


def main():
    if OUTPUT.exists():
        raise ValueError('Preserve earlier source collection; inspect rather than automatically retry')
    original_manifest = ROOT / 'data/alignment-public-v1/manifest.json'
    sources = json.loads(original_manifest.read_text())['preferences']
    if sources['revision'] != 'c72f5cee8eb7b4d2ea5617657f4430d5e333af07' or sources['license'] != 'MIT':
        raise ValueError('Exact existing pinned original public human-ranking provenance required')
    OUTPUT.mkdir()
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), original_manifest_sha256=file_hash(original_manifest),
        source_script_sha256=file_hash(Path(__file__).resolve()), sources=sources, maximum_blob_bytes=256*1024**2,
        response_origin='Responses from Anthropic models, ranked by human annotators.',
        external_teacher_or_model_weights=False, goal_achieved=False)
    atomic_json(OUTPUT/'plan.json', plan)
    def fetch(source):
        name = source['subset']+'-'+source['original_split']+'.jsonl.gz'
        path = OUTPUT/name; pending=path.with_suffix(path.suffix+'.pending')
        digest=hashlib.sha256();size=0
        request=urllib.request.Request(source['url'],headers={'User-Agent':'SLM-Lab public human-feedback source research'})
        with urllib.request.urlopen(request,timeout=90) as response, pending.open('xb') as stream:
            while block:=response.read(1024**2):
                size+=len(block)
                if size>plan['maximum_blob_bytes']:
                    raise ValueError('Original public source exceeded its declared blob budget')
                digest.update(block);stream.write(block)
        if digest.hexdigest()!=source['download_sha256']:
            raise ValueError('Original public source bytes differ from the existing pinned collection receipt')
        rows=0
        with gzip.open(pending,'rt') as stream:
            for line in stream:
                item=json.loads(line)
                if not isinstance(item.get('chosen'),str) or not isinstance(item.get('rejected'),str):
                    raise ValueError('Original public chosen/rejected records required')
                rows+=1
        pending.replace(path)
        record=dict(**source,file=name,compressed_bytes=size,complete_original_rows=rows,actual_sha256=digest.hexdigest())
        print(json.dumps(record),flush=True)
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(fetch,sources['sources']))
    license_path=ROOT/'data/alignment-public-v1/HH-LICENSE.txt'
    (OUTPUT/'LICENSE.txt').write_bytes(license_path.read_bytes())
    manifest=dict(completed_at=datetime.now(timezone.utc).isoformat(),plan_sha256=file_hash(OUTPUT/'plan.json'),
        source_revision=sources['revision'],license='MIT',sources=records,
        files_sha256={r['file']:r['actual_sha256'] for r in records}|{'LICENSE.txt':file_hash(OUTPUT/'LICENSE.txt')},
        response_origin=plan['response_origin'],full_original_gzip_CRC_and_every_JSON_record_checked=True,
        prior_subset_expanded_to_complete_source_files=True,no_model_weights_loaded=True,training_admitted=False,goal_achieved=False)
    atomic_json(OUTPUT/'manifest.json',manifest)


if __name__=='__main__':
    main()
