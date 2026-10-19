"""Retain complete original human word/trace annotations, with no training.

One complete original train shard and the original validation annotation file.
Captions alone already exist here; timing/traces are new source supervision.
"""
import base64
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.request import Request,urlopen

import google_crc32c

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'data/localized-pointer-source-v1'
URLS={
    'publisher-page.html':'https://google.github.io/localized-narratives/',
    'train-shard00000.jsonl':'https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00000-of-00010.jsonl',
    'validation.jsonl':'https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_localized_narratives.jsonl',
}


def fetch(name,url):
    limit=2*1024**3 if name.endswith('.jsonl') else 2*1024**2
    with urlopen(Request(url,method='HEAD',headers={'User-Agent':'slm-lab-public-research/1.0','Accept-Encoding':'identity'}),timeout=30) as head:
        declared=head.headers.get('Content-Length');etag=head.headers.get('ETag')
        if head.status!=200 or (declared and not 0<int(declared)<=limit):
            raise ValueError('Complete original source must fit its sealed byte bound')
    headers={'User-Agent':'slm-lab-public-research/1.0','Accept-Encoding':'identity'}
    if etag and not etag.startswith('W/') and name.endswith('.jsonl'):headers['If-Match']=etag
    sha=hashlib.sha256();crc=google_crc32c.Checksum();count=0;started=time.monotonic()
    target=OUTPUT/name;temporary=target.with_suffix(target.suffix+'.pending')
    # A single bounded request: rate/server failures remain visible and are
    # never converted into source exclusions or automatic rapid retries.
    with urlopen(Request(url,headers=headers),timeout=45) as response,temporary.open('xb') as stream:
        length=response.headers.get('Content-Length');actual_etag=response.headers.get('ETag')
        if (response.status!=200 or (declared and length and length!=declared)
                or (name.endswith('.jsonl') and etag!=actual_etag)):
            raise ValueError('Original full source response/HEAD object identity differs')
        received_headers=dict(response.headers);final_url=response.url
        while True:
            block=response.read(4*1024**2)
            if not block:break
            count+=len(block)
            if count>limit or time.monotonic()-started>900 or shutil.disk_usage(ROOT).free<26*1024**3:
                raise ValueError('Original source byte/time/reserve bound exceeded; pending bytes preserved')
            sha.update(block);crc.update(block);stream.write(block)
            if count%(64*1024**2)==0:
                atomic_json(OUTPUT/'progress.json',dict(observed_at=datetime.now(timezone.utc).isoformat(),
                    downloading=name,bytes=count,declared_bytes=int(length) if length else None,
                    training_admitted=False,model_updates=0,goal_achieved=False))
                print('original_word_trace_source_bytes',name,count,flush=True)
    if (length and count!=int(length)) or (declared and count!=int(declared)):
        raise ValueError('Every original source byte must be retained')
    checks={part.split('=',1)[0].strip():part.split('=',1)[1].strip()
        for part in received_headers.get('x-goog-hash','').split(',') if '=' in part}
    actual_crc=base64.b64encode(crc.digest()).decode()
    if checks.get('crc32c') and checks['crc32c']!=actual_crc:
        raise ValueError('Original publisher CRC32C differs from complete downloaded bytes')
    if name.endswith('.jsonl') and 'crc32c' not in checks:
        raise ValueError('Original trace source needs its publisher transport checksum')
    temporary.replace(target)
    return dict(url=url,final_url=final_url,bytes=count,sha256=sha.hexdigest(),etag=actual_etag,
        last_modified=received_headers.get('Last-Modified'),publisher_crc32c=checks.get('crc32c'),
        actual_crc32c=actual_crc,publisher_checksum_verified='crc32c' in checks,
        collected_at=datetime.now(timezone.utc).isoformat())


def main():
    if OUTPUT.exists():raise ValueError('Preserve original word/trace source and any incomplete transfer; no implicit restart')
    if shutil.disk_usage(ROOT).free<31*1024**3:raise ValueError('Fund4GiB original sources above26GiB reserve')
    OUTPUT.mkdir()
    atomic_json(OUTPUT/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),urls=URLS,
        maximum_bytes_per_annotation_source=2*1024**3,maximum_total_source_bytes=4*1024**3+2*1024**2,
        source_implementation_sha256=file_hash(__file__),train_scope='Complete shard00000of00010 only; not the full train set',
        validation_scope='Complete original localized validation annotations; never optimization input',
        new_source_fields=['timed_caption','traces'],original_images_not_downloaded=True,
        training_admitted=False,model_updates=0,goal_achieved=False))
    receipts={name:fetch(name,url) for name,url in URLS.items()}
    page=(OUTPUT/'publisher-page.html').read_text()
    if 'creativecommons.org/licenses/by/4.0' not in page:
        raise ValueError('Retained original annotation license declaration absent')
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),sources=receipts,
        original_human_caption_and_word_mouse_trace_annotations=True,annotation_license='CC-BY-4.0',
        annotation_attribution='Pont-Tuset, Uijlings, Changpinyo, Soricut and Ferrari. Connecting Vision and Language with Localized Narratives, ECCV2020.',
        annotation_license_page_sha256=file_hash(OUTPUT/'publisher-page.html'),
        original_image_licenses_and_authors_separate=True,original_images_downloaded=0,
        train_scope='One complete publisher train shard,00000of00010',validation_optimization_forbidden=True,
        source_prepared_for_training=False,model_updates=0,external_model_teacher_used=False,goal_achieved=False,
        qualifications=['Existing caption-only source is already used; new fields are word timing and mouse traces.',
            'Mouse attention is approximate spatial supervision, not verified object boxes or rendered text.',
            'Original temporal alignment is retained; no new speech-to-text or paid API is called.',
            'Original image identity/orientation/author/pixel/split checks remain required before data admission.'])
    atomic_json(OUTPUT/'collection.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
