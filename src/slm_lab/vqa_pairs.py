"""Original human visual question pairs with explicit source-photo identities."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import csv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import time
import threading
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit
import zipfile

import numpy as np
from PIL import Image
from tokenizers import Tokenizer
import torch
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash
from .expanded_photos import difference_hash, encode_storage, near_protected
from .masked_visual_pretraining import process_identity
from .model import ModelConfig
from .public_sources import AGENT
from .training import Encoder
from .vqa_attribution import attribution
from .photo_storage_policy_v1 import validate_reserve, migration_snapshot

ROOT = Path(__file__).resolve().parents[2]


def retry_delay(value, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None: date = date.replace(tzinfo=timezone.utc)
            seconds = (date - now).total_seconds()
        except (TypeError, ValueError, OverflowError):
            seconds = 300
    return max(60, seconds)


class FlickrPacing:
    def __init__(self, interval=10):
        self.interval = interval; self.next_request = 0.; self.blocked_until = 0.
        self.lock = threading.Lock(); self.receipt_path = None

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic(); delay = max(self.next_request, self.blocked_until) - now
                if delay <= 0:
                    self.next_request = now + self.interval; return
            time.sleep(min(15, delay))

    def cooldown(self, seconds, reason):
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)
            if self.receipt_path is not None:
                atomic_json(self.receipt_path, {'status':'respecting publisher cooldown; source requests wait',
                            'reason':reason,'recorded_at':datetime.now(timezone.utc).isoformat(),
                            'delay_seconds':seconds,'minimum_page_interval_seconds':self.interval,'goal_achieved':False})


FLICKR = FlickrPacing()


def flickr_photo_id(url):
    parsed = urlsplit(url or '')
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname: return None
    if parsed.hostname in {'flickr.com', 'www.flickr.com'}:
        match = re.fullmatch(r'/photos/[^/]+/([0-9]+)/?', parsed.path)
        if match: return match.group(1)
        if parsed.path == '/photo.gne':
            values = parse_qs(parsed.query).get('id', [])
            if len(values) == 1 and re.fullmatch(r'[0-9]+', values[0]): return values[0]
    elif parsed.hostname.endswith('.staticflickr.com'):
        match = re.match(r'([0-9]+)_', parsed.path.rsplit('/', 1)[-1])
        if match: return match.group(1)
    return None


def known_photo_aliases(row):
    result = {flickr_photo_id(row.get(key)) for key in ['flickr_url', 'source_url', 'original_url', 'OriginalURL', 'OriginalLandingURL']}
    result.discard(None)
    if row.get('source_flickr_photo_id'):
        value = str(row['source_flickr_photo_id'])
        if re.fullmatch(r'[0-9]+', value): result.add(value)
    return result


def canonical_question(pair):
    first = ' '.join(pair['first']['question']['question'].casefold().split())
    second = ' '.join(pair['second']['question']['question'].casefold().split())
    if (not first or first != second or pair['first']['answer'] == pair['second']['answer']
            or pair['first']['question']['image_id'] == pair['second']['question']['image_id']):
        raise ValueError('Original paired question/image/answer contrast differs')
    return first


def source_identity():
    names = ['vqa_pairs', 'vqa_attribution', 'expanded_photos', 'diverse_images', 'model', 'training', 'corpus', 'masked_visual_pretraining', 'photo_storage_policy_v1']
    paths = [ROOT / 'src/slm_lab' / (name + '.py') for name in names]
    paths.extend([ROOT / 'configs/model.json', ROOT / 'scripts/inspect_vqa_sources.py'])
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def screen_candidates(cache):
    inspection_path = ROOT / 'reports/vqa-original-source-inspection-v1.json'
    inspection = json.loads(inspection_path.read_text()); sources = {str(inspection_path): file_hash(inspection_path)}
    if inspection['inspection_source_sha256'] != file_hash(ROOT / 'scripts/inspect_vqa_sources.py'):
        raise ValueError('Original candidate inspection implementation changed')
    pairs = {}
    for split in ['train', 'valid']:
        path = cache / ('candidate-pairs.' + split + '.jsonl')
        if file_hash(path) != inspection['candidate_files_sha256'][path.name]: raise ValueError('Original pair candidate metadata changed')
        sources[str(path)] = file_hash(path); pairs[split] = list(jsonl(path))
    for name, receipt in inspection['sources'].items():
        path = cache / name
        if file_hash(path) != receipt['sha256']: raise ValueError('Original publisher source changed')
        sources[str(path)] = receipt['sha256']
    wanted = set().union(*(known_photo_aliases(row[key]['image']) for rows in pairs.values() for row in rows for key in ['first', 'second']))
    if not wanted: raise ValueError('No actual source photo identities; never match null aliases')
    blocked = {'train': set(), 'valid': set()}
    for split in ['train', 'validation', 'test']:
        path = ROOT / 'data/image-source-cache-v1' / ('openimages-' + split + '.csv')
        receipt = json.loads(path.with_suffix('.csv.source.json').read_text())
        if file_hash(path) != receipt['sha256']: raise ValueError('Original OpenImages metadata changed')
        sources[str(path)] = receipt['sha256']
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                matches = wanted & known_photo_aliases(row)
                blocked['valid' if split == 'train' else 'train'].update(matches)
    with zipfile.ZipFile(cache / 'coco-annotations-trainval2014.zip') as archive:
        for split, part in [('train', 'train2014'), ('valid', 'val2014')]:
            with archive.open('annotations/captions_' + part + '.json') as stream: images = json.load(stream)['images']
            for row in images: blocked['valid' if split == 'train' else 'train'].update(wanted & known_photo_aliases(row))
    test_path = cache / 'coco-image-info-test2015.zip'
    receipt = json.loads(test_path.with_suffix('.zip.source.json').read_text())
    if file_hash(test_path) != receipt['sha256']: raise ValueError('Original COCO test metadata changed')
    sources[str(test_path)] = receipt['sha256']
    with zipfile.ZipFile(test_path) as archive:
        with archive.open('annotations/image_info_test2015.json') as stream: images = json.load(stream)['images']
        for row in images: blocked['train'].update(wanted & known_photo_aliases(row))
    counts = {}; admitted = {}
    for split, rows in pairs.items():
        admitted[split] = []
        for row in rows:
            canonical_question(row)
            aliases = [known_photo_aliases(row[key]['image']) for key in ['first', 'second']]
            if all(len(values) == 1 and not (values & blocked[split]) for values in aliases): admitted[split].append(row)
        counts[split] = {'candidate_pairs': len(rows), 'pairs_after_official_source_alias_screen': len(admitted[split]),
                         'blocked_candidate_photo_aliases': len(blocked[split])}
    return admitted, sources, counts


def fetch(url, limit):
    original_flickr = urlsplit(url).hostname in {'flickr.com','www.flickr.com'}
    while True:
        if original_flickr: FLICKR.acquire()
        request = urllib.request.Request(url, headers={'User-Agent': AGENT})
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                declared = response.headers.get('Content-Length')
                if response.status != 200 or (declared and int(declared) > limit): raise ValueError('Original source response exceeds byte/status bound')
                raw = response.read(limit + 1)
                if len(raw) > limit or (declared and len(raw) != int(declared)): raise ValueError('Original source response incomplete/oversized')
                return raw, response.url, response.headers.get('ETag')
        except urllib.error.HTTPError as error:
            if not original_flickr or error.code != 429: raise
            delay = retry_delay(error.headers.get('Retry-After')); error.close()
            FLICKR.cooldown(delay,'Original Flickr returned HTTP429; honor Retry-After without discarding a source image')
            print('Original Flickr cooldown seconds',delay,flush=True)


def prepare_image(item):
    split, image = item; key = image['id']; photo = flickr_photo_id(image['flickr_url'])
    try:
        if image['license'] != 4 or not photo: raise ValueError('Original photo lacks expected CC-BY2.0/source identity')
        expected = f'COCO_{split}_{key:012d}.jpg'
        if image['file_name'] != expected or image['coco_url'] != f'http://images.cocodataset.org/{split}/{expected}':
            raise ValueError('Original COCO file/source identity differs')
        html, final_url, _ = fetch('https://www.flickr.com/photo.gne?id=' + photo, 2_000_000)
        credit = attribution(html, photo, final_url)
        url = 'https://s3.amazonaws.com/images.cocodataset.org/' + split + '/' + expected
        raw, final_image_url, etag = fetch(url, 8_000_000)
        md5 = hashlib.md5(raw).hexdigest(); md5_verified = bool(etag and re.fullmatch(r'"[0-9a-f]{32}"', etag))
        if md5_verified and etag.strip('"') != md5: raise ValueError('Original image object ETag differs from complete response bytes')
        with Image.open(io.BytesIO(raw)) as original:
            if (original.size != (image['width'], image['height']) or min(original.size) < 96
                    or original.width * original.height > 10_000_000 or original.getexif().get(274, 1) != 1):
                raise ValueError('Original COCO dimensions/orientation exceed defined transform')
            source_pixels = {str(size): pixel_hash(original, size) for size in [224, 448]}
            stored, pixels, before_storage, fingerprint = encode_storage(original)
        page = gzip.compress(html, compresslevel=6, mtime=0)
        record = {**credit, 'coco_image_id': key, 'source_subset': split, 'group_id': 'flickr:' + photo,
                  'image': 'images/' + expected, 'stored_image_bytes': len(stored), 'stored_image_sha256': hashlib.sha256(stored).hexdigest(),
                  'attribution_page': f'attribution/{split}_{key:012d}_{photo}.html.gz', 'attribution_page_bytes': len(page),
                  'attribution_page_sha256': hashlib.sha256(page).hexdigest(), 'attribution_retrieved_at': datetime.now(timezone.utc).isoformat(),
                  'original_COCO_image_metadata': image, 'download_url': url, 'download_final_url': final_image_url,
                  'download_sha256': hashlib.sha256(raw).hexdigest(), 'download_etag': etag,
                  'object_etag_md5_verified': md5_verified, 'download_md5_hex': md5,
                  'source_input_pixels_sha256': source_pixels, 'pre_storage_pixels_sha256': {str(k): v for k, v in before_storage.items()},
                  'model_pixel_sha256': pixels[224], 'model_pixel_sha256_448': pixels[448], 'difference_hash64': str(fingerprint)}
        return record, stored, page
    except (ValueError, OSError, urllib.error.URLError) as error:
        return {'failed_image': expected if 'expected' in locals() else str(key), 'error': str(error)[:300]}, None, None


def collect(output, photos_pid, valid_pair_limit=512, resume=False, minimum_free_gib=35, prior_source_snapshot=None):
    minimum_free_gib = validate_reserve(minimum_free_gib)
    if prior_source_snapshot is not None and not resume:
        raise ValueError('Prior source snapshots apply only to an inspected resume')
    output = output.resolve(); loaded = source_identity()
    if output.exists() and not resume: raise ValueError('Existing VQA collection requires inspection, not overwrite/restart')
    if resume and (not output.exists() or (output/'manifest.json').exists()): raise ValueError('Resume requires incomplete inspected collection')
    photo_identity = process_identity(photos_pid)
    final_photos = ROOT / 'data/expanded-everyday-photos-v1/manifest.json'
    if not final_photos.exists() and (photo_identity is None or b'slm_lab.expanded_photos' not in Path(f'/proc/{photos_pid}/cmdline').read_bytes().split(b'\0')):
        raise ValueError('Original-photo producer is not verified live or finalized')
    if not resume:
        output.mkdir(); (output / 'images').mkdir(); (output / 'attribution').mkdir()
    FLICKR.receipt_path = output/'publisher-throttle.json'
    if resume: FLICKR.cooldown(600,'Previously observed HTTP429; conservative initial wait before paced resume')
    def progress(message, **values):
        atomic_json(output / 'progress.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                             'goal_achieved': False, **values})
    try:
        progress('screening original image identities across official source splits')
        pairs, sources, screen = screen_candidates(ROOT / 'data/vqa-source-cache-v1')
        if not pairs['train'] or not pairs['valid']: raise ValueError('No usable original pair split after alias protection')
        pairs['valid'] = sorted(pairs['valid'], key=lambda row: hashlib.sha256(row['id'].encode()).digest())[:valid_pair_limit]
        selected = {}
        for rows in pairs.values():
            for row in rows:
                for key in ['first', 'second']:
                    sample = row[key]; selected[(row['source_subset'], sample['image']['id'])] = (row['source_subset'], sample['image'])
        configuration = {'implementation_sources_sha256': loaded, 'original_sources_sha256': sources,
                         'official_alias_screen': screen, 'validation_pair_limit': valid_pair_limit, 'workers': 2,
                         'stored_bytes_cap': int(1.5 * 1024**3), 'minimum_free_GiB': minimum_free_gib,
                         'storage': '768px JPEG quality90; original attribution HTML losslessly gzip-compressed',
                         'photo_producer_pid': photos_pid, 'photo_producer_start_ticks': photo_identity,
                         'question_prompt': 'Identical case/whitespace-canonical question for both original paired images; answer briefly.'}
        images = {}; failures = []; stored_bytes = 0
        if resume:
            old = json.loads((output/'configuration.json').read_text())
            prior_source = migration_snapshot(prior_source_snapshot, old['implementation_sources_sha256']['src/slm_lab/vqa_pairs.py'])
            for key in ['original_sources_sha256','official_alias_screen','validation_pair_limit','workers','stored_bytes_cap','storage','question_prompt']:
                if old[key]!=configuration[key]: raise ValueError('Inspected resume configuration/data changed: '+key)
            snapshot = ROOT/'reports/source-snapshots/vqa-pairs-before-rate-limit-resume-v1.py'
            for name,digest in old['implementation_sources_sha256'].items():
                accepted = {file_hash(snapshot),loaded[name]} if name=='src/slm_lab/vqa_pairs.py' else {loaded[name]}
                if name == 'src/slm_lab/vqa_pairs.py' and prior_source is not None:
                    accepted.add(prior_source['sha256'])
                if digest not in accepted: raise ValueError('Prior prepared inputs used another source implementation: '+name)
            journal_path=output/'images.collected.jsonl';raw=journal_path.read_bytes();complete=raw[:raw.rfind(b'\n')+1]
            (output/'images.collected.before-rate-limit-v1.jsonl').write_bytes(raw)
            for row in (json.loads(line) for line in complete.splitlines()):
                for name,digest in [(row['image'],row['stored_image_sha256']),(row['attribution_page'],row['attribution_page_sha256'])]:
                    path=(output/name).resolve()
                    if not path.is_relative_to(output) or file_hash(path)!=digest: raise ValueError('Prior committed image/attribution bytes changed')
                if row['group_id']!='flickr:'+row['source_flickr_photo_id']: raise ValueError('Prior committed source photo identity changed')
                images[(row['source_subset'],row['coco_image_id'])]=row
                stored_bytes+=row['stored_image_bytes']+row['attribution_page_bytes']
            committed={row[key] for row in images.values() for key in ['image','attribution_page']}
            orphaned=[]
            for folder in ['images','attribution']:
                for path in (output/folder).iterdir():
                    relative=str(path.relative_to(output))
                    if relative not in committed:
                        target=output/'uncommitted-before-resume'/relative;target.parent.mkdir(parents=True,exist_ok=True);path.replace(target);orphaned.append(relative)
            journal_path.write_bytes(complete)
            configuration['resume_receipt']={'prior_configuration_sha256':file_hash(output/'configuration.before-rate-limit-v1.json'),
                'prior_committed_records':len(images),'prior_committed_bytes_verified':stored_bytes,'quarantined_uncommitted_files':orphaned,
                'previous_source_snapshot_sha256':file_hash(snapshot),'page_interval_seconds':FLICKR.interval,'initial_cooldown_seconds':600}
            configuration['resume_receipt']['prior_collector_source_snapshot'] = prior_source
            configuration['resume_receipt']['previous_minimum_free_GiB'] = old['minimum_free_GiB']
        atomic_json(output / 'configuration.json', configuration)
        ordered = sorted(selected.values(), key=lambda item: hashlib.sha256((item[0] + ':' + str(item[1]['id'])).encode()).digest())
        progress('fetching original licensed photo bytes and creator declarations', selected_images=len(ordered), fetched_images=0)
        ordered=[item for item in ordered if (item[0],item[1]['id']) not in images]
        with (ThreadPoolExecutor(max_workers=2) as executor,
              (output / 'images.collected.jsonl').open('a' if resume else 'w') as journal,
              (output/'failed-requests.jsonl').open('a') as failed):
            for start in range(0, len(ordered), 2):
                while shutil.disk_usage(ROOT).free < configuration['minimum_free_GiB'] * 1024**3:
                    if source_identity() != loaded: raise ValueError('Collection implementation changed while waiting for disk')
                    progress('waiting for free disk before original photo requests', fetched_images=len(images), stored_bytes=stored_bytes,
                             minimum_free_GiB=configuration['minimum_free_GiB'], actual_free_GiB=shutil.disk_usage(ROOT).free/1024**3)
                    time.sleep(15)
                futures = [executor.submit(prepare_image, item) for item in ordered[start:start + 2]]
                bounded = False
                for future in futures:
                    record, raw, page = future.result()
                    if raw is None: failures.append(record);failed.write(json.dumps(record)+'\n');failed.flush();continue
                    if stored_bytes + len(raw) + len(page) > configuration['stored_bytes_cap']: bounded = True; break
                    (output / record['image']).write_bytes(raw); (output / record['attribution_page']).write_bytes(page)
                    images[(record['source_subset'], record['coco_image_id'])] = record
                    stored_bytes += len(raw) + len(page); journal.write(json.dumps(record) + '\n'); journal.flush()
                if start % 50 == 0:
                    progress('fetching original licensed photo bytes and creator declarations', selected_images=len(ordered), fetched_images=len(images),
                             failures=len(failures), stored_bytes=stored_bytes)
                    print('Original real-photo QA images',len(images),'/',len(ordered),'failures',len(failures),flush=True)
                    if source_identity() != loaded: raise ValueError('Loaded collector implementation changed')
                if bounded: break
        while not final_photos.exists():
            if photo_identity is None or process_identity(photos_pid) != photo_identity: raise ValueError('Photo producer ended without finalized inputs; do not admit VQA')
            if source_identity() != loaded: raise ValueError('Collector implementation changed before source snapshot')
            progress('waiting for finalized ordinary-photo training inputs before VQA split admission', fetched_images=len(images))
            time.sleep(15)
        protected_paths = [ROOT / 'data/alignment-curriculum-v1' / ('images.' + split + '.jsonl') for split in ['train','valid']]
        protected_paths += [ROOT / 'data/visual-pretraining-v1' / (split + '.jsonl') for split in ['train','valid']]
        protected_paths += [ROOT / 'data/expanded-everyday-photos-v1/train.jsonl', ROOT / 'data/expanded-scene-words-v1/train.jsonl']
        protected = {split:{224:set(),448:set()} for split in ['train','valid']}; fingerprints = {'train':set(),'valid':set()}
        aliases = {'train':set(),'valid':set()}; examined = set()
        progress('checking stored VQA photos against finalized training and validation inputs', fetched_images=len(images))
        for path in protected_paths:
            manifest = json.loads((path.parent / 'manifest.json').read_text())
            digest = file_hash(path)
            if digest != manifest['files_sha256'][path.name]: raise ValueError('Protected original image input changed')
            sources[str(path)] = digest; split = 'valid' if 'valid' in path.name else 'train'
            for row in jsonl(path):
                aliases[split].update(known_photo_aliases(row))
                for size,key in [(224,'model_pixel_sha256'),(448,'model_pixel_sha256_448')]:
                    if row.get(key): protected[split][size].add(row[key])
                # Full parent images supply perceptual controls. Word regions
                # retain exact pixels/aliases without pretending crops are full scenes.
                if path.parent.name not in {'visual-pretraining-v1','expanded-everyday-photos-v1'}: continue
                image_path = (path.parent / row['image']).resolve()
                if (split,image_path) in examined: continue
                examined.add((split,image_path))
                if row.get('difference_hash64'): fingerprints[split].add(int(row['difference_hash64']))
                else:
                    with Image.open(image_path) as image: fingerprints[split].add(difference_hash(image))
        excluded = Counter(); approved = {}; accepted_pairs = {}; pixels = {224:set(),448:set()}; valid_fingerprints=set()
        tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'
        encoder = Encoder(Tokenizer.from_file(str(tokenizer_path)), ModelConfig.load(ROOT/'configs/model.json'), output, 448)
        for split in ['valid','train']:
            opposing = 'train' if split == 'valid' else 'valid'
            blocked_pixels = {size: protected[opposing][size] | (pixels[size] if split=='train' else set()) for size in [224,448]}
            blocked_fingerprints = np.asarray(sorted(fingerprints[opposing] | (valid_fingerprints if split=='train' else set())),dtype=np.uint64)
            blocked_aliases = aliases[opposing] | ({r['source_flickr_photo_id'] for r in approved.get('valid',{}).values()} if split=='train' else set())
            admitted_images = {}
            for key, record in images.items():
                if key[0] != ('val2014' if split=='valid' else 'train2014'): continue
                if record['source_flickr_photo_id'] in blocked_aliases: excluded[split+': known opposite source photo']+=1;continue
                pixel_fields = [('model_pixel_sha256',224),('model_pixel_sha256_448',448)]
                if any(record[field] in blocked_pixels[size] or record['source_input_pixels_sha256'][str(size)] in blocked_pixels[size] for field,size in pixel_fields):
                    excluded[split+': protected exact pixels']+=1;continue
                if near_protected(int(record['difference_hash64']),blocked_fingerprints): excluded[split+': protected perceptual distance<=4']+=1;continue
                admitted_images[key[1]]=record
            accepted_pairs[split]=[pair for pair in pairs[split] if all(pair[key]['image']['id'] in admitted_images for key in ['first','second'])]
            retained_ids={pair[key]['image']['id'] for pair in accepted_pairs[split] for key in ['first','second']}
            approved[split]={key:value for key,value in admitted_images.items() if key in retained_ids}
            if not accepted_pairs[split]: raise ValueError('No complete paired image/answer examples after original-image protection: '+split)
            if split=='valid':
                for record in approved[split].values():
                    for size,field in [(224,'model_pixel_sha256'),(448,'model_pixel_sha256_448')]:
                        pixels[size].update([record[field],record['source_input_pixels_sha256'][str(size)],record['pre_storage_pixels_sha256'][str(size)]])
                    valid_fingerprints.add(int(record['difference_hash64']))
            rows={}; pair_rows=[]
            for pair in accepted_pairs[split]:
                question=canonical_question(pair);ids=[]
                for key in ['first','second']:
                    sample=pair[key];qid=sample['question']['question_id'];identity='vqa2:'+pair['source_subset']+':question:'+str(qid);ids.append(identity)
                    if identity in rows:
                        rows[identity]['pair_ids'].append(pair['id']);continue
                    record=approved[split][sample['image']['id']]
                    row={**record,'id':identity,'pair_ids':[pair['id']],'prompt':question+'\nAnswer briefly.','response':sample['answer'],
                         'task':'visual_qa','domain':'everyday_photos','original_question':sample['question'],'original_human_annotation':sample['annotation'],
                         'exact_human_agreement':sample['exact_human_votes'],'annotation_license':'CC-BY-4.0','annotation_source':'VQA Consortium original v2.0',
                         'annotation_source_url':'https://visualqa.org/download.html','external_model_teacher_used':False}
                    sequence,labels=encoder.sequence(row,'response')
                    if encoder.truncated: raise ValueError('Full original visual question/answer exceeds own context')
                    row['context_tokens']=len(sequence);row['target_tokens']=sum(value!=-100 for value in labels[1:]);rows[identity]=row
                pair_rows.append({'id':pair['id'],'question_ids':ids,'canonical_question':question,'source_subset':pair['source_subset']})
            (output/(split+'.jsonl')).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows.values()))
            (output/('pairs.'+split+'.jsonl')).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in pair_rows))
        for record in images.values():
            for name,digest in [(record['image'],record['stored_image_sha256']),(record['attribution_page'],record['attribution_page_sha256'])]:
                if file_hash(output/name)!=digest: raise ValueError('Stored photo/attribution bytes changed before finalization')
        if source_identity()!=loaded or any(file_hash(Path(path))!=digest for path,digest in sources.items()): raise ValueError('Collection/protected publisher inputs changed before finalization')
        stats={split:{'questions':len(list(jsonl(output/(split+'.jsonl')))),'pairs':len(accepted_pairs[split]),'images':len(approved[split])} for split in ['train','valid']}
        atomic_json(output/'manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),'configuration':configuration,'sources_sha256':sources,'splits':stats,
                    'stored_image_and_attribution_bytes':stored_bytes,'downloaded_images':len(images),'failures':failures,'exclusions':dict(excluded),
                    'tokenizer_sha256':file_hash(tokenizer_path),'files_sha256':{name:file_hash(output/name) for name in ['train.jsonl','valid.jsonl','pairs.train.jsonl','pairs.valid.jsonl','images.collected.jsonl']},
                    'trained':False,'goal_achieved':False,'external_model_teacher_used':False,
                    'limitations':['License/creator metadata are original publisher/current Flickr declarations, not independent ownership adjudication.',
                                   'Subset filtered by source, current eligibility and agreement; not full standardized VQA evaluation.',
                                   'Exact/photo/perceptual distance<=4 protections cover declared frozen inputs; semantic and arbitrary near-duplicate independence remain unproved.',
                                   'Only complete accepted pairs enter rows; some downloaded images/attribution pages remain unused.',
                                   'Original raw COCO response hashes/ETags are retained; stored JPEGs resize/recompress those originals.']})
        progress('original real-photo question pairs prepared; not trained',splits=stats,stored_bytes=stored_bytes)
        print('Original real-photo QA pairs ready',stats,flush=True)
    except Exception as error:
        progress('original real-photo pair preparation failed; inspect without automatic restart',error=str(error));raise


def main():
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/vqa-real-pairs-v1')
    parser.add_argument('--photos-pid',type=int,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--minimum-free-gib',type=float,default=35)
    parser.add_argument('--prior-source-snapshot',type=Path)
    args=parser.parse_args();collect(args.output,args.photos_pid,resume=args.resume,
                                   minimum_free_gib=args.minimum_free_gib,prior_source_snapshot=args.prior_source_snapshot)


if __name__=='__main__':main()
