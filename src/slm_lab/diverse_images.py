"""Bounded public photo/caption and scene-text collection, using original annotations.

No model weights or generated captions are downloaded. Source images remain grouped
by Open Images ID across tasks; the curriculum builder resolves cross-source splits.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import heapq
import io
import json
import math
from pathlib import Path
import time
import urllib.request

from PIL import Image, ImageOps
from .corpus import file_hash
from .public_sources import AGENT

GOOGLE = 'https://storage.googleapis.com/'
CAPTIONS = GOOGLE + 'localized-narratives/annotations/open_images_{}_captions.jsonl'
METADATA = {
    'train': GOOGLE + 'openimages/2018_04/train/train-images-boxable-with-rotation.csv',
    'validation': GOOGLE + 'openimages/2018_04/validation/validation-images-with-rotation.csv',
    'test': GOOGLE + 'openimages/2018_04/test/test-images-with-rotation.csv',
}
TEXT_OCR_REVISION = '0d0a421b9f09f2d00358af52091de57bfaee1684'
TEXT_OCR_MIRROR = ('https://huggingface.co/datasets/redactable-llm/TextOCR/resolve/'
                   + TEXT_OCR_REVISION + '/')


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def asset(cache, name, url, limit=800_000_000, log=True):
    """Stream bounded assets to disk; hash complete responses and serialize writers."""
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / name; metadata = path.with_suffix(path.suffix + '.source.json')
    with (cache / (name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists() and metadata.exists():
            record = json.loads(metadata.read_text())
            if record['url'] != url or file_hash(path) != record['sha256']:
                raise ValueError('Cached source URL/hash mismatch: ' + str(path))
            return path, record
        for attempt in range(3):
            temporary = path.with_suffix(path.suffix + '.part')
            try:
                start = time.monotonic(); count = 0; digest = hashlib.sha256()
                request = urllib.request.Request(url, headers={'User-Agent': AGENT})
                with urllib.request.urlopen(request, timeout=30) as response, temporary.open('wb') as out:
                    headers = dict(response.headers)
                    declared = response.headers.get('Content-Length')
                    if declared and int(declared) > limit: raise ValueError('Source exceeds size limit')
                    while block := response.read(1 << 20):
                        count += len(block)
                        if count > limit or time.monotonic() - start > 600:
                            raise ValueError('Source exceeds size/time limit')
                        digest.update(block); out.write(block)
                        if count % (64 << 20) == 0: print('download', name, count, flush=True)
                if declared and count != int(declared): raise OSError('Incomplete source response')
                record = {'url': url, 'bytes': count, 'sha256': digest.hexdigest(),
                          'etag': headers.get('ETag'), 'last_modified': headers.get('Last-Modified'),
                          'collected_at': datetime.now(timezone.utc).isoformat()}
                temporary.replace(path); atomic_json(metadata, record)
                if log:print('download complete', name, count, flush=True)
                return path, record
            except Exception as error:
                temporary.unlink(missing_ok=True)
                if attempt == 2 or isinstance(error, ValueError) or getattr(error, 'code', None) in {401,403,404}:
                    raise
                time.sleep(2 ** attempt)


def jsonl(path):
    with Path(path).open() as source:
        for line in source:
            if line.strip(): yield json.loads(line)


def image_pixels(image,size=224):
    """Exactly the current encoder's pixel transform, for cross-source duplicate checks."""
    return ImageOps.pad(image.convert('RGB'), (size,size), method=Image.Resampling.BICUBIC,
                        color=(128,128,128)).tobytes()


def pixel_hash(image,size=224):
    return hashlib.sha256(image_pixels(image,size)).hexdigest()


def metadata_for(cache, ids):
    found = {}; sources = []; protected = set()
    for split, url in METADATA.items():
        path, record = asset(cache, 'openimages-' + split + '.csv', url)
        sources.append(record)
        with path.open(newline='') as source:
            for row in csv.DictReader(source):
                key = row['ImageID']
                if split != 'train': protected.add('openimages:' + key)
                if key not in ids: continue
                if row['License'].rstrip('/') != 'https://creativecommons.org/licenses/by/2.0': continue
                found[key] = {**row, 'Subset': split}
    return found, sources, protected


def photo(cache, metadata):
    key = metadata['ImageID']; subset = metadata['Subset']
    path, record = asset(Path(cache) / 'source-images' / subset, key + '.jpg',
                         f'https://s3.amazonaws.com/open-images-dataset/{subset}/{key}.jpg', 5_000_000, log=False)
    with Image.open(path) as original:
        image = original.convert('RGB')
    if min(image.size) < 96 or image.width * image.height > 10_000_000:
        raise ValueError('Unsupported source image dimensions')
    return image, record


def attribution(metadata, record):
    return {'source': 'Open Images / CVDF', 'source_image_id': metadata['ImageID'],
            'group_id': 'openimages:' + metadata['ImageID'],
            'source_url': metadata['OriginalLandingURL'], 'author': metadata['Author'],
            'author_profile_url': metadata['AuthorProfileURL'], 'source_title': metadata['Title'],
            'original_url': metadata['OriginalURL'], 'original_md5': metadata['OriginalMD5'],
            'license': 'CC-BY-2.0', 'license_url': metadata['License'],
            'source_subset': metadata['Subset'], 'download_url': record['url'],
            'download_sha256': record['sha256'], 'download_etag': record.get('etag')}


def save_image(root, name, image):
    path = root / 'images' / (name + '.png')
    image = ImageOps.contain(image, (768,768), Image.Resampling.LANCZOS)
    # Retain provenance in JSONL, rather than duplicating EXIF/large ICC blobs in
    # each crop. These metadata can exceed Pillow's PNG decompression guard.
    image.info.clear()
    # Lossless, inexpensive compression keeps CPU available to the live text trainer.
    temporary = path.with_suffix('.tmp'); image.save(temporary, format='PNG',compress_level=1); temporary.replace(path)
    return str(path.relative_to(root)), file_hash(path), pixel_hash(image)


def persist(root, rows, sources, protected, limitations, errors):
    splits = {}
    for split in ['train','valid']:
        records = [r for r in rows if r['split'] == split]
        path = root / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False) + '\n' for r in records))
        splits[split] = {'examples': len(records), 'images': len({r['group_id'] for r in records}),
                         'tasks': dict(Counter(r['task'] for r in records))}
    (root / 'protected-image-ids.txt').write_text('\n'.join(sorted(protected)) + '\n')
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'splits': splits,
                'sources': sources, 'protected_image_ids': len(protected),
                'files_sha256': {n:file_hash(root/n) for n in ['train.jsonl','valid.jsonl','protected-image-ids.txt']},
                'limitations': limitations, 'failed_images': errors,
                'external_model_weights_used': False}
    atomic_json(root / 'manifest.json', manifest); print('dataset completed', splits, flush=True)


def bounded_results(function,items,workers):
    """Keep at most `workers` requests live without a barrier on the slowest batch item."""
    items=iter(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(function,item) for item in (next(items,None) for _ in range(workers)) if item is not None}
        try:
            while pending:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
                    item=next(items,None)
                    if item is not None:pending.add(pool.submit(function,item))
        finally:
            for future in pending:future.cancel()


def captions(cache, split, limit):
    original_split = 'train_v6' if split == 'train' else 'validation'
    path, record = asset(cache, 'narratives-' + split + '.jsonl', CAPTIONS.format(original_split), 180_000_000)
    seen = set()
    def candidates():
        for row in jsonl(path):
            key = row['image_id']; text = ' '.join(row['caption'].split())
            if key in seen or not 25 <= len(text) <= 1600: continue
            seen.add(key); yield {**row, 'caption': text, 'split': split}
    # Deterministic subset across the entire caption source, not its leading lines.
    rows = heapq.nsmallest(limit, candidates(), key=lambda r:hashlib.sha256(r['image_id'].encode()).digest())
    return rows, record


def collect_photos(root, cache, train_images=12000, valid_images=800, workers=6):
    root = Path(root); root.mkdir(parents=True,exist_ok=True); (root/'images').mkdir(exist_ok=True)
    if (root/'manifest.json').exists(): raise ValueError('Completed dataset already exists')
    candidates = []; sources = []
    for split, count in [('train',train_images),('valid',valid_images)]:
        subset, source = captions(cache, split, count * 2); candidates.extend(subset); sources.append(source)
    metadata, records, protected = metadata_for(cache, {r['image_id'] for r in candidates}); sources.extend(records)
    rows = list(jsonl(root/'collected.jsonl')) if (root/'collected.jsonl').exists() else []
    done = {r['group_id'] for r in rows}; counts = Counter(r['split'] for r in rows)
    used_pixels = {r['model_pixel_sha256'] for r in rows}; errors = []
    budgets = {'train':train_images,'valid':valid_images}
    def prepare(item):
        try:
            m = metadata[item['image_id']]
            if item['split']=='train' and 'openimages:'+item['image_id'] in protected: return None
            image, record = photo(cache,m)
            # Captions use intended orientation. The CVDF mirror preserves original aspect ratio.
            angle = float(m.get('Rotation') or 0)
            if math.isfinite(angle) and angle: image = image.rotate(angle,expand=True)
            path, digest, pixels = save_image(root,item['image_id'],image)
            return {**attribution(m,record), 'id':item['image_id']+'-caption',
                    'image':path,'stored_image_sha256':digest,'model_pixel_sha256':pixels,
                    'domain':'everyday_photos','task':'caption','prompt':'Describe this image.',
                    'response':item['caption'],'split':item['split'],
                    'annotation_source':'Localized Narratives','annotation_license':'CC-BY-4.0',
                    'annotation_url':'https://google.github.io/localized-narratives/',
                    'annotation_origin':'Original human spoken image description, manually transcribed.',
                    'annotator_id':item['annotator_id']}
        except Exception as error: return {'error':str(error),'image_id':item['image_id']}
    # Validation first ensures exact pixel copies cannot enter training.
    with (root/'collected.jsonl').open('a') as journal:
        for split in ['valid','train']:
            selected = [r for r in candidates if r['split']==split and r['image_id'] in metadata
                        and 'openimages:'+r['image_id'] not in done]
            if counts[split]>=budgets[split]:continue
            for result in bounded_results(prepare,selected,workers):
                if result is None: continue
                if 'error' in result: errors.append(result); continue
                if result['model_pixel_sha256'] in used_pixels: continue
                journal.write(json.dumps(result,ensure_ascii=False)+'\n'); journal.flush()
                rows.append(result); done.add(result['group_id']); used_pixels.add(result['model_pixel_sha256'])
                counts[split]+=1
                if counts[split]%100==0: print('photos',split,counts[split],flush=True)
                if counts[split]>=budgets[split]:break
    persist(root,rows,sources,protected,[
        'Bounded subset; natural photo diversity does not establish general visual understanding.',
        'Human captions can contain errors, repetition, and statements not visible in pixels.',
        'Licenses are recorded from source metadata; perceptual near-duplicates remain possible.',
        'Cross-dataset image-group exclusions are applied by the combined curriculum builder.'
    ],errors)


def text_annotations(cache, split):
    suffix = 'train' if split=='train' else 'val'
    name = 'TextOCR_0.1_' + suffix + '.json'
    path, record = asset(cache,name,TEXT_OCR_MIRROR+name,350_000_000)
    data = json.loads(path.read_text())
    if not {'imgs','anns','imgToAnns'} <= data.keys(): raise ValueError('Unexpected TextOCR schema')
    record = {**record, 'original_url':'https://dl.fbaipublicfiles.com/textocr/'+name,
              'mirror_revision':TEXT_OCR_REVISION,
              'mirror_reason':'Original download endpoint returned HTTP 403; original-format annotation mirror.',
              'mirror_equivalence_independently_verified':False}
    return data, record


def word_crops(image, width, height, annotations, limit):
    """Use original word boxes; exclude illegible labels and empty/out-of-bounds crops."""
    if min(width,height)<=0 or abs(image.width/image.height-width/height)>0.04*width/height:
        raise ValueError('Source image aspect ratio does not match TextOCR annotations')
    valid = []
    for ann in annotations:
        text = ann.get('utf8_string','').strip(); box = ann.get('bbox',[])
        if not text or text=='.' or len(text)>160 or len(box)!=4: continue
        x,y,w,h = map(float,box)
        if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h)<4: continue
        left=max(0,math.floor((x-.05*w)*image.width/width)); top=max(0,math.floor((y-.1*h)*image.height/height))
        right=min(image.width,math.ceil((x+1.05*w)*image.width/width)); bottom=min(image.height,math.ceil((y+1.1*h)*image.height/height))
        if min(right-left,bottom-top)<4: continue
        valid.append((str(ann['id']),text,image.crop((left,top,right,bottom)),box))
    # A fixed sample per photo prevents text-dense images dominating the curriculum.
    return heapq.nsmallest(limit,valid,key=lambda r:hashlib.sha256(r[0].encode()).digest())


def collect_text(root,cache,train_images=5000,valid_images=400,crops=4,workers=6):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);(root/'images').mkdir(exist_ok=True)
    if (root/'manifest.json').exists(): raise ValueError('Completed dataset already exists')
    datasets={};sources=[];protected=set(); candidates=[]
    for split,count in [('train',train_images),('valid',valid_images)]:
        data,record=text_annotations(cache,split);datasets[split]=data;sources.append(record)
        if split=='valid':protected.update('openimages:'+key for key in data['imgs'])
        ids=heapq.nsmallest(count*2,data['imgs'],key=lambda key:hashlib.sha256(key.encode()).digest())
        candidates.extend((split,key) for key in ids)
    metadata,records,official_protected=metadata_for(cache,{key for _,key in candidates});sources.extend(records)
    # TextOCR has its own official split. Its train images may be in another Open Images subset;
    # conservatively exclude *all* original Open Images validation/test images from training.
    protected.update(official_protected)
    rows=list(jsonl(root/'collected.jsonl')) if (root/'collected.jsonl').exists() else []
    done={r['group_id'] for r in rows};counts=Counter()
    for split in ['train','valid']:counts[split]=len({r['group_id'] for r in rows if r['split']==split})
    errors=[];budgets={'train':train_images,'valid':valid_images}
    def prepare(item):
        split,key=item
        try:
            if split=='train' and 'openimages:'+key in protected:return None
            m=metadata[key];image,record=photo(cache,m);data=datasets[split];im=data['imgs'][key]
            annotations=[data['anns'][ann] for ann in data['imgToAnns'][key]]
            regions=word_crops(image,im['width'],im['height'],annotations,crops)
            if not regions:return None
            common={**attribution(m,record),'domain':'scene_text','split':split,
                    'annotation_source':'TextOCR','annotation_license':'CC-BY-4.0',
                    'annotation_url':'https://textvqa.org/textocr/dataset/',
                    'annotation_origin':'Original human word transcriptions and bounding boxes.',
                    'original_annotation_split':split,'parent_pixel_sha256':pixel_hash(image)}
            examples=[]
            for annid,text,crop,box in regions:
                path,digest,pixels=save_image(root,key+'-'+annid,crop)
                examples.append({**common,'id':key+'-'+annid,'image':path,
                    'task':'word_ocr','prompt':'Read the text in this image.','response':text,
                    'stored_image_sha256':digest,'model_pixel_sha256':pixels,
                    'original_word_id':annid,'original_bbox':box,'parent_image_id':key})
            # Whole scene + normalized box gives a spatially grounded reading task.
            annid,text,_,box=regions[0];x,y,w,h=box
            normalized=[round(v,4) for v in (x/im['width'],y/im['height'],(x+w)/im['width'],(y+h)/im['height'])]
            path,digest,pixels=save_image(root,key+'-scene',image)
            examples.append({**common,'id':key+'-region','image':path,'task':'region_ocr',
                'prompt':'Read the text inside the box '+str(normalized)+'. Coordinates are normalized [left, top, right, bottom].',
                'response':text,'normalized_box':normalized,'original_word_id':annid,
                'stored_image_sha256':digest,'model_pixel_sha256':pixels})
            return examples
        except Exception as error:return {'error':str(error),'image_id':key}
    with (root/'collected.jsonl').open('a') as journal:
        for split in ['valid','train']:
            selected=[(s,key) for s,key in candidates if s==split and key in metadata and 'openimages:'+key not in done]
            if counts[split]>=budgets[split]:continue
            for result in bounded_results(prepare,selected,workers):
                if result is None:continue
                if isinstance(result,dict):errors.append(result);continue
                for row in result:journal.write(json.dumps(row,ensure_ascii=False)+'\n')
                journal.flush();rows.extend(result);done.add(result[0]['group_id']);counts[split]+=1
                if counts[split]%100==0:print('text photos',split,counts[split],flush=True)
                if counts[split]>=budgets[split]:break
    persist(root,rows,sources,protected,[
        'Bounded scene-text subset; original annotation mirror could not be hash-compared to blocked original endpoint.',
        'All crops and region questions share the original photo group; official validation photos are protected.',
        'Word crops help recognize small text; whole-photo region questions can exceed 224px encoder resolution.',
        'OCR labels are transcriptions, not image descriptions or preference feedback.',
        'Perceptual duplicates and erroneous annotations require further auditing.'
    ],errors)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=['photos','text'])
    parser.add_argument('--output',required=True);parser.add_argument('--cache',required=True)
    parser.add_argument('--train-images',type=int,default=12000);parser.add_argument('--valid-images',type=int,default=800)
    parser.add_argument('--workers',type=int,default=6);parser.add_argument('--crops',type=int,default=4)
    args=parser.parse_args()
    if min(args.train_images,args.valid_images,args.workers,args.crops)<1:parser.error('Limits must be positive')
    if args.task=='photos':collect_photos(args.output,args.cache,args.train_images,args.valid_images,args.workers)
    else:collect_text(args.output,args.cache,args.train_images,args.valid_images,args.crops,args.workers)

if __name__=='__main__':main()
