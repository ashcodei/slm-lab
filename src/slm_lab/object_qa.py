"""Derive grounded object/count/spatial questions from original Open Images boxes.

Reuse only photos admitted by the existing pixel/group leakage audit. Questions
are programmatically derived from public annotations, not authored by humans.
"""
import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import google_crc32c
from PIL import Image

from .corpus import file_hash
from .diverse_images import asset, atomic_json, jsonl
from .wikipedia_corpus import protected_questions

SOURCES = {
    'train': ('oidv6-train-bbox.csv', 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv', 3_000_000_000),
    'valid': ('oidv5-valid-bbox.csv', 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv', 100_000_000),
    'classes': ('oidv7-box-classes.csv', 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv', 1_000_000),
}
TASKS = {'object_region', 'object_count', 'object_spatial'}
FLAGS = ['IsGroupOf', 'IsDepiction', 'IsInside', 'IsOccluded', 'IsTruncated']


def checked_asset(cache, source):
    name, url, limit = source
    path, record = asset(cache, name, url, limit)
    with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as response:
        hashes = dict(item.strip().split('=', 1) for item in ','.join(response.headers.get_all('x-goog-hash', [])).split(',') if '=' in item)
        etag = response.headers.get('ETag')
    if etag != record['etag'] or 'crc32c' not in hashes:
        raise ValueError('Official annotation version/checksum cannot be verified')
    if record.get('verified_publisher_crc32c') != hashes['crc32c']:
        checksum = google_crc32c.Checksum()
        with path.open('rb') as handle:
            while block := handle.read(8 << 20):
                checksum.update(block)
        if base64.b64encode(checksum.digest()).decode() != hashes['crc32c']:
            raise ValueError('Publisher CRC32C mismatch: ' + name)
        record = {**record, 'verified_publisher_crc32c': hashes['crc32c']}
        atomic_json(path.with_suffix(path.suffix + '.source.json'), record)
    return path, record


def rotate_box(box, angle):
    left, top, right, bottom = map(float, box)
    if not all(math.isfinite(v) for v in box) or not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
        raise ValueError('Invalid source bounding box')
    if angle == 0:
        return [left, top, right, bottom]
    if angle == 90:
        return [top, 1-right, bottom, 1-left]
    if angle == 180:
        return [1-right, 1-bottom, 1-left, 1-top]
    if angle == 270:
        return [1-bottom, left, 1-top, right]
    raise ValueError('Unsupported photo rotation')


def overlap(first, second):
    intersection = max(0, min(first[2], second[2])-max(first[0], second[0])) * max(0, min(first[3], second[3])-max(first[1], second[1]))
    area = lambda box: (box[2]-box[0])*(box[3]-box[1])
    return intersection/(area(first)+area(second)-intersection)


def clear(box):
    return box['source'] in {'xclick', 'activemil'} and all(box['flags'][name] == 0 for name in FLAGS)


def readable(box, photo):
    if 'stored_image_dimensions' not in photo:
        return True
    width,height=photo['stored_image_dimensions'];size=448
    scaled=(size,round(height/width*size)) if width>=height else (round(width/height*size),size)
    return min((box['box'][2]-box['box'][0])*scaled[0],(box['box'][3]-box['box'][1])*scaled[1])>=24


def questions_for(photo, boxes):
    boxes = sorted(boxes, key=lambda b: (b['label_id'], b['box'], b['row_number']))
    classes = defaultdict(list)
    for box in boxes:
        classes[box['label_id']].append(box)
    results = []

    def add(task, prompt, response, evidence, **extra):
        identity = hashlib.sha256((photo['group_id'] + task + prompt).encode()).hexdigest()
        results.append({**photo, 'id': 'object-' + identity, 'task': task, 'prompt': prompt, 'response': response,
                        'annotation_source': 'Open Images original object boxes', 'annotation_license': 'CC-BY-4.0',
                        'annotation_url': 'https://storage.googleapis.com/openimages/web/download_v7.html',
                        'annotation_origin': 'Programmatic question/answer derived from original manual or human-verified boxes; not human-written VQA or preference feedback.',
                        'box_evidence': evidence, **extra})

    # Teach named regions without discarding the rest of the photo context.
    for box in boxes:
        left, top, right, bottom = box['box']
        ambiguous = any(other['label_id']!=box['label_id'] and overlap(other['box'],box['box'])>.95 for other in boxes)
        if clear(box) and readable(box,photo) and not ambiguous and min(right-left, bottom-top) >= .10:
            normalized = [round(value, 6) for value in box['box']]
            add('object_region', 'Identify the object inside the box ' + str(normalized) + '. Answer with the object class name only.',
                box['label'], [box], normalized_box=normalized)
            if sum(row['task']=='object_region' for row in results) >= 3:
                break
    # Exhaustive class-instance annotations allow counting only when the whole
    # class is clear. Never remove a crowd/uncertain box and count what remains.
    for members in classes.values():
        if not 1 <= len(members) <= 6 or not all(clear(box) and readable(box,photo) for box in members):
            continue
        if any(overlap(first['box'], second['box']) > .65 for i, first in enumerate(members) for second in members[i+1:]):
            continue
        label = members[0]['label']
        add('object_count', "How many instances of the object class '" + label + "' are visible? Answer with only the number.",
            str(len(members)), members)
        if sum(row['task']=='object_count' for row in results) >= 2:
            break
    unique = [members[0] for members in classes.values() if len(members)==1 and clear(members[0]) and readable(members[0],photo)]
    for i, first in enumerate(unique):
        for second in unique[i+1:]:
            if overlap(first['box'], second['box']) > .2:
                continue
            center = lambda box: ((box[0]+box[2])/2, (box[1]+box[3])/2)
            a, b = center(first['box']), center(second['box'])
            axis = 0 if abs(a[0]-b[0]) >= abs(a[1]-b[1]) else 1
            if abs(a[axis]-b[axis]) < .18:
                continue
            seed = hashlib.sha256((photo['group_id']+first['label_id']+second['label_id']).encode()).digest()
            negative_direction = seed[0] % 2 == 0
            relation = ('to the left of' if negative_direction else 'to the right of') if axis==0 else ('above' if negative_direction else 'below')
            answer = (a[axis] < b[axis]) if negative_direction else (a[axis] > b[axis])
            prompt = f"Is the center of the '{first['label']}' object {relation} the center of the '{second['label']}' object in this image? Answer yes or no."
            add('object_spatial', prompt, 'yes' if answer else 'no', [first, second], spatial_axis=axis, spatial_relation=relation)
            return results
    return results


def photos_and_rotations(curriculum, cache, split):
    rows = {row['source_image_id']: row for row in jsonl(curriculum/f'images.{split}.jsonl') if row.get('curriculum_source')=='everyday_photos'}
    metadata = cache/('openimages-train.csv' if split=='train' else 'openimages-validation.csv')
    found = {}
    with metadata.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['ImageID']
            if key not in rows:
                continue
            value = float(row.get('Rotation') or 0)
            # Reproduce the already stored photo collector's finite-angle rule.
            angle = value if math.isfinite(value) else 0
            if angle not in {0, 90, 180, 270}:
                raise ValueError('Unsupported source rotation: ' + key)
            found[key] = {**rows[key], 'applied_source_rotation': int(angle), 'source_rotation_field': row.get('Rotation')}
    if set(found) != set(rows):
        raise ValueError('Photo rotation metadata missing')
    return found, metadata


def extract_subset(path, photos, labels, progress):
    grouped = defaultdict(list); invalid_classes = defaultdict(set); rejected = Counter(); seen = {}; matched=0
    with path.open(newline='') as handle:
        reader = csv.reader(handle); columns = next(reader); indices = {name:i for i,name in enumerate(columns)}
        for ordinal, values in enumerate(reader, 2):
            key = values[indices['ImageID']]
            if key not in photos:
                continue
            label_id = values[indices['LabelName']]
            try:
                if label_id not in labels or float(values[indices['Confidence']]) != 1:
                    raise ValueError('Unknown class/confidence')
                original = [float(values[indices[name]]) for name in ['XMin','YMin','XMax','YMax']]
                box = rotate_box(original, photos[key]['applied_source_rotation'])
                flags = {name:int(values[indices[name]]) for name in FLAGS}
                if any(value not in {-1,0,1} for value in flags.values()):
                    raise ValueError('Unknown attribute value')
                identity = (key, label_id, tuple(box))
                if identity in seen:
                    if seen[identity] != flags:
                        invalid_classes[key].add(label_id)
                    rejected['duplicate_box'] += 1; continue
                seen[identity] = flags
                grouped[key].append({'label_id':label_id,'label':labels[label_id],'source':values[indices['Source']],
                                     'original_box':original,'box':box,'flags':flags,'row_number':ordinal})
                matched+=1
            except (ValueError, IndexError):
                invalid_classes[key].add(label_id); rejected['invalid_annotation'] += 1
            if matched and matched % 10000 == 0:
                progress(len(grouped))
    # An invalid class instance makes its class count ambiguous. Exclude that
    # class altogether, while leaving unrelated classes on the same photo usable.
    for key, bad in invalid_classes.items():
        grouped[key] = [box for box in grouped[key] if box['label_id'] not in bad]
    return grouped, dict(rejected)


def prepare(output, curriculum, cache, questions):
    output, curriculum, cache, questions = map(Path, [output, curriculum, cache, questions])
    output.mkdir(parents=True, exist_ok=True)
    with (output/'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(output, curriculum, cache, questions)


def _prepare(output, curriculum, cache, questions):
    original = json.loads((curriculum/'manifest.json').read_text())
    for split in ['train','valid']:
        name = f'images.{split}.jsonl'
        if file_hash(curriculum/name) != original['files_sha256'][name]:
            raise ValueError('Audited image curriculum changed')
    inputs = {}
    for key, source in SOURCES.items():
        inputs[key] = checked_asset(cache, source)
    with inputs['classes'][0].open(newline='') as handle:
        labels = dict(csv.reader(handle))
    configuration={'version':2,'implementation_sha256':file_hash(Path(__file__)),
                   'question_input_resolution':448,'minimum_object_edge_pixels':24,
                   'source_sha256':{key:record['sha256'] for key,(_,record) in inputs.items()},
                   'image_curriculum_manifest_sha256':file_hash(curriculum/'manifest.json'),
                   'questions_sha256':file_hash(questions),
                   'rotation_metadata_sha256':{name:file_hash(cache/('openimages-train.csv' if name=='train' else 'openimages-validation.csv')) for name in ['train','valid']}}
    configuration_path=output/'configuration.json'
    if configuration_path.exists() and json.loads(configuration_path.read_text())!=configuration:
        raise ValueError('Object question curriculum has a different configuration')
    atomic_json(configuration_path,configuration)
    if (output/'manifest.json').exists():
        manifest=json.loads((output/'manifest.json').read_text())
        if manifest['configuration']!=configuration:raise ValueError('Completed object curriculum changed')
        for name,digest in manifest['files_sha256'].items():
            if file_hash(output/name)!=digest:raise ValueError('Completed object records changed')
        return manifest
    matcher = protected_questions(questions)
    counts = {}; groups = {}; pixels = {}; exclusions = {}
    for split in ['train','valid']:
        photos, metadata = photos_and_rotations(curriculum, cache, split)
        def progress(count):
            atomic_json(output/'progress.json', {'status':'extracting original object annotations','split':split,'matched_photos':count})
        boxes, rejected = extract_subset(inputs[split][0], photos, labels, progress)
        # Store compact original evidence for audit and future question variants.
        with (output/f'annotations.{split}.jsonl').open('w') as evidence:
            for key in sorted(boxes):
                evidence.write(json.dumps({'image_id':key,'boxes':boxes[key]},ensure_ascii=True)+'\n')
        accepted = []; bins = Counter(); group_set = set(); pixel_set = {224:set(),448:set()}; filtered = Counter()
        # Per-answer caps prevent the common count=1/class labels overwhelming
        # all other supervision. They do not make the result a standard benchmark.
        cap = 300 if split=='train' else 30
        for key in sorted(boxes):
            photo = photos[key]
            if file_hash(Path(photo['image'])) != photo['stored_image_sha256']:
                raise ValueError('Stored photo changed: '+key)
            with Image.open(photo['image']) as image:
                photo={**photo,'stored_image_dimensions':list(image.size),'qa_input_resolution':448}
            for row in questions_for(photo, boxes[key]):
                if list(matcher.iter(' '.join((row['prompt']+' '+row['response']).casefold().split()))):
                    filtered['reserved_question'] += 1; continue
                bucket = (row['task'],row['response'].casefold())
                task_cap = cap*5 if row['task']=='object_spatial' else cap
                if bins[bucket]>=task_cap:
                    filtered['answer_frequency_cap'] += 1; continue
                bins[bucket] += 1; accepted.append(row); group_set.add(row['group_id'])
                pixel_set[224].add(row['model_pixel_sha256']);pixel_set[448].add(row['model_pixel_sha256_448'])
        path = output/f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row,ensure_ascii=True)+'\n' for row in accepted))
        counts[split] = {'examples':len(accepted),'original_images':len(group_set),
                         'by_task':dict(Counter(row['task'] for row in accepted)),
                         'by_answer':{task:dict(Counter(row['response'] for row in accepted if row['task']==task)) for task in TASKS}}
        if not accepted or set(counts[split]['by_task'])!=TASKS:
            raise ValueError('Missing object QA task coverage')
        groups[split] = group_set; pixels[split] = pixel_set; exclusions[split] = {**rejected,**filtered}
        inputs['rotation_'+split] = (metadata, {'sha256':file_hash(metadata),'path':str(metadata.resolve())})
    if groups['train'] & groups['valid'] or any(pixels['train'][size] & pixels['valid'][size] for size in [224,448]):
        raise ValueError('Object QA split leakage')
    manifest = {'created_at':datetime.now(timezone.utc).isoformat(),'splits':counts,'configuration':configuration,
                'sources':{key:record for key,(_,record) in inputs.items()},
                'image_curriculum_manifest_sha256':file_hash(curriculum/'manifest.json'),
                'questions_sha256':file_hash(questions),'annotation_license':'CC-BY-4.0',
                'train_valid_image_group_overlap':0,'train_valid_exact_input_pixel_overlap':{'224':0,'448':0},
                'exclusions':exclusions,'external_model_weights_used':False,
                'files_sha256':{name:file_hash(output/name) for name in ['train.jsonl','valid.jsonl','annotations.train.jsonl','annotations.valid.jsonl']},
                'limitations':['Questions are derived from public annotations; annotation mistakes remain possible.',
                               'Counts use specific annotated classes; absent unannotated classes are never treated as negatives.',
                               'Only unambiguous, non-crowd, visible instances form this introductory curriculum.',
                               'Every target must span at least 24 pixels on each edge after the recorded 448px letterbox transform.',
                               'Spatial truth uses box centers, not 3D geometry or semantic interaction labels.',
                               'Photo groups share the existing caption split; this is tuning validation, not a new independent final test.']}
    atomic_json(output/'manifest.json',manifest)
    atomic_json(output/'progress.json',{'status':'object question curriculum prepared; not trained','splits':counts})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--curriculum',required=True)
    parser.add_argument('--cache',required=True);parser.add_argument('--questions',required=True)
    args = parser.parse_args();result = prepare(args.output,args.curriculum,args.cache,args.questions)
    print(json.dumps({key:value for key,value in result.items() if key!='sources'},indent=2),flush=True)


if __name__=='__main__':
    main()
