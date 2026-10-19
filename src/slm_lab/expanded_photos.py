"""Additional human-captioned photos, bounded storage and protected existing splits."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import heapq
import io
import json
import math
from pathlib import Path
import re
import shutil
import time
import urllib.request

import numpy as np
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION
from tokenizers import Tokenizer

from .corpus import file_hash
from .diverse_images import (AGENT, CAPTIONS, asset, attribution, atomic_json,
                            bounded_results, jsonl, metadata_for, pixel_hash)

ROOT = Path(__file__).resolve().parents[2]


def difference_hash(image):
    gray = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
    return int.from_bytes(np.packbits((gray[:, 1:] > gray[:, :-1]).reshape(-1)).tobytes(), 'big')


def near_protected(fingerprint, protected, distance=4):
    if not len(protected):
        return False
    values = np.asarray(protected, dtype=np.uint64) ^ np.uint64(fingerprint)
    distances = np.unpackbits(values.view(np.uint8)).reshape(-1, 64).sum(1)
    return bool((distances <= distance).any())


def encode_storage(image):
    image = ImageOps.contain(image.convert('RGB'), (768, 768), Image.Resampling.LANCZOS)
    image.info.clear()
    original_pixels = {size: pixel_hash(image, size) for size in [224, 448]}
    output = io.BytesIO(); image.save(output, format='JPEG', quality=90, subsampling=2, optimize=False)
    raw = output.getvalue()
    with Image.open(io.BytesIO(raw)) as stored:
        decoded = stored.convert('RGB')
    return raw, {size: pixel_hash(decoded, size) for size in [224, 448]}, original_pixels, difference_hash(decoded)


def select_candidates(path, excluded_groups, limit):
    seen = set()
    def records():
        for row in jsonl(path):
            key = row['image_id']; caption = ' '.join(row['caption'].split())
            if (not re.fullmatch('[0-9a-f]{16}', key) or key in seen or 'openimages:' + key in excluded_groups
                    or not 25 <= len(caption) <= 1600):
                continue
            seen.add(key); yield {**row, 'caption': caption, 'split': 'train'}
    return heapq.nsmallest(limit, records(), key=lambda row: hashlib.sha256(row['image_id'].encode()).digest())


def collect(output, target=100000, workers=2, min_free_gib=35, storage_gib=8):
    output = Path(output); output.mkdir(parents=True, exist_ok=True); (output / 'images').mkdir(exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Completed additional photo collection exists; do not restart')
    cache = ROOT / 'data/image-source-cache-v1'
    protected_inputs = [ROOT / 'data/alignment-curriculum-v1/images.train.jsonl',
                        ROOT / 'data/alignment-curriculum-v1/images.valid.jsonl',
                        ROOT / 'data/everyday-images-v1/train.jsonl', ROOT / 'data/everyday-images-v1/valid.jsonl',
                        ROOT / 'data/visual-pretraining-v1/train.jsonl', ROOT / 'data/visual-pretraining-v1/valid.jsonl']
    for path in protected_inputs:
        legacy = json.loads((path.parent / 'manifest.json').read_text())
        if file_hash(path) != legacy['files_sha256'][path.name]:
            raise ValueError('Existing frozen image records changed')
    configuration = {'version': 1, 'target_additional_training_images': target, 'workers': workers,
                     'minimum_free_GiB': min_free_gib, 'maximum_stored_image_GiB': storage_gib,
                     'storage': '768px longest-side JPEG quality90 subsampling2', 'pillow': PILLOW_VERSION,
                     'implementation_sha256': file_hash(__file__),
                     'protected_inputs': {str(path): file_hash(path) for path in protected_inputs},
                     'protected_manifests': {str(path.parent / 'manifest.json'): file_hash(path.parent / 'manifest.json') for path in protected_inputs},
                     'tokenizer_sha256': file_hash(ROOT / 'data/tokenizer-corpus/tokenizer.json')}
    config_path = output / 'configuration.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError('Additional photo collection configuration/source changed')
    else:
        if any(path.name != 'images' for path in output.iterdir()):
            raise ValueError('Unrecorded photo collection files require inspection')
        atomic_json(config_path, configuration)
    def progress(message, **fields):
        atomic_json(output / 'progress.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                              'goal_achieved': False, **fields})
    progress('preparing protected image/source inventory')
    excluded_groups = set(); excluded_md5 = set(); protected_pixels = {224: set(), 448: set()}
    protected_perceptual = set()
    for path in protected_inputs:
        for row in jsonl(path):
            excluded_groups.add(row['group_id'])
            if row.get('original_md5'): excluded_md5.add(row['original_md5'])
            if path.name in {'images.valid.jsonl', 'valid.jsonl'}:
                if row.get('model_pixel_sha256'): protected_pixels[224].add(row['model_pixel_sha256'])
                if row.get('model_pixel_sha256_448'): protected_pixels[448].add(row['model_pixel_sha256_448'])
                image_path = path.parent / row['image']
                with Image.open(image_path) as image:
                    protected_perceptual.add(difference_hash(image))
                    protected_pixels[448].add(pixel_hash(image, 448))
    protected_perceptual = np.asarray(sorted(protected_perceptual), dtype=np.uint64)
    captions, record = asset(cache, 'narratives-train.jsonl', CAPTIONS.format('train_v6'), 180_000_000)
    candidates = select_candidates(captions, excluded_groups, target * 2)
    metadata, metadata_sources, official_protected = metadata_for(cache, {row['image_id'] for row in candidates})
    original_sources = {item['url']: item['sha256'] for item in
                        json.loads((ROOT / 'data/everyday-images-v1/manifest.json').read_text())['sources']}
    source_inventory = [record, *metadata_sources]
    if any(original_sources.get(item['url']) != item['sha256'] for item in source_inventory):
        raise ValueError('Publisher annotation/metadata assets differ from the original frozen release')
    inventory_path = output / 'source-inventory.json'
    if inventory_path.exists() and json.loads(inventory_path.read_text()) != source_inventory:
        raise ValueError('Collection publisher assets changed across resume')
    atomic_json(inventory_path, source_inventory)
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer-corpus/tokenizer.json'))
    journal = output / 'collected.jsonl'
    rows = list(jsonl(journal)) if journal.exists() else []
    seen_groups = {row['group_id'] for row in rows}; seen_md5 = {row['original_md5'] for row in rows if row.get('original_md5')}
    seen_pixels = {size: {row['model_pixel_sha256' if size == 224 else 'model_pixel_sha256_448'] for row in rows} for size in [224, 448]}
    for row in rows:
        if file_hash(output / row['image']) != row['stored_image_sha256']:
            raise ValueError('Previously collected image changed')
    used_bytes = sum(row['stored_image_bytes'] for row in rows)
    exclusions = Counter(); failures = []

    def prepare(item):
        key = item['image_id']; group = 'openimages:' + key
        try:
            m = metadata.get(key)
            if m is None: return {'excluded': 'missing permitted CC-BY-2.0 metadata'}
            if m['Subset'] != 'train' or group in official_protected: return {'excluded': 'official held-out source group'}
            if m.get('OriginalMD5') in excluded_md5: return {'excluded': 'existing source image MD5'}
            if len(tokenizer.encode(item['caption']).ids) > 2047: return {'excluded': 'caption exceeds own response budget'}
            url = f'https://s3.amazonaws.com/open-images-dataset/train/{key}.jpg'
            request = urllib.request.Request(url, headers={'User-Agent': AGENT})
            with urllib.request.urlopen(request, timeout=25) as response:
                declared = response.headers.get('Content-Length')
                if declared and int(declared) > 5_000_000: return {'excluded': 'source byte limit'}
                raw = response.read(5_000_001); etag = response.headers.get('ETag')
                if len(raw) > 5_000_000 or (declared and len(raw) != int(declared)):
                    return {'excluded': 'source byte limit or incomplete response'}
            transport_md5 = hashlib.md5(raw).hexdigest()
            checked_etag = bool(etag and re.fullmatch('"[0-9a-f]{32}"', etag))
            if checked_etag and etag.strip('"') != transport_md5:
                raise ValueError('Publisher object ETag does not match downloaded bytes')
            with Image.open(io.BytesIO(raw)) as original:
                if min(original.size) < 96 or original.width * original.height > 10_000_000:
                    return {'excluded': 'source dimensions'}
                image = original.convert('RGB')
            angle = float(m.get('Rotation') or 0)
            if not math.isfinite(angle): return {'excluded': 'unknown source orientation'}
            if angle: image = image.rotate(angle, expand=True)
            stored, pixels, original_pixels, perceptual = encode_storage(image)
            if any(pixels[size] in protected_pixels[size] or original_pixels[size] in protected_pixels[size] for size in [224, 448]):
                return {'excluded': 'exact protected input pixels'}
            if near_protected(perceptual, protected_perceptual): return {'excluded': 'held-out perceptual screen distance<=4'}
            provenance = {'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'etag': etag}
            row = {**attribution(m, provenance), 'id': key + '-additional-caption', 'image': 'images/' + key + '.jpg',
                   'stored_image_sha256': hashlib.sha256(stored).hexdigest(), 'stored_image_bytes': len(stored),
                   'model_pixel_sha256': pixels[224], 'model_pixel_sha256_448': pixels[448],
                   'pre_storage_pixel_sha256': {str(size): digest for size, digest in original_pixels.items()},
                   'difference_hash64': str(perceptual), 'download_md5_hex': transport_md5, 'object_etag_md5_verified': checked_etag,
                   'original_flickr_MD5_equivalence_verified': False,
                   'prompt': 'Describe this image.', 'response': item['caption'], 'task': 'caption', 'domain': 'everyday_photos',
                   'split': 'train', 'visual_source': 'additional_everyday_photos',
                   'annotation_source': 'Localized Narratives', 'annotation_license': 'CC-BY-4.0',
                   'annotation_url': 'https://google.github.io/localized-narratives/',
                   'annotation_origin': 'Original human spoken image description, manually transcribed; whitespace collapsed.',
                   'annotator_id': item['annotator_id'], 'source_caption_sha256': hashlib.sha256(item['caption'].encode()).hexdigest()}
            return {'row': row, 'stored': stored}
        except Exception as error:
            return {'error': str(error), 'image_id': key}

    selected = [row for row in candidates if 'openimages:' + row['image_id'] not in seen_groups]
    storage_limit = int(storage_gib * 1024**3); stop_reason = 'candidate source exhausted'
    progress('collecting additional licensed human-captioned training photos', images=len(rows), target=target)
    with journal.open('a') as stream:
        for result in bounded_results(prepare, selected, workers):
            if len(rows) >= target: stop_reason = 'image target reached'; break
            while shutil.disk_usage(output).free < min_free_gib * 1024**3:
                progress('waiting for free disk; active training has priority', images=len(rows), minimum_free_GiB=min_free_gib)
                time.sleep(20)
            if 'excluded' in result: exclusions[result['excluded']] += 1; continue
            if 'error' in result:
                failures.append(result); continue
            row = result['row']
            if row['group_id'] in excluded_groups or row['group_id'] in seen_groups: raise ValueError('Protected/repeated source group admitted')
            if (row.get('original_md5') and row['original_md5'] in seen_md5) or any(row['model_pixel_sha256' if size == 224 else 'model_pixel_sha256_448'] in seen_pixels[size] for size in [224, 448]):
                exclusions['duplicate collected source/pixels'] += 1; continue
            if used_bytes + len(result['stored']) > storage_limit:
                stop_reason = 'bounded image storage limit reached'; break
            path = output / row['image']; temporary = path.with_suffix('.tmp')
            temporary.write_bytes(result['stored']); temporary.replace(path)
            stream.write(json.dumps(row, ensure_ascii=False) + '\n'); stream.flush()
            rows.append(row); seen_groups.add(row['group_id']); used_bytes += row['stored_image_bytes']
            if row.get('original_md5'): seen_md5.add(row['original_md5'])
            for size in [224, 448]: seen_pixels[size].add(row['model_pixel_sha256' if size == 224 else 'model_pixel_sha256_448'])
            if len(rows) % 100 == 0:
                print('additional photos', len(rows), 'stored MiB', round(used_bytes / 1024**2), flush=True)
                progress('collecting additional licensed human-captioned training photos', images=len(rows), target=target,
                         stored_bytes=used_bytes, exclusions=dict(exclusions), download_failures=len(failures))
    if not rows: raise ValueError('No additional licensed photos retained')
    if file_hash(__file__) != configuration['implementation_sha256'] or any(file_hash(path) != digest for path, digest in
            {**configuration['protected_inputs'], **configuration['protected_manifests']}.items()):
        raise ValueError('Collection implementation/protected sources changed')
    (output / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    (output / 'failures.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in failures))
    manifest = {'configuration': configuration, 'train_examples': len(rows), 'stored_image_bytes': used_bytes,
                'sources': [record, *metadata_sources], 'termination_reason': stop_reason,
                'excluded_existing_groups': len(excluded_groups), 'official_protected_groups': len(official_protected),
                'protected_perceptual_fingerprints': len(protected_perceptual), 'exclusions': dict(exclusions),
                'files_sha256': {name: file_hash(output / name) for name in ['train.jsonl', 'collected.jsonl', 'failures.jsonl']},
                'validation': 'Reuse the existing frozen image validation sets; no new validation allocation.',
                'external_model_weights_used': False, 'generated_captions_used': False, 'goal_achieved': False,
                'limitations': ['Additional photos are collected, not proof of trained capability.',
                               'Source metadata provide licenses; ownership and caption factual accuracy are not independently certified.',
                               'JPEG storage changes pixels; all recorded input hashes use the decoded stored JPEG.',
                               'Group, source-MD5, exact-pixel and simple perceptual screens do not certify absence of semantic near-duplicates.',
                               'Full source JPEGs are not stored; download hashes and original URLs/attribution remain.']}
    atomic_json(output / 'manifest.json', manifest)
    progress('additional photo collection completed; not yet trained', images=len(rows), termination_reason=stop_reason)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True); parser.add_argument('--target', type=int, default=100000)
    parser.add_argument('--workers', type=int, default=2); parser.add_argument('--min-free-gib', type=float, default=35)
    parser.add_argument('--storage-gib', type=float, default=8)
    args = parser.parse_args()
    if args.target < 1 or not 1 <= args.workers <= 4 or args.min_free_gib < 30 or args.storage_gib <= 0:
        parser.error('Positive bounded budgets, at most four workers and >=30 GiB reserve required')
    collect(args.output, args.target, args.workers, args.min_free_gib, args.storage_gib)


if __name__ == '__main__':
    main()
