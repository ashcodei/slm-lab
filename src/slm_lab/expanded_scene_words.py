"""Unused real-word annotations from already-admitted TextOCR training photos."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

from PIL import Image, ImageOps
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash, word_crops

ROOT = Path(__file__).resolve().parents[2]


def additional_crops(image, width, height, annotations, used_ids, limit=16):
    unused = [ann for ann in annotations if str(ann['id']) not in used_ids]
    return [item for item in word_crops(image, width, height, unused, len(unused))
            if min(item[2].size) >= 8][:limit]


def prepare(output, extra_per_photo=16, storage_gib=2):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()): raise ValueError('Use a fresh additional scene-word directory')
    if shutil.disk_usage(output).free < 35 * 1024**3:
        raise ValueError('Need 35 GiB reserve alongside active training')
    (output / 'images').mkdir()
    scene = ROOT / 'data/scene-text-images-v1'
    base = ROOT / 'data/alignment-curriculum-v1'
    source_manifest = json.loads((scene / 'manifest.json').read_text())
    base_manifest = json.loads((base / 'manifest.json').read_text())
    paths = [scene / 'train.jsonl', scene / 'valid.jsonl', base / 'images.train.jsonl', base / 'images.valid.jsonl']
    for path in paths:
        manifest = source_manifest if path.parent == scene else base_manifest
        if file_hash(path) != manifest['files_sha256'][path.name]: raise ValueError('Frozen scene/main records changed')
    annotations_path = ROOT / 'data/image-source-cache-v1/TextOCR_0.1_train.json'
    annotation_source = next(item for item in source_manifest['sources'] if item['url'].endswith('/TextOCR_0.1_train.json'))
    if file_hash(annotations_path) != annotation_source['sha256']:
        raise ValueError('Original-format pinned annotation mirror changed')
    annotations = json.loads(annotations_path.read_text())
    original_rows = list(jsonl(scene / 'train.jsonl'))
    admitted = {row['group_id'] for row in jsonl(base / 'images.train.jsonl')}
    protected = {row['group_id'] for row in jsonl(base / 'images.valid.jsonl')}
    protected.update(row['group_id'] for row in jsonl(scene / 'valid.jsonl'))
    used = {row['original_word_id'] for row in original_rows if row['task'] == 'word_ocr'}
    parents = {row['source_image_id']: row for row in original_rows
               if row['group_id'] in admitted and row['group_id'] not in protected and row['source_subset'] == 'train'}
    existing_pixels = {224: set(), 448: set()}
    for path in paths:
        for row in jsonl(path):
            image_path = path.parent / row['image']
            if row.get('model_pixel_sha256'): existing_pixels[224].add(row['model_pixel_sha256'])
            if row.get('model_pixel_sha256_448'): existing_pixels[448].add(row['model_pixel_sha256_448'])
            elif path.name in {'valid.jsonl', 'images.valid.jsonl'}:
                with Image.open(image_path) as image: existing_pixels[448].add(pixel_hash(image, 448))
    config = {'inputs_sha256': {str(path): file_hash(path) for path in paths},
              'annotation_source': annotation_source, 'extra_per_photo': extra_per_photo, 'storage_limit_GiB': storage_gib,
              'implementation_sha256': file_hash(__file__), 'storage': '448px longest side, lossless RGB PNG',
              'parent_policy': 'Only existing admitted train parents; protect all current validation parents/pixels.'}
    atomic_json(output / 'configuration.json', config)
    rows = []; excluded = Counter(); stored_bytes = 0; image_count = 0
    limit = int(storage_gib * 1024**3); full = False
    provenance_fields = ['source', 'source_image_id', 'group_id', 'source_url', 'author', 'author_profile_url',
                         'source_title', 'original_url', 'original_md5', 'license', 'license_url', 'source_subset',
                         'download_url', 'download_sha256', 'download_etag', 'annotation_source', 'annotation_license',
                         'annotation_url', 'annotation_origin', 'original_annotation_split']
    with (output / 'collected.jsonl').open('w') as journal:
        for key, parent in sorted(parents.items()):
            if not re.fullmatch('[0-9a-f]{16}', key): raise ValueError('Unsafe source photo identifier')
            if shutil.disk_usage(output).free < 35 * 1024**3:
                excluded['free disk reserve reached'] += 1; break
            original = ROOT / 'data/image-source-cache-v1/source-images/train' / (key + '.jpg')
            if file_hash(original) != parent['download_sha256']:
                raise ValueError('Cached original training photo changed: ' + key)
            metadata = annotations['imgs'][key]
            word_annotations = [annotations['anns'][identifier] for identifier in annotations['imgToAnns'][key]]
            if any(annotation['image_id'] != key for annotation in word_annotations):
                raise ValueError('Word annotation/photo mapping differs')
            with Image.open(original) as image:
                image = image.convert('RGB')
                crops = additional_crops(image, metadata['width'], metadata['height'], word_annotations, used, extra_per_photo)
            accepted_here = 0
            for annotation_id, text, crop, box in crops:
                crop = ImageOps.contain(crop.convert('RGB'), (448, 448), Image.Resampling.LANCZOS); crop.info.clear()
                hashes = {size: pixel_hash(crop, size) for size in [224, 448]}
                if any(hashes[size] in existing_pixels[size] for size in hashes):
                    excluded['existing train/held-out input pixels'] += 1; continue
                identifier = hashlib.sha256((key + ':' + annotation_id).encode()).hexdigest()
                target = output / 'images' / (identifier + '.png')
                crop.save(target, format='PNG', compress_level=1)
                size = target.stat().st_size
                if stored_bytes + size > limit:
                    target.unlink(); full = True; excluded['bounded storage limit reached'] += 1; break
                row = {**{name: parent[name] for name in provenance_fields},
                       'id': 'additional-word-' + identifier, 'image': str(target.resolve()),
                       'task': 'word_ocr', 'domain': 'scene_text', 'split': 'train',
                       'prompt': 'Read the text in this image.', 'response': text, 'original_word_id': annotation_id,
                       'original_bbox': box, 'parent_image_id': key, 'stored_image_sha256': file_hash(target),
                       'model_pixel_sha256': hashes[224], 'model_pixel_sha256_448': hashes[448],
                       'curriculum_source': 'additional_scene_words', 'stored_image_bytes': size,
                       'annotation_mirror_equivalence_independently_verified': False}
                journal.write(json.dumps(row, ensure_ascii=False) + '\n'); journal.flush()
                rows.append(row); stored_bytes += size; used.add(annotation_id); accepted_here += 1
                for resolution in hashes: existing_pixels[resolution].add(hashes[resolution])
            image_count += bool(accepted_here)
            if image_count % 100 == 0 and accepted_here:
                print('additional scene-word parents', image_count, 'new word crops', len(rows), flush=True)
                atomic_json(output / 'progress.json', {'status': 'preparing additional real-word crops', 'parents': image_count,
                                                       'new_word_crops': len(rows), 'stored_bytes': stored_bytes, 'goal_achieved': False})
            if full: break
    if not rows: raise ValueError('No unused real-word labels retained')
    if any(file_hash(path) != digest for path, digest in config['inputs_sha256'].items()):
        raise ValueError('Protected source records changed during preparation')
    (output / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'configuration': config,
                'train_examples': len(rows), 'train_parent_images': image_count, 'stored_bytes': stored_bytes,
                'existing_annotations_reused': False, 'new_parent_images_added': 0, 'validation_parent_overlap': 0,
                'excluded': dict(excluded), 'files_sha256': {name: file_hash(output / name) for name in ['train.jsonl', 'collected.jsonl']},
                'goal_achieved': False, 'external_model_weights_used': False,
                'limitations': ['Additional human-labelled words from the same admitted training photos, not new independent scenes.',
                                'Original-format annotation mirror remains unverified against the blocked original endpoint.',
                                'Word-box labels can contain transcription errors; tiny/illegible labels are filtered but not exhaustively audited.',
                                'Preparation is not training or proof of native arbitrary-scene/document reading.']}
    atomic_json(output / 'manifest.json', manifest)
    atomic_json(output / 'progress.json', {'status': 'additional real-word crops prepared; not trained',
                                          'parents': image_count, 'new_word_crops': len(rows), 'goal_achieved': False})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    parser.add_argument('--extra-per-photo', type=int, default=16); parser.add_argument('--storage-gib', type=float, default=2)
    args = parser.parse_args()
    if not 1 <= args.extra_per_photo <= 32 or args.storage_gib <= 0: parser.error('Use bounded positive crop/storage budgets')
    prepare(args.output, args.extra_per_photo, args.storage_gib)


if __name__ == '__main__':
    main()
