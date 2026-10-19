"""Unused original human word labels, cropped from already-licensed native photos.

Preparation writes references and geometry only. It preserves the earlier
validation population and does not copy or download image data.
"""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch

from .byte_word_reader import HEIGHT, WIDTH, labels, load_rows, pixels
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl
from .evaluation_provenance import capture, pixels_sha256
from .rendered_text import text_group
from .vqa_pairs import known_photo_aliases

ROOT = Path(__file__).resolve().parents[2]


def word_geometry(annotation, original_dimensions, native_dimensions):
    text = annotation.get('utf8_string', '').strip()
    if not text or text == '.': raise ValueError('original label unreadable/empty')
    labels(text)
    width, height = original_dimensions; native_width, native_height = native_dimensions
    if min(width, height, native_width, native_height) <= 0: raise ValueError('invalid original/native photo dimensions')
    if abs(native_width / native_height - width / height) > .04 * width / height: raise ValueError('original/native photo aspect differs')
    rectangle = annotation['bbox']
    if len(rectangle) != 4 or not all(math.isfinite(float(value)) for value in rectangle): raise ValueError('invalid original word rectangle')
    x, y, w, h = map(float, rectangle)
    if min(w, h) < 4: raise ValueError('original word rectangle below4px')
    bounds = [max(0, math.floor((x - .05 * w) * native_width / width)),
              max(0, math.floor((y - .1 * h) * native_height / height)),
              min(native_width, math.ceil((x + 1.05 * w) * native_width / width)),
              min(native_height, math.ceil((y + 1.1 * h) * native_height / height))]
    if min(bounds[2] - bounds[0], bounds[3] - bounds[1]) < 8: raise ValueError('native margin crop below8px on a side')
    return text, bounds


def crop_tensor(image, bounds):
    if len(bounds) != 4 or any(not isinstance(value, int) for value in bounds): raise ValueError('Native word crop requires integer pixel bounds')
    x0, y0, x1, y1 = bounds
    if not 0 <= x0 < x1 <= image.width or not 0 <= y0 < y1 <= image.height: raise ValueError('Native crop outside actual source photo')
    crop = image.crop(tuple(bounds)).convert('RGB').resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
    array = np.asarray(crop, dtype=np.float32) / 127.5 - 1
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


def additional_pixels(row):
    path = Path(row['image']).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Additional native photo path leaves project')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['source_image_sha256']: raise ValueError('Admitted native photo bytes changed')
    with Image.open(io.BytesIO(raw)) as image:
        if list(image.size) != row['native_dimensions']: raise ValueError('Admitted native photo dimensions changed')
        value = crop_tensor(image.convert('RGB'), row['native_crop_bounds'])
    if pixels_sha256(value) != row['reader_input_pixels_sha256']: raise ValueError('Actual native word crop differs from prepared input')
    return value


def prepare(output):
    output = Path(output)
    if output.exists(): raise ValueError('Preserve existing additional real-word preparation; inspect before continuing')
    old_data = ROOT / 'data/byte-word-reader-v1'; previous, old_manifest = load_rows(old_data)
    previous_manifest_sha = file_hash(old_data / 'manifest.json')
    used = {(row['group_id'], str(row.get('original_word_id', row['id']))) for row in previous['train']}
    protected_words = {text_group(row['response']) for row in previous['valid'] if row['byte_family'] == 'rendered'}
    protected_pixels = {pixels_sha256(pixels(row)) for row in previous['valid']}
    protected_groups = {row['group_id'] for row in previous['valid']}; protected_aliases = set()
    inventories = {}; parents = []
    for name in ['alignment-curriculum-v1/images.train.jsonl', 'alignment-curriculum-v1/images.valid.jsonl', 'visual-pretraining-v1/valid.jsonl']:
        path = ROOT / 'data' / name; digest = file_hash(path)
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        if digest != manifest['files_sha256'][path.name]: raise ValueError('Previously frozen photo inventory changed')
        inventories[str(path)] = digest
        for row in jsonl(path):
            if name.endswith('images.train.jsonl'):
                if row.get('task') == 'region_ocr' and row.get('domain') == 'scene_text': parents.append(row)
            else:
                protected_groups.add(row['group_id']); protected_aliases.update(known_photo_aliases(row))
    if len(parents) != 5000 or len({row['group_id'] for row in parents}) != 5000: raise ValueError('Original full training-photo scope changed')
    annotation_path = ROOT / 'data/image-source-cache-v1/TextOCR_0.1_train.json'
    annotation_receipt = json.loads(annotation_path.with_suffix('.json.source.json').read_text())
    if file_hash(annotation_path) != annotation_receipt['sha256']: raise ValueError('Original human training annotations changed')
    original = json.loads(annotation_path.read_text())
    files = [Path(__file__)] + [ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['byte_word_reader', 'corpus', 'diverse_images', 'evaluation_provenance', 'rendered_text', 'vqa_pairs']]
    runtime = capture(files); output.mkdir(parents=True); exclusions = Counter(); parents_with_new_words = set(); native_sources = {}
    statistics = Counter(); seen = set(); decoded = hashlib.sha256()
    with gzip.open(output / 'additional.train.jsonl.gz', 'wb', compresslevel=6) as writer:
        for index, parent in enumerate(sorted(parents, key=lambda row: row['group_id'])):
            key = parent['source_image_id']; metadata = original['imgs'][key]
            if not re.fullmatch('[0-9a-f]{16}', key) or parent['source_subset'] not in {'train', 'validation', 'test'}: raise ValueError('Invalid source photo path identity')
            if metadata['set'] != 'train' or parent['group_id'] in protected_groups or known_photo_aliases(parent) & protected_aliases:
                raise ValueError('Additional word photo differs from original protected training split')
            if parent['license'] != 'CC-BY-2.0' or not parent['author'] or not parent['author_profile_url']: raise ValueError('Native photo lacks original creator/rights metadata')
            source = ROOT / 'data/image-source-cache-v1/source-images' / parent['source_subset'] / (key + '.jpg')
            raw = source.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
            if digest != parent['download_sha256']: raise ValueError('Cached native training photo differs from published source receipt')
            native_sources[str(source)] = digest
            with Image.open(io.BytesIO(raw)) as opened:
                image = opened.convert('RGB')
                for annotation_id in original['imgToAnns'][key]:
                    annotation = original['anns'][str(annotation_id)]
                    if str(annotation['image_id']) != key or str(annotation['id']) != str(annotation_id): raise ValueError('Original word annotation/photo identity differs')
                    identity = (parent['group_id'], str(annotation_id))
                    if identity in used:
                        exclusions['already in original byte-reader training'] += 1; continue
                    if identity in seen: raise ValueError('Repeated additional original word annotation')
                    try: text, bounds = word_geometry(annotation, (metadata['width'], metadata['height']), image.size)
                    except ValueError as error:
                        exclusions[str(error)] += 1; continue
                    if text_group(text) in protected_words:
                        exclusions['protected rendered-validation word'] += 1; continue
                    actual = crop_tensor(image, bounds); input_sha = pixels_sha256(actual)
                    if input_sha in protected_pixels:
                        exclusions['exact reader-input validation pixel match'] += 1; continue
                    row = {name: parent[name] for name in ['source_image_id', 'group_id', 'source_url', 'author', 'author_profile_url', 'license', 'license_url']}
                    row.update(id='unused-original-word:' + str(annotation_id), original_word_id=str(annotation_id), image=str(source),
                               source_image_sha256=digest, native_dimensions=list(image.size), native_crop_bounds=bounds,
                               original_bbox=annotation['bbox'], original_annotation_dimensions=[metadata['width'], metadata['height']],
                               response=text, task='word_ocr', split='train', byte_family='additional_native_real_word',
                               reader_input_pixels_sha256=input_sha, annotation_source_sha256=annotation_receipt['sha256'],
                               annotation_license='CC-BY-4.0', annotation_origin='Original human word transcript/geometry, not generated by a model.')
                    encoded = (json.dumps(row, ensure_ascii=False) + '\n').encode(); writer.write(encoded); decoded.update(encoded)
                    statistics['new_original_words'] += 1; statistics['target_UTF8_bytes'] += len(text.encode()); statistics['non_ASCII_transcripts'] += int(not text.isascii())
                    seen.add(identity); parents_with_new_words.add(parent['group_id'])
            if index % 250 == 0: print('Unused native real words prepared', index + 1, len(parents), dict(statistics), flush=True)
    for path, digest in {**inventories, str(annotation_path): annotation_receipt['sha256'], **native_sources}.items():
        if file_hash(path) != digest: raise ValueError('Admitted source changed during additional word preparation')
    if capture(files) != runtime or file_hash(old_data / 'manifest.json') != previous_manifest_sha: raise ValueError('Additional preparation implementation/previous word source changed')
    report = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'source_provenance': runtime,
              'previous_reader_manifest_sha256': previous_manifest_sha, 'previous_reader_sources_sha256': old_manifest['source_files_sha256'],
              'photo_inventories_sha256': inventories, 'native_source_photos_sha256': native_sources,
              'original_annotation_source': annotation_receipt, 'additional_statistics': dict(statistics),
              'new_word_photo_parents': len(parents_with_new_words), 'excluded': dict(exclusions),
              'files_sha256': {'additional.train.jsonl.gz': file_hash(output / 'additional.train.jsonl.gz')},
              'decoded_JSONL_sha256': decoded.hexdigest(), 'protected_rendered_validation_words': len(protected_words),
              'protected_validation_reader_pixel_hashes': len(protected_pixels), 'validation_data_changed': False,
              'new_images_downloaded_or_copied': 0, 'training_started': False, 'goal_achieved': False,
              'crop_protocol': 'Native licensed JPEG; original human bbox; fixed5% horizontal/10% vertical margins; integer floor/ceil/clamp; native side>=8px; one bicubic32x512 resize.',
              'limitations': ['Preparation only; additional learning and its capability/retention tests remain unexecuted.',
                              'Same original5000 training parents; additional annotation coverage is not new independent photo diversity.',
                              'Frozen prior validation populations preserved; not an independent final benchmark.',
                              'Native JPEGs may still be resized publisher copies; native-side threshold is not a certified readability label.',
                              'Original-format mirrored human annotations remain independently unverified against the blocked original endpoint.',
                              'No orientation rectification or new whole-photo policy integration.']}
    atomic_json(output / 'manifest.json', report); return report


if __name__ == '__main__':
    result = prepare(ROOT / 'data/more-native-real-words-v1')
    print(json.dumps({key: value for key, value in result.items() if key not in ['native_source_photos_sha256', 'photo_inventories_sha256', 'source_provenance']}), flush=True)
