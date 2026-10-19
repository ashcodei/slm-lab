"""Original word-region targets on existing full photos; no oracle crops as inputs."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from PIL import Image

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash
from .training import letterbox_box
from .vqa_pairs import known_photo_aliases

ROOT = Path(__file__).resolve().parents[2]


def projected_target(annotation, dimensions, stored_dimensions, size=768):
    width, height = dimensions
    if min(width, height, *stored_dimensions) <= 0:
        raise ValueError('Invalid original/stored photo dimensions')
    if abs(stored_dimensions[0] / stored_dimensions[1] - width / height) > .04 * width / height:
        raise ValueError('Stored photo aspect differs from original word geometry')
    box = annotation['bbox']
    if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
        raise ValueError('Original word box must have four finite coordinates')
    x, y, w, h = map(float, box)
    if min(w, h) <= 0: raise ValueError('Original word box is empty')
    clipped = [max(0., x), max(0., y), min(float(width), x + w), min(float(height), y + h)]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError('Original word box lies outside its photo')
    normal = [clipped[0] / width, clipped[1] / height, clipped[2] / width, clipped[3] / height]
    padded = letterbox_box(normal, stored_dimensions, size)
    text = annotation.get('utf8_string', '').strip()
    # Preserve unreadable-region geometry as an ignore target, not background.
    return {'original_annotation_id': str(annotation['id']), 'original_bbox': box,
            'normalized_source_box': normal, 'normalized_padded_box': padded,
            'original_points': annotation.get('points'), 'legible': bool(text and text != '.'),
            'clipped_to_image_bounds': clipped != [x, y, x + w, y + h]}


def prepare(output):
    output = Path(output)
    if output.exists(): raise ValueError('Preserve existing localization data; inspect before re-preparation')
    curriculum = ROOT / 'data/alignment-curriculum-v1'
    manifest = json.loads((curriculum / 'manifest.json').read_text())
    cache = ROOT / 'data/image-source-cache-v1'
    originals = {}; sources = {}; initial = {}
    for split, suffix in [('train', 'train'), ('valid', 'val')]:
        path = cache / ('TextOCR_0.1_' + suffix + '.json')
        receipt = json.loads(path.with_suffix('.json.source.json').read_text())
        if file_hash(path) != receipt['sha256']: raise ValueError('Original-format human word annotations changed')
        originals[split] = json.loads(path.read_text()); sources[split] = receipt
        initial[str(path)] = receipt['sha256']
    input_rows = {}; all_groups = {}; all_aliases = {}; all_pixels = {}
    for split in ['train', 'valid']:
        path = curriculum / ('images.' + split + '.jsonl'); digest = file_hash(path)
        if digest != manifest['files_sha256'][path.name]: raise ValueError('Frozen photo curriculum changed')
        initial[str(path)] = digest
        rows = list(jsonl(path))
        all_groups[split] = {row['group_id'] for row in rows}
        all_aliases[split] = {alias for row in rows for alias in known_photo_aliases(row)}
        all_pixels[split] = {value for row in rows for value in [row.get('model_pixel_sha256'), row.get('model_pixel_sha256_448')] if value}
        input_rows[split] = [row for row in rows if row.get('task') == 'region_ocr' and row.get('domain') == 'scene_text']
        if len({row['group_id'] for row in input_rows[split]}) != len(input_rows[split]):
            raise ValueError('Repeated full-photo localization parent')
    output.mkdir(parents=True); statistics = {}; hashes = {}; excluded = []
    for split in ['train', 'valid']:
        source = originals[split]; rows = []; total = 0; legible = 0
        opposing = 'valid' if split == 'train' else 'train'
        for record in input_rows[split]:
            image_id = record['source_image_id']; original = source['imgs'][image_id]
            if original['set'] != ('train' if split == 'train' else 'val'):
                raise ValueError('Human word annotation/image split differs')
            image_path = (curriculum / record['image']).resolve()
            if not image_path.is_relative_to(ROOT) or file_hash(image_path) != record['stored_image_sha256']:
                raise ValueError('Actual admitted full-photo bytes/path changed')
            if (record['group_id'] in all_groups[opposing] or known_photo_aliases(record) & all_aliases[opposing]
                    or record['model_pixel_sha256'] in all_pixels[opposing]
                    or record['model_pixel_sha256_448'] in all_pixels[opposing]):
                raise ValueError('Localization photo overlaps an opposing prepared image split')
            with Image.open(image_path) as image:
                stored_size = image.size; pixels_sha = pixel_hash(image, 768)
            targets = []
            for annotation_id in source['imgToAnns'][image_id]:
                annotation = source['anns'][annotation_id]
                if str(annotation['image_id']) != image_id or str(annotation['id']) != str(annotation_id):
                    raise ValueError('Original human annotation/photo identity mismatch')
                try: target = projected_target(annotation, (original['width'], original['height']), stored_size)
                except ValueError as error:
                    excluded.append({'image_id': image_id, 'original_annotation_id': str(annotation_id),
                                     'split': split, 'reason': str(error)})
                    continue
                targets.append(target)
            if not targets: raise ValueError('Prepared full text photo has no valid original word geometry')
            row = {key: record[key] for key in ['group_id', 'source_image_id', 'source_url', 'author',
                   'author_profile_url', 'license', 'license_url', 'download_sha256', 'stored_image_sha256']}
            row.update(id='text-localization:' + image_id, image=str(image_path), split=split,
                       task='full_photo_word_localization', targets=targets,
                       stored_dimensions=list(stored_size), original_annotation_dimensions=[original['width'], original['height']],
                       model_pixel_sha256_768=pixels_sha, annotation_source_sha256=sources[split]['sha256'],
                       annotation_license='CC-BY-4.0', annotation_origin='Original human TextOCR word regions; no model-generated boxes.',
                       transcript_supplied_to_detector=False)
            rows.append(row); total += len(targets); legible += sum(target['legible'] for target in targets)
            if len(rows) % 500 == 0: print('Original full-photo word geometry prepared', split, len(rows), flush=True)
        path = output / (split + '.jsonl'); path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        hashes[path.name] = file_hash(path)
        statistics[split] = {'full_photos': len(rows), 'word_regions': total, 'legible_regions': legible,
                             'unreadable_ignore_regions': total - legible}
    excluded_path = output / 'excluded-geometry.jsonl'
    excluded_path.write_text(''.join(json.dumps(row) + '\n' for row in excluded)); hashes[excluded_path.name] = file_hash(excluded_path)
    for path, digest in initial.items():
        if file_hash(path) != digest: raise ValueError('Original annotation/curriculum changed during preparation')
    report = {'prepared_at': datetime.now(timezone.utc).isoformat(), 'implementation_sha256': file_hash(__file__),
              'source_files_sha256': initial, 'original_annotation_sources': sources, 'files_sha256': hashes,
              'splits': statistics, 'excluded_geometry': dict(Counter(row['reason'] for row in excluded)),
              'input_size': 768, 'train_validation_parent_overlap': 0, 'new_images_downloaded': 0,
              'trained': False, 'goal_achieved': False,
              'limitations': ['Prepared localization supervision only; no trained detector or native full-photo reading result.',
                              'Existing frozen development photo splits; not independent final broad evaluation.',
                              'Original annotation endpoint was blocked; mirror byte equivalence remains independently unverified as in prior source records.',
                              'Axis-aligned bounding boxes are projected; raw original polygon points are retained, not yet used by a rotated detector.',
                              'Unreadable word regions are ignore targets; transcripts are not input features or generated labels.']}
    atomic_json(output / 'manifest.json', report)
    return report


if __name__ == '__main__': print(json.dumps(prepare(ROOT / 'data/text-localization-v1'), indent=2))
