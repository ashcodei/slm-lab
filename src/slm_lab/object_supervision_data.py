"""Object-label supervision from bounded admitted photos and original human boxes."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageOps
from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash
from .object_feature_probe import crop_image
from .object_qa import clear, extract_subset, overlap, photos_and_rotations

ROOT = Path(__file__).resolve().parents[2]


def acceptable(box, others):
    return clear(box) and not any(other['label_id'] != box['label_id'] and overlap(box['box'], other['box']) > .95 for other in others)


def prepare(output, additional_photos=30000, cap_per_class=500, storage_gib=2):
    output = Path(output)
    if output.exists(): raise ValueError('Use a fresh supervised-object data directory')
    if shutil.disk_usage(ROOT).free < 40 * 1024**3: raise ValueError('Keep40GiB alongside active main training')
    output.mkdir(); (output / 'images').mkdir()
    source = ROOT / 'data/object-qa-v1'; probe = ROOT / 'data/object-feature-probe-v1'; cache = ROOT / 'data/image-source-cache-v1'
    source_manifest = json.loads((source / 'manifest.json').read_text()); probe_manifest = json.loads((probe / 'manifest.json').read_text())
    classes = probe_manifest['classes']; class_set = set(classes)
    for folder, manifest in [(source, source_manifest), (probe, probe_manifest)]:
        for name, digest in manifest['files_sha256'].items():
            if file_hash(folder / name) != digest: raise ValueError('Frozen original human-box/probe records changed')
    bbox_path = cache / 'oidv6-train-bbox.csv'; rotation_path = cache / 'openimages-train.csv'
    if file_hash(bbox_path) != source_manifest['sources']['train']['sha256'] or file_hash(rotation_path) != source_manifest['sources']['rotation_train']['sha256']:
        raise ValueError('Pinned publisher box/rotation data changed')
    photos, _ = photos_and_rotations(ROOT / 'data/alignment-curriculum-v1', cache, 'train')
    for row in photos.values(): row['image'] = str((ROOT / 'data/alignment-curriculum-v1' / row['image']).resolve())
    expanded = ROOT / 'data/expanded-everyday-photos-v1'
    snapshot = list(itertools.islice(jsonl(expanded / 'collected.jsonl'), additional_photos))
    if len(snapshot) != additional_photos: raise ValueError('Additional accepted-photo prefix not complete')
    (output / 'additional-photo-prefix.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in snapshot))
    pending = {row['source_image_id']: {**row, 'image': str((expanded / row['image']).resolve())} for row in snapshot}
    excluded = Counter()
    with rotation_path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = row['ImageID']
            if key not in pending: continue
            angle = float(row.get('Rotation') or 0)
            if not math.isfinite(angle) or angle not in {0, 90, 180, 270}:
                excluded['unsupported source rotation'] += 1; del pending[key]; continue
            pending[key]['applied_source_rotation'] = int(angle)
            pending[key]['source_rotation_field'] = row.get('Rotation')
    if any('applied_source_rotation' not in row for row in pending.values()): raise ValueError('Source orientation missing')
    held = list(jsonl(probe / 'valid.jsonl')); protected_groups = {row['group_id'] for row in held}
    for name in ['images.valid.jsonl']:
        protected_groups.update(row['group_id'] for row in jsonl(ROOT / 'data/alignment-curriculum-v1' / name))
    if any(row['group_id'] in protected_groups for row in pending.values()): raise ValueError('Additional training parents overlap protected validation')
    if set(pending) & set(photos): raise ValueError('Additional photos repeat prior training parents')
    for key, row in pending.items():
        if file_hash(row['image']) != row['stored_image_sha256']: raise ValueError('Additional stored training photo changed')
    with (cache / 'oidv7-box-classes.csv').open(newline='') as handle: labels = dict(csv.reader(handle))
    grouped, rejected = extract_subset(bbox_path, pending, labels, lambda count: print('scanned additional boxed photos', count, flush=True))
    for row in jsonl(source / 'annotations.train.jsonl'):
        if row['image_id'] in grouped: raise ValueError('Repeated original/additional object parent')
        grouped[row['image_id']] = row['boxes']
    photos.update(pending)
    candidates = defaultdict(list)
    for key, boxes in grouped.items():
        if key not in photos or photos[key]['group_id'] in protected_groups: continue
        for box in boxes:
            if box['label'] not in class_set or not acceptable(box, boxes): continue
            candidate = {**photos[key], 'normalized_box': box['box'], 'response': box['label'], 'box_evidence': [box],
                         'id': 'supervised-object-' + hashlib.sha256((key + ':' + str(box['row_number'])).encode()).hexdigest(),
                         'annotation_source': source_manifest['sources']['train'], 'annotation_license': 'CC-BY-4.0'}
            candidates[box['label']].append(candidate)
    selected = []
    for name in classes:
        seen = set()
        for row in sorted(candidates[name], key=lambda r: hashlib.sha256(r['id'].encode()).digest()):
            if row['group_id'] in seen: continue
            selected.append(row); seen.add(row['group_id'])
            if len(seen) >= cap_per_class: break
    protected_pixels = {size: {row['crop_input_sha256'][str(size)] for row in held} for size in [224, 448]}
    rows = {'train': [], 'valid': []}; seen_pixels = {}; conflicting_pixels = set(); stored = 0; limit = int(storage_gib * 1024**3)
    # Preserve the original221 held object crops exactly, including both pixel resolutions.
    for split, population in [('valid', held), ('train', selected)]:
        for index, row in enumerate(population):
            image = crop_image(row)
            if min(image.size) < 24:
                if split == 'valid': raise ValueError('Original held crop unexpectedly too small')
                excluded['small training crop'] += 1; continue
            if split == 'train': image = ImageOps.pad(image, (224, 224), Image.Resampling.BICUBIC, color=(128, 128, 128))
            hashes = {size: pixel_hash(image, size) for size in [224, 448]}
            if split == 'valid' and any(hashes[size] != row['crop_input_sha256'][str(size)] for size in hashes):
                raise ValueError('Held object pixels changed')
            if split == 'train' and any(hashes[size] in protected_pixels[size] for size in hashes):
                excluded['protected held input pixels'] += 1; continue
            if split == 'train' and hashes[224] in seen_pixels:
                if seen_pixels[hashes[224]] != row['response']: conflicting_pixels.add(hashes[224])
                excluded['duplicate training crop pixels'] += 1; continue
            path = output / 'images' / (split + '-' + hashlib.sha256(row['id'].encode()).hexdigest() + '.png')
            image.save(path, format='PNG', compress_level=1)
            size = path.stat().st_size
            if stored + size > limit:
                path.unlink(); raise ValueError('Object crop storage budget exhausted; inspect before changing selection')
            stored += size
            rows[split].append({**row, 'parent_image': row['image'], 'parent_stored_image_sha256': row['stored_image_sha256'],
                                'image': str(path.resolve()), 'stored_image_sha256': file_hash(path),
                                'label': classes.index(row['response']), 'model_pixel_sha256': hashes[224],
                                'model_pixel_sha256_448': hashes[448], 'split': split})
            if split == 'train': seen_pixels[hashes[224]] = row['response']
            if (index + 1) % 1000 == 0: print('stored supervised object crops', split, len(rows[split]), flush=True)
    accepted = []
    for row in rows['train']:
        if row['model_pixel_sha256'] in conflicting_pixels:
            path = Path(row['image']); stored -= path.stat().st_size; path.unlink()
            excluded['conflicting class for identical training pixels'] += 1
        else: accepted.append(row)
    rows['train'] = accepted
    counts = Counter(row['response'] for row in rows['train'])
    if len(rows['valid']) != len(held) or any(counts[name] < 20 for name in classes): raise ValueError('Lost original held cases or insufficient training support')
    if {r['group_id'] for r in rows['train']} & {r['group_id'] for r in rows['valid']}: raise ValueError('Object parent group overlap')
    for split, values in rows.items():
        (output / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in values))
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'classes': classes,
                'configuration': {'implementation_sha256': file_hash(__file__), 'additional_photos': additional_photos,
                                  'cap_per_class': cap_per_class, 'image_size': 224, 'storage_gib': storage_gib,
                                  'source_manifest_sha256': file_hash(source / 'manifest.json'), 'probe_manifest_sha256': file_hash(probe / 'manifest.json'),
                                  'source_boxes_sha256': file_hash(bbox_path), 'source_orientation_sha256': file_hash(rotation_path),
                                  'additional_photo_prefix_sha256': file_hash(output / 'additional-photo-prefix.jsonl')},
                'splits': {split: {'examples': len(values), 'parent_images': len({r['group_id'] for r in values}),
                                  'by_class': dict(Counter(r['response'] for r in values))} for split, values in rows.items()},
                'stored_bytes': stored, 'excluded': dict(excluded), 'annotation_exclusions': rejected,
                'all_original_validation_cases_preserved': True, 'train_validation_parent_overlap': 0,
                'files_sha256': {name: file_hash(output / name) for name in ['train.jsonl', 'valid.jsonl', 'additional-photo-prefix.jsonl']},
                'goal_achieved': False, 'external_model_weights_used': False,
                'limitations': ['Original object boxes are supplied for training/evaluation: oracle cropped classification, not detection or native multimodal answers.',
                                'Same61 train-supported classes and221 reused development crops; rare categories and independent final generalization are unverified.',
                                'Right-angle source rotations only; hierarchy/annotation errors and semantic near-duplicates remain possible.',
                                'Training crops are stored at224px; held original crop pixels remain exact at224/448.',
                                'Preparation is not trained vision or measured generic image capability.']}
    atomic_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    parser.add_argument('--additional-photos', type=int, default=30000); parser.add_argument('--cap-per-class', type=int, default=500)
    args = parser.parse_args(); print(json.dumps(prepare(args.output, args.additional_photos, args.cap_per_class), indent=2), flush=True)


if __name__ == '__main__': main()
