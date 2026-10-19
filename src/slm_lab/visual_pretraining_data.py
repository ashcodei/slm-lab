"""Freeze original full photos, scenes, pages and completed Commons for visual learning."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from PIL import Image

from .corpus import file_hash
from .diverse_images import atomic_json, jsonl, pixel_hash

FULL_IMAGE_TASKS = {'caption', 'region_ocr', 'document_region_ocr'}
PRIORITY = {'caption': 0, 'region_ocr': 1, 'document_region_ocr': 2}


def prepare(base, commons, output):
    base, commons, output = map(Path, [base, commons, output])
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((base / 'manifest.json').read_text())
    completed = json.loads((commons / 'manifest.json').read_text())
    names = {base: ['images.train.jsonl', 'images.valid.jsonl'], commons: ['train.jsonl', 'valid.jsonl']}
    inputs = []
    for root, paths in names.items():
        metadata = original if root == base else completed
        for name in ['manifest.json', *paths]:
            digest = file_hash(root / name)
            if name != 'manifest.json' and digest != metadata['files_sha256'][name]:
                raise ValueError('Frozen visual source changed: ' + str(root / name))
            inputs.append({'path': str((root / name).resolve()), 'sha256': digest})
    configuration = {'version': 1, 'sources': inputs, 'image_resolutions': [224, 448],
                     'full_image_tasks': sorted(FULL_IMAGE_TASKS), 'one_image_per_parent': True,
                     'implementation_sha256': file_hash(__file__)}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['configuration'] != configuration:
            raise ValueError('Use a new visual pretraining dataset directory')
        for name, digest in manifest['files_sha256'].items():
            if file_hash(output / name) != digest:
                raise ValueError('Prepared visual pretraining records changed')
        return manifest
    candidates = {'train': [], 'valid': []}
    base_rows = {}
    for split in candidates:
        base_rows[split] = list(jsonl(base / ('images.' + split + '.jsonl')))
        for root, rows in [(base, base_rows[split]), (commons, list(jsonl(commons / (split + '.jsonl'))))]:
            for row in rows:
                if root == commons:
                    row = {**row, 'task': row.get('task', 'caption')}
                if row.get('curriculum_source') == 'fashion' or row.get('task') not in FULL_IMAGE_TASKS:
                    continue
                candidates[split].append({**row, 'image': str((root / row['image']).resolve()),
                                          'visual_source': row.get('curriculum_source', 'commons'),
                                          'visual_source_root': str(root.resolve()),
                                          'visual_objective': 'image-only masked patch reconstruction'})
    old_train_pixels = {size: {row['model_pixel_sha256' if size == 224 else 'model_pixel_sha256_448']
                               for row in base_rows['train']} for size in [224, 448]}
    protected_groups = {row['group_id'] for row in base_rows['valid']}
    protected_pixels = {size: {row['model_pixel_sha256' if size == 224 else 'model_pixel_sha256_448']
                               for row in base_rows['valid']} for size in [224, 448]}
    exclusions = []
    accepted = {}
    for split in ['valid', 'train']:
        rows = []
        groups = set()
        seen_pixels = {size: set() for size in [224, 448]}
        ordered = sorted(candidates[split], key=lambda row: (PRIORITY[row['task']], row['group_id'], row['id']))
        for row in ordered:
            if row['group_id'] in groups:
                continue
            path = Path(row['image'])
            if file_hash(path) != row['stored_image_sha256']:
                raise ValueError('Stored source image changed: ' + str(path))
            with Image.open(path) as image:
                hashes = {size: pixel_hash(image, size) for size in [224, 448]}
            reason = None
            if split == 'valid' and row['visual_source_root'] == str(commons.resolve()):
                if any(hashes[size] in old_train_pixels[size] for size in hashes):
                    reason = 'additional validation image was used in earlier training'
            if split == 'train':
                if row['group_id'] in protected_groups:
                    reason = 'protected held-out parent'
                elif any(hashes[size] in protected_pixels[size] for size in hashes):
                    reason = 'exact held-out input pixels'
            if any(hashes[size] in seen_pixels[size] for size in hashes):
                reason = 'duplicate visual input within split'
            if reason:
                exclusions.append({'id': row['id'], 'group_id': row['group_id'], 'split': split, 'reason': reason})
                continue
            groups.add(row['group_id'])
            for size in hashes:
                seen_pixels[size].add(hashes[size])
            rows.append({**row, 'model_pixel_sha256': hashes[224], 'model_pixel_sha256_448': hashes[448]})
        accepted[split] = rows
        if split == 'valid':
            protected_groups.update(row['group_id'] for row in rows)
            for size in protected_pixels:
                protected_pixels[size].update(seen_pixels[size])
        print('prepared full visual inputs', split, len(rows), flush=True)
    overlap = {size: len({row['model_pixel_sha256' if size == 224 else 'model_pixel_sha256_448'] for row in accepted['train']}
                         & {row['model_pixel_sha256' if size == 224 else 'model_pixel_sha256_448'] for row in accepted['valid']})
               for size in [224, 448]}
    group_overlap = len({row['group_id'] for row in accepted['train']} & {row['group_id'] for row in accepted['valid']})
    if group_overlap or any(overlap.values()) or not all(accepted.values()):
        raise ValueError('Visual pretraining splits are empty or overlap')
    for split, rows in accepted.items():
        (output / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    (output / 'excluded.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in exclusions))
    # Retain the full completed source metadata separately from the old live curriculum.
    (output / 'commons-source-manifest.json').write_text(json.dumps(completed, indent=2) + '\n')
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'configuration': configuration,
                'splits': {split: {'images': len(rows), 'by_source': dict(Counter(row['visual_source'] for row in rows)),
                                   'by_original_task': dict(Counter(row['task'] for row in rows))}
                           for split, rows in accepted.items()},
                'completed_commons_images': completed['images'], 'parent_group_overlap': group_overlap,
                'exact_input_pixel_overlap': {str(size): value for size, value in overlap.items()},
                'excluded': dict(Counter(row['reason'] for row in exclusions)),
                'files_sha256': {name: file_hash(output / name) for name in
                                 ['train.jsonl', 'valid.jsonl', 'excluded.jsonl', 'commons-source-manifest.json']},
                'external_model_weights_used': False, 'fashion_training_images': 0,
                'limitations': ['One full image per parent; caption/OCR labels are retained as provenance but not used by the reconstruction objective.',
                                'Existing development splits, not independent final visual benchmarks.',
                                'Exact canonical/pixel protection does not certify absence of perceptual near-duplicates.',
                                'Masked reconstruction loss is not evidence of semantic understanding, OCR or native language-model skill.']}
    atomic_json(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='data/alignment-curriculum-v1')
    parser.add_argument('--commons', default='data/commons-images-v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.base, args.commons, args.output), indent=2))


if __name__ == '__main__':
    main()
