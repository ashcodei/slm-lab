"""Require the completed original input audit before queued real-reading training."""
import json
from pathlib import Path

from .corpus import file_hash
from .diverse_images import jsonl
from .evaluation_provenance import capture


def admit(root):
    root = Path(root).resolve()
    data = root / 'data/textvqa-human-reading-v1'
    manifest_path = data / 'manifest.json'
    audit_path = root / 'reports/textvqa-original-input-audit-v1.json'
    manifest = json.loads(manifest_path.read_text()); audit = json.loads(audit_path.read_text())
    if (audit.get('passed') is not True or audit['manifest_sha256'] != file_hash(manifest_path)
            or audit['preparation_source_sha256'] != file_hash(root / 'src/slm_lab/textvqa_data.py')
            or audit['audit_source_sha256'] != file_hash(root / 'src/slm_lab/textvqa_input_audit.py')
            or audit['files_sha256'] != manifest['files_sha256']
            or audit['tokenizer_sha256'] != manifest['tokenizer_sha256']
            or audit['validation_group_overlap'] != 0 or audit['actual_model_pixel_sizes_rechecked'] != [224, 448]):
        raise ValueError('Completed real-reading source audit differs from current inputs/implementation')
    files = [root / name for name in audit['implementation_runtime']['files_sha256']]
    if capture(files) != audit['implementation_runtime']:
        raise ValueError('Original reading audit implementation/runtime changed')
    for path, digest in audit['original_source_and_inventory_sha256'].items():
        source = Path(path).resolve()
        if not source.is_relative_to(root) or file_hash(source) != digest:
            raise ValueError('Audited original reading source/protected image inventory changed')
    verified = {}; rows = {}
    for split in ['train', 'valid']:
        path = data / (split + '.jsonl')
        if file_hash(path) != manifest['files_sha256'][path.name]:
            raise ValueError('Audited original reading records changed')
        rows[split] = list(jsonl(path))
        counts = {'examples': len(rows[split]), 'original_image_groups': len({row['group_id'] for row in rows[split]}),
                  'target_tokens': sum(row['target_tokens'] for row in rows[split])}
        if (counts != audit['splits'][split]
                or any(counts[key] != manifest['splits'][split][key] for key in counts)
                or len({row['id'] for row in rows[split]}) != len(rows[split])):
            raise ValueError('Audited real-reading target/count/identity receipt differs')
        for row in rows[split]:
            image = Path(row['image']).resolve()
            if not image.is_relative_to(root): raise ValueError('Admitted original reading photo escapes project')
            if image not in verified: verified[image] = file_hash(image)
            if verified[image] != row['stored_image_sha256']:
                raise ValueError('Previously audited real-reading photo bytes changed')
    if {row['group_id'] for row in rows['train']} & {row['group_id'] for row in rows['valid']}:
        raise ValueError('Original real-reading training/validation photos overlap')
    receipt = {'passed': True, 'audit_sha256': file_hash(audit_path), 'manifest_sha256': file_hash(manifest_path),
               'admission_source_sha256': file_hash(__file__), 'actual_photo_bytes_rehashed': len(verified),
               'splits': audit['splits'], 'goal_achieved': False,
               'scope': 'Completed original question/full-photo/pixel/prompt/label audit required; current source inventories and photo bytes rechecked before training.'}
    return rows['train'], receipt
