"""Recover bounded corpus space while preserving every old model byte exactly."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from slm_lab.checkpoint_xor import archive_checkpoint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'reports/inactive-model-exact-xor-archives-v2.json'
    if output.exists(): raise ValueError('Existing archive batch must be inspected before any continuation')
    protected = set()
    for path in (ROOT / 'checkpoints').glob('**/manifest.json'):
        if not path.parent.name.endswith(('.xor', '.archive')): continue
        manifest = json.loads(path.read_text())
        for key in ['anchor', 'source_checkpoint']:
            if manifest.get(key): protected.add(Path(manifest[key]).resolve())
    pairs = [
        ('code-execution-dpo-v1', 'verified-code-foundation-probe-v1/base'),
        ('verified-code-foundation-probe-v1/candidate', 'verified-code-foundation-probe-v1/base'),
        ('human-feedback-pilot-v1/reward', 'multimodal-pilot-v1/joint-instruction-images'),
        ('human-feedback-pilot-v1/ppo', 'multimodal-pilot-v1/joint-instruction-images'),
        ('clevr-dense-learning-v1/questions', 'clevr-dense-learning-v1/vision'),
        ('exercism-module-probe-v1/candidate', 'exercism-module-probe-v1/base'),
        ('multimodal-pilot-v1/instruction', 'broad-public-pilot-v1'),
        ('broad-public-pilot-v1', 'l40-ddp-readiness'),
    ]
    # A newly used exact anchor must remain full, including later candidates in this list.
    protected.update((ROOT / 'checkpoints' / anchor / 'model.pt').resolve() for _, anchor in pairs)
    before = shutil.disk_usage(ROOT).free
    record = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'archiving verified inactive completed checkpoints',
              'free_bytes_before': before, 'desired_free_GiB': 37, 'archives': [], 'skipped': [],
              'protected_full_anchors': sorted(str(path) for path in protected), 'goal_achieved': False,
              'scope': 'Exact original-file reconstruction checked before removing redundant full files; full anchors must remain.'}
    atomic_json(output, record)
    for target_name, anchor_name in pairs:
        if shutil.disk_usage(ROOT).free >= 37 * 1024**3: break
        target = ROOT / 'checkpoints' / target_name / 'model.pt'
        anchor = ROOT / 'checkpoints' / anchor_name / 'model.pt'
        if target.resolve() in protected or target.stat().st_nlink != 1:
            record['skipped'].append({'checkpoint': str(target), 'reason': 'Required full anchor or shared hard link'})
            atomic_json(output, record); continue
        archive = target.parent / 'model.pt.xor'
        print('Exact inactive-model archive starting', target_name, flush=True)
        result = archive_checkpoint(target, anchor, archive, remove_original=True)
        record['archives'].append({key: value for key, value in result.items() if key != 'segments'})
        record['free_bytes_observed'] = shutil.disk_usage(ROOT).free
        atomic_json(output, record)
        print('Exact inactive-model archive completed', target_name, 'saved bytes', result['original_bytes'] - result['payload_bytes'], flush=True)
    for path in protected:
        if not path.exists(): raise ValueError('Required immutable full archive anchor is missing')
    record.update(status='inactive completed models preserved in exact XOR archives',
                  completed_at=datetime.now(timezone.utc).isoformat(), free_bytes_after=shutil.disk_usage(ROOT).free,
                  reconstructed_original_bytes=sum(item['original_bytes'] for item in record['archives']),
                  saved_payload_bytes=sum(item['original_bytes'] - item['payload_bytes'] for item in record['archives']))
    atomic_json(output, record)


if __name__ == '__main__': main()
