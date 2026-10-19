"""Freeze one published reader snapshot and assess actual word generation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import torch

from slm_lab.byte_word_reader import ByteWordReader, evaluate, load_rows, pixels
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture, pixels_sha256
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.ocr_evaluation import transcription_scores

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    reports = ROOT / 'reports/byte-word-reader-v1'
    status_path = reports / 'intermediate-assessment-status.json'
    output = reports / 'intermediate-reading.json'
    if output.exists() or status_path.exists(): raise ValueError('Preserve existing intermediate assessment; inspect before resuming')
    pid = 310980; identity = '11314830'
    if process_identity(pid) != identity or b'scripts/train_byte_word_reader.py' not in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'):
        raise ValueError('Expected byte-word trainer is not verified live')
    files = [Path(__file__), ROOT / 'scripts/train_byte_word_reader.py', ROOT / 'src/slm_lab/byte_word_reader.py',
             ROOT / 'src/slm_lab/ocr_evaluation.py', ROOT / 'src/slm_lab/rendered_text.py',
             ROOT / 'src/slm_lab/corpus.py', ROOT / 'src/slm_lab/diverse_images.py']
    runtime = capture(files)
    plan_path = reports / 'plan.json'; plan_sha = file_hash(plan_path); plan = json.loads(plan_path.read_text())
    data = ROOT / 'data/byte-word-reader-v1'; manifest_sha = file_hash(data / 'manifest.json')
    source = ROOT / 'checkpoints/byte-word-reader-v1/latest.pt'
    minimum = (plan['train_examples'] + plan['batch_size'] - 1) // plan['batch_size']

    def status(message, **values):
        atomic_json(status_path, {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                                 'goal_achieved': False, **values})
    def verify():
        if capture(files) != runtime or file_hash(plan_path) != plan_sha or file_hash(data / 'manifest.json') != manifest_sha:
            raise ValueError('Loaded reader assessment/source plan changed')

    try:
        status('waiting for published snapshot after at least one full word-training pass', minimum_step=minimum,
               trainer_pid=pid, trainer_start_ticks=identity)
        while True:
            verify()
            if source.exists():
                # Atomic writer replacement cannot alter an already-open inode.
                with source.open('rb') as stream:
                    before = os.fstat(stream.fileno())
                    state = torch.load(stream, map_location='cpu', weights_only=True)
                    after = os.fstat(stream.fileno())
                    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                        raise ValueError('Published snapshot inode changed during read')
                    if state['step'] >= minimum:
                        if state['plan'] != plan: raise ValueError('Reader snapshot belongs to another declared experiment')
                        stream.seek(0); snapshot = ROOT / 'checkpoints/byte-word-reader-v1/intermediate-assessment.pt'
                        if snapshot.exists(): raise ValueError('Existing frozen reader snapshot must be inspected')
                        with snapshot.open('xb') as target: shutil.copyfileobj(stream, target, length=1024 * 1024)
                        break
            if process_identity(pid) != identity:
                raise ValueError('Reader ended before the requested snapshot; inspect terminal evidence')
            time.sleep(15)
        digest = file_hash(snapshot); model = ByteWordReader()
        model.load_state_dict(state['model']); step = state['step']; del state
        status('assessing actual intermediate word transcripts', snapshot_step=step, snapshot_sha256=digest)
        rows, manifest = load_rows(data)
        device = 'cuda:2'; model = model.to(device)
        result = evaluate(model, rows['valid'], output, device)
        selected = {row['id']: row for row in rows['valid']}; by_id = {row['id']: row for row in result['results']}
        if len(by_id) != 256 or len({row['blank_generated'] for row in result['results']}) != 1:
            raise ValueError('Incomplete word cases or inconsistent identical-blank generation')
        for row in result['results']:
            other = by_id[row['wrong_image_id']]
            if row['wrong_generated'] != other['proper_generated']:
                raise ValueError('Same word-crop input produced different actual/control transcripts')
            row['input_pixels_sha256'] = pixels_sha256(pixels(selected[row['id']]))
            row['wrong_pixels_sha256'] = pixels_sha256(pixels(selected[row['wrong_image_id']]))
            for control in ['proper', 'blank', 'wrong']:
                if row[control + '_scores'] != transcription_scores(row['expected'], row[control + '_generated']):
                    raise ValueError('Stored word score differs from actual generated transcript')
        baseline = json.loads((reports / 'before.json').read_text())
        key = lambda row: (row['id'], row['group_id'], row['expected'], row['wrong_image_id'], row['stored_image_sha256'])
        if [key(row) for row in baseline['results']] != [key(row) for row in result['results']]:
            raise ValueError('Unmatched original word cases/targets/counterparts')
        verify()
        result.update(snapshot_sha256=digest, snapshot_step=step, plan_sha256=plan_sha,
                      data_manifest_sha256=manifest_sha, assessment_implementation_runtime=runtime,
                      captured_during_live_training=True, policy_promoted=False)
        atomic_json(output, result)
        atomic_json(reports / 'intermediate-comparison.json', {'snapshot_step': step, 'snapshot_sha256': digest,
                   'before': baseline['overall'], 'intermediate': result['overall'],
                   'by_family_intermediate': result['by_family'], 'case_and_control_contract_verified': True,
                   'scope': 'One development checkpoint; full finite training/final native integration remain incomplete.',
                   'goal_achieved': False})
        status('intermediate actual word generation assessed; final/native reading unverified', snapshot_step=step)
    except Exception as error:
        status('intermediate reader assessment failed; inspect without restarting live work', error=str(error)); raise


if __name__ == '__main__': main()
