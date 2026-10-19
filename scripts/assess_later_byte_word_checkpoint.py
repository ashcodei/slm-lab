"""Freeze a later actual word-reader checkpoint and assess the unchanged256 cases."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import torch

from slm_lab.byte_word_reader import ByteWordReader, evaluate, load_rows, pixels
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture, pixels_sha256
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.ocr_evaluation import transcription_scores

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); reports = ROOT / 'reports/byte-word-reader-v1'
    output = reports / 'later-reading.json'; snapshot = ROOT / 'checkpoints/byte-word-reader-v1/later-assessment.pt'
    if output.exists() or snapshot.exists(): raise ValueError('Preserve the existing later actual assessment')
    pid = 310980; ticks = '11314830'
    if process_identity(pid) != ticks or b'scripts/train_byte_word_reader.py' not in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'):
        raise ValueError('Later snapshot requires the exact verified live word-reader trainer')
    files = [Path(__file__)] + [ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['byte_word_reader', 'ocr_evaluation', 'rendered_text', 'corpus', 'diverse_images', 'evaluation_provenance']]
    runtime = capture(files); plan_path = reports / 'plan.json'; plan_sha = file_hash(plan_path); plan = json.loads(plan_path.read_text())
    data = ROOT / 'data/byte-word-reader-v1'; data_sha = file_hash(data / 'manifest.json')
    with (ROOT / 'checkpoints/byte-word-reader-v1/latest.pt').open('rb') as source:
        before = os.fstat(source.fileno()); state = torch.load(source, map_location='cpu', weights_only=True); after = os.fstat(source.fileno())
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino): raise ValueError('Published reader snapshot inode changed')
        minimum = 6 * ((plan['train_examples'] + plan['batch_size'] - 1) // plan['batch_size'])
        if state['step'] < minimum or state['plan'] != plan: raise ValueError('Later assessment requires at least six full passes from the declared own experiment')
        source.seek(0)
        with snapshot.open('xb') as target: shutil.copyfileobj(source, target, length=1024 * 1024)
    model = ByteWordReader(); model.load_state_dict(state['model']); step = state['step']; del state
    rows, _ = load_rows(data); device = 'cuda:2'; model = model.to(device)
    report = evaluate(model, rows['valid'], output, device)
    baseline = json.loads((reports / 'before.json').read_text()); by_id = {row['id']: row for row in report['results']}
    original = {row['id']: row for row in rows['valid']}
    key = lambda row: (row['id'], row['group_id'], row['expected'], row['wrong_image_id'], row['stored_image_sha256'])
    if len(by_id) != 256 or [key(row) for row in baseline['results']] != [key(row) for row in report['results']]: raise ValueError('Later assessment changed actual cases/targets/counterparts')
    if len({row['blank_generated'] for row in report['results']}) != 1: raise ValueError('Identical blank crops generated different words')
    correct = []; histogram = Counter()
    for row in report['results']:
        if row['wrong_generated'] != by_id[row['wrong_image_id']]['proper_generated']: raise ValueError('Actual counterpart transcript differs from its own proper generation')
        row['input_pixels_sha256'] = pixels_sha256(pixels(original[row['id']]))
        row['wrong_pixels_sha256'] = pixels_sha256(pixels(original[row['wrong_image_id']]))
        for control in ['proper', 'blank', 'wrong']:
            if row[control + '_scores'] != transcription_scores(row['expected'], row[control + '_generated']): raise ValueError('Later stored transcription score differs from actual transcript')
        if row['proper_scores']['exact_match']:
            correct.append(row); histogram[row['expected']] += 1
    if capture(files) != runtime or file_hash(plan_path) != plan_sha or file_hash(data / 'manifest.json') != data_sha: raise ValueError('Later reader assessment source/runtime/data changed')
    report.update(snapshot_step=step, snapshot_sha256=file_hash(snapshot), plan_sha256=plan_sha, data_manifest_sha256=data_sha,
                  implementation_runtime=runtime, captured_during_live_training=True, policy_promoted=False)
    atomic_json(output, report)
    comparison = {'snapshot_step': step, 'snapshot_sha256': file_hash(snapshot), 'before': baseline['overall'], 'later': report['overall'],
                  'by_family': report['by_family'], 'correct_transcript_histogram': dict(histogram.most_common()),
                  'actual_case_pixel_target_counterpart_matching_verified': True, 'completed_final_training': False,
                  'scope': 'A later unchanged256-case development assessment while final finite word training continues; no whole-photo/native policy evidence.', 'goal_achieved': False}
    atomic_json(reports / 'later-comparison.json', comparison)
    atomic_json(reports / 'later-correct-cases.json', {'cases': correct, 'goal_achieved': False})
    print(json.dumps(comparison), flush=True)


if __name__ == '__main__': main()
