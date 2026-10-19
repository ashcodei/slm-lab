"""After exact reader completion, infer every prepared validation word crop."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

import torch

from slm_lab.byte_word_reader import ByteWordReader, load_rows
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.full_byte_reading_evaluation import evaluate
from slm_lab.masked_visual_pretraining import process_identity

ROOT = Path(__file__).resolve().parents[1]
DONE = 'scratch byte-word component trained and assessed; native full-photo reading unverified'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--reader-pid', type=int, required=True)
    parser.add_argument('--reader-start-ticks', required=True); args = parser.parse_args()
    reports = ROOT / 'reports/full-byte-reading-v1'
    if reports.exists(): raise ValueError('Preserve existing full validation assessment; inspect before resuming')
    if process_identity(args.reader_pid) != args.reader_start_ticks or b'scripts/train_byte_word_reader.py' not in Path(f'/proc/{args.reader_pid}/cmdline').read_bytes().split(b'\0'):
        raise ValueError('Expected word reader predecessor is not verified live')
    torch.set_num_threads(1)
    files = [Path(__file__), ROOT / 'scripts/train_byte_word_reader.py'] + [ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['full_byte_reading_evaluation', 'byte_word_reader', 'ocr_evaluation', 'rendered_text', 'corpus', 'diverse_images', 'evaluation_provenance']]
    runtime = capture(files); data = ROOT / 'data/byte-word-reader-v1'; data_sha = file_hash(data / 'manifest.json')
    plan_path = ROOT / 'reports/byte-word-reader-v1/plan.json'; plan_sha = file_hash(plan_path); plan = json.loads(plan_path.read_text()); reports.mkdir()
    atomic_json(reports / 'queued-plan.json', {'created_at': datetime.now(timezone.utc).isoformat(), 'reader_pid': args.reader_pid,
                'reader_start_ticks': args.reader_start_ticks, 'validation_examples_available': plan['validation_examples_available'],
                'proper_batch_size': 32, 'device': 'cuda:2', 'data_manifest_sha256': data_sha, 'reader_plan_sha256': plan_sha,
                'source_provenance': runtime, 'new_training_updates': 0, 'automatic_policy_promotion': False, 'goal_achieved': False})
    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **values})
    def verify():
        if capture(files) != runtime or file_hash(data / 'manifest.json') != data_sha or file_hash(plan_path) != plan_sha:
            raise ValueError('Full word-reading implementation/runtime/data/plan changed')
    try:
        status('waiting for verified own byte reader training and final256-case assessment', reader_pid=args.reader_pid, reader_start_ticks=args.reader_start_ticks)
        while process_identity(args.reader_pid) == args.reader_start_ticks:
            verify(); time.sleep(15)
        verify()
        reader_reports = ROOT / 'reports/byte-word-reader-v1'; comparison = json.loads((reader_reports / 'comparison.json').read_text())
        if json.loads((reader_reports / 'status.json').read_text())['status'] != DONE: raise ValueError('Reader ended without declared completed training/assessment')
        checkpoint = ROOT / 'checkpoints/byte-word-reader-v1/model.pt'; checkpoint_sha = file_hash(checkpoint)
        if checkpoint_sha != comparison['model_sha256']: raise ValueError('Completed reader model bytes changed')
        snapshot = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if snapshot['plan'] != plan or snapshot['step'] != plan['steps']: raise ValueError('Full reading assessment received a partial or different experiment')
        original_files = [ROOT / value for value in plan['source_provenance']['files_sha256']]
        if capture(original_files) != plan['source_provenance']: raise ValueError('Completed reader training implementation/runtime changed')
        rows, manifest = load_rows(data)
        if len(rows['valid']) != plan['validation_examples_available']: raise ValueError('Complete original validation word scope changed')
        device = 'cuda:2'; torch.cuda.set_per_process_memory_fraction(.15, 2)
        model = ByteWordReader().to(device); model.load_state_dict(snapshot['model']); del snapshot
        status('inferring every original held-out word crop', validation_examples=len(rows['valid']))
        report = evaluate(model, rows['valid'], reports / 'reading.json', device=device, batch_size=32)
        previous_path = reader_reports / 'after.json'; previous_sha = file_hash(previous_path); previous = json.loads(previous_path.read_text())
        full = {row['id']: row for row in report['results']}; differences = []
        for row in previous['results']:
            actual = full[row['id']]
            if any(row[key] != actual[key] for key in ['id', 'group_id', 'expected', 'stored_image_sha256']): raise ValueError('Previous256 reading identities changed in complete validation')
            if row['proper_generated'] != actual['proper_generated']:
                differences.append({'id': row['id'], 'earlier_single_crop': row['proper_generated'], 'new_batched_crop': actual['proper_generated'], 'expected': row['expected']})
        if len(previous['results']) != 256: raise ValueError('Original256-case assessment coverage changed')
        verify()
        if file_hash(checkpoint) != checkpoint_sha or file_hash(previous_path) != previous_sha: raise ValueError('Completed reader/reference report changed during full assessment')
        for path, digest in manifest['source_files_sha256'].items():
            if file_hash(path) != digest: raise ValueError('Original reader crop source inventory changed')
        report.update(checkpoint_sha256=checkpoint_sha, checkpoint_updates=plan['steps'], data_manifest_sha256=data_sha,
                      source_provenance=runtime, matched_previous256_cases=256, previous256_report_sha256=previous_sha,
                      proper_transcript_batch_rounding_differences=differences, all_available_validation_cases_inferred=True,
                      automatic_policy_promotion=False)
        atomic_json(reports / 'reading.json', report)
        atomic_json(reports / 'summary.json', {key: value for key, value in report.items() if key != 'results'})
        status('all prepared word crops assessed; broader independent/native capability unverified', validation_examples=len(rows['valid']))
        print(json.dumps({'overall': report['overall'], 'by_family': report['by_family'], 'previous256_batch_differences': len(differences)}), flush=True)
    except Exception as error:
        status('full validation word reading assessment failed; inspect actual evidence before resuming', error=str(error)); raise


if __name__ == '__main__': main()
