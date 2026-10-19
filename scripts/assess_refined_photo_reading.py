"""Wait for verified reader refinement, then repeat the original400 full photos."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.refined_photo_evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
DONE = 'native-word refinement trained and assessed; full-photo/native-policy benefit unverified'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--reader-pid', type=int, required=True)
    parser.add_argument('--reader-start-ticks', required=True); args = parser.parse_args()
    reports = ROOT / 'reports/refined-photo-reading-v1'
    if reports.exists(): raise ValueError('Preserve existing refined-photo assessment; inspect before resuming')
    if process_identity(args.reader_pid) != args.reader_start_ticks or b'scripts/train_native_word_refinement.py' not in Path(f'/proc/{args.reader_pid}/cmdline').read_bytes().split(b'\0'):
        raise ValueError('Expected own reader refinement is not verified live')
    baseline_path = ROOT / 'reports/photo-word-reading-v1/reading.json'; baseline_sha = file_hash(baseline_path)
    baseline = json.loads(baseline_path.read_text())
    files = [ROOT / path for path in baseline['source_provenance']['files_sha256']] + [Path(__file__), ROOT / 'src/slm_lab/refined_photo_evaluation.py']
    runtime = capture(files)
    for name, digest in baseline['source_provenance']['files_sha256'].items():
        if runtime['files_sha256'][name] != digest: raise ValueError('Original complete-photo evaluator changed before refined assessment')
    plan_path = ROOT / 'reports/native-word-refinement-v1/plan.json'; plan_sha = file_hash(plan_path)
    data_path = ROOT / 'data/text-localization-v1/manifest.json'; data_sha = file_hash(data_path)
    detector = ROOT / 'checkpoints/word-localizer-v1/model.pt'; detector_sha = file_hash(detector)
    if data_sha != baseline['data_manifest_sha256'] or detector_sha != baseline['detector_sha256']: raise ValueError('Original detector/evaluation input changed')
    reports.mkdir(); torch.set_num_threads(1)
    atomic_json(reports / 'queued-plan.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                'reader_pid': args.reader_pid, 'reader_start_ticks': args.reader_start_ticks, 'reader_plan_sha256': plan_sha,
                'baseline_full_photo_report_sha256': baseline_sha, 'data_manifest_sha256': data_sha,
                'detector_sha256': detector_sha, 'source_provenance': runtime, 'full_photos': 400,
                'input_size': 768, 'device': 'cuda:3', 'additional_training_updates': 0, 'automatic_policy_promotion': False,
                'scope': 'Fixed original detector/photos/word crops and unchanged inference; only refined own reader weights differ.', 'goal_achieved': False})
    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(), 'goal_achieved': False, **values})
    def verify():
        if capture(files) != runtime or file_hash(baseline_path) != baseline_sha or file_hash(plan_path) != plan_sha or file_hash(data_path) != data_sha or file_hash(detector) != detector_sha:
            raise ValueError('Fixed refined-reader assessment implementation/baseline/inputs changed')
    try:
        status('waiting for own reader refinement and full original-crop assessment', reader_pid=args.reader_pid, reader_start_ticks=args.reader_start_ticks)
        while process_identity(args.reader_pid) == args.reader_start_ticks:
            verify(); time.sleep(15)
        verify(); controller = ROOT / 'reports/native-word-refinement-v1'
        if json.loads((controller / 'status.json').read_text())['status'] != DONE: raise ValueError('Refinement ended without its declared completed learning/assessment')
        plan = json.loads(plan_path.read_text()); comparison = json.loads((controller / 'comparison.json').read_text())
        checkpoint = ROOT / 'checkpoints/native-word-refinement-v1/model.pt'; digest = file_hash(checkpoint)
        if digest != comparison['model_sha256'] or plan['own_source_model_sha256'] != baseline['reader_sha256']:
            raise ValueError('Different refined reader/model lineage than original baseline')
        snapshot = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if snapshot['step'] != plan['steps'] or snapshot['plan'] != plan: raise ValueError('Refined whole-photo assessment received a partial/different model')
        del snapshot
        if capture([ROOT / path for path in plan['source_provenance']['files_sha256']]) != plan['source_provenance']:
            raise ValueError('Loaded reader-refinement implementation/runtime changed')
        device = 'cuda:3'; torch.cuda.set_per_process_memory_fraction(.2, 3)
        status('generating refined own-reader words on unchanged complete photos')
        report, results = evaluate(checkpoint, reports / 'reading.json', runtime, files, device=device)
        verify()
        status('matched refined-reader complete-photo assessment finished; native broad capability unverified',
               correct_words=report['totals']['proper']['true_positive'], gained_word_targets=len(results['gained_correct_word_targets']),
               lost_word_targets=len(results['lost_previously_correct_word_targets']))
    except Exception as error:
        status('refined-photo reading assessment failed; inspect actual evidence before resuming', error=str(error)); raise


if __name__ == '__main__': main()
