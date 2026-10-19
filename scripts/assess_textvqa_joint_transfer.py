"""Measure real-photo reading transfer after the verified live scene/QA experiment."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.textvqa_evaluation import compare_reports

ROOT = Path(__file__).resolve().parents[1]
PID = 295624
IDENTITY = '10604208'


def main():
    reports = ROOT / 'reports/textvqa-joint-transfer-v1'
    if reports.exists(): raise ValueError('Preserve existing transfer assessment; inspect before resuming')
    if process_identity(PID) != IDENTITY or b'scripts/train_clevr_joint_rehearsal.py' not in Path(f'/proc/{PID}/cmdline').read_bytes().split(b'\0'):
        raise ValueError('Expected native joint-rehearsal controller is not verified live')
    controller = ROOT / 'reports/clevr-joint-rehearsal-v1'
    controller_plan = file_hash(controller / 'plan.json')
    dataset = ROOT / 'data/textvqa-human-reading-v1'
    data_manifest = json.loads((dataset / 'manifest.json').read_text())
    data_hashes = {str(dataset / 'manifest.json'): file_hash(dataset / 'manifest.json')}
    for name, digest in data_manifest['files_sha256'].items():
        if file_hash(dataset / name) != digest: raise ValueError('Finalized real-reading records changed')
        data_hashes[str(dataset / name)] = digest
    files = [ROOT / name for name in capture()['files_sha256']] + [Path(__file__),
             ROOT / 'src/slm_lab/textvqa_data.py', ROOT / 'src/slm_lab/textvqa_evaluation.py',
             ROOT / 'src/slm_lab/vqa_pairs.py', ROOT / 'src/slm_lab/vqa_pair_evaluation.py',
             ROOT / 'src/slm_lab/clevr_evaluation.py']
    loaded = capture(files); reports.mkdir()

    def status(message, **values):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                    'goal_achieved': False, 'preceding_pid': PID, 'preceding_start_ticks': IDENTITY, **values})

    def verify():
        if capture(files) != loaded or file_hash(controller / 'plan.json') != controller_plan:
            raise ValueError('Loaded transfer implementation/controller plan changed')
        for path, digest in data_hashes.items():
            if file_hash(path) != digest: raise ValueError('Loaded real-reading inputs changed')

    atomic_json(reports / 'plan.json', {'created_at': datetime.now(timezone.utc).isoformat(),
                'preceding_pid': PID, 'preceding_start_ticks': IDENTITY, 'controller_plan_sha256': controller_plan,
                'dataset_files_sha256': data_hashes, 'source_provenance': loaded, 'device': 'cuda:2',
                'max_examples': 128, 'max_new_tokens': 32, 'policy_promoted': False, 'goal_achieved': False,
                'scope': 'Real-photo zero-shot reading transfer from own rendered-scene task intervention; TextVQA is not training data for this intervention.'})
    try:
        status('waiting for verified live native scene/QA controller')
        while json.loads((controller / 'status.json').read_text())['status'] != 'joint native scene/QA rehearsal assessed; broader capability unverified':
            verify()
            if process_identity(PID) != IDENTITY:
                raise ValueError('Observed preceding controller terminated without successful completion; inspect its actual evidence')
            time.sleep(15)
        receipt = json.loads((controller / 'stage-receipt.json').read_text())
        before = ROOT / 'checkpoints/clevr-dense-learning-v1/vision/model.pt'
        after = ROOT / 'checkpoints/clevr-joint-rehearsal-v1/joint/model.pt'
        if file_hash(before) != receipt['source_checkpoint_sha256'] or file_hash(after) != receipt['model_sha256']:
            raise ValueError('Completed own transfer checkpoints differ from recorded training receipt')
        results = {}
        for name, checkpoint in [('before', before), ('after', after)]:
            verify(); status('assessing native reading on original real photographs', stage=name)
            output = reports / (name + '.json')
            subprocess.run([sys.executable, '-u', '-m', 'slm_lab.textvqa_evaluation', '--checkpoint', str(checkpoint),
                            '--data', str(dataset), '--output', str(output), '--device', 'cuda:2'], check=True, cwd=ROOT,
                           env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
            results[name] = json.loads(output.read_text())
        verify()
        atomic_json(reports / 'comparison.json', {'guard': compare_reports(results['before'], results['after']),
                    'before': results['before']['overall'], 'after': results['after']['overall'],
                    'policy_promoted': False, 'goal_achieved': False,
                    'scope': 'Reused original real-photo development subset; not independent final testing or full generic image capability.'})
        status('original real-photo native reading transfer assessed; broader capability unverified')
    except Exception as error:
        status('real-photo transfer assessment failed; inspect actual preceding/live evidence', error=str(error)); raise


if __name__ == '__main__': main()
