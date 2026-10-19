"""After a verified live MAE job, compare frozen object features with its base."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.masked_visual_pretraining import process_identity
from slm_lab.object_feature_probe import SOURCE_FILES, evaluate

ROOT = Path(__file__).resolve().parents[1]


def paired_difference(before, after):
    if [(row['id'], row['group_id'], row['expected'], row['input_pixels_sha256'], row['wrong_image_id'], row['wrong_pixels_sha256'])
        for row in before] != [(row['id'], row['group_id'], row['expected'], row['input_pixels_sha256'], row['wrong_image_id'], row['wrong_pixels_sha256']) for row in after]:
        raise ValueError('Object feature case/input/control identities differ')
    groups = {}
    for first, second in zip(before, after):
        change = int(second['predicted'] == second['expected']) - int(first['predicted'] == first['expected'])
        values = groups.setdefault(first['group_id'], [0, 0])
        values[0] += change; values[1] += 1
    values = np.asarray(list(groups.values()), dtype=np.float64)
    draws = np.random.default_rng(7823).integers(0, len(values), size=(2000, len(values)))
    samples = values[draws].sum(1)
    differences = samples[:, 0] / samples[:, 1]
    return {'accuracy_change': float(values[:, 0].sum() / values[:, 1].sum()),
            'photo_group_bootstrap_95_interval': np.quantile(differences, [.025, .975]).tolist(),
            'photo_groups': len(values), 'resamples': 2000, 'seed': 7823,
            'interpretation': 'Development estimate with photo groups as resampling units; not independent final capability proof.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-pid', type=int, required=True)
    args = parser.parse_args()
    reports = ROOT / 'reports/object-feature-masked-v1'; reports.mkdir(exist_ok=True)
    files = SOURCE_FILES + [Path(__file__), ROOT / 'src/slm_lab/masked_visual_pretraining.py']
    loaded = capture(files)

    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
                    'goal_achieved': False, **fields})

    try:
        pid = args.training_pid; identity = process_identity(pid)
        if identity is None:
            raise ValueError('Requested MAE process is not live; inspect completed artifacts before assessment')
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if b'slm_lab.masked_visual_pretraining' not in command:
            raise ValueError('Requested process is not the expected masked-vision trainer')
        status('waiting for verified live masked-vision training', training_pid=pid, process_start_ticks=identity)
        while process_identity(pid) == identity:
            time.sleep(15)
        if capture(files) != loaded:
            raise ValueError('Loaded feature assessment implementation/runtime changed while waiting')
        stage = ROOT / 'reports/masked-visual-pretraining-v1'
        completed = json.loads((stage / 'status.json').read_text())
        plan = json.loads((stage / 'plan.json').read_text())
        if (completed['status'] != 'masked visual pretraining/reconstruction checks completed; semantic/native capability unverified'
                or completed['steps'] != plan['steps']):
            raise ValueError('Preceding masked-vision stage did not complete its declared plan')
        candidate = ROOT / 'checkpoints/masked-visual-pretraining-v1/model.pt'
        state = torch.load(candidate, map_location='cpu', weights_only=True, mmap=True)
        if state['plan'] != plan or state['steps'] != plan['steps']:
            raise ValueError('Masked-vision checkpoint differs from the completed plan')
        del state
        baseline_dir = ROOT / 'reports/object-feature-baseline-v1'
        baseline = json.loads((baseline_dir / 'summary.json').read_text())
        original = next(row for row in baseline['encoders'] if row['name'] == 'original')
        if original['checkpoint_sha256'] != file_hash(plan['base_checkpoint']):
            raise ValueError('Object feature baseline was not the MAE source encoder')
        status('assessing frozen masked-vision object features', checkpoint=str(candidate))
        result = evaluate([('masked', candidate)], ROOT / 'data/object-feature-probe-v1', reports,
                          device='cuda:2', random_control=False)
        for name in ['manifest_sha256', 'config', 'classes', 'regularization', 'feature_protocol',
                     'evaluation_provenance', 'training_examples', 'validation_examples',
                     'inference_device_type', 'inference_precision']:
            if baseline[name] != result[name]:
                raise ValueError('Feature before/after protocol differs: ' + name)
        candidate_report = json.loads((reports / 'masked.json').read_text())
        original_report = json.loads((baseline_dir / 'original.json').read_text())
        differences = paired_difference(original_report['cases'], candidate_report['cases'])
        if capture(files) != loaded:
            raise ValueError('Feature assessment implementation/runtime changed during assessment')
        atomic_json(reports / 'comparison.json', {
            'baseline': original['validation'], 'candidate': result['encoders'][0]['validation'],
            'paired': differences, 'baseline_random': next(row['validation'] for row in baseline['encoders'] if row['name'] == 'random'),
            'appearance_baselines': baseline['baselines'], 'baseline_source': str(baseline_dir / 'summary.json'),
            'candidate_source': str(reports / 'summary.json'), 'goal_achieved': False, 'production_promoted': False,
            'limitations': ['Diagnostic heads train on labels while encoders stay frozen; this is oracle-box classification.',
                            'Neither reconstruction nor a probe gain establishes native multimodal/general language capability.',
                            'Caption realignment and native visual/code/knowledge preservation remain required before adoption.']})
        status('matched object feature comparison completed; native multimodal capability remains unverified',
               accuracy_change=differences['accuracy_change'], production_promoted=False)
    except Exception as error:
        status('requires inspection; masked-vision feature assessment stopped', error=str(error))
        raise


if __name__ == '__main__':
    main()
