"""Read every tall real validation crop at0/90/270 degrees; oracle diagnostic only."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from slm_lab.byte_word_reader import ByteWordReader, HEIGHT, WIDTH, collapse, load_rows, pixels
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture, pixels_sha256
from slm_lab.ocr_evaluation import transcription_scores

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); output = ROOT / 'reports/full-byte-reading-v1/tall-orientation-diagnostic.json'
    if output.exists(): raise ValueError('Preserve existing orientation diagnostic')
    checkpoint = ROOT / 'checkpoints/byte-word-reader-v1/model.pt'; checkpoint_sha = file_hash(checkpoint)
    comparison = json.loads((ROOT / 'reports/byte-word-reader-v1/comparison.json').read_text())
    if checkpoint_sha != comparison['model_sha256']: raise ValueError('Different completed own word-reader bytes')
    snapshot = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if snapshot['step'] != snapshot['plan']['steps']: raise ValueError('Orientation diagnostic requires completed own word training')
    files = [Path(__file__)] + [ROOT / 'src/slm_lab' / (name + '.py') for name in
             ['byte_word_reader', 'ocr_evaluation', 'rendered_text', 'corpus', 'diverse_images', 'evaluation_provenance']]
    runtime = capture(files); rows, manifest = load_rows(ROOT / 'data/byte-word-reader-v1')
    reference = ROOT / 'reports/full-byte-reading-v1/reading.json'; reference_sha = file_hash(reference)
    before = {row['id']: row for row in json.loads(reference.read_text())['results']}
    device = 'cuda:2'; torch.cuda.set_per_process_memory_fraction(.15, 2)
    model = ByteWordReader().to(device); model.load_state_dict(snapshot['model']); del snapshot; model.eval()
    cases = []
    for row in sorted(rows['valid'], key=lambda row: row['id']):
        if row['byte_family'] != 'real_word': continue
        raw = Path(row['image']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['stored_image_sha256']: raise ValueError('Actual orientation crop bytes changed')
        with Image.open(io.BytesIO(raw)) as opened:
            image = opened.convert('RGB')
            if image.height <= 1.5 * image.width: continue
            values = [pixels(row)]
            for operation in [Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270]:
                rotated = image.transpose(operation).resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
                values.append(torch.from_numpy((np.asarray(rotated, dtype=np.float32) / 127.5 - 1).copy()).permute(2, 0, 1))
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): logits = model(torch.stack(values).to(device))
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite orientation reading logits')
        predictions = {}
        for degrees, path, actual in zip([0, 90, 270], logits.argmax(-1).cpu().tolist(), values):
            text, valid = collapse(path)
            predictions[str(degrees)] = {'generated': text, 'valid_utf8': valid, 'scores': transcription_scores(row['response'], text),
                                         'input_pixels_sha256': pixels_sha256(actual)}
        cases.append({'id': row['id'], 'group_id': row['group_id'], 'expected': row['response'],
                      'stored_image_sha256': row['stored_image_sha256'], 'predictions': predictions,
                      'unrotated_prediction_equals_full_baseline': predictions['0']['generated'] == before[row['id']]['proper_generated']})
    if capture(files) != runtime or file_hash(checkpoint) != checkpoint_sha or file_hash(reference) != reference_sha:
        raise ValueError('Orientation diagnostic source/model/baseline changed')
    report = {'completed_at': datetime.now(timezone.utc).isoformat(), 'tall_real_validation_crops': len(cases),
              'checkpoint_sha256': checkpoint_sha, 'full_validation_report_sha256': reference_sha, 'source_provenance': runtime,
              'per_fixed_rotation_correct': {str(angle): sum(row['predictions'][str(angle)]['scores']['exact_match'] for row in cases) for angle in [0, 90, 270]},
              'any_rotation_correct_ORACLE': sum(any(value['scores']['exact_match'] for value in row['predictions'].values()) for row in cases),
              'unrotated_batch_rounding_differences': sum(not row['unrotated_prediction_equals_full_baseline'] for row in cases),
              'results': cases, 'oracle_diagnostic': True, 'deployed_orientation_selector': False, 'goal_achieved': False,
              'scope': 'All tall real development crops selected by pixel dimensions only;0/90/270 degree candidate predictions are actual own-model outputs. Any-candidate correctness uses reference labels AFTER inference and is an oracle upper-bound diagnostic, not a deployable reading score or an improvement to the unchanged baseline.'}
    atomic_json(output, report); print(json.dumps({key: value for key, value in report.items() if key not in ['results', 'source_provenance']}), flush=True)


if __name__ == '__main__': main()
