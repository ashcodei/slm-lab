"""Wait for the exact live collector, then admit/index all finalized photo QA."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_paired_photo_replay_v1 import FAMILY, PairedPhotoReplay, prepare_paired_replay
from slm_lab.codenet_runner_v1 import VerifiedReplayEncoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.model import ModelConfig
from slm_lab.vqa_pair_admission import admit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/vqa-real-pairs-v1'
OUTPUT = ROOT / 'data/codenet-paired-photo-replay-v1'
REPORTS = ROOT / 'reports/codenet-paired-photo-replay-v1'
MAXIMUM = 32 * 1024**2


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message,
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))


def live_collector(pid, start_ticks):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
        argv = (proc / 'cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:
        return False
    expected = [str(ROOT / '.venv/bin/python').encode(), b'-u', b'-m', b'slm_lab.vqa_pairs']
    return fields[0] != 'Z' and fields[19] == start_ticks and argv[:4] == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collector-pid', type=int, required=True)
    parser.add_argument('--collector-start-ticks', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only original paired-photo source preparation required')
    if OUTPUT.exists() or REPORTS.exists():
        raise ValueError('Retain earlier complete/partial preparation; no implicit restart')
    if not (DATA / 'manifest.json').exists() and not live_collector(args.collector_pid, args.collector_start_ticks):
        raise ValueError('Exact currently live original collector or final manifest required')
    token = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    files = [Path(__file__), token, ROOT / 'configs/model.json',
        ROOT / 'tests/test_codenet_paired_photo_replay_v1.py',
        ROOT / 'reports/codenet-paired-photo-replay-v1-focused-tests.log']
    files += [ROOT / 'src/slm_lab' / (name + '.py') for name in ['codenet_paired_photo_replay_v1',
        'codenet_checkpoint_v1', 'codenet_replay_v1', 'codenet_runner_v1', 'vqa_pair_admission',
        'vqa_pairs', 'vqa_attribution', 'vqa_pair_evaluation', 'clevr_evaluation']]
    bindings = {str(path): file_hash(path) for path in files}
    bindings.update({str(ROOT / name): digest for name, digest in capture()['files_sha256'].items()})
    runtime = capture(files)
    guard = FileBindings(bindings)
    REPORTS.mkdir()
    try:
        while not (DATA / 'manifest.json').exists():
            if not live_collector(args.collector_pid, args.collector_start_ticks):
                raise ValueError('Original collector ended without finalization; inspect without automatic restart')
            guard.check()
            if capture(files) != runtime:
                raise ValueError('Source preparation implementation/runtime changed while waiting')
            status('waiting for exact live original photo-pair collector to finalize; no partial rows admitted',
                collector_pid=args.collector_pid, collector_start_ticks=args.collector_start_ticks,
                original_collector_verified_live=True, output_data_created=False)
            time.sleep(15)
        if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 2 * MAXIMUM:
            raise ValueError('Fund32MiB indices and32MiB reports above unchanged26GiB reserve')
        guard.check(full=True)
        status('reconstructing every finalized original human question, pair, creator declaration and224/448px photo')
        admitted, receipt = admit(ROOT)
        tokenizer = Tokenizer.from_file(str(token))
        config = ModelConfig.load(ROOT / 'configs/model.json')
        encoder = VerifiedReplayEncoder(tokenizer, config, DATA, 448)
        manifest = prepare_paired_replay(OUTPUT, DATA, admitted, receipt, encoder, file_hash(token))
        reader = PairedPhotoReplay(OUTPUT)
        expected = {row['id']: row for row in admitted}
        checked, target_tokens, seen = 0, 0, set()
        for position in range(reader.size(FAMILY)):
            row = reader.row(FAMILY, position)
            original = {key: value for key, value in row.items() if not key.startswith('_replay_')}
            if original != expected.get(row['id']) or row['id'] in seen:
                raise ValueError('Every exact admitted original photo question/metadata must be read back once')
            seen.add(row['id'])
            ids, labels = encoder.sequence(row, 'response')
            if (len(ids) != row['_replay_context_tokens']
                    or sum(value != -100 for value in labels[1:]) != row['_replay_target_tokens']
                    or hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest() != row['_replay_input_sha256']):
                raise ValueError('Every actual original full-mask input must match its source index')
            checked += 1
            target_tokens += row['_replay_target_tokens']
        if seen != set(expected) or encoder.truncated:
            raise ValueError('Complete admitted original source coverage without clipping required')
        selected = sorted({0, reader.size(FAMILY) - 1, int(np.argmax(reader.index['context_tokens'])),
            int(np.argmax(reader.index['target_tokens']))})
        for position in selected:
            row = reader.row(FAMILY, position)
            batch = encoder.batch([row], 'response', 'cpu')
            if tuple(batch['pixel_values'].shape) != (1, 3, 448, 448):
                raise ValueError('Actual original448px golden-verified image batch required')
        size = sum(path.stat().st_size for path in OUTPUT.iterdir() if path.is_file())
        if size > MAXIMUM:
            raise ValueError('Bounded original paired replay artifacts exceeded32MiB commitment')
        original = json.loads((DATA / 'manifest.json').read_text())
        source_bindings = {str(DATA / name): digest for name, digest in original['files_sha256'].items()}
        source_bindings.update(original['sources_sha256'])
        source_bindings[str(DATA / 'manifest.json')] = manifest['original_manifest_sha256']
        source_bindings[str(DATA / 'configuration.json')] = manifest['original_configuration_sha256']
        guard.check(full=True)
        FileBindings(source_bindings).check(full=True)
        if capture(files) != runtime:
            raise ValueError('Actual original preparation runtime changed during source readback')
        atomic_json(REPORTS / 'results.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256={**bindings, **source_bindings}, manifest_sha256=file_hash(OUTPUT / 'manifest.json'),
            complete_original_questions=checked, complete_original_response_targets=target_tokens,
            unique_original_train_photos=manifest['unique_original_training_images'],
            selected_actual448px_batches=len(selected), all_original_source_admissions_reconstructed=True,
            all_actual_replay_rows_inputs_masks_and_metadata_read_back=True,
            original_validation_not_returned_for_training=True, original_image_and_source_bytes_copied=0,
            stored_artifact_bytes=size, new_neural_training_started=False,
            full_final_source_kernel_and_production_registration_pending=True, goal_achieved=False))
        status('all finalized original paired human photo questions indexed and read back; production registration pending',
            complete_original_questions=checked, complete_original_response_targets=target_tokens,
            stored_artifact_bytes=size, neural_training_started=False)
        print(json.dumps(dict(complete_original_questions=checked,
            complete_original_response_targets=target_tokens, stored_artifact_bytes=size)), flush=True)
    except Exception as error:
        status('original paired replay preparation stopped; retain evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
