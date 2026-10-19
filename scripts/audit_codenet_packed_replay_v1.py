"""Read every indexed causal window against original packed shard bytes."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_packed_replay_v1 import FAMILY, PackedPretrainingEncoder, PackedReplayCorpus
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/codenet-packed-pretraining-replay-v1'
PREP = ROOT / 'reports/codenet-packed-pretraining-replay-v1/results.json'
OUTPUT = ROOT / 'reports/codenet-packed-pretraining-replay-v1-independent-audit'


def status(message, **fields):
    atomic_json(OUTPUT / 'status.json', dict(status=message,
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only complete original token readback required')
    if OUTPUT.exists():
        raise ValueError('Preserve prior complete/partial audit; no implicit restart')
    prepared = json.loads(PREP.read_text())
    if file_hash(DATA / 'manifest.json') != prepared['manifest_sha256']:
        raise ValueError('Actual completed source preparation required')
    bindings = prepared['source_files_sha256'].copy()
    for path in [PREP, DATA / 'manifest.json', Path(__file__)]:
        bindings[str(path)] = file_hash(path)
    manifest = json.loads((DATA / 'manifest.json').read_text())
    for name, digest in manifest['artifact_files_sha256'].items():
        bindings[str(DATA / name)] = digest
    packed_root = Path(manifest['packed_root'])
    packed = json.loads((packed_root / 'corpus.json').read_text())
    bindings[str(packed_root / 'corpus.json')] = manifest['packed_manifest_sha256']
    for split in ('train', 'valid'):
        for item in packed['splits'][split]['shards']:
            bindings[str(packed_root / item['path'])] = item['sha256']
    guard = FileBindings(bindings)
    guard.check(full=True)
    OUTPUT.mkdir()
    status('checking every original packed token, window hash, causal target and final partial window')
    try:
        reader = PackedReplayCorpus(DATA)
        tokenizer = Tokenizer.from_file(manifest['tokenizer'])
        config = ModelConfig(**json.loads((ROOT / 'configs/model.json').read_text()))
        encoder = PackedPretrainingEncoder(tokenizer, config, ROOT / 'data', image_size=448)
        observed = {}
        for split in ('train', 'valid'):
            parts, start = [], 0
            for item in packed['splits'][split]['shards']:
                array = np.memmap(packed_root / item['path'], dtype='<u2', mode='r')
                parts.append((start, start + len(array), array))
                start += len(array)
            total, expected_start, targets, crossings, held_rejections = start, 0, 0, 0, 0
            selected, all_input_digest = [], hashlib.sha256()
            for position in range(reader.size(FAMILY, split=split)):
                row = reader.row(FAMILY, position, split=split)
                stop = min(total, expected_start + config.max_text_length)
                slices = [array[max(0, expected_start - begin):min(len(array), stop - begin)]
                    for begin, end, array in parts if end > expected_start and begin < stop]
                original = np.concatenate(slices)
                if (row['packed_offset'] != expected_start or row['tokens'] != original.astype(np.int64).tolist()
                        or row['_replay_context_tokens'] != len(original)
                        or row['_replay_target_tokens'] != len(original) - 1
                        or row['_replay_input_sha256'] != hashlib.sha256(original.tobytes()).hexdigest()):
                    raise ValueError('Original direct shard bytes/offsets/window digest/target count differ')
                if split == 'train':
                    ids, labels = encoder.sequence(row, None)
                    if ids != labels or ids != row['tokens']:
                        raise ValueError('Every original next-token input and causal label required')
                else:
                    try:
                        encoder.sequence(row, None)
                    except ValueError as error:
                        if 'held/source' not in str(error):
                            raise
                        held_rejections += 1
                    else:
                        raise ValueError('Original validation data must reject the training encoder')
                if len(slices) > 1:
                    crossings += 1
                if position == 0 or stop == total or len(slices) > 1:
                    selected.append(dict(position=position, offset=expected_start, length=len(original),
                        target_tokens=len(original) - 1, input_sha256=row['_replay_input_sha256']))
                all_input_digest.update(original.tobytes())
                targets += len(original) - 1
                if stop == total:
                    if position != reader.size(FAMILY, split=split) - 1:
                        raise ValueError('Extra indexed windows after the complete original stream')
                expected_start = stop - 1
                if position and position % 10000 == 0:
                    status('checking every original packed token, window hash, causal target and final partial window',
                        split=split, windows_checked=position + 1, original_causal_targets_checked=targets)
            if expected_start != total - 1 or targets != total - 1:
                raise ValueError('Every original adjacent-token target must be exposed exactly once')
            if split == 'train':
                batch_rows = [reader.row(FAMILY, position) for position in [0, reader.size(FAMILY) - 1]]
                batch = encoder.batch(batch_rows, None, 'cpu')
                if int((batch['labels'][:, 1:] != -100).sum()) != sum(len(row['tokens']) - 1 for row in batch_rows):
                    raise ValueError('Actual first/tail batch padding must preserve all causal labels')
            observed[split] = dict(original_tokens=total, windows=reader.size(FAMILY, split=split),
                actual_target_tokens=targets, shard_boundary_windows=crossings,
                final_window_length=selected[-1]['length'], all_window_bytes_sha256=all_input_digest.hexdigest(),
                every_validation_window_rejected_for_training=held_rejections if split == 'valid' else None,
                independently_checked_boundary_windows=selected)
        if encoder.truncated:
            raise ValueError('No clipped or fabricated packed text is permitted')
        guard.check(full=True)
        atomic_json(OUTPUT / 'results.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            passed=True, source_files_sha256=bindings, manifest_sha256=prepared['manifest_sha256'],
            splits=observed, all_original_packed_input_bytes_and_causal_masks_checked=True,
            every_original_adjacent_pair_targeted_once=True, artificial_special_tokens_added=0,
            validation_windows_rejected_for_training=True, original_source_bytes_copied=0,
            new_neural_training_started=False, final_base_match=False, production_runner_implemented=False,
            all_later_sources_covered=False, goal_achieved=False,
            scope='Every prepared packed train/valid window independently read from original shard slices, '
                'then compared with actual reader and train-only next-token encoder. Complete original '
                'systems file/token reconstruction is bound from the source preparer. Existing old code '
                'is rehearsal; no new data collection, semantic correctness, final model or capability claim.'))
        status('every original packed input and causal target checked; production integration and training remain pending',
            splits={split: {key: value for key, value in report.items() if key != 'independently_checked_boundary_windows'}
                for split, report in observed.items()})
        print(json.dumps(dict(passed=True, original_train_tokens=observed['train']['original_tokens'],
            actual_train_target_tokens=observed['train']['actual_target_tokens'],
            train_windows=observed['train']['windows'])), flush=True)
    except Exception as error:
        status('complete packed source readback stopped; retain evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
