"""Read every original replay exposure back through its bound complete reader."""
from collections import Counter
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from tokenizers import Tokenizer
import torch

from slm_lab.codenet_replay_v1 import FullReplayEncoder, ReplayCorpus
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def wait_for_preparation():
    launch = json.loads((ROOT / 'reports/codenet-replay-v1-launch.json').read_text())
    own_digest = file_hash(Path(__file__))
    while True:
        if file_hash(Path(__file__)) != own_digest:
            raise ValueError('Waiting replay auditor source changed')
        for name, digest in launch['source_files_sha256'].items():
            if file_hash(Path(name)) != digest:
                raise ValueError('Original replay preparation runtime changed')
        proc = Path('/proc') / str(launch['pid'])
        try:
            fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
            live = fields[19] == str(launch['start_ticks']) and fields[0] != 'Z'
            if live and (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode() != launch['observed_argv']:
                raise ValueError('Preparation process argv changed')
        except FileNotFoundError:
            live = False
        if not live:
            if not (ROOT / 'data/codenet-replay-v1/manifest.json').exists():
                raise ValueError('Preparation terminated without a complete manifest; do not restart it automatically')
            return
        atomic_json(ROOT / 'reports/codenet-replay-v1-independent-audit-wait.json', {
            'status': 'waiting for exact live replay preparation process before auditing complete artifacts',
            'preparation_pid': launch['pid'], 'preparation_start_ticks': launch['start_ticks'],
            'observed_at': datetime.now(timezone.utc).isoformat(), 'waiting_audit_source_sha256': own_digest,
            'new_model_training_started': False, 'training_stage_ready': False, 'goal_achieved': False})
        print('waiting_for_live_replay_preparation', launch['pid'], launch['start_ticks'], flush=True)
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-preparation', action='store_true')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Keep all GPUs dedicated to the live curriculum')
    if args.wait_for_preparation:
        wait_for_preparation()
    torch.set_num_threads(1)
    folder = ROOT / 'reports/codenet-replay-v1-independent-audit'
    if folder.exists():
        raise ValueError('Preserve previous complete replay audit')
    folder.mkdir()
    started = time.monotonic()

    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'new_model_training_started': False, 'training_stage_ready': False, 'goal_achieved': False, **fields})

    try:
        status('verifying all replay artifacts and original training/image files')
        corpus = ReplayCorpus(ROOT / 'data/codenet-replay-v1')
        manifest = corpus.manifest
        plan_path = Path(manifest['original_frozen_plan'])
        frozen = next(s for s in json.loads(plan_path.read_text())['stages'] if s['name'] == 'joint')
        if (file_hash(plan_path) != manifest['original_frozen_plan_sha256']
            or Path(frozen['data']) != corpus.source or frozen['data_sha256'] != file_hash(corpus.source)):
            raise ValueError('Exact earlier frozen training role/source required')
        config_path = ROOT / 'configs/model.json'
        tokenizer_path = Path(manifest['tokenizer'])
        if file_hash(tokenizer_path) != manifest['tokenizer_sha256']:
            raise ValueError('Exact original own tokenizer required')
        encoder = FullReplayEncoder(Tokenizer.from_file(str(tokenizer_path)), ModelConfig.load(config_path),
                                    corpus.source.parent, image_size=448)
        files = [Path(__file__), ROOT / 'src/slm_lab/codenet_replay_v1.py', ROOT / 'src/slm_lab/training.py',
                 ROOT / 'src/slm_lab/model.py', config_path, tokenizer_path, plan_path,
                 corpus.folder / 'manifest.json', corpus.source]
        bound = {path: file_hash(path) for path in files}
        states = {path: corpus._state(path) for path in files}

        def guard(full=False):
            if any(corpus._state(p) != states[p] or (full and file_hash(p) != digest) for p, digest in bound.items()):
                raise ValueError('Bound complete replay audit source changed')

        atomic_json(folder / 'plan.json', {'source_files_sha256': {str(p): h for p, h in bound.items()},
            'scope': 'Every original frozen main-joint train row, complete-mask readback and actual448px encoder tensors. No model update or capability evaluation.',
            'no_model_loaded': True, 'device': 'cpu', 'new_model_training_started': False, 'goal_achieved': False})
        family_counts, all_families, contexts, targets = Counter(), Counter(), Counter(), Counter()
        ids, actual_pixels = Counter(), {}
        checked = fit = overflow = legacy_long = reserved = pixel_rows = 0
        records_path, reviews_path = corpus.folder / 'records.jsonl', corpus.folder / 'review.jsonl'
        with corpus.source.open('rb') as source, records_path.open() as records, reviews_path.open() as reviews:
            while True:
                offset = source.tell()
                raw = source.readline()
                if not raw:
                    break
                if not raw.strip():
                    continue
                row = json.loads(raw)
                checked += 1
                ids[row.get('id')] += 1
                for key in ['split', 'source_split']:
                    if row.get(key) not in {None, 'train'}:
                        raise ValueError('Explicit held/source role found in original frozen train source')
                # Independently identify the expected rehearsal objective from source metadata.
                if row.get('image'):
                    name = 'image_reading' if row.get('task') in {
                        'word_ocr', 'region_ocr', 'document_ocr', 'document_region_ocr'} else 'generic_visual'
                elif any(s in row.get('source', '') for s in ['TheAlgorithms/Python', 'MBPP']):
                    name = 'verified_programs'
                elif row.get('language'):
                    name = 'code:' + row['language']
                elif row.get('domain') == 'math_reasoning' or row.get('source', '').startswith('GSM8K'):
                    name = 'math'
                elif row.get('source') == 'HuggingFaceH4/no_robots':
                    name = 'human_instruction'
                else:
                    raise ValueError('Original source has no reviewed objective')
                all_families[name] += 1
                prompt = encoder.tokens(encoder.prompt(row))
                response = encoder.tokens(row['response'])
                expected_ids = [encoder.bos] + prompt + response + [encoder.eos]
                raw_digest = hashlib.sha256(raw).hexdigest()
                if len(expected_ids) > encoder.config.max_text_length:
                    reviewed = json.loads(next(reviews))
                    if (reviewed['source_offset'] != offset or reviewed['source_bytes'] != len(raw)
                        or reviewed['source_row_sha256'] != raw_digest or reviewed['family'] != name
                        or reviewed['prompt_tokens'] != len(prompt) or reviewed['response_tokens'] != len(response)
                        or reviewed['original_full_pair_tokens'] != len(expected_ids)):
                        raise ValueError('Whole original over-window row review differs')
                    overflow += 1
                else:
                    item = corpus.index[fit]
                    if (int(item['offset']) != offset or int(item['bytes']) != len(raw)
                        or item['row_sha256'].tobytes().hex() != raw_digest):
                        raise ValueError('Exact source row exposure was lost, duplicated or changed')
                    loaded = corpus.row(name, family_counts[name])
                    originals = {key: value for key, value in loaded.items() if not key.startswith('_replay_')}
                    if originals != row:
                        raise ValueError('Original response or metadata changed during replay readback')
                    actual_ids, labels = encoder.sequence(loaded, 'response')
                    expected_labels = [-100] * (len(prompt) + 1) + response + [encoder.eos]
                    metadata = json.loads(next(records))
                    if (actual_ids != expected_ids or labels != expected_labels
                        or metadata['source_offset'] != offset or metadata['source_bytes'] != len(raw)
                        or metadata['family'] != name or metadata['source_row_sha256'] != raw_digest
                        or metadata['complete_context_tokens'] != len(actual_ids)
                        or metadata['complete_target_tokens'] != len(response) + 1
                        or metadata['complete_input_ids_sha256'] != hashlib.sha256(np.asarray(actual_ids, dtype='<u2').tobytes()).hexdigest()):
                        raise ValueError('Complete original prompt/response IDs or masks differ')
                    family_counts[name] += 1
                    contexts[name] += len(actual_ids)
                    targets[name] += len(response) + 1
                    legacy_long += len(response) > encoder.config.max_text_length // 2 - 1
                    reserved += any(value in {encoder.bos, encoder.eos, encoder.pad} for value in actual_ids[1:-1])
                    fit += 1
                if row.get('image'):
                    path = Path(row['image'])
                    path = (path if path.is_absolute() else corpus.source.parent / path).resolve()
                    if path not in actual_pixels:
                        if corpus._state(path) != corpus.images[str(path)]:
                            raise ValueError('Image bytes changed after readback initialization')
                        tensor = encoder.pixels(row)
                        if tensor.shape != (3, 448, 448) or tensor.dtype != torch.float32:
                            raise ValueError('Actual encoder pixel shape/dtype differs')
                        # Golden hashes are uint8 RGB HWC; invert the actual float CHW encoder normalization.
                        rgb = np.rint((tensor.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
                        actual_pixels[path] = hashlib.sha256(rgb.tobytes()).hexdigest()
                    if not row.get('model_pixel_sha256_448') or actual_pixels[path] != row['model_pixel_sha256_448']:
                        raise ValueError('Actual448px replay encoder pixels differ from original golden hash')
                    pixel_rows += 1
                if checked % 1000 == 0:
                    guard()
                    status('checking every original exposure, complete mask and448px image tensor',
                           original_rows_checked=checked, complete_fit_rows=fit, reviewed_over_window_rows=overflow,
                           golden_pixel_rows_verified=pixel_rows, unique_actual_pixel_tensors=len(actual_pixels))
                    print('verified_replay_rows', checked, 'golden_pixel_rows', pixel_rows, flush=True)
            if next(records, None) is not None or next(reviews, None) is not None:
                raise ValueError('Replay metadata has unexplained extra exposures')
        if (checked != manifest['source_rows'] or fit != manifest['complete_fit_rows'] or fit != len(corpus.index)
            or overflow != manifest['over_window_review_rows'] or dict(family_counts) != manifest['family_counts']
            or dict(all_families) != manifest['original_family_counts']
            or legacy_long != manifest['complete_targets_longer_than_legacy_cap']
            or reserved != manifest['rows_with_literal_reserved_ids_requiring_review']
            or len(actual_pixels) != manifest['unique_training_image_files'] or encoder.truncated):
            raise ValueError('Whole-source replay accounting differs')
        if any(manifest['family_complete_token_counts'][name] != {
            'context_tokens': contexts[name], 'target_tokens': targets[name]} for name in family_counts):
            raise ValueError('Complete objective token census differs')
        for path, state in corpus.images.items():
            if corpus._state(Path(path)) != state:
                raise ValueError('Original image changed during pixel audit')
        guard(full=True)
        results = {'completed_at': datetime.now(timezone.utc).isoformat(), 'replay_manifest_sha256': bound[corpus.folder / 'manifest.json'],
            'original_source_rows_verified': checked, 'complete_fit_rows_verified': fit,
            'original_over_window_rows_preserved_for_review': overflow, 'source_rows_dropped_or_duplicated': 0,
            'family_counts_verified': dict(family_counts), 'family_context_tokens': dict(contexts),
            'family_complete_target_tokens': dict(targets), 'literal_reserved_id_review_rows': reserved,
            'complete_answers_longer_than_legacy_cap': legacy_long, 'encoder_truncated_sequences': encoder.truncated,
            'golden448px_image_rows_verified': pixel_rows, 'unique_actual448px_pixel_tensors_verified': len(actual_pixels),
            'repeated_nonempty_original_id_exposures_retained': sum(count - 1 for key, count in ids.items() if key is not None),
            'elapsed_seconds': time.monotonic() - started, 'plan_sha256': file_hash(folder / 'plan.json'),
            'no_model_loaded': True, 'CUDA_initialized': torch.cuda.is_initialized(),
            'dataset_admitted_to_training': False, 'new_model_training_started': False,
            'final_checkpoint_replay_match_verified': False, 'training_stage_ready': False, 'goal_achieved': False,
            'scope': 'Exact old frozen main-joint train exposures and actual complete masks/448px pixels. Later queued dialogue, fact, own-context and paired-photo stages require final replay binding. Not a model capability score or broad semantic/pretraining contamination certificate.'}
        atomic_json(folder / 'results.json', results)
        status('every original frozen main-joint exposure and image verified; final model/replay stage binding pending',
               complete_fit_rows=fit, over_window_review_rows=overflow, golden_pixel_rows_verified=pixel_rows)
        print(json.dumps(results), flush=True)
    except Exception as error:
        status('independent complete replay audit stopped; preserve evidence and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
