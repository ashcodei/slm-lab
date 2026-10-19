"""Index every original fixed joint-training row for complete-mask rehearsal."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_replay_v1 import FullReplayEncoder, INDEX_DTYPE, ReplayContextDoesNotFit, family
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    output, reports = ROOT / 'data/codenet-replay-v1', ROOT / 'reports/codenet-replay-v1'
    if output.exists() or reports.exists():
        raise ValueError('Preserve earlier complete replay preparation')
    maximum_bytes = 128 * 1024**2
    if shutil.disk_usage(ROOT).free < 26 * 1024**3 + maximum_bytes:
        raise ValueError('Require replay index budget above persistent writers')
    output.mkdir()
    reports.mkdir()
    def status(message, **fields):
        atomic_json(reports / 'status.json', {'status': message, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'new_model_training_started': False, 'training_stage_ready': False, 'goal_achieved': False, **fields})
    try:
        source = ROOT / 'data/extended-curriculum-v1/joint.train.jsonl'
        prior_plan_path = ROOT / 'reports/extended-curriculum-v1/plan.json'
        prior = json.loads(prior_plan_path.read_text())
        prior_stage = next(stage for stage in prior['stages'] if stage['name'] == 'joint')
        if Path(prior_stage['data']) != source or file_hash(source) != prior_stage['data_sha256']:
            raise ValueError('Exact original fixed joint training source required')
        token_path = ROOT / 'data/codenet-durable-pack-v1/tokenizer.json'
        config_path = ROOT / 'configs/model.json'
        files = [Path(__file__), ROOT / 'src/slm_lab/codenet_replay_v1.py', ROOT / 'src/slm_lab/training.py',
            source, prior_plan_path, token_path, config_path]
        bound = {path: file_hash(path) for path in files}
        states = {path: (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) for path in bound}
        images, image_states = {}, {}
        def guard(full=False):
            if (sum(path.stat().st_size for path in output.iterdir() if path.is_file()) > maximum_bytes
                or shutil.disk_usage(ROOT).free < 512 * 1024**2):
                raise ValueError('Reserved replay index capacity exceeded')
            for path, digest in bound.items():
                s = path.stat()
                if (s.st_ino, s.st_size, s.st_mtime_ns) != states[path] or (full and file_hash(path) != digest):
                    raise ValueError('Bound original replay source/runtime changed')
        config = ModelConfig.load(config_path)
        encoder = FullReplayEncoder(Tokenizer.from_file(str(token_path)), config, source.parent, image_size=448)
        atomic_json(reports / 'plan.json', {'recorded_at': datetime.now(timezone.utc).isoformat(),
            'source_files_sha256': {str(path): digest for path, digest in bound.items()},
            'source': str(source), 'source_fold': 'Exact prior frozen joint training file; no hash resplitting.',
            'maximum_persistent_artifact_bytes': maximum_bytes, 'initial_free_gib_required': 26.125,
            'complete_response_masks': True, 'image_size': 448, 'new_model_training_started': False,
            'training_stage_ready': False, 'goal_achieved': False})
        entries, names, family_ids = [], [], {}
        all_counts, fit_counts, totals = Counter(), Counter(), Counter()
        checked = reviewed = legacy_clipped = reserved_rows = 0
        with source.open('rb') as stream, (output / 'records.jsonl').open('x') as records, (output / 'review.jsonl').open('x') as review:
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                if not raw.strip():
                    continue
                row = json.loads(raw)
                name = family(row)
                checked += 1
                all_counts[name] += 1
                if name not in family_ids:
                    family_ids[name] = len(names)
                    names.append(name)
                image = row.get('image')
                if image:
                    path = Path(image)
                    path = (path if path.is_absolute() else source.parent / path).resolve(strict=True)
                    if path not in images:
                        images[path] = file_hash(path)
                        s = path.stat()
                        image_states[path] = (s.st_ino, s.st_size, s.st_mtime_ns)
                record = {'source_offset': offset, 'source_bytes': len(raw), 'id': row.get('id'), 'family': name,
                    'source_row_sha256': hashlib.sha256(raw).hexdigest(), 'explicit_role': row.get('split'),
                    'original_source_role': row.get('source_split')}
                try:
                    ids, labels = encoder.sequence(row, 'response')
                except ReplayContextDoesNotFit as error:
                    reviewed += 1
                    review.write(json.dumps({**record, 'qualification': 'original_full_replay_context_exceeds_window',
                        'prompt_tokens': error.prompt_tokens, 'response_tokens': error.response_tokens,
                        'original_full_pair_tokens': error.prompt_tokens + error.response_tokens + 2}) + '\n')
                else:
                    targets = sum(value != -100 for value in labels)
                    digest = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).digest()
                    if labels[-targets:] != ids[-targets:] or labels[:-targets] != [-100] * (len(ids) - targets):
                        raise ValueError('Original complete replay response mask differs')
                    literal_reserved = any(value in {encoder.bos, encoder.eos, encoder.pad} for value in ids[1:-1])
                    older_cap = targets - 1 > config.max_text_length // 2 - 1
                    entries.append((offset, len(raw), family_ids[name], len(ids), targets,
                        np.void(hashlib.sha256(raw).digest()), np.void(digest)))
                    fit_counts[name] += 1
                    totals[(name, 'context_tokens')] += len(ids)
                    totals[(name, 'target_tokens')] += targets
                    legacy_clipped += older_cap
                    reserved_rows += literal_reserved
                    records.write(json.dumps({**record, 'complete_context_tokens': len(ids), 'complete_target_tokens': targets,
                        'complete_input_ids_sha256': digest.hex(), 'response_longer_than_legacy_cap': older_cap,
                        'literal_reserved_id_review_needed': literal_reserved}) + '\n')
                if checked % 1000 == 0:
                    records.flush()
                    review.flush()
                    guard()
                    status('indexing original joint-training replay with complete masks', source_rows_checked=checked,
                        complete_fit_rows=len(entries), over_window_review_rows=reviewed, unique_images=len(images))
                    print('original_replay_rows', checked, 'complete_fit', len(entries), flush=True)
        if not entries or encoder.truncated:
            raise ValueError('Complete nonempty replay with zero clipping required')
        guard(full=True)
        image_manifest = output / 'image-files.jsonl'
        with image_manifest.open('x') as target:
            for path, digest in sorted(images.items()):
                s = path.stat()
                if (s.st_ino, s.st_size, s.st_mtime_ns) != image_states[path] or file_hash(path) != digest:
                    raise ValueError('Original training image changed during preparation')
                target.write(json.dumps({'path': str(path), 'sha256': digest}) + '\n')
        np.save(output / 'index.npy', np.asarray(entries, dtype=INDEX_DTYPE), allow_pickle=False)
        required = {'code:Python', 'code:JavaScript', 'code:Java', 'code:Go', 'code:PHP', 'code:Ruby',
                    'verified_programs', 'human_instruction', 'math', 'image_reading', 'generic_visual'}
        if not required <= set(fit_counts):
            raise ValueError('All current joint rehearsal fronts must remain present')
        artifacts = ['index.npy', 'image-files.jsonl', 'records.jsonl', 'review.jsonl']
        size = sum((output / name).stat().st_size for name in artifacts)
        if size > maximum_bytes:
            raise ValueError('Final replay artifact budget exceeded')
        manifest = {'format': 'slm-complete-joint-replay-v1', 'completed_at': datetime.now(timezone.utc).isoformat(),
            'original_training_source': str(source), 'original_training_source_sha256': bound[source],
            'original_frozen_plan': str(prior_plan_path), 'original_frozen_plan_sha256': bound[prior_plan_path],
            'tokenizer': str(token_path), 'tokenizer_sha256': bound[token_path], 'image_size': 448,
            'source_rows': checked, 'complete_fit_rows': len(entries), 'over_window_review_rows': reviewed,
            'family_names': names, 'original_family_counts': dict(all_counts), 'family_counts': dict(fit_counts),
            'family_complete_token_counts': {name: {key: totals[(name, key)] for key in ['context_tokens', 'target_tokens']} for name in names},
            'unique_training_image_files': len(images), 'complete_targets_longer_than_legacy_cap': legacy_clipped,
            'rows_with_literal_reserved_ids_requiring_review': reserved_rows, 'encoder_truncated_sequences': encoder.truncated,
            'artifact_files_sha256': {name: file_hash(output / name) for name in artifacts}, 'stored_artifact_bytes': size,
            'preparation_plan_sha256': file_hash(reports / 'plan.json'), 'new_model_training_started': False,
            'training_stage_ready': False, 'goal_achieved': False,
            'scope': 'Existing frozen main-joint training source only. Complete masks/bytes and explicit-role checks, not certification of semantic/pretraining independence. Later queued paired-photo/dialogue/fact/own-context streams are not automatically covered; final replay must match the chosen starting model.'}
        atomic_json(output / 'manifest.json', manifest)
        status('original joint replay indexed; independent coverage audit and final stage binding pending',
            source_rows_checked=checked, complete_fit_rows=len(entries), over_window_review_rows=reviewed,
            family_counts=dict(fit_counts), stored_bytes=size)
        print('prepared_complete_original_replay_rows', len(entries), 'of', checked, flush=True)
    except Exception as error:
        status('complete replay preparation stopped; preserve partial output and inspect', error=str(error))
        raise


if __name__ == '__main__':
    main()
