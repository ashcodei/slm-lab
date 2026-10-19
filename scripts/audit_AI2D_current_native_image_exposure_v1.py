"""Reconstruct every current native training image position for AI2D overlap.

Checks exact original file bytes and exact actual normalized448 input tensors.
Repeated source/protocol/expected-hash inputs reuse only a verified digest;
every source position is still visited and accounted for. No model is built.
This is exact inspected-current-pool exposure, not perceptual/semantic matching
or proof of what all historical checkpoints actually consumed.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D, load_pixels, tensor_sha
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.shared_word_data_v1 import image_arrays, array_digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/AI2D-current-native-image-exposure-v1'


def state(path):
    s = Path(path).stat(); return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden complete native image audit required')
    torch.set_num_threads(1); started = time.monotonic(); OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(OUT / 'owner.json', dict(owner=owner, CPU_threads=1, CUDA_VISIBLE_DEVICES='', model_generations=0, model_updates=0))
    try:
        phase_path = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(phase_path.read_text())
        initial = phase['initial_dense_settings']; architecture = Path(initial['initializer'])
        if file_hash(architecture) != initial['initializer_sha256']: raise ValueError('Exact own configuration source required')
        payload = torch.load(architecture, map_location='cpu', weights_only=True, mmap=True)
        config = ModelConfig(**payload['config']); del payload
        data = ExpandedJointSources(ROOT, config); population = FullAI2D(ROOT)
        inventory_path = ROOT / 'data/ai2d-all-original-test-eval-v2/image-inventory.jsonl'
        inventory = list(map(json.loads, inventory_path.open()))
        expected = {k: v['examples'] for k, v in phase['complete_source_budget'].items() if v['kind'] in ['image', 'word']}
        if ({k: data.populations[k] for k in data.populations if data.kinds[k] in ['image', 'word']} != expected
                or len(expected) != 9 or expected['words'] != 91585): raise ValueError('Every exact original image/word source family required')
        files = dict(phase['source_files_sha256']); files.update(population.bindings)
        files.update({str(p): file_hash(p) for p in [Path(__file__), phase_path,
            *python_sources(ROOT, [Path(__file__)])]})
        guard = FileBindings(files); guard.check(full=True)
        raw_targets = defaultdict(list); native_targets = defaultdict(list)
        for image in inventory:
            if not image['original_image_available']: continue
            raw_targets[image['original_image_sha256']].append(dict(image_id=image['image_id'], source='original_publisher_image_bytes'))
            for variant, info in image['native_variants'].items():
                value = load_pixels(dict(info, image=info['path']))
                if tensor_sha(value) != info['model_input_tensor_sha256_448']: raise ValueError('Every target actual native448 tensor required')
                native_targets[info['model_input_tensor_sha256_448']].append(dict(image_id=image['image_id'], variant=variant))
        atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
            expected_complete_source_positions=expected, expected_positions=sum(expected.values()),
            AI2D_original_publisher_images=981, AI2D_native448_variants=1555, AI2D_scored_original_questions=3088,
            original_population_sha256=population.identity_sha256, source_files_sha256=files,
            architecture_configuration_only=dict(path=str(architecture), sha256=initial['initializer_sha256']),
            exact_original_bytes_and_actual_normalized_float32_CHW448_inputs=True,
            same_source_protocol_and_expected_hash_digest_reuse_not_position_skipping=True,
            word_CTC64x448_separate_from_native448_and_not_miscompared=True,
            all_historical_images_semantic_perceptual_or_crop_ancestry_not_certified=True,
            models_constructed=0, model_generations=0, model_updates=0, goal_achieved=False))
        raw_files = {}; raw_states = {}; cache = {}; counts = Counter(); matches = []; last_progress = 0.
        raw_hits = defaultdict(list); native_hits = defaultdict(list)
        visits_path = OUT / 'all-original-native-image-source-visits.jsonl'
        unique_path = OUT / 'all-freshly-reconstructed-native-image-inputs.jsonl'
        with visits_path.open('x') as visits, unique_path.open('x') as unique:
            for name in expected:
                for position in range(expected[name]):
                    if name == 'words':
                        row = data.words.rows[position]; path = Path(row['image']).resolve()
                        expected_pixel = row['native_input_pixels_sha256']; mode = 'word'; encoder = None
                    else:
                        group, family = name.split(':', 1); source = data.replay if group == 'replay' else data.images
                        row = source.row(family, position); encoder = data.replay_encoder
                        path = (encoder.data_dir / row['image']).resolve(); expected_pixel = row['model_pixel_sha256_448']; mode = 'replay'
                    if not path.is_relative_to(ROOT): raise ValueError('Only the original repository source image may be inspected')
                    if str(path) not in raw_files:
                        raw_files[str(path)] = file_hash(path); raw_states[str(path)] = state(path)
                    if state(path) != raw_states[str(path)]: raise ValueError('Original source image changed during audit')
                    raw_sha = raw_files[str(path)]
                    if row.get('stored_image_sha256') not in [None, raw_sha]: raise ValueError('Actual source image bytes differ from original row')
                    key = (str(path), raw_sha, mode, expected_pixel)
                    if key not in cache:
                        word_sha = None
                        if mode == 'word':
                            with Image.open(path) as image: word, native = image_arrays(image)
                            native_sha = array_digest(native); word_sha = array_digest(word)
                            if native_sha != expected_pixel or word_sha != row['word_input_pixels_sha256']:
                                raise ValueError('Every actual complete word/native pixel view must match its source admission')
                            shape = list(native.shape)
                        else:
                            pixels = encoder.pixels(row)
                            if pixels is None or pixels.dtype != torch.float32: raise ValueError('An actual native float32 image input is required')
                            native_sha = tensor_sha(pixels); shape = list(pixels.shape)
                        if shape != [3, 448, 448]: raise ValueError('Exact native448 CHW protocol required for image comparison')
                        cache[key] = dict(actual_native448_tensor_sha256=native_sha, actual_CTC64x448_tensor_sha256=word_sha)
                        unique.write(json.dumps(dict(source_image=str(path), original_file_sha256=raw_sha, native_protocol=mode,
                            expected_source_pixel_hash=expected_pixel, **cache[key]), separators=(',', ':')) + '\n')
                    observation = dict(family=name, original_position=position, original_source_id=row.get('id'),
                        source_image=str(path), source_file_sha256=raw_sha, **cache[key])
                    visits.write(json.dumps(observation, separators=(',', ':')) + '\n'); counts[name] += 1
                    for target in raw_targets.get(raw_sha, []):
                        match = dict(kind='exact_original_file_bytes', target=target, training=observation)
                        matches.append(match); raw_hits[target['image_id']].append(match)
                    for target in native_targets.get(cache[key]['actual_native448_tensor_sha256'], []):
                        match = dict(kind='exact_actual_native448_tensor', target=target, training=observation)
                        matches.append(match); native_hits[(target['image_id'], target['variant'])].append(match)
                    now = time.monotonic()
                    if now - last_progress >= 15:
                        visits.flush(); unique.flush(); os.fsync(visits.fileno()); os.fsync(unique.fileno())
                        progress = dict(owner=owner, family=name, completed_source_positions=dict(counts),
                            source_positions=sum(counts.values()), unique_original_image_files=len(raw_files),
                            unique_reconstructed_source_protocol_inputs=len(cache), exact_matches=len(matches), elapsed_seconds=now - started,
                            models_constructed=0, model_updates=0, goal_achieved=False)
                        atomic_json(OUT / 'progress.json', progress); print(json.dumps(progress), flush=True); last_progress = now
            visits.flush(); unique.flush(); os.fsync(visits.fileno()); os.fsync(unique.fileno())
        if dict(counts) != expected: raise ValueError('Every original selected native image source position must be accounted for')
        guard.check(full=True)
        for path, sha in raw_files.items():
            if state(path) != raw_states[path] or file_hash(path) != sha: raise ValueError('Every original raw image must remain byte-exact')
        if torch.cuda.is_initialized(): raise ValueError('Read-only CPU audit initialized CUDA')
        atomic_json(OUT / 'all-original-source-image-digests.json', raw_files)
        match_path = OUT / 'exact-matches.jsonl'; match_path.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in matches))
        by_question = []
        for case in population.cases:
            variant = 'abc_opaque' if case['original_ABC_labels'] else 'original'
            by_question.append(dict(id=case['id'], image_id=case['image_id'], original_ABC_labels=case['original_ABC_labels'],
                exact_original_diagram_file_in_inspected_current_pool=bool(raw_hits[case['image_id']]),
                exact_actual_proper_native448_input_in_inspected_current_pool=bool(native_hits[(case['image_id'], variant)]),
                no_match_is_not_semantic_perceptual_or_all_historical_independence=True))
        qpath = OUT / 'all3088-original-question-image-exposure.jsonl'
        qpath.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in by_question))
        atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, complete=True,
            plan_sha256=file_hash(OUT / 'plan.json'), original_population_sha256=population.identity_sha256,
            complete_current_native_image_source_positions=dict(counts), actual_source_positions=sum(counts.values()),
            actual_unique_original_image_files=len(raw_files), actual_unique_reconstructed_source_protocol_inputs=len(cache),
            all_source_bytes_and_actual_native448_pixels_checked=True,
            actual_word_native448_and_CTC64x448_protocols_checked_without_shape_mismatch=True,
            exact_original_diagram_file_matched_questions=sum(r['exact_original_diagram_file_in_inspected_current_pool'] for r in by_question),
            exact_actual_proper_native448_input_matched_questions=sum(r['exact_actual_proper_native448_input_in_inspected_current_pool'] for r in by_question),
            exact_training_position_target_match_instances=len(matches), source_files_sha256={str(p): file_hash(p) for p in [
                OUT / 'plan.json', visits_path, unique_path, OUT / 'all-original-source-image-digests.json', match_path, qpath]},
            no_original_question_or_source_position_excluded=True, inspected_pool_not_exact_historical_consumption=True,
            other_ancestral_images_unannotated_text_semantic_perceptual_or_crop_ancestry_not_certified=True,
            models_constructed=0, model_generations=0, model_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_source_positions=sum(counts.values()),
            exact_matches=len(matches), models_constructed=0)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, type=type(error).__name__, message=str(error),
            traceback=traceback.format_exc(), complete=False, automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__': main()
