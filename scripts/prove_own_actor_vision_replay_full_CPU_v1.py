"""Actual full-size native photo/word updates, recovery and small dev cohorts.

Use our immutable6144 foundation and one neutral own actor. Two human train
photos plus two original real words supply two updates. Eight dev photos and
four dev real words are fixed before loading the NN and never receive gradients.
This is not a complete vision curriculum, human alignment or GPU admission.
"""
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint, capture_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.fast_joint_inference_v2 import load as load_base
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.own_actor_vision_alignment_v1 import pair_positives
from slm_lab.own_actor_vision_replay_CPU_v1 import OwnActorVisionReplayCPU
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_tokenizer_package_v1 import ensure
from slm_lab.own_vision_replay_bundle_CPU_v1 import export as export_bundle, load as load_bundle
from slm_lab.own_vision_replay_state_CPU_v1 import fixed_digest, snapshot, restore
from slm_lab.shared_word_data_v1 import SharedWordData
from slm_lab.shared_word_evaluation_v1 import ctc_records
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/own-actor-vision-replay-full-CPU-v1'
OUTPUT = ROOT / 'checkpoints/own-actor-vision-replay-full-CPU-v1'
ORIGIN = 'Original human spoken image description, manually transcribed.'


def ordered(rows, tag):
    return sorted(rows, key=lambda r: hashlib.sha256((tag + ':' + r['id']).encode()).hexdigest())


def equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or REPORTS.exists() or OUTPUT.exists():
        raise ValueError('Fresh CPU-hidden full-size vision/word experiment required; no automatic retry')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    words_folder = ROOT / 'data/shared-byte-words-v1'
    words = SharedWordData(words_folder, 'train', 'optimization'); held_words = SharedWordData(words_folder, 'valid', 'evaluation')
    photo_folder = ROOT / 'data/alignment-curriculum-v1'; manifest = json.loads((photo_folder / 'manifest.json').read_text())
    photos = {}
    for split in ['train', 'valid']:
        path = photo_folder / ('contrastive.' + split + '.jsonl')
        if file_hash(path) != manifest['files_sha256'][path.name]: raise ValueError('Exact original caption inventory required')
        photos[split] = [r for r in map(json.loads, path.read_text().splitlines()) if r.get('annotation_origin') == ORIGIN]
        if any(r.get('split') != split for r in photos[split]): raise ValueError('Every selected human caption needs its original explicit role')
    train_groups = {r['group_id'] for r in photos['train']} | {r['group_id'] for r in words.rows}
    photo_train_hashes = {r['model_pixel_sha256_448'] for r in photos['train']}
    word_train_native = {r['native_input_pixels_sha256'] for r in words.rows}
    word_train_pixels = {r['word_input_pixels_sha256'] for r in words.rows}
    chosen_photos = ordered(photos['train'], 'vision-replay-actual-train-v1')[:2]
    eligible_photo_dev = [r for r in photos['valid'] if r['group_id'] not in train_groups and r['model_pixel_sha256_448'] not in photo_train_hashes]
    dev_photos = ordered(eligible_photo_dev, 'vision-replay-actual-dev-v1')[:8]
    word_positions = []
    for family in ['real_word', 'additional_real_word']:
        candidates = ordered([dict(r, original_position=i) for i, r in enumerate(words.rows) if r['byte_family'] == family], 'vision-replay-original-word-train-v1')
        word_positions.append(candidates[0]['original_position'])
    eligible_word_dev = [dict(r, original_position=i) for i, r in enumerate(held_words.rows)
        if r['byte_family'] == 'real_word' and r['group_id'] not in train_groups
        and r['native_input_pixels_sha256'] not in word_train_native and r['word_input_pixels_sha256'] not in word_train_pixels]
    dev_word_positions = []; seen_groups = set(); seen_texts = set()
    for row in ordered(eligible_word_dev, 'vision-replay-original-word-dev-v1'):
        if row['group_id'] in seen_groups or row['response'] in seen_texts: continue
        dev_word_positions.append(row['original_position']); seen_groups.add(row['group_id']); seen_texts.add(row['response'])
        if len(dev_word_positions) == 4: break
    if len(chosen_photos) != 2 or len(dev_photos) != 8 or len(dev_word_positions) != 4:
        raise ValueError('Predeclared complete2/8 photo and2/4 word cohorts must exist without relaxing source separation')
    tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'; tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if file_hash(tokenizer_path) != words.manifest['tokenizer_sha256'] or tokenizer.get_vocab_size() != 32768:
        raise ValueError('Exact own original tokenizer for all native targets required')
    inputs = [Path(__file__), tokenizer_path, photo_folder / 'manifest.json',
        photo_folder / 'contrastive.train.jsonl', photo_folder / 'contrastive.valid.jsonl',
        words_folder / 'manifest.json', words_folder / 'admission.json', words_folder / 'independent-source-readback.json',
        ROOT / 'tests/test_own_actor_vision_replay_CPU_v1.py', ROOT / 'reports/own-actor-vision-replay-CPU-v1-final-tests.log',
        *[words_folder / name for name in words.manifest['files_sha256']]]
    photo_data = {}
    for role, rows in [('train', chosen_photos), ('dev', dev_photos)]:
        pixel_values = []; sequences = []; texts = []
        for row in rows:
            path = Path(row['image']); inputs.append(path)
            if file_hash(path) != row['stored_image_sha256']: raise ValueError('Original stored photograph changed')
            with Image.open(path) as image:
                array = np.asarray(ImageOps.pad(image.convert('RGB'), (448, 448), method=Image.Resampling.BICUBIC, color=(128, 128, 128)), dtype=np.uint8).copy()
            if hashlib.sha256(array.tobytes()).hexdigest() != row['model_pixel_sha256_448']:
                raise ValueError('Exact native448 original photo pixels required')
            pixel_values.append(torch.from_numpy(array.astype(np.float32) / 127.5 - 1).permute(2, 0, 1))
            prompt = [1, *tokenizer.encode('User: ' + row['prompt'] + '\nAssistant:').ids]
            response = [*tokenizer.encode(row['response']).ids, 2]
            seq = prompt + response
            if len(seq) > 4096 or any(t in [0, 1, 2] for t in response[:-1]): raise ValueError('Whole unchanged caption and genuine EOS required')
            sequences.append((seq, [-100] * len(prompt) + response)); texts.append([1, *response])
        def padded(items, fill):
            out = torch.full((len(items), max(map(len, items))), fill, dtype=torch.long)
            for i, item in enumerate(items): out[i, :len(item)] = torch.tensor(item)
            return out
        ids = padded([s[0] for s in sequences], 0); labels = padded([s[1] for s in sequences], -100); caption_ids = padded(texts, 0)
        photo_data[role] = dict(batch=dict(input_ids=ids, labels=labels, attention_mask=ids != 0, pixel_values=torch.stack(pixel_values)),
            caption_ids=caption_ids, caption_mask=caption_ids != 0, positive=pair_positives([r['model_pixel_sha256_448'] for r in rows], [r['response'] for r in rows]),
            source_rows=rows)
    for owner, positions in [(words, word_positions), (held_words, dev_word_positions)]:
        for position in positions: inputs.append(Path(owner.rows[position]['image']))
        owner.batch(positions, 'cpu')  # Hash/reconstruct every selected complete crop and both actual pixel views before NN loading.
    proof_path = ROOT / 'reports/phase-joint-step6144-readback-v1/complete.json'; proof = json.loads(proof_path.read_text())
    source_checkpoint = ROOT / 'checkpoints/phase-joint-v3/step-6144/model.pt'; source_plan = ROOT / 'reports/phase-joint-v3/plan.json'
    if proof['actual_completed_updates'] != 6144 or not proof['native_and_resume_all_model_tensors_bit_exact']:
        raise ValueError('Actual independently read complete own6144 native source required')
    inputs += [proof_path, source_checkpoint, source_plan, *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_own_actor_vision_replay_CPU_v1.py'])]
    bindings = {str(path): file_hash(path) for path in inputs}
    if bindings[str(source_checkpoint)] != proof['source_files_sha256'][str(source_checkpoint)]: raise ValueError('Exact original own6144 weights required')
    for name, expected in words.manifest['source_files_sha256'].items():
        if name in bindings and bindings[name] != expected: raise ValueError('Conflicting original word provenance source')
        bindings[name] = expected
    guard = FileBindings(bindings)
    owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    REPORTS.mkdir(); OUTPUT.mkdir()
    atomic_json(REPORTS / 'source-cohort-before-neural-loading.json', dict(owner=owner,
        train_photo_ids=[r['id'] for r in chosen_photos], dev_photo_ids=[r['id'] for r in dev_photos],
        original_train_word_positions=word_positions, original_dev_word_positions=dev_word_positions,
        original_train_word_ids=[words.rows[p]['id'] for p in word_positions], original_dev_word_ids=[held_words.rows[p]['id'] for p in dev_word_positions],
        eligible_group_separated_dev_photos=len(eligible_photo_dev), eligible_group_pixel_separated_dev_words=len(eligible_word_dev),
        source_files_sha256=bindings, selection_before_neural_quality_observation=True, no_held_training_gradients=True, goal_achieved=False))
    base, _ = load_base(source_checkpoint, source_plan, 'step-6144', expected_checkpoint_sha256=bindings[str(source_checkpoint)],
        expected_plan_sha256=bindings[str(source_plan)], device='cpu')
    actor = OwnDenseActor(base, rank=16, seed=971117).eval(); view = OwnActorVisionReplayCPU(actor).eval()
    if actor.counts()['actual_unique'] != 999706442 or torch.cuda.is_initialized(): raise ValueError('Actual full-size own CPU-only under1B student required')
    selected = view.named_selected(); parameters = [p for _, p in selected]
    settings = dict(learning_rate=1e-4, betas=[.9, .95], weight_decay=.01, matching_weight=.1, temperature=.07,
        CTC_weight=.25, native_word_CE_weight=.25, CPU_threads=1, original_word_positions=word_positions,
        original_photo_ids=[r['id'] for r in chosen_photos], precision='CPU_FP32')
    plan = dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, own_source_checkpoint=str(source_checkpoint), own_source_actual_updates=6144,
        source_files_sha256=bindings, actual_counts=actor.counts(), actual_selected_tensors=len(selected), actual_selected_values=sum(p.numel() for p in parameters),
        added_neural_values=0, settings=settings, planned_logical_updates=2, planned_physical_updates=3,
        photo_cohort_role='train2/dev8', word_cohort_role='train2/dev4',
        all_real_words_and_source_caption_EOS_targets_preserved=True, all_original_vision_sources_consumed=False,
        actual_full_size_student=True, held_cohorts_are_small_development_not_independent_final_benchmarks=True,
        no_external_teacher_or_new_human_feedback=True, full_four_GPU_admission=False, automatic_retry=False, candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'plan.json', plan); digest = file_hash(REPORTS / 'plan.json')
    native_sources = dict(vision_replay_run_plan_sha256=digest, own6144_checkpoint_sha256=bindings[str(source_checkpoint)],
        own_original_word_manifest_sha256=bindings[str(words_folder / 'manifest.json')], own_original_caption_manifest_sha256=bindings[str(photo_folder / 'manifest.json')])
    ensure(tokenizer_path, OUTPUT, expected_sha256=bindings[str(tokenizer_path)], expected_vocab_size=32768)
    frozen = fixed_digest(view); initial_selected = {n: p.detach().clone() for n, p in selected}
    probe = photo_data['train']['caption_ids']; probe_mask = photo_data['train']['caption_mask']
    with torch.no_grad():
        text_before = actor(probe, attention_mask=probe_mask)['logits'].clone()
        source_text = base(probe, attention_mask=probe_mask)['logits']
        neutral_difference = float((text_before - source_text).abs().max())
        if not torch.allclose(text_before, source_text, atol=.0001, rtol=1e-5): raise ValueError('Neutral actual full-size actor must preserve its own6144 base computation')
        if not torch.equal(actor(probe, pixel_values=photo_data['train']['batch']['pixel_values'], attention_mask=probe_mask)['logits'],
                           view.native(probe, pixel_values=photo_data['train']['batch']['pixel_values'], attention_mask=probe_mask)['logits']):
            raise ValueError('Full-size vision training view must preserve actual initial native image outputs exactly')
    atomic_json(REPORTS / 'actual-full-size-initial-native-parity.json', dict(neutral_actor_vs_own6144_base_max_logit_difference=neutral_difference,
        original_native_actor_and_trainable_vision_view_image_logits_bitexact=True, added_neural_values=0, model_updates=0, goal_achieved=False))
    @torch.no_grad()
    def assess(label):
        photo_results = {}; embeddings = {}
        for role, data in photo_data.items():
            batch = data['batch']; records = []; images = []; texts = []
            for i, row in enumerate(data['source_rows']):
                alternatives = [j for j, donor in enumerate(data['source_rows']) if j != i and donor['response'] != row['response']]
                if not alternatives: raise ValueError('Different original caption donor required')
                wrong = alternatives[i % len(alternatives)]; one = {k: value[i:i+1] for k, value in batch.items()}
                nll = {}
                for condition, pixels in [('proper', one['pixel_values']), ('blank', torch.zeros_like(one['pixel_values'])), ('wrong', batch['pixel_values'][wrong:wrong+1])]:
                    nll[condition] = float(actor(**dict(one, pixel_values=pixels))['loss'])
                image_feature, text_feature = view.features(data['caption_ids'][i:i+1], data['caption_mask'][i:i+1], one['pixel_values'])
                images.append(image_feature); texts.append(text_feature)
                records.append(dict(id=row['id'], source_group=row['group_id'], source_pixel_sha256=row['model_pixel_sha256_448'],
                    native_input_ids=one['input_ids'][0].tolist(), native_labels=one['labels'][0].tolist(), native_EOS_targets=int((one['labels'][:, 1:] != -100).sum()),
                    wrong_source_id=data['source_rows'][wrong]['id'], native_CE=nll))
            image_features = torch.cat(images).float(); text_features = torch.cat(texts).float()
            scores = F.normalize(image_features, dim=-1) @ F.normalize(text_features, dim=-1).T
            positive = data['positive']; image_ok = positive[torch.arange(len(scores)), scores.argmax(1)]
            text_ok = positive[scores.argmax(0), torch.arange(len(scores))]
            embeddings[role] = dict(images=image_features, texts=text_features, positive=positive)
            photo_results[role] = dict(records=records, image_to_text_correct=int(image_ok.sum()), text_to_image_correct=int(text_ok.sum()),
                images=len(records), positive_pair_chance=float(positive.float().mean()), pointing_projection_matching_not_native_caption_generation=True)
            atomic_json(REPORTS / (label + '-photo-assessment-prefix.json'), photo_results)
        word_results = {}
        for role, data, positions in [('train', words, word_positions), ('dev', held_words, dev_word_positions)]:
            records = []
            for position in positions:
                batch = data.batch([position], 'cpu'); logits = actor.base.word_logits(batch['word_pixels'])
                ctc = ctc_records(logits, batch['texts'])[0]
                native = {k: batch[k] for k in ['input_ids', 'labels', 'attention_mask', 'pixel_values']}
                records.append(dict(position=position, id=data.rows[position]['id'], expected=batch['texts'][0],
                    source_word_pixels_sha256=data.rows[position]['word_input_pixels_sha256'],
                    actual_byte_CTC=ctc, native_response_EOS_CE=float(actor(**native)['loss'])))
            word_results[role] = dict(records=records, complete_words=len(records), CTC_exact=sum(r['actual_byte_CTC']['exact'] for r in records),
                UTF8_byte_error_rate=sum(r['actual_byte_CTC']['byte_edits'] for r in records) / sum(r['actual_byte_CTC']['reference_bytes'] for r in records))
            atomic_json(REPORTS / (label + '-word-assessment-prefix.json'), word_results)
        torch.save(embeddings, REPORTS / (label + '-actual-matching-embeddings.pt'))
        result = dict(photo=photo_results, words=word_results, development_data_not_final_independent_assessment=True, model_updates=0, goal_achieved=False)
        atomic_json(REPORTS / (label + '-assessment-complete.json'), result); return result
    before = assess('initial')
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False); metrics = []
    def step(label):
        train = photo_data['train']
        photo = view.objective(train['batch'], train['caption_ids'], train['caption_mask'], train['positive'])
        result = view.backward_with_replay(photo, words, word_positions)
        result['gradient_norm_before_clip'] = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
        optimizer.step(); optimizer.zero_grad(set_to_none=True); metrics.append(result)
        if fixed_digest(view) != frozen: raise ValueError('Any original fixed text/actor/critic/head value changed')
        with torch.no_grad():
            if not torch.equal(text_before, actor(probe, attention_mask=probe_mask)['logits']): raise ValueError('Text-only native computation changed')
        atomic_json(REPORTS / (label + '-actual-native-update.json'), result)
        atomic_json(REPORTS / 'progress.json', dict(actual_logical_updates=len(metrics), planned_updates=2, latest=result, goal_achieved=False))
        print(json.dumps(dict(phase='actual-full-size-vision-replay-update', label=label, logical_updates=len(metrics), objective=result['actual_summed_objective'])), flush=True)
        return result
    step('first'); first = OUTPUT / 'after-first-private.pt'
    atomic_checkpoint(first, snapshot(view, optimizer, completed=1, planned=2, sources=native_sources, settings=settings, metrics=metrics), 1024**3, 26 * 1024**3)
    second = step('expected-second'); expected_weights = {n: p.detach().clone() for n, p in selected}; expected_adam = copy.deepcopy(optimizer.state_dict()); expected_rng = capture_rng('cpu')
    raw = torch.load(first, map_location='cpu', weights_only=True, mmap=True); bad = copy.deepcopy(raw); bad['sources']['vision_replay_run_plan_sha256'] = 'f' * 64
    versions = [(id(p), p._version, p.data_ptr()) for p in actor.parameters()]
    try: restore(view, optimizer, bad, planned=2, expected_sources=native_sources, expected_settings=settings)
    except ValueError: pass
    else: raise ValueError('Wrong actual replay source must reject before mutation')
    if versions != [(id(p), p._version, p.data_ptr()) for p in actor.parameters()] or not equal(expected_rng, capture_rng('cpu')):
        raise ValueError('Rejected recovery mutated model or private RNG')
    _, metrics = restore(view, optimizer, raw, planned=2, expected_sources=native_sources, expected_settings=settings)
    replayed = step('replayed-second')
    if (not equal(second, replayed) or any(not torch.equal(p, expected_weights[n]) for n, p in selected)
            or not equal(expected_adam, optimizer.state_dict()) or not equal(expected_rng, capture_rng('cpu'))):
        raise ValueError('Actual full-size next native update/Adam/private CPU RNG must replay exactly')
    del expected_weights, expected_adam, raw, bad
    final_state = OUTPUT / 'final-private.pt'
    atomic_checkpoint(final_state, snapshot(view, optimizer, completed=2, planned=2, sources=native_sources, settings=settings, metrics=metrics), 1024**3, 26 * 1024**3)
    changed = [n for n, p in selected if not torch.equal(initial_selected[n], p)]
    if not changed or len(optimizer.state) != len(selected) or any(float(s['step']) != 2 for s in optimizer.state.values()): raise ValueError('Every actual selected clock2 and changed values required')
    after = assess('trained')
    with torch.no_grad():
        expected_native = dict(text=tensor_digest([('logits', actor(probe, attention_mask=probe_mask)['logits'])]),
            image=tensor_digest([('logits', actor(probe, pixel_values=photo_data['train']['batch']['pixel_values'], attention_mask=probe_mask)['logits'])]),
            words=tensor_digest([('logits', actor.base.word_logits(words.batch(word_positions)['word_pixels']))]))
    native = OUTPUT / 'native-student.pt'
    atomic_checkpoint(native, export_bundle(actor, sources=native_sources, completed=2, full_size_student=True), 6 * 1024**3, 26 * 1024**3)
    del optimizer, parameters, selected, actor, view, base, initial_selected; gc.collect()
    cold, meta = load_bundle(native, expected_checkpoint_sha256=file_hash(native), expected_sources=native_sources)
    with torch.no_grad():
        actual_native = dict(text=tensor_digest([('logits', cold(probe, attention_mask=probe_mask)['logits'])]),
            image=tensor_digest([('logits', cold(probe, pixel_values=photo_data['train']['batch']['pixel_values'], attention_mask=probe_mask)['logits'])]),
            words=tensor_digest([('logits', cold.base.word_logits(words.batch(word_positions)['word_pixels']))]))
    if actual_native != expected_native or torch.cuda.is_initialized(): raise ValueError('Self-contained cold text/image/word logits must be bit exact with no CUDA')
    guard.check(full=True)
    artifact_bindings = dict(bindings)
    for path in [*REPORTS.glob('*'), *OUTPUT.glob('*')]:
        if path.is_file(): artifact_bindings[str(path)] = file_hash(path)
    atomic_json(REPORTS / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), actual_counts=cold.counts(), actual_full_size_student=True,
        logical_updates=2, physical_optimizer_calls=3, actual_selected_changed_tensors=changed,
        actual_original_train_photos=2, actual_original_train_words=2, actual_dev_photos=8, actual_dev_words=4,
        next_update_weights_Adam_private_CPU_RNG_bitexact=True, wrong_replay_source_restore_rejected_before_mutation=True,
        all_fixed_text_actor_critic_other_heads_unchanged=True, actual_text_only_logits_bitexact_after_updates=True,
        complete_self_contained_native_cold_text_image_word_logits_bitexact=True, added_neural_values=0,
        native_checkpoint_sha256=file_hash(native), source_files_sha256=artifact_bindings,
        actual_initial_assessment='initial-assessment-complete.json', actual_final_assessment='trained-assessment-complete.json',
        independent_state_pixel_embedding_and_outcome_readback_pending=True,
        tiny_development_cohorts_not_broad_visual_or_independent_final_quality=True, full_new_GPU_admission=False,
        separation_is_known_group_and_within_protocol_pixel_matching_not_full_semantic_or_ancestry_independence=True,
        complete_source_epoch_or_human_RLHF=False, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        import traceback
        target = REPORTS / 'failure.json' if REPORTS.exists() else ROOT / 'reports/own-actor-vision-replay-full-CPU-v1-preflight-failure.json'
        atomic_json(target, dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False)); raise
