"""Independent raw-state, matching/CTC outcomes and actual native IO readback.

Reconstruct the trained model from its own6144 source plus the selected private
state, then compare a fresh self-contained bundle load. Repeat the actual word
readout forwards at both endpoints. Caption losses remain producer observations;
their complete same-case weighted aggregates are independently rederived.
"""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from slm_lab.code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.fast_joint_inference_v2 import load as load_base
from slm_lab.own_actor_vision_replay_CPU_v1 import OwnActorVisionReplayCPU
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.own_vision_replay_bundle_CPU_v1 import load as load_bundle
from slm_lab.own_vision_replay_state_CPU_v1 import fixed_digest
from slm_lab.shared_word_data_v1 import SharedWordData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/own-actor-vision-replay-full-CPU-v1'
CHECKPOINTS = ROOT / 'checkpoints/own-actor-vision-replay-full-CPU-v1'
OUT = ROOT / 'reports/own-actor-vision-replay-full-CPU-v1-readback'


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        new = [i]
        for j, right in enumerate(b, 1): new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (left != right)))
        row = new
    return row[-1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden full-size native readback required')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    complete_path = RUN / 'complete.json'; complete = json.loads(complete_path.read_text())
    plan_path = RUN / 'plan.json'; plan = json.loads(plan_path.read_text()); plan_sha = file_hash(plan_path)
    cohort = json.loads((RUN / 'source-cohort-before-neural-loading.json').read_text())
    if (complete['logical_updates'] != 2 or complete['physical_optimizer_calls'] != 3
            or complete['actual_counts']['actual_unique'] != 999706442 or plan['actual_selected_tensors'] != 130
            or not complete['complete_self_contained_native_cold_text_image_word_logits_bitexact']):
        raise ValueError('Complete actual full-size two-update native replay proof required')
    for name, digest in complete['source_files_sha256'].items():
        if file_hash(name) != digest: raise ValueError('Original complete vision/word evidence changed')
    source_checkpoint = Path(plan['own_source_checkpoint']); source_plan = ROOT / 'reports/phase-joint-v3/plan.json'
    native = CHECKPOINTS / 'native-student.pt'; raw = torch.load(native, map_location='cpu', weights_only=True, mmap=True)
    first = torch.load(CHECKPOINTS / 'after-first-private.pt', map_location='cpu', weights_only=True, mmap=True)
    final = torch.load(CHECKPOINTS / 'final-private.pt', map_location='cpu', weights_only=True, mmap=True)
    sources = raw['complete_actor']['sources']
    if sources['vision_replay_run_plan_sha256'] != plan_sha or raw['complete_base']['sources'] != sources:
        raise ValueError('Exact native source plan and all own source bindings required')
    base, _ = load_base(source_checkpoint, source_plan, 'step-6144',
        expected_checkpoint_sha256=plan['source_files_sha256'][str(source_checkpoint)],
        expected_plan_sha256=plan['source_files_sha256'][str(source_plan)], device='cpu')
    actor = OwnDenseActor(base, **raw['actor_settings']).eval(); view = OwnActorVisionReplayCPU(actor).eval()
    selected = view.named_selected(); selected_ids = {id(p) for _, p in selected}
    if len(selected) != 130 or sum(p.numel() for _, p in selected) != 52818498:
        raise ValueError('Every actual existing image/matching/word value must match the original selected inventory')
    if any(not torch.equal(t, actor.parts.state_dict()[n]) for n, t in raw['complete_actor']['parts'].items()):
        raise ValueError('Every original own actor/critic value must remain its exact deterministic own initialization')
    initial_frozen = fixed_digest(view)
    changed = []
    for state, count in [(first, 1), (final, 2)]:
        if (state['completed'] != count or state['planned'] != 2 or state['sources'] != sources or state['settings'] != plan['settings']
                or state['optimized_names'] != [n for n, _ in selected] or len(state['metrics']) != count
                or state['fixed_language_actor_critic_and_other_heads_sha256'] != initial_frozen):
            raise ValueError('Exact complete private selected state, source/cursor/profile/frozen weights required')
        _validate_adam(state['optimizer'], [p for _, p in selected], count); _validate_rng(state['rng'], 'cpu')
    for name, p in selected:
        value = final['selected_values'][name]
        if not bool(torch.isfinite(value).all()) or value.shape != p.shape or value.dtype != p.dtype:
            raise ValueError('Every actual selected finite tensor must have its exact original shape/dtype')
        if not torch.equal(value, p): changed.append(name)
        section, key = ('network', name) if name.startswith('vision.') else ('heads', name)
        if not torch.equal(value, raw['complete_base'][section][key]): raise ValueError('Every trained selected native/private tensor must match')
    if changed != complete['actual_selected_changed_tensors']: raise ValueError('Every changed selected tensor must independently reconstruct')
    # Compare all fixed native base values to the actual own source. Tied tables
    # are equal aliases in the live model and intentionally counted only once.
    for section, module in [('network', base.network), ('heads', base.heads)]:
        for name, live in module.named_parameters():
            if id(live) not in selected_ids and not torch.equal(live, raw['complete_base'][section][name]):
                raise ValueError('Any saved fixed original text/critic/other-head value differs')
    if not torch.equal(raw['complete_base']['network']['embedding.weight'], raw['complete_base']['network']['lm_head.weight']):
        raise ValueError('Native tied vocabulary/head values must agree')
    counted = sum(t.numel() for n, t in raw['complete_base']['network'].items() if n != 'lm_head.weight')
    counted += sum(t.numel() for t in raw['complete_base']['heads'].values()) + sum(t.numel() for t in raw['complete_actor']['parts'].values())
    if counted != 999706442: raise ValueError('Every actual native base/head/actor/critic value must count strictly under1B')
    assessments = {stage: json.loads((RUN / (stage + '-assessment-complete.json')).read_text()) for stage in ['initial', 'trained']}
    words = {role: SharedWordData(ROOT / 'data/shared-byte-words-v1', 'train' if role == 'train' else 'valid',
        'optimization' if role == 'train' else 'evaluation') for role in ['train', 'dev']}
    positions = dict(train=cohort['original_train_word_positions'], dev=cohort['original_dev_word_positions'])
    @torch.no_grad()
    def actual_word_paths(model, stage):
        for role, data in words.items():
            rows = assessments[stage]['words'][role]['records']
            for position, row in zip(positions[role], rows):
                batch = data.batch([position], 'cpu'); path = model.base.word_logits(batch['word_pixels']).argmax(-1)[0].tolist()
                if path != row['actual_byte_CTC']['path'] or batch['texts'][0] != row['expected']:
                    raise ValueError('Fresh actual native word NN forward differs from the original source/path record')
    actual_word_paths(actor, 'initial')
    with torch.no_grad():
        for name, p in selected: p.copy_(final['selected_values'][name])
    if fixed_digest(view) != initial_frozen: raise ValueError('Reconstructed selected overlay changed fixed text/actor/critic/head values')
    photo_rows = {r['id']: r for r in map(json.loads, (ROOT / 'data/alignment-curriculum-v1/contrastive.train.jsonl').read_text().splitlines())}
    selected_photos = [photo_rows[i] for i in cohort['train_photo_ids']]
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer-corpus/tokenizer.json'))
    seqs = [[1, *tokenizer.encode(r['response']).ids, 2] for r in selected_photos]
    probe = torch.zeros(2, max(map(len, seqs)), dtype=torch.long)
    for i, seq in enumerate(seqs): probe[i, :len(seq)] = torch.tensor(seq)
    mask = probe != 0; encoder = Encoder(tokenizer, actor.config, ROOT, 448)
    pixels = torch.stack([encoder.pixels(row) for row in selected_photos]); word_pixels = words['train'].batch(positions['train'])['word_pixels']
    @torch.no_grad()
    def native_probe(model):
        return dict(text=tensor_digest([('logits', model(probe, attention_mask=mask)['logits'])]),
            image=tensor_digest([('logits', model(probe, pixels, attention_mask=mask)['logits'])]),
            words=tensor_digest([('logits', model.base.word_logits(word_pixels))]))
    expected = native_probe(actor)
    del selected, actor, view, base; gc.collect()
    cold, metadata = load_bundle(native, expected_checkpoint_sha256=complete['native_checkpoint_sha256'], expected_sources=sources)
    if native_probe(cold) != expected: raise ValueError('Independent private-state reconstruction and self-contained cold NN outputs differ')
    actual_word_paths(cold, 'trained')
    summary = {}; raw_outcomes = {}
    for stage, assessment in assessments.items():
        embeddings = torch.load(RUN / (stage + '-actual-matching-embeddings.pt'), map_location='cpu', weights_only=True)
        photo_summary = {}; word_summary = {}
        for role, report in assessment['photo'].items():
            evidence = embeddings[role]; scores = F.normalize(evidence['images'].float(), dim=-1) @ F.normalize(evidence['texts'].float(), dim=-1).T
            positive = evidence['positive']; image_correct = int(positive[torch.arange(len(scores)), scores.argmax(1)].sum())
            text_correct = int(positive[scores.argmax(0), torch.arange(len(scores))].sum())
            if image_correct != report['image_to_text_correct'] or text_correct != report['text_to_image_correct']:
                raise ValueError('Every original matching rank/aggregate must rederive from retained actual embeddings')
            rows = report['records']; targets = sum(r['native_EOS_targets'] for r in rows)
            means = {condition: sum(r['native_CE'][condition] * r['native_EOS_targets'] for r in rows) / targets for condition in ['proper', 'blank', 'wrong']}
            photo_summary[role] = dict(images=len(rows), matching_image_to_text_correct=image_correct, matching_text_to_image_correct=text_correct,
                complete_native_caption_targets=targets, token_mean_native_caption_CE=means,
                blank_minus_proper_NLL=means['blank'] - means['proper'], wrong_minus_proper_NLL=means['wrong'] - means['proper'])
        for role, report in assessment['words'].items():
            byte_edits = byte_count = exact = 0; outcomes = []
            for row in report['records']:
                actual = row['actual_byte_CTC']; values = []; previous = None
                for value in actual['path']:
                    if type(value) is not int or not 0 <= value <= 256: raise ValueError('Every real140-frame native byte/blank path required')
                    if value != 256 and value != previous: values.append(value)
                    previous = value
                if len(actual['path']) != 140: raise ValueError('Whole unchanged actual word140-frame path required')
                try: decoded = bytes(values).decode('utf-8'); valid = True
                except UnicodeDecodeError: decoded = bytes(values).decode('utf-8', errors='replace'); valid = False
                expected_bytes = list(row['expected'].encode('utf-8')); edits = distance(expected_bytes, values)
                if (actual['generated_UTF8_bytes'] != values or actual['generated'] != decoded or actual['valid_UTF8'] != valid
                        or actual['byte_edits'] != edits or actual['reference_bytes'] != len(expected_bytes)
                        or actual['exact'] != (decoded == row['expected']) or actual['character_edits'] != distance(row['expected'], decoded)):
                    raise ValueError('Every real collapsed UTF8 output/edit/exact outcome must reconstruct')
                byte_edits += edits; byte_count += len(expected_bytes); exact += int(decoded == row['expected'])
                outcomes.append(dict(id=row['id'], expected=row['expected'], actual=decoded, actual_UTF8_bytes=values, exact=decoded == row['expected']))
            if exact != report['CTC_exact'] or byte_edits / byte_count != report['UTF8_byte_error_rate']:
                raise ValueError('Every full actual word exact/byte-error aggregate must rederive')
            word_summary[role] = dict(words=len(report['records']), exact=exact, byte_edits=byte_edits, reference_bytes=byte_count, byte_error_rate=byte_edits / byte_count)
            raw_outcomes[stage + ':' + role] = outcomes
        summary[stage] = dict(photo=photo_summary, words=word_summary)
    if torch.cuda.is_initialized(): raise ValueError('CPU readback must not initialize CUDA')
    OUT.mkdir(); atomic_json(OUT / 'actual-raw-CTC-word-outcomes.json', raw_outcomes)
    files = [Path(__file__), complete_path, plan_path, OUT / 'actual-raw-CTC-word-outcomes.json']
    files += [RUN / (stage + '-assessment-complete.json') for stage in assessments]
    files += [RUN / (stage + '-actual-matching-embeddings.pt') for stage in assessments]
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256={str(p): file_hash(p) for p in files},
        actual_native_values_counted=999706442, all130_selected_native_private_tensors_match=True,
        all_selected_Adam_clocks=2, all_original_fixed_base_language_actor_critic_other_heads_exact=True,
        independent_actual_private_state_reconstruction_vs_self_contained_cold_text_image_word_NN_logits_bitexact=True,
        fresh_actual_initial_and_trained_word_readout_NN_forwards=12, all_retained_actual_word_paths_reproduced=True,
        actual_matching_rankings_and_UTF8_word_edits_independently_rederived=True, actual_same_case_summaries=summary,
        caption_CE_values_are_producer_observations_independently_aggregated_not_repeated_neural_forwards=True,
        all_development_OCR_exact_scores_still_zero=all(s['words']['dev']['exact'] == 0 for s in summary.values()),
        byte_error_reduction_not_reading_accuracy_or_broad_visual_quality=True,
        development_cohorts_only_not_complete_curriculum_or_independent_final_benchmark=True,
        new_training_updates_by_reader=0, new_GPU_admission=False, candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_values=999706442, independent_actual_native_reload=True,
        summaries=summary)), flush=True)


if __name__ == '__main__': main()
