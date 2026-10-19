"""Actual full-size own alignment/vision graph and disposable CPU recovery.

Uses complete source training rows. This is a mechanics/capacity probe, not a
production alignment run, RLHF outcome, OCR score or broad capability result.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import unicodedata

import ahocorasick
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
from slm_lab.code_tasks import TASKS
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_aligned_assistant_v1 import OwnAlignedAssistant, byte_ctc_loss
from slm_lab.own_alignment_checkpoint_v1 import make_state, restore_state
from slm_lab.own_onpolicy_inference_v1 import load, attach_original_head
from slm_lab.point_grounding_data_v1 import PointerData
from slm_lab.point_grounding_evaluation_v1 import state_digest
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/own-aligned-assistant-full-CPU-v1'
GROUPS = ('language', 'vision', 'reward', 'word')
RESERVE = 26 * 1024**3
MAX_SNAPSHOT = 256 * 1024**2


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def whole_sequence(encoder, row, field):
    prefix = [encoder.bos, *encoder.tokens(encoder.prompt(row))]
    response = [*encoder.tokens(row[field]), encoder.eos]
    if not row[field].strip() or len(prefix) + len(response) > encoder.config.max_text_length:
        raise ValueError('Whole original response and prompt must fit; no truncation is permitted')
    ids = prefix + response; labels = [-100] * len(prefix) + response
    return dict(input_ids=torch.tensor([ids]), labels=torch.tensor([labels]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.bool))


def held_questions():
    matcher = ahocorasick.Automaton(); files = {}; count = {}
    for family, name, field in [('knowledge', 'data/knowledge-eval-v1/questions.jsonl', 'question'),
        ('math', 'data/math-reasoning-v2/test.jsonl', 'question'),
        ('public_code', 'data/public-code-assessment-v1/questions.jsonl', 'specification')]:
        path = ROOT / name; files[str(path)] = file_hash(path); count[family] = 0
        for line in path.open():
            row = json.loads(line); text = normalized(row[field])
            if not text:
                raise ValueError('Every original held question must be present')
            matcher.add_word(text, family); count[family] += 1
    apps = sorted((ROOT / 'data/public-apps-publisher-test-evaluation-v1/original-test').glob('*/question.txt'))
    if len(apps) != 5000 or count != dict(knowledge=559, math=1319, public_code=175):
        raise ValueError('Complete original named held question populations required')
    for path in apps:
        files[str(path)] = file_hash(path)
        matcher.add_word(normalized(path.read_text()), 'original_APPS_test')
    count['original_APPS_test'] = len(apps); count['local_code'] = len(TASKS)
    for task in TASKS:
        matcher.add_word(normalized(task[2]), 'local_code')
    matcher.make_automaton()
    return matcher, files, count


def source_rows(encoder):
    matcher, protected, held_counts = held_questions()
    hroot = ROOT / 'data/human-dialogue-v1'
    proot = ROOT / 'data/human-preference-curriculum-v1'
    wroot = ROOT / 'data/expanded-scene-words-v1'
    source_files = {}
    for root in (hroot, proot, wroot):
        path = root / 'manifest.json'; source_files[str(path)] = file_hash(path)
        manifest = json.loads(path.read_text())
        for name, sha in manifest['files_sha256'].items():
            if file_hash(root / name) != sha:
                raise ValueError('Exact retained original public source bytes required')
            source_files[str(root / name)] = sha
    valid_human = list(map(json.loads, (hroot / 'valid.jsonl').read_text().splitlines()))
    valid_groups = {r['group_id'] for r in valid_human}
    valid_pairs = list(map(json.loads, (proot / 'preferences.valid.jsonl').read_text().splitlines()))
    valid_prompts = {normalized(r['prompt']) for r in valid_pairs}
    def outside_named_held(row, fields):
        return not any(next(matcher.iter(normalized(row[field])), None) is not None for field in fields)
    human = None
    for position, line in enumerate((hroot / 'train.jsonl').open()):
        row = json.loads(line)
        batch = whole_sequence(encoder, row, 'response')
        if (row['language'] == 'en' and batch['input_ids'].shape[1] <= 192
                and outside_named_held(row, ['prompt', 'response'])):
            if (row['split'] != 'train' or row['synthetic'] is not False or row['license'] != 'Apache-2.0'
                    or row['group_id'] in valid_groups or row['human_quality']['value'] < .5
                    or row['human_quality']['count'] < 1
                    or hashlib.sha256(row['response'].encode()).hexdigest() != row['source_text_sha256']
                    or int((batch['labels'][:, 1:] != -100).sum()) != row['target_tokens']):
                raise ValueError('Actual reviewed human training source, complete source response and protected original tree required')
            human = dict(source_position=position, row=row, batch=batch); break
    pair = None
    for position, line in enumerate((proot / 'preferences.train.jsonl').open()):
        row = json.loads(line)
        batches = {name: whole_sequence(encoder, row, name) for name in ('chosen', 'rejected')}
        if max(b['input_ids'].shape[1] for b in batches.values()) <= 192 and outside_named_held(row, ['prompt', 'chosen', 'rejected']):
            if (row['feedback_source'] != 'human' or row['original_split'] != 'train'
                    or row['revision'] != 'c72f5cee8eb7b4d2ea5617657f4430d5e333af07'
                    or row['response_origin'] != 'Responses from Anthropic models, ranked by human annotators.'
                    or normalized(row['prompt']) in valid_prompts):
                raise ValueError('Original source-train genuine human ranking and exact response-origin qualification required')
            pair = dict(source_position=position, row=row, batches=batches); break
    # Original additional human word crops are train-only; parent isolation is
    # covered by their retained manifest. Its mirror-equivalence caveat stays.
    word = None
    for position, line in enumerate((wroot / 'train.jsonl').open()):
        row = json.loads(line); raw = list(row['response'].encode())
        need = len(raw) + sum(a == b for a, b in zip(raw, raw[1:]))
        if raw and need <= 28 and outside_named_held(row, ['prompt', 'response']):
            if row['split'] != 'train' or row['original_annotation_split'] != 'train' or row['task'] != 'word_ocr':
                raise ValueError('Actual original public human word training annotation required')
            image = Path(row['image'])
            if file_hash(image) != row['stored_image_sha256']:
                raise ValueError('Actual stored original human word crop bytes required')
            source_files[str(image)] = row['stored_image_sha256']
            with Image.open(image) as picture:
                def pixels(width, height):
                    array = np.asarray(picture.convert('RGB').resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)/127.5 - 1
                    return torch.from_numpy(array.copy()).permute(2, 0, 1)[None]
                word = dict(source_position=position, row=row, CTC_pixels=pixels(448, 64),
                            native_pixels=pixels(448, 448), batch=whole_sequence(encoder, row, 'response'))
            break
    if any(row is None for row in (human, pair, word)):
        raise ValueError('All three complete real original training sources required')
    return human, pair, word, {**source_files, **protected}, held_counts


def signature(module):
    return [(id(p), p._version, p.data_ptr(), p.requires_grad) for p in module.parameters()]


def optimizer_digest(optimizer):
    digest = hashlib.sha256()
    for index, state in enumerate(optimizer.state.values()):
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            tensor = state[key].detach().cpu().contiguous()
            digest.update(json.dumps([index, key, list(tensor.shape), str(tensor.dtype)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    if REPORTS.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Fresh GPU-hidden CPU full-model probe required')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    test_log = ROOT / 'reports/own-aligned-assistant-and-recovery-v1-tests.log'
    if '6 passed' not in test_log.read_text():
        raise ValueError('Full extension and strict Adam/RNG/cursor recovery checks required')
    prior = ROOT / 'reports/own-onpolicy-pilot-v2'
    old_plan_path = prior / 'plan.json'; old_plan = json.loads(old_plan_path.read_text())
    receipt = json.loads((prior / 'CE_control-training-complete.json').read_text())
    code = ROOT / 'checkpoints/own-onpolicy-pilot-v2/CE_control/model.pt'
    reference = prior / 'reference.json'
    language, metadata = load(code, old_plan_path, 'CE_control', reference,
        expected_checkpoint_sha256=receipt['trained_components_sha256'], expected_plan_sha256=file_hash(old_plan_path),
        expected_reference_sha256=receipt['reference_scores_sha256'], device='cpu')
    original, attachment = attach_original_head(language, old_plan)
    original.eval(); encoder = Encoder(Tokenizer.from_file(old_plan['tokenizer']), original.config, ROOT)
    human, pair, word, source_files, held_counts = source_rows(encoder)
    names = ['src/slm_lab/own_aligned_assistant_v1.py', 'src/slm_lab/own_alignment_checkpoint_v1.py',
        'scripts/probe_own_aligned_assistant_full_cpu_v1.py', 'tests/test_own_aligned_assistant_v1.py',
        'tests/test_own_alignment_checkpoint_v1.py', 'reports/own-aligned-assistant-and-recovery-v1-tests.log',
        'src/slm_lab/own_code_delta_v1.py', 'src/slm_lab/own_assistant_v1.py', 'src/slm_lab/model.py',
        'src/slm_lab/own_onpolicy_inference_v1.py', 'src/slm_lab/code_residual_checkpoint_v1.py',
        'src/slm_lab/codenet_checkpoint_v1.py', 'src/slm_lab/code_tasks.py', 'src/slm_lab/training.py',
        'reports/own-onpolicy-pilot-v2/CE_control-training-complete.json',
        'reports/sub1b-alignment-vision-budget-v2/complete.json']
    files = {**source_files, **{str(ROOT / n): file_hash(ROOT / n) for n in names}}
    for path in (code, old_plan_path, reference, Path(old_plan['base_checkpoint']), Path(old_plan['tokenizer']), Path(old_plan['initial_manifest'])):
        files[str(path)] = file_hash(path)
    guard = FileBindings(files)
    REPORTS.mkdir()
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        CPU_threads=2, precision='CPU/FP32', original_code_initialization='CE_control128update experimental component; not promoted',
        actual_sources=dict(human=dict(position=human['source_position'], id=human['row']['id']),
            ranking=dict(position=pair['source_position'], id=pair['row']['id']),
            word=dict(position=word['source_position'], id=word['row']['id'])),
        logical_updates=2, physical_optimizer_calls=3, optimizer_groups=list(GROUPS), learning_rate=2e-5,
        beta=[.9, .95], weight_decay=.01, gradient_norm_clip=1.,
        scope='Disposable full-graph source-gradient and exact recovery check; no production/genuine on-policy RLHF or capability result',
        no_source_text_or_response_truncation=True, no_original_or_code_or_grounding_optimizer_parameters=True,
        matched_held_question_literal_screen_counts=held_counts, semantic_independence_not_certified=True,
        original_human_source_extraction_proofs_reused_not_fresh_full_tree_audit=True,
        word_annotation_mirror_equivalence_independently_verified=False, candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'plan.json', plan)
    print(json.dumps(dict(phase='loaded_complete_original_and_selected_whole_train_sources', elapsed=time.monotonic()-started)), flush=True)
    fixed_digest = state_digest(original); versions = signature(original)
    pointer = PointerData(ROOT / 'data/localized-pointer-candidate-v1', 'train', 'audit')
    probe_position = json.loads((ROOT / 'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())['original_source_position']
    native = pointer.batch([probe_position], 'cpu')
    ids, pixels = native['input_ids'], native['pixel_values']
    def retained():
        original.eval()
        with torch.no_grad():
            return {route: original(ids, pixels, route=route)['logits'].clone() for route in ('general', 'code', 'grounding')}
    before = retained()
    with torch.no_grad():
        old_cache = original.language.code.prefill(ids)['cache']
        old_next = original.language.code.decode(torch.tensor([[4]]), old_cache)['logits'].clone()
    rng = torch.get_rng_state().clone()
    model = OwnAlignedAssistant(original).eval()
    if (not torch.equal(rng, torch.get_rng_state()) or signature(original) != versions
            or model.counts()['total_unique'] != 999860418):
        raise ValueError('Actual complete sub1B allocation and untouched original parameters/RNG required')
    counts = model.counts()
    with torch.no_grad():
        initial = model(ids, pixels)['logits']
        initial_error = float((initial - before['general']).abs().max())
        if not torch.allclose(initial, before['general'], atol=2e-5, rtol=2e-5):
            raise ValueError('Zero factors and own copied norms must retain initial general logits within explicit FP32 rounding tolerance')
    print(json.dumps(dict(phase='actual_full_graph_instantiated', counts=counts, initial_general_max_abs_error=initial_error)), flush=True)
    sources = dict(original_base_sha256=old_plan['base_checkpoint_sha256'],
        original_code_sha256=receipt['trained_components_sha256'], tokenizer_sha256=old_plan['tokenizer_sha256'],
        original_policy_manifest_sha256=old_plan['initial_manifest_sha256'], run_plan_sha256=file_hash(REPORTS / 'plan.json'))
    initial_export = model.export(sources)
    atomic_checkpoint(REPORTS / 'initial-extensions.pt', initial_export, 64*1024**2, RESERVE)
    selected = model.extension_parameters(*GROUPS)
    optimizer = torch.optim.AdamW(selected, lr=2e-5, betas=(.9, .95), weight_decay=.01, foreach=False)
    random.seed(947133); np.random.seed(947133); torch.random.default_generator.manual_seed(947133)
    model.aligned.gradient_checkpointing = True
    def update():
        optimizer.zero_grad(set_to_none=True)
        model.aligned.train()
        draw = random.random()+float(np.random.random())+float(torch.rand(()))
        # Native teacher forcing on complete original response/EOS targets.
        sft = model(**human['batch'])['loss']
        (sft * (1 + draw*.01)).backward()
        native_loss = model(**word['batch'], pixel_values=word['native_pixels'])['loss']
        native_loss.backward()
        scores = {name: model.preference_score(batch['input_ids'], attention_mask=batch['attention_mask'])
                  for name, batch in pair['batches'].items()}
        reward = F.softplus(-(scores['chosen']-scores['rejected'])).mean()
        reward.backward()
        logits = model.word_logits(word['CTC_pixels'])
        ctc = byte_ctc_loss(logits, [word['row']['response']]); (.1*ctc).backward()
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in selected)
                or any(p.grad is not None for p in original.parameters())
                or model.parts['critic'].weight.grad is not None):
            raise ValueError('Every selected new parameter and only selected new parameters need finite gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(selected, 1., error_if_nonfinite=True))
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        record = dict(stochastic_weight_probe_draw=draw, human_SFT=float(sft.detach()), human_ranking=float(reward.detach()),
            native_word_SFT=float(native_loss.detach()), word_CTC=float(ctc.detach()), gradient_norm_before_clip=norm,
            complete_human_targets=int((human['batch']['labels'][:, 1:]!=-100).sum()),
            complete_native_word_targets=int((word['batch']['labels'][:, 1:]!=-100).sum()),
            original_word_UTF8_bytes=len(word['row']['response'].encode()), CTC_frames=logits.shape[1])
        print(json.dumps(dict(phase='disposable_update', elapsed=time.monotonic()-started, **record)), flush=True)
        return record
    first = update()
    cursor = dict(human_source_position=human['source_position'], ranking_source_position=pair['source_position'],
                  word_source_position=word['source_position'], completed_draws_each=1, next_draw_repeats_same_train_probe=True)
    state = make_state(model, optimizer, groups=GROUPS, sources=sources, completed_steps=1, planned_steps=2,
                       sampler_state=cursor, metrics=[first], device='cpu')
    snapshot = REPORTS / 'disposable-resume.pt'
    atomic_checkpoint(snapshot, state, MAX_SNAPSHOT, RESERVE)
    expected = update(); parts_after = state_digest(model.parts); adam_after = optimizer_digest(optimizer)
    restored, metrics = restore_state(model, optimizer, torch.load(snapshot, map_location='cpu', weights_only=True, mmap=True),
        groups=GROUPS, expected_sources=sources, planned_steps=2, expected_sampler_state=cursor, device='cpu')
    actual = update()
    if (restored != 1 or metrics != [first] or expected != actual or parts_after != state_digest(model.parts)
            or adam_after != optimizer_digest(optimizer)):
        raise ValueError('Actual full-size next source update, all extension weights and all Adam moments must recover exactly')
    model.eval()
    after = retained()
    with torch.no_grad():
        cached = original.language.code.decode(torch.tensor([[4]]), old_cache)['logits']
        changed_aligned = model(ids, pixels)['logits']
        changed_word = model.word_logits(word['CTC_pixels'])
    if (any(not torch.equal(before[r], after[r]) for r in before) or not torch.equal(cached, old_next)
            or fixed_digest != state_digest(original) or versions != signature(original)
            or torch.equal(initial, changed_aligned)
            or torch.cuda.is_initialized()):
        raise ValueError('Every original tensor/version/native route/cache must remain exact while real new weights change')
    trained_extension_state = model.extension_state()
    changed_families = {family: sum(not torch.equal(t, initial_export['extensions'][name])
        for name, t in trained_extension_state.items() if name.startswith(family))
        for family in ('language_', 'vision_', 'reward.', 'word.')}
    if (any(n == 0 for n in changed_families.values())
            or not torch.equal(trained_extension_state['critic.weight'], initial_export['extensions']['critic.weight'])):
        raise ValueError('Actual new language/vision/reward/word weights must change; unused separate critic stays unchanged')
    trained_path = REPORTS / 'disposable-trained-extensions.pt'
    atomic_checkpoint(trained_path, model.export(sources), 64*1024**2, RESERVE)
    # Reload new-only tensors into a fresh wrapper around the identical original
    # object; no second full backbone is constructed or loaded for the reload.
    cold = OwnAlignedAssistant(original).eval()
    cold.import_export(torch.load(trained_path, weights_only=True), expected_sources=sources)
    with torch.no_grad():
        if (not torch.equal(changed_aligned, cold(ids, pixels)['logits'])
                or not torch.equal(changed_word, cold.word_logits(word['CTC_pixels']))
                or any(not torch.equal(t, cold.parts.state_dict()[n]) for n,t in trained_extension_state.items())):
            raise ValueError('Actual full-size new-only cold reload must match trained aligned and word routes exactly')
    guard.check(full=True)
    complete = dict(completed_at=datetime.now(timezone.utc).isoformat(), actual_counts=counts,
        actual_full_original_base_objects=1, original_backbones_loaded=metadata['original_backbones_loaded'],
        existing_original_and_code_and_grounding_bytes_versions_and_native_route_logits_exact=True,
        pre_existing_original_code_cache_decode_bit_identical=True, original_state_digest=fixed_digest,
        actual_all_unique_extensions=sum(p.numel() for p in model.parts.parameters()),
        actual_changed_extension_tensor_counts=changed_families,
        actual_selected_new_parameters=sum(p.numel() for p in selected), complete_Adam_states=len(optimizer.state),
        actual_logical_disposable_updates=2, actual_physical_optimizer_calls=3,
        actual_next_update_every_extension_Adam_and_RNG_recovery_exact=True,
        all_selected_gradients_finite_and_all_original_gradients_absent=True,
        initial_new_route_vs_original_max_absolute_logit_error=initial_error,
        actual_all_new_only_tensor_cold_reload_and_aligned_and_word_routes_exact=True,
        actual_model_text_vocabulary_object_shared=True, no_external_model_or_teacher_weights_loaded=True,
        original_human_dialogue_id=human['row']['id'], original_human_ranking_id=pair['row']['id'],
        ranking_response_origin=pair['row']['response_origin'], human_rankings_not_new_labels_on_own_outputs=True,
        original_train_word_id=word['row']['id'], whole_word_transcript=word['row']['response'],
        word_ctc_image_shape=list(word['CTC_pixels'].shape), word_native_image_shape=list(word['native_pixels'].shape),
        source_sequences=dict(human={k:v.tolist() for k,v in human['batch'].items()},
            ranking={n:{k:v.tolist() for k,v in b.items()} for n,b in pair['batches'].items()},
            word={k:v.tolist() for k,v in word['batch'].items()}),
        updates=[first, expected], full_native_retention_probe_original_photo_id=pointer.rows[probe_position]['id'],
        original_critic_and_separate_new_critic_unchanged=True, PPO_or_DPO_not_run_in_this_probe=True,
        source_complete_not_truncated=True, matched_held_literal_screen_counts=held_counts,
        semantic_independence_not_certified=True, source_word_annotation_mirror_caveat_retained=True,
        artifacts={p.name:dict(bytes=p.stat().st_size,sha256=file_hash(p)) for p in
            (REPORTS/'initial-extensions.pt',snapshot,trained_path)},
        elapsed_seconds=time.monotonic()-started, CPU_threads=2, CUDA_initialized=False, production_updates=0,
        GPU_capacity_full_human_training_and_broad_capability_assessments_remaining=True,
        no_learned_capability_OR_frontier_performance_claim=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'complete.json', complete)
    print(json.dumps({k:complete[k] for k in ['actual_counts','actual_all_unique_extensions','complete_Adam_states',
        'actual_next_update_every_extension_Adam_and_RNG_recovery_exact','existing_original_and_code_and_grounding_bytes_versions_and_native_route_logits_exact',
        'elapsed_seconds','production_updates','goal_achieved']},indent=2),flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if REPORTS.exists():
            atomic_json(REPORTS / 'failure.json', dict(failed_at=datetime.now(timezone.utc).isoformat(), error=repr(error),
                        production_updates=0, goal_achieved=False))
        raise
