"""All original AI2D test questions, with explicit native image interventions.

Evaluation only. Original choices, duplicate/null options and gold indices stay
intact. Correct, gray and source-fixed wrong diagrams use the same whole prompt
and 32-token greedy budget. This font/native448 protocol is not a published
leaderboard score or a semantic/ancestral contamination certificate.
"""
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from tokenizers import Tokenizer

from .ai2d_native_eval_inputs_v1 import PROTOCOL, native_prompt, source_gold
from .corpus import file_hash
from .diverse_images import atomic_json

FORMAT = 'own-all3088-original-AI2D-native-intervention-evaluation-v1'
CONDITIONS = ('proper_image', 'gray_image', 'wrong_image')
CANDIDATES = (('spaced_letter', ' '), ('direct_letter', ''))
BUDGET = 32


def dispositions(inventory):
    counts = Counter()
    for row in inventory:
        if not row['original_image_available']:
            if row['image_id'] != 4325 or row['original_question_count'] or row['native_variants']:
                raise ValueError('Only the explicit absent publisher ID4325 may lack an image')
            counts['absent_image'] += 1
        elif row['original_question_file_sha256'] is None:
            if row['original_question_count']: raise ValueError('Questions require a publisher question file')
            counts['present_without_question_file'] += 1
        elif not row['original_question_count']:
            counts['present_with_empty_question_file'] += 1
        else:
            counts['present_with_nonempty_questions'] += 1
    return dict(counts)


def wrong_images(cases):
    """Fixed cyclic replacement, same ABC role, distinct source and pixels.

    Membership depends only on source order, diagram identity and input hashes.
    Gold labels and question/answer text never affect the replacement selection.
    """
    result = {}
    for index, case in enumerate(cases):
        for offset in range(1, len(cases)):
            other = cases[(index + offset) % len(cases)]
            if (other['image_id'] != case['image_id'] and other['original_ABC_labels'] == case['original_ABC_labels']
                    and other['model_input_tensor_sha256_448'] != case['model_input_tensor_sha256_448']):
                result[case['id']] = other; break
        else: raise ValueError('Every question requires a distinct same-ABC-role wrong source diagram')
    return result


def load_pixels(case):
    with Image.open(case['image']) as image:
        if image.mode != 'RGB' or image.size != (448, 448): raise ValueError('Whole prepared RGB448 image required')
        rgb = np.asarray(image, dtype=np.uint8).copy()
    value = np.ascontiguousarray((rgb.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1))
    if (hashlib.sha256(rgb.tobytes()).hexdigest() != case['model_pixel_sha256_448']
            or hashlib.sha256(value.tobytes()).hexdigest() != case['model_input_tensor_sha256_448']):
        raise ValueError('Actual supplied image pixels must match their publisher-derived native input')
    return torch.from_numpy(value).unsqueeze(0)


def gray_pixels():
    rgb = np.full((448, 448, 3), 128, dtype=np.uint8)
    value = np.ascontiguousarray((rgb.astype(np.float32) / 127.5 - 1).transpose(2, 0, 1))
    return torch.from_numpy(value).unsqueeze(0)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def native_inputs(tokenizer, case):
    if (case['split'] != 'test' or case['purpose'] != 'evaluation_only' or case['admitted_to_training'] is not False
            or case['protocol'] != PROTOCOL or case['original_options_not_pruned'] is not True
            or case['input_gold_labels_or_annotations_exposed_as_text'] is not False):
        raise ValueError('Whole original evaluation-only question with no gold/annotation hint required')
    prompt = native_prompt(case['question'], case['options']); tokens = [1, *tokenizer.encode(prompt).ids]
    if (prompt != case['whole_native_prompt'] or tokens != case['complete_native_prompt_ids']
            or case['native_generation_budget'] != BUDGET or len(tokens) + BUDGET > 4096):
        raise ValueError('Exact whole preflighted original prompt and unchanged32-action budget required')
    return prompt, tokens


class FullAI2D:
    purpose = 'evaluation_only'
    optimization_forbidden = True

    def __init__(self, root):
        self.root = Path(root).resolve(); folder = self.root / 'data/ai2d-all-original-test-eval-v2'
        manifest_path = folder / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
        admission_path = folder / 'evaluation-input-reconstruction-admission.json'
        admission = json.loads(admission_path.read_text())
        reader_path = self.root / 'reports/ai2d-all-original-test-eval-v2-readback/complete.json'
        reader = json.loads(reader_path.read_text())
        if (manifest['protocol'] != PROTOCOL or manifest['all_original_test_questions'] != 3088
                or manifest['admitted_to_training'] is not False or admission['training_admitted'] is not False
                or admission['input_reconstruction_admitted_for_evaluation_only'] is not True
                or file_hash(manifest_path) != admission['manifest_sha256']
                or file_hash(reader_path) != admission['independent_reader_sha256']
                or reader['all_original_questions_options_order_gold_ambiguities_prompt_ids_and_image_pixels_rederived'] is not True):
            raise ValueError('Complete reconstructed original publisher inputs, private evaluation only, required')
        self.bindings = dict(manifest['source_files_sha256'])
        self.bindings.update({str(folder / name): sha for name, sha in manifest['artifact_metadata_sha256'].items()})
        self.bindings.update({str(p): file_hash(p) for p in [manifest_path, admission_path, reader_path, Path(__file__)]})
        for path, digest in self.bindings.items():
            if file_hash(path) != digest: raise ValueError('Every original prepared source binding must match')
        self.tokenizer_path = self.root / 'data/tokenizer-corpus/tokenizer.json'
        self.tokenizer_sha256 = file_hash(self.tokenizer_path)
        if self.tokenizer_sha256 != manifest['tokenizer_sha256']: raise ValueError('Exact own tokenizer required')
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.cases = list(map(json.loads, (folder / 'questions.jsonl').open()))
        inventory = list(map(json.loads, (folder / 'image-inventory.jsonl').open()))
        expected_dispositions = dict(present_with_nonempty_questions=814, present_with_empty_question_file=91,
            present_without_question_file=76, absent_image=1)
        self.dispositions = dispositions(inventory)
        if (len(self.cases) != 3088 or len({c['id'] for c in self.cases}) != 3088 or len(inventory) != 982
                or len({c['image_id'] for c in self.cases}) != 814 or self.dispositions != expected_dispositions
                or sum(c['original_ABC_labels'] for c in self.cases) != 1184):
            raise ValueError('All3088 original questions and all982 exact publisher ID dispositions required')
        by_image = {r['image_id']: r for r in inventory}; expected_ids = set()
        for image in inventory:
            if not image['original_image_available']: continue
            self.bindings[image['original_image']] = image['original_image_sha256']
            for key, subdir in [('original_question_file_sha256', 'original-question-files'),
                                ('original_annotation_file_sha256', 'original-annotation-files')]:
                if image[key] is not None:
                    path = folder / subdir / (image['image_name'] + '.json')
                    self.bindings[str(path)] = image[key]
                    if file_hash(path) != image[key]: raise ValueError('Unchanged original question/annotation bytes required')
                    if key == 'original_question_file_sha256':
                        raw = json.loads(path.read_text())
                        if len(raw['questions']) != image['original_question_count']: raise ValueError('Exact original question count required')
                        expected_ids.update('ai2d-original-test:' + r['questionId'] for r in raw['questions'].values())
            for variant in image['native_variants'].values():
                # Fresh reconstruction is in the bound reader; actual pixel
                # tensors are checked again immediately before each NN input.
                self.bindings[variant['path']] = file_hash(variant['path'])
        if expected_ids != {c['id'] for c in self.cases}: raise ValueError('No original scored question may be skipped or invented')
        max_tokens = 0
        for case in self.cases:
            raw = case['original_question_record']; source = by_image[case['image_id']]
            original = json.loads((folder / 'original-question-files' / (source['image_name'] + '.json')).read_text())
            if original['questions'].get(case['question']) != raw: raise ValueError('Exact original question record required')
            variant = source['native_variants']['abc_opaque' if raw['abcLabel'] else 'original']
            if (case['options'] != raw['answerTexts'] or case['answer_index'] != source_gold(raw)
                    or case['image'] != variant['path'] or case['original_ABC_labels'] != raw['abcLabel']
                    or case['original_question_id'] != raw['questionId']
                    or any(case[k] != variant[k] for k in ['model_pixel_sha256_448', 'model_input_tensor_sha256_448'])
                    or case['exact_duplicate_equivalent_answer_indices'] != [i for i, s in enumerate(case['options']) if s == case['options'][case['answer_index']]]):
                raise ValueError('Original options, gold index, ambiguity and image membership must reconstruct')
            _, ids = native_inputs(self.tokenizer, case); max_tokens = max(max_tokens, len(ids))
        if max_tokens != manifest['maximum_complete_native_prompt_tokens']: raise ValueError('Complete native context preflight changed')
        self.wrong = wrong_images(self.cases)
        # The older reader labels905 existing question FILES as question-bearing;
        # 91 of those files are empty. The actual rows/manifest correctly use814.
        self.legacy_disposition_label_correction = dict(original_questions=3088, diagrams_with_questions=814,
            existing_question_files=905, empty_question_files=91, present_no_question_file=76,
            absent_source_IDs=[4325], no_original_scored_case_changed=True)
        self.identity_sha256 = hashlib.sha256(json.dumps(dict(protocol=PROTOCOL, cases=self.cases,
            wrong={k: v['id'] for k, v in self.wrong.items()}, conditions=CONDITIONS,
            dispositions=self.dispositions), sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def verdict(case, text, scores):
    n = len(case['options']); gold = case['answer_index']
    if type(gold) is not int or not 0 <= gold < n: raise ValueError('Original ordered gold index required')
    found = re.match(r'\s*(?:(?:the\s+)?answer(?:\s+is|:)\s*)?([A-Z])\b', text, re.I)
    prediction = ord(found[1].upper()) - 65 if found else None
    if prediction is not None and prediction >= n: prediction = None
    equivalent = [i for i, option in enumerate(case['options']) if option == case['options'][gold]]
    result = dict(generated_index=prediction, answer_parseable=prediction is not None,
        strict_correct=prediction == gold, exact_duplicate_equivalent_correct=prediction in equivalent,
        original_gold_index=gold, exact_duplicate_equivalent_answer_indices=equivalent,
        strict_chance=1 / n, duplicate_equivalent_chance=len(equivalent) / n)
    for name, _ in CANDIDATES:
        values = scores[name]
        if len(values) != n or any(not math.isfinite(v) or v > 0 for v in values):
            raise ValueError('Every unchanged original letter requires a finite full-vocabulary log probability')
        index = max(range(n), key=lambda i: values[i])
        result[name + '_index'] = index; result[name + '_strict_correct'] = index == gold
        result[name + '_exact_duplicate_equivalent_correct'] = index in equivalent
    return result


@torch.inference_mode()
def evaluate_case(model, tokenizer, case, wrong):
    params = list(model.parameters()); device = params[0].device
    if (model.training or any(p.grad is not None for p in params) or tokenizer.get_vocab_size() != model.config.vocab_size
            or [tokenizer.token_to_id(t) for t in ['<pad>', '<bos>', '<eos>']] != [0, 1, 2]):
        raise ValueError('Exact native own evaluation model/tokenizer and zero gradients required')
    prompt, tokens = native_inputs(tokenizer, case)
    if len(tokens) + BUDGET > model.config.max_text_length: raise ValueError('Whole prompt and unchanged budget must fit')
    if (wrong['image_id'] == case['image_id'] or wrong['original_ABC_labels'] != case['original_ABC_labels']
            or wrong['model_input_tensor_sha256_448'] == case['model_input_tensor_sha256_448']):
        raise ValueError('Distinct predeclared same-ABC-role source diagram required')
    ids = torch.tensor([tokens], device=device)
    images = dict(proper_image=load_pixels(case), gray_image=gray_pixels(), wrong_image=load_pixels(wrong))
    result = {}
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    for condition, pixels in images.items():
        actual_input_sha = tensor_sha(pixels); pixels = pixels.to(device)
        with amp():
            initial = model.prefill(ids, pixel_values=pixels); scores = {}; evidence = {}
            for name, prefix in CANDIDATES:
                values = []; rows = []
                for index in range(len(case['options'])):
                    candidate = tokenizer.encode(prefix + chr(65 + index)).ids
                    if not candidate: raise ValueError('Whole native answer letter must have actual own tokens')
                    state = initial; logps = []
                    for position, token in enumerate(candidate):
                        logps.append(float(state['logits'][0, -1].float().log_softmax(-1)[token]))
                        if position + 1 < len(candidate): state = model.decode(torch.tensor([[token]], device=device), state['cache'])
                    values.append(math.fsum(logps)); rows.append(dict(candidate_ids=candidate, per_token_logprobs=logps))
                scores[name] = values; evidence[name] = rows
            state = initial; actions = []; greedy_logps = []
            for position in range(BUDGET):
                distribution = state['logits'][0, -1].float(); token = int(distribution.argmax())
                greedy_logps.append(float(distribution.log_softmax(-1)[token])); actions.append(token)
                if token == 2: break
                if position + 1 < BUDGET: state = model.decode(torch.tensor([[token]], device=device), state['cache'])
        text = tokenizer.decode(actions)
        result[condition] = dict(actual_model_input_tensor_sha256_448=actual_input_sha,
            supplied_image_id=None if condition == 'gray_image' else (case if condition == 'proper_image' else wrong)['image_id'],
            actual_generated_ids=actions, actual_generated_response=text, actual_greedy_token_logprobs=greedy_logps,
            genuine_generated_EOS=actions[-1] == 2, choice_scores=scores, candidate_token_evidence=evidence,
            verdict=verdict(case, text, scores))
        del initial, state
    return dict(id=case['id'], original_question_id=case['original_question_id'], image_id=case['image_id'],
        protocol=PROTOCOL, original_prompt=prompt, original_native_prompt_ids=tokens,
        original_native_prompt_ids_sha256=hashlib.sha256(np.asarray(tokens, dtype='<u2').tobytes()).hexdigest(),
        generation_budget=BUDGET, original_ABC_labels=case['original_ABC_labels'], source_category=case['source_category'],
        wrong_source_question_id=wrong['id'], conditions=result,
        inference_device_type=device.type, inference_precision='BF16_autocast' if device.type == 'cuda' else 'FP32',
        original_options_not_pruned=True, input_gold_labels_or_annotations_exposed_as_text=False,
        text_image_and_semantic_exposure_audit_pending=True, optimization_forbidden=True, model_updates=0)


def verify_record(tokenizer, case, wrong, record):
    prompt, tokens = native_inputs(tokenizer, case)
    if (record['id'] != case['id'] or record['original_question_id'] != case['original_question_id']
            or record['image_id'] != case['image_id'] or record['protocol'] != PROTOCOL
            or record['original_prompt'] != prompt or record['original_native_prompt_ids'] != tokens
            or record['original_native_prompt_ids_sha256'] != hashlib.sha256(np.asarray(tokens, dtype='<u2').tobytes()).hexdigest()
            or record['generation_budget'] != BUDGET or record['wrong_source_question_id'] != wrong['id']
            or record['original_ABC_labels'] != case['original_ABC_labels'] or record['source_category'] != case['source_category']
            or record['optimization_forbidden'] is not True or record['model_updates'] != 0
            or record['original_options_not_pruned'] is not True or record['input_gold_labels_or_annotations_exposed_as_text'] is not False
            or record['text_image_and_semantic_exposure_audit_pending'] is not True or set(record['conditions']) != set(CONDITIONS)):
        raise ValueError('Every original whole prompt, source diagram and fixed native protocol must rederive')
    for condition in CONDITIONS:
        row = record['conditions'][condition]; actions = row['actual_generated_ids']
        expected_sha = tensor_sha(gray_pixels()) if condition == 'gray_image' else (case if condition == 'proper_image' else wrong)['model_input_tensor_sha256_448']
        expected_id = None if condition == 'gray_image' else (case if condition == 'proper_image' else wrong)['image_id']
        logps = row['actual_greedy_token_logprobs']
        if (row['actual_model_input_tensor_sha256_448'] != expected_sha or row['supplied_image_id'] != expected_id
                or not 1 <= len(actions) <= BUDGET or (actions[-1] != 2 and len(actions) != BUDGET) or 2 in actions[:-1]
                or any(type(t) is not int or not 0 <= t < tokenizer.get_vocab_size() for t in actions)
                or row['actual_generated_response'] != tokenizer.decode(actions) or row['genuine_generated_EOS'] != (actions[-1] == 2)
                or len(logps) != len(actions) or any(not math.isfinite(p) or p > 0 for p in logps)):
            raise ValueError('Actual supplied image and complete first-EOS or full-budget greedy actions required')
        if set(row['choice_scores']) != {n for n, _ in CANDIDATES} or set(row['candidate_token_evidence']) != {n for n, _ in CANDIDATES}:
            raise ValueError('Both fixed full-vocabulary letter likelihood protocols required')
        for name, prefix in CANDIDATES:
            evidence = row['candidate_token_evidence'][name]; values = row['choice_scores'][name]
            if len(evidence) != len(case['options']) or len(values) != len(evidence): raise ValueError('Every original ordered option required')
            for index, item in enumerate(evidence):
                ids = tokenizer.encode(prefix + chr(65 + index)).ids; probs = item['per_token_logprobs']
                if (item['candidate_ids'] != ids or len(probs) != len(ids) or any(not math.isfinite(p) or p > 0 for p in probs)
                        or math.fsum(probs) != values[index]): raise ValueError('Every whole native letter score must match its retained actual token likelihoods')
        if row['verdict'] != verdict(case, row['actual_generated_response'], row['choice_scores']):
            raise ValueError('Original strict and duplicate-equivalent gold verdicts must independently rederive')


def summarize(cases, records):
    if len(records) != len(cases) or [r['id'] for r in records] != [c['id'] for c in cases]:
        raise ValueError('Exactly one source-ordered actual delivery per selected original question required')
    metrics = ('strict_correct', 'exact_duplicate_equivalent_correct', 'answer_parseable',
        'spaced_letter_strict_correct', 'spaced_letter_exact_duplicate_equivalent_correct',
        'direct_letter_strict_correct', 'direct_letter_exact_duplicate_equivalent_correct')
    def group(rows):
        result = dict(questions=len(rows), conditions={})
        for condition in CONDITIONS:
            values = [r['conditions'][condition]['verdict'] for r in rows]
            item = {name: sum(v[name] for v in values) for name in metrics}
            item['strict_accuracy'] = item['strict_correct'] / len(values) if values else None
            item['mean_strict_chance'] = math.fsum(v['strict_chance'] for v in values) / len(values) if values else None
            result['conditions'][condition] = item
        result['proper_image_interventions'] = {}
        for control in ['gray_image', 'wrong_image']:
            result['proper_image_interventions'][control] = {
                metric: dict(proper_only=sum(r['conditions']['proper_image']['verdict'][metric] and not r['conditions'][control]['verdict'][metric] for r in rows),
                    control_only=sum(not r['conditions']['proper_image']['verdict'][metric] and r['conditions'][control]['verdict'][metric] for r in rows))
                for metric in metrics if metric != 'answer_parseable'}
            result['proper_image_interventions'][control]['mean_gold_direct_letter_logprob_gain'] = (
                math.fsum(r['conditions']['proper_image']['choice_scores']['direct_letter'][r['conditions']['proper_image']['verdict']['original_gold_index']]
                    - r['conditions'][control]['choice_scores']['direct_letter'][r['conditions'][control]['verdict']['original_gold_index']] for r in rows) / len(rows) if rows else None)
        return result
    groups = defaultdict(list)
    for case, row in zip(cases, records):
        for condition in CONDITIONS:
            if row['conditions'][condition]['verdict'] != verdict(case, row['conditions'][condition]['actual_generated_response'], row['conditions'][condition]['choice_scores']):
                raise ValueError('Actual generated answer and original gold must reproduce every summary verdict')
        groups['ABC' if case['original_ABC_labels'] else 'original_image'].append(row)
        groups['duplicate_or_null_options' if len(set(case['options'])) != len(case['options']) or '{}' in case['options'] else 'unambiguous_original_options'].append(row)
    categories = defaultdict(list)
    for case, row in zip(cases, records): categories[str(case['source_category'])].append(row)
    return dict(all_original=group(records), by_input_and_original_option_stratum={k: group(v) for k, v in groups.items()},
        by_original_category={k: group(v) for k, v in categories.items()})


def evaluate_shard(model, tokenizer, population, output, *, rank, world, checkpoint_sha256, checked=lambda: None, record=lambda _: None):
    if type(population) is not FullAI2D or type(world) is not int or not 1 <= world <= 4 or type(rank) is not int or not 0 <= rank < world:
        raise ValueError('Complete original AI2D population and exact native shard required')
    if Path(output).exists(): raise ValueError('Preserve every completed original visual QA shard')
    if json.loads(tokenizer.to_str()) != json.loads(population.tokenizer_path.read_text()): raise ValueError('Actual whole own tokenizer required')
    count = sum(p.numel() for p in model.parameters())
    if not 0 < count < 10**9: raise ValueError('All loaded native student values must fit under1B')
    selected = population.cases[rank::world]; rows = []
    for case in selected:
        checked(); row = evaluate_case(model, tokenizer, case, population.wrong[case['id']])
        verify_record(tokenizer, case, population.wrong[case['id']], row); record(row); rows.append(row)
    for path, digest in population.bindings.items():
        if file_hash(path) != digest: raise ValueError('Original input evidence changed during native assessment')
    result = dict(format=FORMAT, rank=rank, world=world, protocol=PROTOCOL, checkpoint_sha256=checkpoint_sha256,
        original_population_sha256=population.identity_sha256, tokenizer_sha256=population.tokenizer_sha256,
        actual_student_parameter_values=count, inference_device_type=next(model.parameters()).device.type,
        inference_precision='BF16_autocast' if next(model.parameters()).device.type == 'cuda' else 'FP32',
        generation_budget=BUDGET, conditions=list(CONDITIONS), original_selected_ids=[c['id'] for c in selected],
        results=rows, summary=summarize(selected, rows), source_files_sha256=population.bindings,
        literal_publisher_ID_dispositions=population.dispositions, legacy_disposition_label_correction=population.legacy_disposition_label_correction,
        full_original_questions=3088, text_image_and_semantic_exposure_audit_pending=True, not_an_official_leaderboard_score=True,
        optimization_forbidden=True, model_updates=0, candidate_promoted=False, goal_achieved=False)
    atomic_json(Path(output), result); return result


def reconcile(population, paths, output):
    if type(population) is not FullAI2D or Path(output).exists(): raise ValueError('Real full population and fresh output required')
    reports = [json.loads(Path(p).read_text()) for p in paths]
    if not reports: raise ValueError('Every declared actual shard required')
    first = reports[0]; world = first['world']
    if (type(world) is not int or not 1 <= world <= 4 or len(reports) != world
            or {r['rank'] for r in reports} != set(range(world))): raise ValueError('Every distinct native image shard required')
    indexed = {}
    for report in reports:
        selected = population.cases[report['rank']::world]; lookup = {c['id']: c for c in selected}
        if (report['format'] != FORMAT or report['protocol'] != PROTOCOL or report['original_population_sha256'] != population.identity_sha256
                or report['tokenizer_sha256'] != population.tokenizer_sha256 or report['source_files_sha256'] != population.bindings
                or report['generation_budget'] != BUDGET or report['conditions'] != list(CONDITIONS)
                or any(report[k] != first[k] for k in ['world', 'checkpoint_sha256', 'actual_student_parameter_values', 'inference_device_type', 'inference_precision'])
                or report['original_selected_ids'] != [c['id'] for c in selected] or report['summary'] != summarize(selected, report['results'])):
            raise ValueError('Every full original native image/text/score protocol must match')
        for row in report['results']:
            if row['id'] in indexed: raise ValueError('Repeated original diagram question')
            case = lookup[row['id']]; verify_record(population.tokenizer, case, population.wrong[case['id']], row); indexed[row['id']] = row
    rows = [indexed[c['id']] for c in population.cases]
    result = dict(format=FORMAT, complete=True, protocol=PROTOCOL, original_population_sha256=population.identity_sha256,
        checkpoint_sha256=first['checkpoint_sha256'], actual_student_parameter_values=first['actual_student_parameter_values'],
        inference_device_type=first['inference_device_type'], inference_precision=first['inference_precision'],
        generation_budget=BUDGET, conditions=list(CONDITIONS), original_questions=3088, actual_question_condition_assessments=len(rows) * 3,
        results=rows, summary=summarize(population.cases, rows), source_files_sha256={str(p): file_hash(p) for p in paths},
        literal_publisher_ID_dispositions=population.dispositions, legacy_disposition_label_correction=population.legacy_disposition_label_correction,
        text_image_and_semantic_exposure_audit_pending=True, not_an_official_leaderboard_score=True,
        optimization_forbidden=True, model_updates=0, candidate_promoted=False, goal_achieved=False)
    atomic_json(Path(output), result); return result
