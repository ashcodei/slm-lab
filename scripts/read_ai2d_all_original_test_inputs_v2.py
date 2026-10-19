"""Reconcile all original AI2D test inputs, including the missing publisher ID.

No model is loaded, no gradients are computed, and training stays forbidden.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
from tokenizers import Tokenizer

from slm_lab.ai2d_native_eval_inputs_v1 import native_pixels, native_prompt, opaque_labels, source_gold
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/ai2d-all-original-test-eval-v2'
OUT = ROOT / 'reports/ai2d-all-original-test-eval-v2-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden complete input readback required')
    manifest_path = SOURCE / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
    if (manifest['all_original_test_questions'] != 3088 or manifest['admitted_to_training']
            or manifest['publisher_test_IDs_without_any_original_archive_entry'] != [4325]
            or manifest['original_test_images_physically_present'] != 981):
        raise ValueError('Complete declared original test inventory with exact missing-source disposition required')
    bindings = dict(manifest['source_files_sha256'])
    bindings.update({str(SOURCE / name): digest for name, digest in manifest['artifact_metadata_sha256'].items()})
    for path, digest in bindings.items():
        if file_hash(path) != digest: raise ValueError('Original source or prepared metadata changed')
    tokenizer = Tokenizer.from_file(str(ROOT / 'data/tokenizer-corpus/tokenizer.json'))
    font = Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf')
    rows = [json.loads(line) for line in (SOURCE / 'questions.jsonl').read_text().splitlines()]
    inventory = [json.loads(line) for line in (SOURCE / 'image-inventory.jsonl').read_text().splitlines()]
    ids = [int(line) for line in (ROOT / 'data/ai2d-source-cache-v1/ai2d_test_ids.csv').read_text().splitlines() if line.strip()]
    if [r['image_id'] for r in inventory] != sorted(ids) or len({r['id'] for r in rows}) != 3088:
        raise ValueError('Every original publisher ID and all3088 unique question IDs required')
    lookup = {r['original_question_id']: r for r in rows}; seen = set(); counts = Counter(); maximum = 0
    for image in inventory:
        if not image['original_image_available']:
            if image['image_id'] != 4325 or image['original_question_count'] != 0 or image['native_variants']:
                raise ValueError('Only publisher ID4325 is absent; no invented image/question/score permitted')
            counts['absent_archive_image'] += 1; continue
        original_path = Path(image['original_image'])
        if file_hash(original_path) != image['original_image_sha256']: raise ValueError('Every original stored test image must match source bytes')
        qpath = SOURCE / 'original-question-files' / (image['image_name'] + '.json')
        apath = SOURCE / 'original-annotation-files' / (image['image_name'] + '.json')
        if image['original_question_file_sha256'] is None:
            if qpath.exists() or image['original_question_count'] != 0: raise ValueError('No publisher questions may be fabricated')
            questions = []; counts['present_image_no_question'] += 1
        else:
            if file_hash(qpath) != image['original_question_file_sha256']: raise ValueError('Original publisher question bytes changed')
            original = json.loads(qpath.read_text()); questions = list(original['questions'].items()); counts['question_bearing_image'] += 1
            if original['imageName'] != image['image_name'] or len(questions) != image['original_question_count']: raise ValueError('Exact original diagram/question membership required')
        annotation = None
        if image['original_annotation_file_sha256'] is not None:
            if file_hash(apath) != image['original_annotation_file_sha256']: raise ValueError('Original annotation bytes changed')
            annotation = json.loads(apath.read_text())
        with Image.open(original_path) as original:
            if list(original.size) != image['original_dimensions']: raise ValueError('Whole original diagram dimensions changed')
            for variant, record in image['native_variants'].items():
                if variant == 'original': source = original
                elif variant == 'abc_opaque':
                    source, trace = opaque_labels(original, annotation, font)
                    if trace != image['actual_ABC_render_trace']: raise ValueError('Every original ABC geometry/label/font disposition must reproduce')
                else: raise ValueError('Undeclared image input variant')
                fresh, rgb_sha, tensor_sha = native_pixels(source)
                if rgb_sha != record['model_pixel_sha256_448'] or tensor_sha != record['model_input_tensor_sha256_448']:
                    raise ValueError('Every fresh original/ABC native448 pixel/tensor view must reconstruct')
                with Image.open(record['path']) as stored:
                    if stored.mode != 'RGB' or stored.size != (448, 448) or stored.tobytes() != fresh.tobytes():
                        raise ValueError('Every actual prepared native image must match the fresh publisher-derived input')
                counts['native_image_variants'] += 1
        for question, raw in questions:
            row = lookup[raw['questionId']]; variant = image['native_variants']['abc_opaque' if raw['abcLabel'] else 'original']
            prompt = native_prompt(question, raw['answerTexts']); ids = [1, *tokenizer.encode(prompt).ids]; gold = source_gold(raw)
            if (row['original_question_record'] != raw or row['question'] != question or row['options'] != raw['answerTexts']
                    or row['answer_index'] != gold or row['image_id'] != image['image_id'] or row['image'] != variant['path']
                    or row['model_pixel_sha256_448'] != variant['model_pixel_sha256_448']
                    or row['model_input_tensor_sha256_448'] != variant['model_input_tensor_sha256_448']
                    or row['whole_native_prompt'] != prompt or row['complete_native_prompt_ids'] != ids
                    or row['native_generation_budget'] != 32 or len(ids) + 32 > 4096
                    or row['admitted_to_training'] or row['purpose'] != 'evaluation_only'
                    or row['exact_duplicate_equivalent_answer_indices'] != [i for i, value in enumerate(raw['answerTexts']) if value == raw['answerTexts'][gold]]):
                raise ValueError('Every original full question/options/order/gold ambiguity and whole native prompt must reconstruct')
            seen.add(raw['questionId']); maximum = max(maximum, len(ids)); counts['questions'] += 1
            counts['ABC' if raw['abcLabel'] else 'original_image'] += 1
    if (len(seen) != 3088 or seen != set(lookup) or counts['ABC'] != 1184 or counts['original_image'] != 1904
            or counts['absent_archive_image'] != 1 or counts['present_image_no_question'] != 76 or counts['question_bearing_image'] != 905
            or maximum != manifest['maximum_complete_native_prompt_tokens']):
        raise ValueError('Every original case, native input and publisher source disposition must reconcile exactly')
    OUT.mkdir(); bindings[str(manifest_path)] = file_hash(manifest_path); bindings[str(Path(__file__))] = file_hash(__file__)
    result = dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256=bindings, complete=True,
        independently_reconstructed_original_questions=3088, independent_source_dispositions=dict(counts), maximum_whole_prompt_tokens=maximum,
        all_original_questions_options_order_gold_ambiguities_prompt_ids_and_image_pixels_rederived=True,
        all1184_ABC_inputs_use_original_geometry_replacement_labels_and_pinned_font=True,
        zero_scored_original_questions_omitted=True, absent_source_ID4325_explicit_not_an_invented_test=True,
        training_admitted=False, private_evaluation_only_under_original_packaged_license=True,
        source_text_image_and_semantic_exposure_audit_pending=True, new_model_assessment_run=False,
        model_updates=0, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUT / 'complete.json', result)
    atomic_json(SOURCE / 'evaluation-input-reconstruction-admission.json', dict(input_reconstruction_admitted_for_evaluation_only=True,
        manifest_sha256=file_hash(manifest_path), independent_reader_sha256=file_hash(OUT / 'complete.json'),
        training_admitted=False, no_independence_or_leaderboard_certificate=True, model_updates=0, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), original_questions=3088, dispositions=dict(counts), training_admitted=False)), flush=True)


if __name__ == '__main__': main()
