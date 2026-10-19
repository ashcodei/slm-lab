"""Prepare every original AI2D test question for private evaluation only.

The cached packaged license remains authoritative for this preparation:
evaluation only, no training admission, no public release/redistribution.
Every original choice survives; this declared font/input variant is not a
claim of official leaderboard protocol or contamination-free independence.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

from PIL import Image
from tokenizers import Tokenizer

from slm_lab.ai2d_native_eval_inputs_v1 import PROTOCOL, native_pixels, native_prompt, opaque_labels, source_gold
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/ai2d-all-original-test-eval-v2'
CACHE = ROOT / 'data/ai2d-source-cache-v1'
FONT = Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden evaluation-only preparation required')
    archive = CACHE / 'ai2d-all.zip'; archive_meta = json.loads((CACHE / 'ai2d-all.zip.source.json').read_text())
    ids_path = CACHE / 'ai2d_test_ids.csv'; ids_meta = json.loads((CACHE / 'ai2d_test_ids.csv.source.json').read_text())
    if (file_hash(archive) != archive_meta['sha256'] or archive.stat().st_size != archive_meta['bytes']
            or file_hash(ids_path) != ids_meta['sha256']): raise ValueError('Exact original publisher archive/test split required')
    test_values = [int(line.strip()) for line in ids_path.read_text().splitlines() if line.strip()]
    if len(test_values) != len(set(test_values)) or len(test_values) != 982: raise ValueError('Every982 original unique publisher test image ID required')
    test_ids = set(test_values); tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'; tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size() != 32768 or tokenizer.token_to_id('<bos>') != 1: raise ValueError('Original own tokenizer required')
    files = [Path(__file__), archive, CACHE / 'ai2d-all.zip.source.json', ids_path, CACHE / 'ai2d_test_ids.csv.source.json', FONT, tokenizer_path,
        ROOT / 'reports/ai2d-source-review-v1.json', ROOT / 'data/ai2d-eval-protocol-cache-v1/ai2d.py',
        ROOT / 'data/ai2d-eval-protocol-cache-v1/ai2d.py.source.json', ROOT / 'data/ai2d-eval-protocol-cache-v1/LICENSE',
        ROOT / 'tests/test_ai2d_native_eval_inputs_v1.py', ROOT / 'reports/ai2d-native-eval-inputs-v1-tests.log',
        *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_ai2d_native_eval_inputs_v1.py'])]
    bindings = {str(p): file_hash(p) for p in files}
    OUT.mkdir(); (OUT / 'original-images').mkdir(); (OUT / 'native448').mkdir(); (OUT / 'original-question-files').mkdir(); (OUT / 'original-annotation-files').mkdir()
    rows = []; inventory = []; counts = Counter(); maximum = 0; names = set(); seen_qids = set()
    with zipfile.ZipFile(archive) as z:
        members = set(z.namelist())
        license_bytes = z.read('ai2d/license.txt'); (OUT / 'original-packaged-license.txt').write_bytes(license_bytes)
        if hashlib.sha256(license_bytes).hexdigest() != json.loads((ROOT / 'reports/ai2d-source-review-v1.json').read_text())['license_sha256']:
            raise ValueError('Exact original reviewed restrictive packaged license required')
        (OUT / 'original-publisher-README.txt').write_bytes(z.read('ai2d/README.txt'))
        categories_raw = z.read('ai2d/categories.json'); (OUT / 'original-categories.json').write_bytes(categories_raw); categories = json.loads(categories_raw)
        for image_id in sorted(test_ids):
            image_name = str(image_id) + '.png'; image_member = 'ai2d/images/' + image_name
            if image_member not in members:
                related = [name for name in members if name.startswith('ai2d/') and ('/' + str(image_id) + '.') in name]
                if image_id != 4325 or related:
                    raise ValueError('Unexpected missing publisher test image or missing image with original question/annotation')
                inventory.append(dict(image_id=image_id, image_name=image_name, official_test=True,
                    original_image_available=False, original_question_count=0, native_variants={},
                    no_original_question_is_not_a_missing_or_skipped_case=True,
                    source_disposition='Official test ID4325 has no entry in the exact original archive; no question/image is fabricated.'))
                continue
            image_raw = z.read(image_member); image_path = OUT / 'original-images' / image_name; image_path.write_bytes(image_raw)
            qmember = 'ai2d/questions/' + image_name + '.json'; amember = 'ai2d/annotations/' + image_name + '.json'
            questions_raw = z.read(qmember) if qmember in members else None
            obj = json.loads(questions_raw) if questions_raw is not None else None
            if obj is not None and obj['imageName'] != image_name: raise ValueError('Exact original image/question identity required')
            questions = [] if obj is None else list(obj['questions'].items())
            if questions_raw is not None: (OUT / 'original-question-files' / (image_name + '.json')).write_bytes(questions_raw)
            annotation_raw = z.read(amember) if amember in members else None
            if annotation_raw is not None: (OUT / 'original-annotation-files' / (image_name + '.json')).write_bytes(annotation_raw)
            with Image.open(io.BytesIO(image_raw)) as original:
                original.load(); original_rgb = original.convert('RGB'); width, height = original.size
                native, pixel_sha, tensor_sha = native_pixels(original_rgb)
                original_native = OUT / 'native448' / (str(image_id) + '-original.png'); native.save(original_native)
                variants = dict(original=dict(path=str(original_native.resolve()), model_pixel_sha256_448=pixel_sha, model_input_tensor_sha256_448=tensor_sha))
                trace = None
                if any(raw['abcLabel'] for _, raw in questions):
                    if annotation_raw is None: raise ValueError('Every ABC test question needs original geometry/replacement annotations')
                    rendered, trace = opaque_labels(original_rgb, json.loads(annotation_raw), FONT)
                    abc, pixel_sha, tensor_sha = native_pixels(rendered)
                    abc_path = OUT / 'native448' / (str(image_id) + '-abc-opaque.png'); abc.save(abc_path)
                    variants['abc_opaque'] = dict(path=str(abc_path.resolve()), model_pixel_sha256_448=pixel_sha, model_input_tensor_sha256_448=tensor_sha)
            image_info = dict(image_id=image_id, image_name=image_name, official_test=True, original_dimensions=[width, height],
                original_image_available=True, original_image=str(image_path.resolve()), original_image_sha256=hashlib.sha256(image_raw).hexdigest(),
                original_question_file_sha256=hashlib.sha256(questions_raw).hexdigest() if questions_raw is not None else None,
                original_annotation_file_sha256=hashlib.sha256(annotation_raw).hexdigest() if annotation_raw is not None else None,
                original_question_count=len(questions), no_original_question_is_not_a_missing_or_skipped_case=not bool(questions),
                native_variants=variants, actual_ABC_render_trace=trace)
            inventory.append(image_info)
            for question, raw in questions:
                choices = raw['answerTexts']; gold = source_gold(raw); qid = raw['questionId']
                if qid in seen_qids or type(raw['abcLabel']) is not bool: raise ValueError('Every original question identity and ABC role must be unique/explicit')
                seen_qids.add(qid); variant = variants['abc_opaque' if raw['abcLabel'] else 'original']
                prompt = native_prompt(question, choices); ids = [1, *tokenizer.encode(prompt).ids]
                if len(ids) + 32 > 4096: raise ValueError('Every whole original question/options must fit the native32-action assessment without clipping')
                maximum = max(maximum, len(ids)); counts['questions'] += 1; counts['ABC' if raw['abcLabel'] else 'original_image'] += 1
                counts['duplicate_options'] += int(len(set(choices)) != len(choices)); counts['null_placeholder_options'] += int('{}' in choices)
                rows.append(dict(id='ai2d-original-test:' + qid, original_question_id=qid, image_id=image_id, image_name=image_name,
                    split='test', purpose='evaluation_only', admitted_to_training=False, protocol=PROTOCOL,
                    question=question, options=choices, original_question_record=raw, answer_index=gold,
                    exact_duplicate_equivalent_answer_indices=[i for i, option in enumerate(choices) if option == choices[gold]],
                    source_category=categories.get(image_name, categories.get(str(image_id))),
                    image=variant['path'], original_image_sha256=image_info['original_image_sha256'],
                    model_pixel_sha256_448=variant['model_pixel_sha256_448'], model_input_tensor_sha256_448=variant['model_input_tensor_sha256_448'],
                    original_ABC_labels=raw['abcLabel'], original_options_not_pruned=True, whole_native_prompt=prompt,
                    complete_native_prompt_ids=ids, native_generation_budget=32, input_gold_labels_or_annotations_exposed_as_text=False,
                    image_from_literal_geometry_and_replacement_labels_not_answer_values=True))
            if len(inventory) % 100 == 0: print(json.dumps(dict(original_test_images=len(inventory), complete_questions=len(rows))), flush=True)
    if len(rows) != 3088 or counts['ABC'] != 1184 or counts['original_image'] != 1904 or len(inventory) != 982:
        raise ValueError('Every original question/image/ABC population must reconcile')
    for name, values in [('questions.jsonl', rows), ('image-inventory.jsonl', inventory)]:
        with (OUT / name).open('x') as stream:
            for row in values: stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    for path, digest in bindings.items():
        if file_hash(path) != digest: raise ValueError('Original input source changed during evaluation preparation')
    atomic_json(OUT / 'manifest.json', dict(format='own-all-original-AI2D-publisher-test-private-evaluation-inputs-v1',
        at=datetime.now(timezone.utc).isoformat(), protocol=PROTOCOL, original_test_image_IDs=982,
        original_test_images_with_questions=sum(bool(row['original_question_count']) for row in inventory),
        original_test_images_without_publisher_questions=sum(not row['original_question_count'] for row in inventory),
        original_test_images_physically_present=sum(row.get('original_image_available', False) for row in inventory),
        publisher_test_IDs_without_any_original_archive_entry=[row['image_id'] for row in inventory if not row.get('original_image_available', False)],
        present_original_test_images_without_publisher_questions=sum(row.get('original_image_available', False) and not row['original_question_count'] for row in inventory),
        all_original_test_questions=3088, original_ABC_questions=1184, original_image_questions=1904,
        all_original_questions_options_order_gold_indices_preserved=True, original_ambiguities=dict(counts),
        maximum_complete_native_prompt_tokens=maximum, native_generation_budget=32, native_image_side=448,
        tokenizer_sha256=bindings[str(tokenizer_path)], source_files_sha256=bindings,
        artifact_metadata_sha256={p.name: file_hash(p) for p in OUT.iterdir() if p.is_file()},
        packaged_source_license='Non-commercial-use license with evaluation elsewhere permitted; retained unchanged.',
        source_listing_license_conflict_preserved_not_resolved=True, admitted_to_training=False,
        private_evaluation_only_no_public_release_or_redistribution=True, source_model_weights_used=False, model_updates=0,
        text_and_image_exposure_audit_pending=True, new_model_assessment_not_run=True,
        not_an_official_leaderboard_or_full_contamination_certificate=True, candidate_promoted=False, goal_achieved=False,
        limitations=['All3088 original test questions are retained across905 diagrams. Of982 official IDs,981 images are present:76 have no question, and ID4325 has no archive entry. All dispositions are explicit; no question/image/score is fabricated.',
            'ABC rendering follows published Ai2 geometry/color with an explicitly pinned LiberationSans font instead of Arial.',
            'Original duplicate/null-placeholder options remain in original order; strict and exact-duplicate-equivalent scoring must be kept separate.',
            'Native448 images/prompting differ from some published benchmark settings; report the declared protocol and no leaderboard-equivalence claim.',
            'Exact/semantic training exposure and original image ancestry still need audit before claims of independence.']))
    print(json.dumps(dict(prepared=str(OUT / 'manifest.json'), original_test_images=982, original_questions=3088,
        training_admitted=False, model_updates=0)), flush=True)


if __name__ == '__main__': main()
