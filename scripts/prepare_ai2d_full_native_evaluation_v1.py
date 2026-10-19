"""Preflight all3088 original visual QA inputs and deterministic controls.

This records a correction to older905-question-bearing wording:905 original
question files include91 empty files. Exactly814 diagrams contain3088 questions.
No original source, question, option, input or earlier evidence is modified.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D, load_pixels, tensor_sha, gray_pixels
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/ai2d-full-native-evaluation-v1-preparation'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden whole-input preparation required')
    population = FullAI2D(ROOT)
    inventory = list(map(json.loads, (ROOT / 'data/ai2d-all-original-test-eval-v2/image-inventory.jsonl').open()))
    inspected = 0
    for image in inventory:
        for variant in image['native_variants'].values():
            value = load_pixels(dict(variant, image=variant['path']))
            if tensor_sha(value) != variant['model_input_tensor_sha256_448']: raise ValueError('Every actual native448 tensor must reproduce')
            inspected += 1
    if inspected != 1555: raise ValueError('Every available original and ABC native448 image variant required')
    files = dict(population.bindings)
    tests = ROOT / 'tests/test_ai2d_full_native_evaluation_v1.py'
    log = ROOT / 'reports/ai2d-full-native-evaluation-v1-tests.log'
    files.update({str(p): file_hash(p) for p in [Path(__file__), tests, log, *python_sources(ROOT, [Path(__file__), tests])]})
    FileBindings(files).check(full=True)
    OUT.mkdir()
    replacements = OUT / 'fixed-wrong-image-identities.jsonl'
    with replacements.open('x') as stream:
        for case in population.cases:
            other = population.wrong[case['id']]
            stream.write(json.dumps(dict(id=case['id'], proper_image_id=case['image_id'], wrong_source_question_id=other['id'],
                wrong_image_id=other['image_id'], proper_tensor_sha256=case['model_input_tensor_sha256_448'],
                wrong_tensor_sha256=other['model_input_tensor_sha256_448'], same_ABC_input_role=True), separators=(',', ':')) + '\n')
    files[str(replacements)] = file_hash(replacements)
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
        original_population_sha256=population.identity_sha256, original_questions=3088, original_nonempty_diagrams=814,
        actual_native448_variants_checked=1555, original_publisher_IDs=982,
        literal_publisher_ID_dispositions=population.dispositions,
        legacy_disposition_label_correction=population.legacy_disposition_label_correction,
        every_complete_original_question_choice_order_and32_action_budget_preserved=True,
        conditions=['proper_image', 'gray_image', 'wrong_image'], gray_tensor_sha256=tensor_sha(gray_pixels()),
        all3088_wrong_image_controls_fixed_before_scores_and_independent_of_gold=True,
        original_duplicate_and_null_options_not_pruned=True, strict_and_exact_duplicate_equivalent_verdicts_separate=True,
        source_files_sha256=files, actual_neural_assessments=0, model_updates=0,
        evaluation_only_under_packaged_license=True, training_admitted=False,
        text_image_and_semantic_exposure_audit_pending=True, not_an_official_leaderboard_score=True,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), original_questions=3088, original_nonempty_diagrams=814,
        actual_native448_variants_checked=inspected, actual_neural_assessments=0)), flush=True)


if __name__ == '__main__': main()
