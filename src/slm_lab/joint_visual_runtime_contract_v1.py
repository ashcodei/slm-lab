"""Full-scope admission for visual continuation of the completed aligned native.

The original canonical human/code/machine/standalone assessments run first.
Their exact live owners and all source readers stay untouched. Passing this
gate grants no GPU lease; the new controller must separately hold the actual
exclusive lock and prove all full-size visual capacity/recovery/cold cases.
"""
import hashlib
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal

PIPELINE = 'reports/machine-standalone-full-assessment-v2'
PIPELINE_QUEUE = PIPELINE + '-assessment-recovery-gate-v1-queue.json'
FINALIZER = 'reports/machine-preference-final-standalone-CPU-v2'
FINALIZER_QUEUE = FINALIZER + '-assessment-recovery-gate-v1-queue.json'
SOURCE = 'reports/full-corrected-multimodal-source-admission-CPU-v1'
ADMISSION = 'reports/full-multimodal-corrected-source-index-CPU-v1/source-training-admission.json'
SCOPES = dict(full_QA_questions=16961, full_AI2D_questions=3088, AI2D_image_conditions=3,
    full_MBPP_problems=600, held_human_pairs=4642, held_word_crops=3429,
    word_image_conditions=['proper', 'blank', 'wrong'], native_word_generation_budget=64)
COMPLETION = ['all16961QA_3088visual_times3conditions_600MBPP_and4642held_human_pairs_complete',
    'all_original_native_code_and_small_broad_checks_complete', 'actor_to_standalone_comparisons_complete',
    'all3429held_word_crops_CTC_and_native_text_with_three_image_controls_complete',
    'no_old_actor_or_reference_weights_loaded_for_neural_inference',
    'source_exposure_and_nonleaderboard_protocol_disclosures_preserved']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ready(root, *, pipeline_owner, finalizer_owner, source_owner):
    root = Path(root)
    pipeline = root / PIPELINE
    source = root / SOURCE
    finalizer = root / FINALIZER
    expected = [(pipeline, pipeline_owner), (finalizer, finalizer_owner), (source, source_owner)]
    if any(not terminal(owner) for _, owner in expected):
        return False
    for folder, owner in expected:
        if ((folder / 'failure.json').exists() or not (folder / 'plan.json').exists()
                or not (folder / 'complete.json').exists() or read(folder / 'plan.json')['owner'] != owner
                or read(folder / 'complete.json').get('complete') is not True):
            raise ValueError('Every exact original complete aligned pipeline/finalizer/source owner must finish successfully')
    if (read(root / PIPELINE_QUEUE)['owner'] != pipeline_owner
            or read(root / FINALIZER_QUEUE)['owner'] != finalizer_owner):
        raise ValueError('The exact originally captured canonical predecessor births must remain unchanged')
    scope = read(root / PIPELINE_QUEUE)
    if (scope.get('expected_student_values') != 984399690 or any(scope.get(k) != v for k, v in SCOPES.items())
            or scope.get('all_original_case_populations_budgets_and_source_exposure_disclosures_preserved') is not True):
        raise ValueError('All full original canonical populations and native budgets must be retained')
    done = read(pipeline / 'complete.json')
    plan = read(pipeline / 'plan.json')
    if (done.get('actual_unique_student_values') != 984399690 or any(done.get(k) is not True for k in COMPLETION)
            or read(pipeline / 'restored.json').get('all_registered_point_actors_terminal') is not True
            or any(not terminal(owner) for owner in read(pipeline / 'children.json'))):
        raise ValueError('Every actual final native assessment and registered GPU worker must be complete and terminal')
    for filename, count in [('full-QA.json', 16961), ('full-AI2D.json', 3088),
            ('full-original-MBPP.json', 600), ('held-preferences.json', 4642)]:
        if len(read(pipeline / 'standalone' / filename).get('results', [])) != count:
            raise ValueError('Every actual original final assessment result must be present before visual continuation')
    word = read(pipeline / 'standalone/all-held-words.json')
    rows = word.get('results', [])
    if (len(rows) != 3429 or [r.get('source_position') for r in rows] != list(range(3429))
            or word.get('generation_budget') != 64 or word.get('complete_held_source_words') is not True
            or word.get('held_source_gradients') is not False
            or word.get('all_original_global_wrong_donors_preserved') is not True
            or any(set(r.get('conditions', {})) != {'proper', 'blank', 'wrong'} for r in rows)):
        raise ValueError('All3429original held words and all3native image conditions must be present')
    final = read(finalizer / 'complete.json')
    if (final.get('completed_updates') != 102 or final.get('actual_counts', {}).get('total_unique') != 984399690
            or any(final.get(k) is not True for k in ['all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact',
                'real_word_logits_exact_through_fold_and_cold_native', 'old_checkpoint_actor_optimizer_or_reference_not_needed_for_inference',
                'fixed_machine_DPO_export_not_quality_winner', 'all_original_machine_source_updates_complete'])
            or sha(final['checkpoint']) != final.get('checkpoint_sha256')):
        raise ValueError('The entire fixed aligned native student and exact finalizer cold proof must be preserved')
    spec = plan['standalone']
    if (spec['checkpoint_sha256'] != final['checkpoint_sha256'] or Path(spec['checkpoint']) != Path(final['checkpoint'])
            or sha(finalizer / 'complete.json') != spec['receipt_sha256']):
        raise ValueError('The actual assessed final native student must be the visual initializer')
    for suite in ['code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words']:
        for rank in range(4):
            proof = read(pipeline / 'standalone' / f'assessment-{suite}-{rank}-complete.json')
            if (proof.get('physical_GPU') != rank or proof.get('actual_student_values') != 984399690
                    or proof.get('all_loaded_values_unchanged') is not True
                    or proof.get('only_standalone_weights_and_tokenizer_for_neural_inference') is not True
                    or proof.get('checkpoint_sha256') != final['checkpoint_sha256'] or proof.get('model_updates') != 0):
                raise ValueError('All28actual complete final native assessment worker receipts must agree')
    admission = read(root / ADMISSION)
    receipt = read(source / 'complete.json')
    if (not receipt.get('passed') or not receipt.get('data_only_training_admitted')
            or receipt['admission_sha256'] != sha(root / ADMISSION)
            or admission.get('owner') != source_owner or admission.get('training_admitted') is not True
            or admission.get('admitted_role') != 'train'
            or any(admission.get(k) != v for k, v in dict(complete_source_views=302627, complete_physical_paths=195121,
                complete_global_steps=4733, complete_rank_calls=18932, complete_context_tokens=17681556,
                complete_response_EOS_targets=4669243, whole_word_views=91585, whole_CTC_UTF8_bytes=644063,
                four_complete_original_negative_targets=80).items())
            or admission.get('runtime_or_kernel_or_model_admission_not_granted') is not True
            or admission.get('historical_failed_word_and_prefix_numeric_controls_not_waived') is not True):
        raise ValueError('Only the exact complete visual data-only admission may precede separate runtime capacity')
    return True
