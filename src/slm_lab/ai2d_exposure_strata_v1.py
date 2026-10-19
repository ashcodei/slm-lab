"""Pre-score source-exposure groups without removing any original question.

Generic question matches, stronger ordered-choice/prompt matches and exact
current-pool image matches remain separate. Absence of these exact inspected
matches is not a clean-data, semantic or full historical independence claim.
"""
import hashlib
import json

from .ai2d_exposure_patterns_v1 import KINDS, SCOPES

FLAGS = ('generic_question_in_inspected_text', 'ordered_options_in_inspected_text', 'whole_native_prompt_in_inspected_text',
    'original_diagram_file_in_current_image_pool', 'proper_native448_in_current_image_pool')
COHORTS = ('all_original', *FLAGS, 'no_generic_question_match_in_inspected_text', 'no_stronger_exact_match_in_inspected_text_and_current_image_pools')
CONDITIONS = ('proper_image', 'gray_image', 'wrong_image')


def join(cases, text_rows, image_rows):
    ids = [r['id'] for r in cases]
    if (not ids or len(set(ids)) != len(ids) or [r['id'] for r in text_rows] != ids or [r['id'] for r in image_rows] != ids):
        raise ValueError('Every original question exactly once and in source order required in both complete audits')
    rows = []
    for case, text, image in zip(cases, text_rows, image_rows):
        if case['purpose'] != 'evaluation_only' or case['admitted_to_training'] is not False or case['split'] != 'test':
            raise ValueError('Every original publisher question stays evaluation only')
        if (text['original_question_id'] != case['original_question_id'] or text['image_id'] != case['image_id']
                or image['image_id'] != case['image_id'] or any(r['original_ABC_labels'] != case['original_ABC_labels'] for r in [text, image])):
            raise ValueError('Exact original question/diagram/ABC membership required')
        scopes = {}
        for kind in KINDS:
            value = text['literal_text_patterns'][kind]; found = value['inspected_exact_match_scopes']
            if (not isinstance(found, list) or len(set(found)) != len(found) or not set(found) <= set(SCOPES)
                    or type(value['found']) is not bool or value['found'] != bool(found)):
                raise ValueError('Every declared literal signature needs consistent explicit source scopes')
            scopes[kind] = found
        if (not set(scopes['whole_native_prompt']) <= set(scopes['question_with_ordered_options'])
                or not set(scopes['question_with_ordered_options']) <= set(scopes['original_question'])):
            raise ValueError('A literal stronger signature must also contain its unchanged weaker signatures')
        raw = image['exact_original_diagram_file_in_inspected_current_pool']
        native = image['exact_actual_proper_native448_input_in_inspected_current_pool']
        if type(raw) is not bool or type(native) is not bool: raise ValueError('Explicit original-byte and actual-native-pixel exposure booleans required')
        flags = dict(zip(FLAGS, [bool(scopes['original_question']), bool(scopes['question_with_ordered_options']),
            bool(scopes['whole_native_prompt']), raw, native]))
        stronger = any(flags[k] for k in FLAGS[1:])
        memberships = [name for name in COHORTS if name == 'all_original' or flags.get(name, False)
            or (name == 'no_generic_question_match_in_inspected_text' and not flags[FLAGS[0]])
            or (name == 'no_stronger_exact_match_in_inspected_text_and_current_image_pools' and not stronger)]
        rows.append(dict(id=case['id'], original_question_id=case['original_question_id'], image_id=case['image_id'],
            original_ABC_labels=case['original_ABC_labels'], literal_text_match_scopes=scopes, flags=flags, memberships=memberships,
            gold_index_or_model_scores_not_used_for_group_assignment=True,
            no_match_is_not_semantic_perceptual_or_all_ancestral_independence=True))
    return rows


def cohort_identities(rows):
    return {name: [r['id'] for r in rows if name in r['memberships']] for name in COHORTS}


def identity_sha256(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def summarize(exposure, deliveries):
    if [r['id'] for r in exposure] != [r['id'] for r in deliveries]: raise ValueError('Every original actual delivery and source flag must match in source order')
    metrics = ('strict_correct', 'exact_duplicate_equivalent_correct', 'answer_parseable',
        'spaced_letter_strict_correct', 'spaced_letter_exact_duplicate_equivalent_correct',
        'direct_letter_strict_correct', 'direct_letter_exact_duplicate_equivalent_correct')
    for row in deliveries:
        if set(row['conditions']) != set(CONDITIONS): raise ValueError('Every unchanged native image intervention required')
        if any(type(row['conditions'][c]['verdict'][m]) is not bool for c in CONDITIONS for m in metrics):
            raise ValueError('Actual independently reconstructed original-choice verdicts required')
    results = {}
    for cohort in COHORTS:
        selected = [d for e, d in zip(exposure, deliveries) if cohort in e['memberships']]
        results[cohort] = dict(questions=len(selected), original_diagrams=len({r['image_id'] for r in selected}), conditions={})
        for condition in CONDITIONS:
            values = {name: sum(r['conditions'][condition]['verdict'][name] for r in selected) for name in metrics}
            values['strict_accuracy'] = values['strict_correct'] / len(selected) if selected else None
            results[cohort]['conditions'][condition] = values
    return dict(cohorts=results, cohorts_overlap_and_must_not_be_added_together=True,
        all_original_questions_retained_as_primary_population=True, exposure_groups_not_independent_or_clean_data_certificates=True)
