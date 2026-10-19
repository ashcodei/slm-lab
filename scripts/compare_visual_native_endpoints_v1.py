"""Full fixed aligned-native baseline versus complete visual-native candidate.

Reuse the already completed same-protocol baseline; retain all before/after
populations, failures, image controls and source exposure disclosures.
"""
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def paired_words(before, after):
    for report in [before, after]:
        rows = report.get('results', [])
        if (len(rows) != 3429 or [r['source_position'] for r in rows] != list(range(3429))
                or report.get('generation_budget') != 64 or report.get('complete_held_source_words') is not True
                or report.get('held_source_gradients') is not False):
            raise ValueError('All3429original held words and complete64token native budgets are required')
    keys = ['source_position', 'id', 'group_id', 'family', 'expected', 'wrong_image_id', 'source_image_sha256', 'native_prompt_ids']
    identical = 0
    for a, b in zip(before['results'], after['results']):
        if any(a[k] != b[k] for k in keys) or set(a['conditions']) != {'proper', 'blank', 'wrong'} or set(b['conditions']) != {'proper', 'blank', 'wrong'}:
            raise ValueError('Every original word, actual source image, donor, complete prompt and all3conditions must match')
        identical += all(a['conditions'][c]['native']['generated_ids'] == b['conditions'][c]['native']['generated_ids'] for c in a['conditions'])
    return dict(before=before['overall'], after=after['overall'], original_words=3429,
        all_three_condition_native_sequences_identical=identical,
        same_original_sources_wrong_donors_and_full_native_budgets=True, candidate_promoted=False)


def compare(baseline, candidate, output):
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from assess_full_held_QA_pipeline_v5 import paired as paired_QA
    from assess_full_AI2D_pipeline_v5 import paired as paired_AI2D
    baseline, candidate, output = Path(baseline), Path(candidate), Path(output)
    if output.exists():
        raise ValueError('Preserve the complete original comparison and every partial failure')
    paths = []
    def both(name):
        paths.extend([baseline / name, candidate / name])
        return read(baseline / name), read(candidate / name)
    QA = paired_QA(*both('full-QA.json'))
    visual = paired_AI2D(*both('full-AI2D.json'))
    before, after = both('full-original-MBPP.json')
    if (len(before['results']) != 600 or len(after['results']) != 600
            or any(before[k] != after[k] for k in ['original_population_sha256', 'own_tokenizer_sha256', 'generation_budget'])):
        raise ValueError('All600original MBPPproblems and the same native protocol are required')
    gained, lost = [], []
    same = 0
    for a, b in zip(before['results'], after['results']):
        if any(a[k] != b[k] for k in ['original_task_id', 'publisher_role', 'original_native_prompt_ids',
                'generation_budget', 'inference_device_type', 'inference_precision']):
            raise ValueError('Every original code task, native prompt and runtime must match')
        p, q = a['execution']['functional_passed'], b['execution']['functional_passed']
        if not p and q:
            gained.append(a['original_task_id'])
        if p and not q:
            lost.append(a['original_task_id'])
        same += a['actual_generated_ids'] == b['actual_generated_ids']
    MBPP = dict(before=before['summary'], after=after['summary'], gained_original_ids=gained,
        lost_original_ids=lost, identical_generated_action_sequences=same)
    code = {}
    for frame in ['completion', 'instruction']:
        a, b = both('code-' + frame + '.json')
        if (a['task_count'] != 24 or b['task_count'] != 24 or any(a[k] != b[k] for k in
                ['generation_budget', 'decoding', 'prompt_mode', 'suite_sha256', 'tokenizer_sha256', 'inference_device_type', 'inference_precision'])
                or [(r['language'], r['task']) for r in a['results']] != [(r['language'], r['task']) for r in b['results']]):
            raise ValueError('All24original native code deliveries per frame and exact protocols must match')
        code[frame] = dict(before={k: a[k] for k in ['syntax_pass_rate', 'functional_pass_at_1', 'efficiency_stress_pass_rate']},
            after={k: b[k] for k in ['syntax_pass_rate', 'functional_pass_at_1', 'efficiency_stress_pass_rate']},
            identical_raw_responses=sum(x['raw_response'] == y['raw_response'] for x, y in zip(a['results'], b['results'])),
            fresh_independent_program_and_action_readbacks_still_required=True)
    human_before, human_after = both('held-preferences.json')
    if any(len(p['results']) != 4642 for p in [human_before, human_after]):
        raise ValueError('Every original4642held human pair is required')
    for a, b in zip(human_before['results'], human_after['results']):
        if any(a[k] != b[k] for k in ['original_held_position', 'id', 'source_record_sha256', 'source_fingerprint',
                'chosen_response_EOS_targets', 'rejected_response_EOS_targets', 'chosen_target_ids_sha256', 'rejected_target_ids_sha256']):
            raise ValueError('Every original full human source and both whole response/EOS masks must match')
    words = paired_words(*both('all-held-words.json'))
    broad = {n: dict(zip(['before', 'after'], both(n + '.json'))) for n in ['knowledge', 'math', 'ocr', 'objects', 'vision']}
    report = dict(complete=True, QA=QA, visual=visual, MBPP=MBPP, native_code=code,
        held_human_before=human_before['overall'], held_human_after=human_after['overall'], held_words=words, broad=broad,
        full_original_populations_and_all_image_conditions_preserved=True,
        exact_aligned_native_baseline_reused_without_model_selection=True,
        independent_neural_score_and_program_readbacks_still_required=True,
        source_files_sha256={str(p): file_hash(p) for p in paths}, candidate_promoted=False, goal_achieved=False)
    atomic_json(output, report)
    return report
