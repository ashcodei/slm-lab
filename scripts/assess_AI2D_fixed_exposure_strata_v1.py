"""Freeze full source-exposure groups before any new AI2D score, then reconcile.

First waits for three exact source-audit owners and preserves all3088 original
questions. Later waits for the original full visual GPU job/worker restoration,
then checks every native delivery and reports every fixed source group. Uses
no GPU, constructs no neural network, trains nothing and selects no winner.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

from slm_lab.ai2d_exposure_strata_v1 import COHORTS, cohort_identities, identity_sha256, join, summarize
from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D, PROTOCOL, summarize as original_summary, verify_record
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/AI2D-fixed-exposure-strata-v1'
VISUAL = ROOT / 'reports/full-AI2D-pipeline-v1'
VISUAL_QUEUE = ROOT / 'reports/full-AI2D-pipeline-v1-queue.json'
AUDITS = ('AI2D-full-text-exposure-audit-v1', 'AI2D-current-native-image-exposure-v1', 'AI2D-all-text-exposure-witnesses-v1-readback')
ENDPOINTS = ('phase_initial', 'phase_trained', 'human_chosen_CE', 'human_DPO', 'code_anchor_CE', 'code_anchor_execution')


def live(identity):
    try: fields = (Path('/proc') / str(identity['pid']) / 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return False
    return fields[19] == identity['ticks'] and fields[0] not in ['Z', 'X']


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or VISUAL.exists():
        raise ValueError('Fresh CPU-hidden source-group declaration before the original GPU assessment starts required')
    OUT.mkdir(); expires = time.time() + 14 * 24 * 3600
    owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    source_plans = {name: json.loads((ROOT / 'reports' / name / 'plan.json').read_text()) for name in AUDITS}
    visual_queue = json.loads(VISUAL_QUEUE.read_text())
    files = {str(p): file_hash(p) for p in [VISUAL_QUEUE, ROOT / 'tests/test_ai2d_exposure_strata_v1.py',
        ROOT / 'reports/ai2d-exposure-strata-v1-tests.log', *(ROOT / 'reports' / name / 'plan.json' for name in AUDITS),
        *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_ai2d_exposure_strata_v1.py'])]}
    guard = FileBindings(files)
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        original_source_audit_owners={name: p['owner'] for name, p in source_plans.items()}, original_visual_owner=visual_queue['owner'],
        fixed_cohort_names=list(COHORTS), all_original_questions_remain_primary=True, source_files_sha256=files,
        all_cohort_formulas_fixed_before_new_visual_assessment=True, no_source_gold_index_or_model_score_used_for_assignment=True,
        actual_neural_assessments=0, no_GPU_or_neural_model_constructed=True, expires_at_unix=expires,
        automatic_retry=False, candidate_promoted=False, goal_achieved=False))
    try:
        for name, source in source_plans.items():
            folder = ROOT / 'reports' / name
            while not (folder / 'complete.json').is_file():
                if not live(source['owner']) or (folder / 'failure.json').exists(): raise ValueError('Exact source auditor ended without a complete original population')
                if time.time() > expires: raise ValueError('Declared source-group owner lifetime exceeded; no automatic retry')
                guard.check(full=False); time.sleep(20)
        if VISUAL.exists(): raise ValueError('Source groups must finish before original visual scores begin')
        population = FullAI2D(ROOT); complete = {}
        for name in AUDITS:
            path = ROOT / 'reports' / name / 'complete.json'; complete[name] = json.loads(path.read_text()); files[str(path)] = file_hash(path)
            for p, sha in complete[name].get('source_files_sha256', {}).items():
                if file_hash(p) != sha: raise ValueError('Every completed source-audit artifact must match')
                files[str(p)] = sha
        text = complete[AUDITS[0]]; image = complete[AUDITS[1]]; reader = complete[AUDITS[2]]
        if (text['all_six_declared_scopes_completed'] is not True or text['actual_original_AI2D_questions'] != 3088
                or text['actual_original_text_patterns'] != 9264 or image['complete'] is not True or image['actual_source_positions'] != 302627
                or reader['complete'] is not True or reader['original_AI2D_questions'] != 3088
                or reader['independent_matched_questions_by_signature'] != text['matched_questions_by_pattern']
                or any(p['original_population_sha256'] != population.identity_sha256 for p in [text, image])):
            raise ValueError('Complete exact all-question text/pixel audits and independent witness readback required')
        text_path = ROOT / 'reports' / AUDITS[0] / 'original-question-pattern-results.jsonl'
        image_path = ROOT / 'reports' / AUDITS[1] / 'all3088-original-question-image-exposure.jsonl'
        if file_hash(text_path) != text['original_question_pattern_results_sha256']: raise ValueError('Every original text exposure verdict must match')
        if file_hash(image_path) != image['source_files_sha256'][str(image_path)]: raise ValueError('Every original native-image exposure verdict must match')
        files.update({str(text_path): file_hash(text_path), str(image_path): file_hash(image_path)})
        rows = join(population.cases, list(map(json.loads, text_path.open())), list(map(json.loads, image_path.open())))
        identities = cohort_identities(rows); sha = identity_sha256(rows)
        output = OUT / 'all3088-fixed-source-exposure-flags.jsonl'
        output.write_text(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows))
        atomic_json(OUT / 'fixed-original-cohort-identities.json', identities)
        files.update({str(output): file_hash(output), str(OUT / 'fixed-original-cohort-identities.json'): file_hash(OUT / 'fixed-original-cohort-identities.json')})
        atomic_json(OUT / 'preparation-complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
            original_questions=3088, original_population_sha256=population.identity_sha256, fixed_exposure_rows_sha256=sha,
            source_group_sizes={k: len(v) for k, v in identities.items()}, all_original_questions_remain_primary=True,
            source_groups_fixed_before_original_GPU_visual_assessment=True, source_gold_and_model_scores_not_used=True,
            source_files_sha256=files, actual_neural_assessments=0, model_updates=0,
            source_groups_not_semantic_perceptual_or_all_historical_independence_certificates=True, goal_achieved=False))
        print(json.dumps(dict(preparation_complete=True, original_questions=3088, fixed_source_group_sizes={k: len(v) for k, v in identities.items()})), flush=True)
        guard = FileBindings(files)
        while True:
            me = visual_queue['owner']; children = VISUAL / 'children.json'
            active = live(me) or (children.exists() and any(live(p) for p in json.loads(children.read_text())))
            if not active:
                if ((VISUAL / 'failure.json').exists() or not (VISUAL / 'complete.json').is_file() or not (VISUAL / 'restored.json').is_file()):
                    raise ValueError('Exact original visual assessment ended without all endpoints and restoration')
                break
            if time.time() > expires: raise ValueError('Declared full visual readback lifetime exceeded; no automatic retry')
            guard.check(full=False); time.sleep(20)
        visual_plan_path = VISUAL / 'plan.json'; visual_plan = json.loads(visual_plan_path.read_text())
        if (visual_plan['owner']['pid'] != visual_queue['owner']['pid'] or visual_plan['owner']['ticks'] != visual_queue['owner']['ticks']
                or visual_plan['full_population_sha256'] != population.identity_sha256 or set(visual_plan['endpoints']) != set(ENDPOINTS)):
            raise ValueError('All six exact original own-model endpoints and complete unchanged image/text population required')
        outcomes = {}; result_files = []
        for endpoint in ENDPOINTS:
            path = VISUAL / endpoint / 'full-AI2D.json'; report = json.loads(path.read_text())
            expected_count = 984399690 if endpoint.startswith('phase_') else 999706442
            if (report['complete'] is not True or report['protocol'] != PROTOCOL or report['original_questions'] != 3088
                    or report['original_population_sha256'] != population.identity_sha256 or report['actual_student_parameter_values'] != expected_count
                    or report['inference_device_type'] != 'cuda' or report['inference_precision'] != 'BF16_autocast'
                    or report['generation_budget'] != 32 or report['conditions'] != ['proper_image', 'gray_image', 'wrong_image']
                    or report['actual_question_condition_assessments'] != 9264
                    or report['checkpoint_sha256'] != visual_plan['endpoints'][endpoint]['checkpoint_sha256']):
                raise ValueError('Every complete full native image/text/score delivery and original under1B endpoint required')
            deliveries = report['results']
            if [r['id'] for r in deliveries] != [c['id'] for c in population.cases]: raise ValueError('All3088 original native questions required')
            for case, row in zip(population.cases, deliveries): verify_record(population.tokenizer, case, population.wrong[case['id']], row)
            if report['summary'] != original_summary(population.cases, deliveries): raise ValueError('All primary original strict/duplicate-equivalent scores must rederive')
            scores = summarize(rows, deliveries); outcomes[endpoint] = scores
            destination = OUT / (endpoint + '-actual-fixed-source-group-scores.json')
            atomic_json(destination, dict(endpoint=endpoint, checkpoint_sha256=report['checkpoint_sha256'], fixed_exposure_rows_sha256=sha,
                original_population_sha256=population.identity_sha256, actual_original_question_deliveries=3088, scores=scores,
                raw_deliveries_and_primary_full_population_scores_preserved=True, model_updates=0, candidate_promoted=False, goal_achieved=False))
            files[str(path)] = file_hash(path); result_files.append(destination)
        guard.check(full=True)
        atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
            original_endpoints=list(ENDPOINTS), actual_native_question_endpoint_deliveries_rederived=6 * 3088,
            actual_question_endpoint_image_condition_verdicts_rederived=6 * 3088 * 3,
            all_original_primary_populations_and_native_budgets_preserved=True, every_predeclared_source_group_scored=True,
            source_groups_overlap_not_semantic_or_all_historical_independence_certificates=True,
            native_probability_observations_retained_not_recomputed_by_new_NN_forwards=True,
            source_files_sha256={str(p): file_hash(p) for p in [OUT / 'preparation-complete.json', visual_plan_path,
                VISUAL / 'complete.json', VISUAL / 'restored.json', *result_files, *(VISUAL / n / 'full-AI2D.json' for n in ENDPOINTS)]},
            model_generations=0, model_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_native_deliveries_rederived=6 * 3088)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__': main()
