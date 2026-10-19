"""Reuse already fixed source-exposure groups and reconcile all nine endpoints.

Reuses complete original source audits and all3088 fixed question memberships.
Waits for the exact code-first visual GPU job and worker restoration,
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
from slm_lab.public_pretraining_lineage_tests_v1 import bind_tests
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / 'reports/AI2D-fixed-exposure-strata-v1'
OUT = ROOT / 'reports/AI2D-fixed-exposure-strata-assessment-recovery-gate-v1'
QUEUE = ROOT / 'reports/AI2D-fixed-exposure-strata-v5-assessment-recovery-gate-v1-queue.json'
VISUAL = ROOT / 'reports/full-AI2D-pipeline-v5'
VISUAL_QUEUE = ROOT / 'reports/full-AI2D-pipeline-v5-assessment-recovery-gate-v1-queue.json'
AUDITS = ('AI2D-full-text-exposure-audit-v1', 'AI2D-current-native-image-exposure-v1', 'AI2D-all-text-exposure-witnesses-v1-readback')
from slm_lab.public_pretraining_endpoints_v1 import ENDPOINTS,expected_values,descriptors


def live(identity):
    try: fields = (Path('/proc') / str(identity['pid']) / 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError: return False
    return fields[19] == identity['ticks'] and fields[0] not in ['Z', 'X']


def prepared(population,folder):
    preparation=json.loads((folder/'preparation-complete.json').read_text())
    path=folder/'all3088-fixed-source-exposure-flags.jsonl'
    identities=folder/'fixed-original-cohort-identities.json'
    rows=list(map(json.loads,path.open()))
    if (preparation.get('complete') is not True or preparation.get('original_questions')!=3088
            or preparation.get('original_population_sha256')!=population.identity_sha256
            or preparation.get('source_groups_fixed_before_original_GPU_visual_assessment') is not True
            or preparation.get('source_gold_and_model_scores_not_used') is not True
            or [r['id'] for r in rows]!=[c['id'] for c in population.cases]
            or identity_sha256(rows)!=preparation['fixed_exposure_rows_sha256']
            or cohort_identities(rows)!=json.loads(identities.read_text())
            or any(file_hash(p)!=preparation['source_files_sha256'][str(p)] for p in [path,identities])):
        raise ValueError('Exact all3088 pre-score flags and original fixed cohort identities required')
    return rows,preparation


def main():
    if '16 passed' not in (ROOT/'reports/public-pretraining-lineage-drivers-v1-final2-tests.log').read_text():
        raise ValueError('Actual complete revised public-pretraining lineage and every original benchmark endpoint must pass')

    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or QUEUE.exists() or VISUAL.exists():
        raise ValueError('Fresh CPU-hidden reader queued before visual scoring required')
    population=FullAI2D(ROOT);rows,preparation=prepared(population,PREPARED)
    sha=preparation['fixed_exposure_rows_sha256'];visual_queue=json.loads(VISUAL_QUEUE.read_text())
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    files=dict(preparation['source_files_sha256'])
    extra=[PREPARED/'preparation-complete.json',VISUAL_QUEUE,ROOT/'reports/state-code-first-visual-reader-v3-tests.log',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_code_first_visual_reader_v4.py'])]
    files.update({str(p):file_hash(p) for p in extra});bind_tests(files,ROOT);guard=FileBindings(files);guard.check(full=True)
    expires=time.time()+14*86400;OUT.mkdir()
    plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,original_visual_owner=visual_queue['owner'],
        source_files_sha256=files,fixed_exposure_rows_sha256=sha,fixed_cohort_names=list(COHORTS),
        source_groups_reused_without_post_score_assignment=True,actual_neural_assessments=0,model_updates=0,goal_achieved=False)
    atomic_json(OUT/'plan.json',plan);atomic_json(QUEUE,plan)
    try:
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
            raise ValueError('All eight exact original own-model endpoints and complete unchanged image/text population required')
        outcomes = {}; result_files = []
        for endpoint in ENDPOINTS:
            path = VISUAL / endpoint / 'full-AI2D.json'; report = json.loads(path.read_text())
            expected_count = expected_values(descriptors(ROOT)[endpoint])
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
            original_endpoints=list(ENDPOINTS), actual_native_question_endpoint_deliveries_rederived=len(ENDPOINTS) * 3088,
            actual_question_endpoint_image_condition_verdicts_rederived=len(ENDPOINTS) * 3088 * 3,
            all_original_primary_populations_and_native_budgets_preserved=True, every_predeclared_source_group_scored=True,
            source_groups_overlap_not_semantic_or_all_historical_independence_certificates=True,
            native_probability_observations_retained_not_recomputed_by_new_NN_forwards=True,
            source_files_sha256={str(p): file_hash(p) for p in [PREPARED / 'preparation-complete.json', visual_plan_path,
                VISUAL / 'complete.json', VISUAL / 'restored.json', *result_files, *(VISUAL / n / 'full-AI2D.json' for n in ENDPOINTS)]},
            model_generations=0, model_updates=0, candidate_promoted=False, goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_native_deliveries_rederived=len(ENDPOINTS) * 3088)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False)); raise


if __name__ == '__main__': main()
