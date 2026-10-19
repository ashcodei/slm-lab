"""Record completed shared-model evidence while preserving the full goal scope."""
from datetime import datetime, timezone
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exclusive_inference_resources_v1 import identity

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'reports/shared-assistant-and-decoding-v1-review.json'


def live(actor):
    found = identity(actor['pid'])
    return found if found and found['state'] not in {'Z', 'X'} and found['ticks'] == actor['ticks'] else None


def main():
    if OUTPUT.exists():
        raise ValueError('Keep the original dated review; no implicit overwrite')
    paths = dict(
        manifest='checkpoints/own-assistant-candidate-v1/manifest.json',
        trained_CPU_validation='reports/own-assistant-candidate-v1/validation-complete.json',
        code_readback='reports/own-code-generation-repair-v1/independent-comparison.json',
        code_report_repair='reports/own-code-generation-repair-endpoint-v2/complete.json',
        head_readback='reports/frozen-base-pointer-v1/complete-head-source-readback.json',
        head_results='reports/frozen-base-pointer-v1/matched-attention-and-native-comparison.json',
        decoding_pilot='reports/own-view-contrastive-v1/comparison.json',
        decoding_independent_execution='reports/own-view-contrastive-v1-readback/complete.json',
        restoration='reports/own-view-contrastive-v1/restored.json',
        algorithm_tests='reports/own-view-contrastive-v1-tests.log',
        controller_tests='reports/own-view-contrastive-driver-v1-tests.log',
        review_source=str(Path(__file__).resolve().relative_to(ROOT)),
    )
    evidence = {key: json.loads((ROOT / path).read_text()) for key, path in paths.items() if path.endswith('.json')}
    counts = evidence['trained_CPU_validation']['actual_counts']
    if counts['total_unique'] != 986450944 or not evidence['trained_CPU_validation']['actual_all_three_routes_validated']:
        raise ValueError('Actual full trained single-backbone CPU validation required')
    if not evidence['decoding_independent_execution']['all_48_actual_model_deliveries_independently_reexecuted']:
        raise ValueError('Complete independent original-case execution readback required')
    metrics = evidence['decoding_pilot']['metrics']
    if (metrics['0.0']['all_finite_cases_pass'], metrics['0.5']['all_finite_cases_pass'], metrics['1.0']['all_finite_cases_pass']) != (1, 0, 0):
        raise ValueError('Actual negative calibration result required, without promoting another setting')
    pilot = ROOT / 'reports/own-view-contrastive-v1'
    plan = json.loads((pilot / 'plan.json').read_text())
    if live(plan['owner']) or any(live(actor) for actor in json.loads((pilot / 'children.json').read_text())):
        raise ValueError('All actual original pilot actors must end')
    waiter = live(plan['waiter'])
    if not waiter or waiter['state'] != 'S' or not evidence['restoration']['all_registered_point_actors_terminal']:
        raise ValueError('Exact original idle waiter must have been restored')
    capacities = []
    for shard in range(4):
        path = pilot / f'capacity-{shard}.json'; capacity = json.loads(path.read_text())
        if capacity['actual_model_counts'] != counts or not capacity['actual_all_three_trained_routes_exercised']:
            raise ValueError('All four actual complete trained GPU route probes required')
        capacities.append(dict(path=str(path.relative_to(ROOT)), sha256=file_hash(path)))
    attention = evidence['head_results']['attention']
    grounding = dict(updates=1137, original_training_queries=36354, held_queries=2027,
        all_native_image_outputs_and_scores_exactly_retained=evidence['head_results']['every_original_native_image_output_retained'],
        training_position_prior_hit_percent=100 * attention['candidate']['conditions']['training_position']['query_mean']['top1_observed_attention_cell'],
        approximate_human_attention_not_general_vision=True)
    for phase in ['baseline', 'candidate']:
        proper = attention[phase]['conditions']['proper']['query_mean']
        grounding[phase] = dict(query_mean_soft_target_CE=proper['soft_target_cross_entropy'],
            attention_cell_hit_percent=100 * proper['top1_observed_attention_cell'])
    now = datetime.now(timezone.utc).isoformat()
    requirements = [
        'Improve actual functional and efficient code generation across languages and broad domain/math/vision capability; current scores remain weak.',
        'Diagnose original train problem/input/interface conditioning before another neural intervention; this failed decoder has no selected held follow-up.',
        'Finish original paired real-photo/full-source preparation and the guarded final production handoff; preserve publisher429/RetryAfter limits and original completeness gates.',
        'Complete fresh genuine-human alignment and broad appropriately independent final-policy evaluation. No frontier capability claim follows from assembly, syntax or token loss.',
    ]
    classification = dict(classification='progress', reason='Completed full four-L40986M shared-route/cached decoding pilot, added independent reexecution of all48original train deliveries, rejected both contrast settings by the predeclared criterion, and restored the exact original waiter. Source/state/negative capability evidence is retained.')
    previous = dict(classification='progress', reason='Completed the original code endpoint repair/readback, frozen-base head epoch/controlled retention comparison and actual trained shared-model CPU factory validation. The separate disk-space response verified cleanup.')
    report = dict(observed_at=now, previous_goal_turn_classification=previous, current_goal_turn_classification=classification,
        actual_counts=counts, completed_code_training=dict(updates=815, response_EOS_targets=4645006,
            public_development=evidence['code_readback']['public_subset'], local_functional_and_efficiency_still_zero=True),
        completed_frozen_base_grounding=grounding, decoding_train_only_metrics=metrics,
        no_decoding_setting_selected=True, all_48_deliveries_independently_reexecuted=True,
        actual_shared_trained_CPU_and_all_four_GPU_routes_exercised=True, actual_GPU_capacity_reports=capacities,
        source_reports={name: dict(path=path, sha256=file_hash(ROOT / path)) for name, path in paths.items()},
        original_waiter=waiter, original_live_collector=live(dict(pid=130149, ticks='20300752')),
        separately_live_CPU_math=[identity(pid) for pid in [2852035, 2904126]],
        photo_collection_observation=json.loads((ROOT / 'data/vqa-real-pairs-v1/progress.json').read_text()),
        publisher_cooldown_observation=json.loads((ROOT / 'data/vqa-real-pairs-v1/publisher-throttle.json').read_text()),
        next_required_work=requirements, goal_achieved=False, at_impasse=False, candidate_promoted=False)
    atomic_json(OUTPUT, report)
    audit_path = ROOT / 'reports/completion-audit.json'; audit = json.loads(audit_path.read_text())
    audit.update(goal_achieved=False, at_impasse=False, updated_at=now, previous_goal_turn_classification=previous,
        current_goal_turn_classification=classification, current_turn_classification=classification, next_required_work=requirements)
    summary = dict(review=str(OUTPUT.relative_to(ROOT)), sha256=file_hash(OUTPUT), actual_counts=counts,
        CPU_and_four_GPU_routes_verified=True, decoder_train_metrics=metrics,
        all_48_independently_reexecuted=True, setting_selected=None, candidate_promoted=False, goal_achieved=False)
    audit['shared_assistant_and_decoding_v1'] = summary
    for key, proof in [('own_code_generation_repair_v1', 'code_readback'), ('frozen_base_pointer_v1', 'head_readback')]:
        audit[key] = dict(status='complete; actual source/state and measured outcomes independently read back',
            latest_review=summary['review'], latest_review_sha256=summary['sha256'], complete_evidence=paths[proof],
            previous_historical_entry=audit.get(key), goal_achieved=False)
    r = audit['requirements']
    r['under_1b_parameters'] = 'Actual trained shared composite986450944unique parameters:706012672original backbone+279651840own code+786432grounding. One actual backbone loaded; trained route/cache/byte/pointer parity verified on fullCPU and all4L40s. Explicit routes only; automatic routing unproven.'
    r['text_and_images'] = 'Actual shared trained text/code/grounding model runs. Full frozen-base head epoch improves2027held attention-cell hits2.86%to5.77% and CE5.046to4.933; training-position prior top-cell hits6.96%. Controlled CE gains and exact original OCR/object/caption retention verified. NativeOCR0/170, object46/164proper versus50/164wrong and captionretrieval3/128remain weak. Approximate mouse attention is not general vision. Original paired-photo finalization/production remains pending.'
    r['serious_full_model_training'] = 'Original own706M broad-code936updates and systems1138updates, all102487admitted human APPS programs/3203updates, own279651840parameter code815updates and786432parameter frozen-base head1137updates are complete with retained source/export/resume/Adam/RNG/cursor evidence. Actual986450944shared trained candidate loads without external weights. Integration adds no joint end-to-end training or fresh final-policy human alignment.'
    r['high_end_llm_capability'] = 'NOT ACHIEVED. Latest public-development coding4/175 versus5/175before; allfiveearlier passing IDs lost. Local functional/efficiency0/24bothformats andC++0/12. General559questions13generated/74likelihood. Earlier fullCUDA/BF16math18/1319belongs to a different own dense-code checkpoint. No high-end general/coding/vision/multilingual/reasoning parity demonstrated.'
    r['any_code_correctness_and_efficiency'] = 'NOT ACHIEVED. Latest local completion syntax11to20/24 and publicsyntax130to132/175improve, but publicfunctional5to4/175 and localPython/JS/C++functional/efficiencyremainzero. Runtime qualifications are not relabelled algorithmic wrong values. Train-only shared-view decoder:greedy1/16, bothcontrasts0/16withmorecompilefailures; all48deliveries independentlyreexecuted and no settingselected. Finite publishedcases/stress are not universalcorrectness/asymptoticproof.'
    r['gpu_runtime'] = 'Four original NVIDIA L40s used for complete code/head production/capacity/recovery and fulltrained986M shared-model decoding pilot. Latest owner/allregisteredworkers confirmedterminal, GPUmemoryreleased and exact sleeping childless original waiter2754743/birth27602681resumed. SeparateCPU/FP32math and original publisher photo collector remain live;429/RetryAfter honored.'
    audit['blocked_audit'] = dict(at_overall_impasse=False, same_original_storage_condition_persists=False,
        reason='Concrete trained shared-model GPU/inference and complete train-only execution evidence added. Neural capability gains, original-source finalization and fresh final-policy human alignment remain actionable incomplete work.')
    atomic_json(audit_path, audit)
    print(json.dumps(dict(review=str(OUTPUT.relative_to(ROOT)), review_sha256=file_hash(OUTPUT),
        actual_counts=counts, decoding_train_metrics=metrics, goal_achieved=False), indent=2))


if __name__ == '__main__':
    main()
