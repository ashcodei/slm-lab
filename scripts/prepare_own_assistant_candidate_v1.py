"""Create an experimental reference bundle from actual completed components.

This bundles a measured candidate, not a promoted best policy. It duplicates
no weight files and does not call a dataset, training job or external model.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import assess_joint_java_comparison_v1 as actors
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_assistant_v1 import MANIFEST_FORMAT

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'checkpoints/own-assistant-candidate-v1'
REPORTS = ROOT / 'reports/own-assistant-candidate-v1'
CODE_REPORTS = ROOT / 'reports/own-code-generation-repair-v1'
HEAD_REPORTS = ROOT / 'reports/frozen-base-pointer-v1'


def completed(folder):
    plan = json.loads((folder / 'plan.json').read_text())
    if actors.same(plan['owner']) or any(actors.same(a) for a in json.loads((folder / 'children.json').read_text())):
        raise ValueError('Actual component owner and all registered actors must end before packaging')
    if not (folder / 'restored.json').exists():
        raise ValueError('Completed component resource restoration evidence required')
    return plan


def main():
    if OUTPUT.exists() or REPORTS.exists():
        raise ValueError('Preserve prior experimental bundle; no overwrite')
    if '3 passed' not in (ROOT / 'reports/own-assistant-v1-tests.log').read_text():
        raise ValueError('Actual shared-base factory/route/cache/provenance checks required')
    code_plan = completed(CODE_REPORTS); head_plan = completed(HEAD_REPORTS)
    code = json.loads((CODE_REPORTS / 'independent-comparison.json').read_text())
    code_audit = json.loads((ROOT / 'reports/own-code-generation-repair-endpoint-v2/complete.json').read_text())
    head = json.loads((HEAD_REPORTS / 'complete-head-source-readback.json').read_text())
    head_results = json.loads((HEAD_REPORTS / 'matched-attention-and-native-comparison.json').read_text())
    code_receipt = json.loads((CODE_REPORTS / 'training-complete.json').read_text())
    head_receipt = json.loads((HEAD_REPORTS / 'training-complete.json').read_text())
    if (code['actual_steps'] != 815 or head['actual_steps'] != 1137
            or not code['all_exported_components_equal_complete_resume']
            or not code['all_original_source_draws_and_complete_target_counts_reconciled']
            or not head['exported_projections_equal_complete_resume'] or not head['all_original_draws_and_targets_reconciled']
            or not head_results['every_original_native_image_output_retained']
            or code_plan['base_checkpoint_sha256'] != head_plan['base_checkpoint_sha256']
            or code_plan['tokenizer_sha256'] != head_plan['tokenizer_sha256']):
        raise ValueError('Complete source-bound own components with one identical original base/tokenizer required')
    references = dict(base_checkpoint=Path(code_plan['base_checkpoint']),
        tokenizer=Path(code_plan['base_checkpoint']).parent / 'tokenizer.json',
        code_checkpoint=ROOT / 'checkpoints/own-code-generation-repair-v1/model.pt',
        code_initial_plan=ROOT / 'reports/own-code-delta-v1/plan.json', code_training_plan=CODE_REPORTS / 'plan.json',
        grounding_head=ROOT / 'checkpoints/frozen-base-pointer-v1/head.pt', grounding_plan=HEAD_REPORTS / 'plan.json')
    expected = dict(base_checkpoint=code_plan['base_checkpoint_sha256'], tokenizer=code_plan['tokenizer_sha256'],
        code_checkpoint=code_receipt['trained_components_sha256'], code_initial_plan=code_plan['initial_plan_sha256'],
        code_training_plan=file_hash(CODE_REPORTS / 'plan.json'), grounding_head=head_receipt['head_sha256'],
        grounding_plan=file_hash(HEAD_REPORTS / 'plan.json'))
    files = {str(path): expected[name] for name, path in references.items()}
    names = ['scripts/prepare_own_assistant_candidate_v1.py', 'src/slm_lab/own_assistant_v1.py',
        'tests/test_own_assistant_v1.py', 'reports/own-assistant-v1-tests.log',
        'reports/own-code-generation-repair-v1/independent-comparison.json',
        'reports/own-code-generation-repair-endpoint-v2/complete.json',
        'reports/frozen-base-pointer-v1/complete-head-source-readback.json',
        'reports/frozen-base-pointer-v1/matched-attention-and-native-comparison.json',
        'reports/frozen-base-pointer-v1/restored.json']
    for name in names:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    guard = FileBindings(files); guard.check(full=True)
    OUTPUT.mkdir(); REPORTS.mkdir()
    spec = dict(format=MANIFEST_FORMAT, created_at=datetime.now(timezone.utc).isoformat(),
        own_code_target_sha256=code_plan['own_code_target_sha256'],
        grounding_data_admission_sha256=head_plan['data_admission_sha256'],
        actual_counts=dict(shared_original_backbone=706012672, additional_code=279651840,
                           additional_grounding=786432, total_unique=986450944),
        component_training_updates=dict(code=815, grounding=1137),
        explicit_routes=['general', 'code', 'grounding'], automatic_routing_unproven=True,
        component_metric_scope='Original public/local development and approximate-human-attention component assessments; not a final standardized combined-model score.',
        measured_code_public_subset=code['public_subset'],
        measured_local_code_functional={name: value['candidate']['functional'] for name, value in code['native_comparisons'].items()
                                        if name in ['code-completion.json', 'code-instruction.json', 'C++.json']},
        raw_native_image_outputs_preserved_by_frozen_base=True,
        fresh_genuine_human_alignment_of_this_composite_completed=False,
        experimental_candidate_not_a_best_policy_selection=True, candidate_promoted=False, goal_achieved=False)
    for name, path in references.items():
        spec[name] = os.path.relpath(path, OUTPUT)
        spec[name + '_sha256'] = expected[name]
    manifest = OUTPUT / 'manifest.json'; atomic_json(manifest, spec)
    atomic_json(REPORTS / 'packaging.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        manifest=str(manifest), manifest_sha256=file_hash(manifest), source_files_sha256=files,
        all_component_weights_actually_trained_and_completed=True, weight_files_copied=0,
        original_backbone_count_to_load=1, declared_total_unique_parameters=986450944,
        full_combined_loader_and_route_parity_validation_still_required=True,
        code_candidate_not_promoted_despite_syntax_gain=True,
        original_code_reporting_failure_preserved_and_independent_CPU_recount_complete=True,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(manifest=str(manifest), manifest_sha256=file_hash(manifest), actual_counts=spec['actual_counts'],
                         trained_updates=spec['component_training_updates'], candidate_promoted=False), indent=2), flush=True)


if __name__ == '__main__':
    main()
