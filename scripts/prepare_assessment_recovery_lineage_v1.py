"""Prepare versioned routes after completed training and terminal eval failure.

All old controllers are already terminal. Keep their failed evidence. The
new routes change readiness and paths only; neural objectives, original
source budgets, model factories and final benchmark populations stay fixed.
"""
import ast
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / 'reports/assessment-recovery-lineage-v1-file-map.json'


def replace_function(text, name, replacement):
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
    lines = text.splitlines(keepends=True)
    return ''.join(lines[:fn.lineno - 1]) + replacement + '\n' + ''.join(lines[fn.end_lineno:])


def transform(text, literals, human=False):
    for before, after in literals.items():
        text = text.replace(before, after)
    if human:
        text = text.replace('from slm_lab.function_repair_GPU_contract_v1 import study_released',
            'from slm_lab.function_repair_GPU_contract_v2 import study_released, public_ready')
        text = replace_function(text, 'previous_ready',
            'def previous_ready(identity):\n'
            '    if not study_released(ROOT):return False\n'
            '    return public_ready(ROOT,identity)\n')
        text = text.replace("PREVIOUS/'complete.json', PREVIOUS/'restored.json'",
            "ROOT/'reports/public-pretraining-assessment-recovery-v1/complete.json', "
            "ROOT/'reports/public-pretraining-assessment-recovery-v1/restored.json', "
            "ROOT/'reports/public-pretraining-complete-epoch-value-readback-CPU-v2/complete.json'")
        text = text.replace("ROOT/'src/slm_lab/function_repair_GPU_contract_v1.py'",
            "ROOT/'src/slm_lab/function_repair_GPU_contract_v2.py'")
        text = text.replace("ROOT/'tests/test_function_repair_GPU_lineage_v1.py'",
            "ROOT/'tests/test_assessment_recovery_lineage_v1.py'")
        text = text.replace("ROOT/'reports/function-repair-GPU-lineage-v1-tests.log'",
            "ROOT/'reports/assessment-recovery-lineage-v1-tests.log'")
    return text


def main():
    if MAP.exists():
        raise ValueError('Preserve every prior prepared recovery lineage')
    previous = json.loads((ROOT / 'reports/function-repair-GPU-lineage-v1-file-map.json').read_text())
    routes = []
    literals = {}
    for row in previous['stages']:
        new = dict(stage=row['stage'], previous_script=row['recovery_script'], previous_queue=row['queue'],
            recovery_script=row['recovery_script'].replace('_repair_gate_v1.py', '_assessment_recovery_gate_v1.py'),
            queue=row['queue'].replace('-repair-gate-v1-', '-assessment-recovery-gate-v1-'))
        routes.append(new)
        literals[row['queue']] = new['queue']
    literals['reports/broad-anchored-code-v7-repair-gate-v1-held-APPS-cohort.json'] = 'reports/broad-anchored-code-v7-assessment-recovery-gate-v1-held-APPS-cohort.json'
    literals['reports/AI2D-fixed-exposure-strata-repair-gate-v1'] = 'reports/AI2D-fixed-exposure-strata-assessment-recovery-gate-v1'
    files = {}
    for i, row in enumerate(routes):
        source = ROOT / 'scripts' / row['previous_script']
        target = ROOT / 'scripts' / row['recovery_script']
        if target.exists() or (ROOT / row['queue']).exists():
            raise ValueError('Only fresh controllers and queues may be prepared')
        target.write_text(transform(source.read_text(), literals, human=i == 0))
        files[str(source)] = file_hash(source)
        files[str(target)] = file_hash(target)
    source = ROOT / 'src/slm_lab/function_repair_GPU_contract_v1.py'
    target = source.with_name('function_repair_GPU_contract_v2.py')
    if target.exists():
        raise ValueError('Fresh contract required')
    text = source.read_text().replace('full-function-repair-comparison-GPU-v1', 'full-function-repair-comparison-GPU-v2')
    text = text.replace('function-repair-GPU-lineage-v1', 'assessment-recovery-lineage-v1')
    text = text.replace("RUNTIME='reports/public-pretraining-recovery-v1'", "RUNTIME='reports/public-pretraining-assessment-recovery-v1'")
    text = replace_function(text, 'public_ready',
        'def public_ready(root,identity):\n'
        '    from .public_pretraining_assessment_recovery_v1 import ORIGINAL, assessment_ready\n'
        "    if {k:identity[k] for k in ['pid','ticks']}!=ORIGINAL:\n"
        "        raise ValueError('Exact original complete public source owner required')\n"
        '    return assessment_ready(root)\n')
    target.write_text(text)
    files[str(source)] = file_hash(source)
    files[str(target)] = file_hash(target)
    source = ROOT / 'scripts/train_full_function_repair_comparison_GPU_v1.py'
    target = source.with_name('train_full_function_repair_comparison_GPU_v2.py')
    if target.exists():
        raise ValueError('Fresh GPU controller required')
    text = source.read_text().replace('slm_lab.function_repair_GPU_contract_v1', 'slm_lab.function_repair_GPU_contract_v2')
    text = text.replace('reports/function-repair-GPU-lineage-v1/admission.json', 'reports/assessment-recovery-lineage-v1/admission.json')
    text = text.replace("ROOT/PUBLIC/'complete.json',ROOT/PUBLIC/'restored.json'",
        "ROOT/'reports/public-pretraining-complete-epoch-value-readback-CPU-v2/complete.json',"
        "ROOT/'reports/public-pretraining-recovery-v1/failure.json',ROOT/'reports/public-pretraining-recovery-v1/restored.json'")
    text = text.replace("paths.extend(python_sources(ROOT,[Path(__file__),ROOT/'tests/test_function_repair_GPU_controller_v1.py']))",
        "paths.append(ROOT/'reports/assessment-recovery-lineage-v1-tests.log')\n"
        "    paths.extend(python_sources(ROOT,[Path(__file__),ROOT/'tests/test_function_repair_GPU_controller_v1.py',"
        "ROOT/'tests/test_assessment_recovery_lineage_v1.py']))")
    target.write_text(text)
    files[str(source)] = file_hash(source)
    files[str(target)] = file_hash(target)
    atomic_json(MAP, dict(stages=routes, permitted_literal_replacements=literals, source_files_sha256=files,
        only_human_readiness_and_public_research_readiness_and_evidence_paths_change=True,
        all_original_neural_objectives_factories_populations_and_nine_final_endpoints_preserved=True,
        failed_original_runtime_and_controller_sources_preserved=True, goal_achieved=False))


if __name__ == '__main__':
    main()
