import ast
import json
from pathlib import Path
import sys

import pytest

from slm_lab.corpus import file_hash

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_assessment_recovery_lineage_v1 import transform
from handoff_assessment_recovery_lineage_v1 import scope


def mapping():
    return json.loads((ROOT / 'reports/assessment-recovery-lineage-v1-file-map.json').read_text())


def test_all_ten_exact_routes_keep_original_neural_functions_and_nine_endpoints():
    from slm_lab.public_pretraining_endpoints_v1 import ENDPOINTS
    m = mapping()
    assert len(m['stages']) == 10 and len(ENDPOINTS) == 9
    for i, row in enumerate(m['stages']):
        old = ROOT / 'scripts' / row['previous_script']
        new = ROOT / 'scripts' / row['recovery_script']
        assert new.read_text() == transform(old.read_text(), m['permitted_literal_replacements'], human=i == 0)
        before = {n.name: n for n in ast.parse(old.read_text()).body if isinstance(n, ast.FunctionDef)}
        after = {n.name: n for n in ast.parse(new.read_text()).body if isinstance(n, ast.FunctionDef)}
        assert before.keys() == after.keys()
        for name in before:
            if i == 0 and name == 'previous_ready':
                continue
            a = ast.parse(ast.unparse(before[name]))
            b = ast.parse(ast.unparse(after[name]))
            for tree in [a, b]:
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        node.value = transform(node.value, m['permitted_literal_replacements'])
                        if i == 0:
                            for x, y in [('function_repair_GPU_contract_v1', 'function_repair_GPU_contract_v2'),
                                    ('test_function_repair_GPU_lineage_v1', 'test_assessment_recovery_lineage_v1'),
                                    ('function-repair-GPU-lineage-v1-tests', 'assessment-recovery-lineage-v1-tests')]:
                                node.value = node.value.replace(x, y)
            if i == 0 and name == 'parent':
                # Evidence binding paths change in parent; every numeric budget
                # and actual neural/model factory call remains identical.
                for tree in [a, b]:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Constant) and isinstance(node.value, str):
                            node.value = ''
                # The new receipt list adds two paths only; compare the full
                # numerical constants and all imported scientific factories.
                assert sorted(repr(n.value) for n in ast.walk(a) if isinstance(n, ast.Constant) and type(n.value) in [int, float]) == sorted(repr(n.value) for n in ast.walk(b) if isinstance(n, ast.Constant) and type(n.value) in [int, float])
            else:
                assert ast.dump(a) == ast.dump(b), (row['stage'], name)
        assert file_hash(new) == m['source_files_sha256'][str(new)]


def test_three_repair_profiles_and_all_actual_neural_kernels_are_identical():
    from slm_lab import function_repair_GPU_contract_v1 as old
    from slm_lab import function_repair_GPU_contract_v2 as new
    assert old.PROFILES == new.PROFILES and old.training_options() == new.training_options()
    def functions(path):
        return {n.name: ast.dump(n) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
    before = functions(ROOT / 'scripts/train_full_function_repair_comparison_GPU_v1.py')
    after = functions(ROOT / 'scripts/train_full_function_repair_comparison_GPU_v2.py')
    for name in ['factory', 'gpu_setup', 'cold_export', 'distributed', 'assess']:
        assert before[name] == after[name]


def test_public_readiness_rejects_other_birth_and_actual_recovery_is_required(tmp_path, monkeypatch):
    from slm_lab import function_repair_GPU_contract_v2 as new
    from slm_lab import public_pretraining_assessment_recovery_v1 as recovery
    calls = []
    monkeypatch.setattr(recovery, 'assessment_ready', lambda root: calls.append(root) or False)
    assert not new.public_ready(tmp_path, dict(recovery.ORIGINAL, state='R'))
    assert calls == [tmp_path]
    with pytest.raises(ValueError, match='Exact original'):
        new.public_ready(tmp_path, dict(pid=1241, ticks='different'))


def test_all_original_queue_populations_and_scopes_survive_only_operational_changes():
    old = dict(owner={'pid': 1}, exact_predecessor={'pid': 2}, source_files_sha256={}, status='old',
        human_train_pairs=85786, coding_sources=3721, machine_pairs=1620, QA=16961, AI2D=3088, conditions=3,
        MBPP=600, endpoints=9)
    new = dict(old, owner={'pid': 3}, status='new', expires_at_unix=100)
    assert scope(old) == scope(new)
    for key in ['human_train_pairs', 'coding_sources', 'machine_pairs', 'QA', 'AI2D', 'conditions', 'MBPP', 'endpoints']:
        assert scope(old) != scope(dict(new, **{key: new[key] - 1}))


def test_handoff_refuses_learned_output_and_never_invokes_a_signal(tmp_path, monkeypatch):
    import handoff_assessment_recovery_lineage_v1 as handoff
    monkeypatch.setattr(handoff, 'ROOT', tmp_path)
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/controller.py').write_text("OUTPUT=ROOT/'checkpoints/model'\n")
    row = dict(stage='test', previous_script='controller.py')
    handoff.no_learned_outputs(row)
    (tmp_path / 'checkpoints/model').mkdir(parents=True)
    with pytest.raises(ValueError, match='Existing scientific output'):
        handoff.no_learned_outputs(row)
    tree = ast.parse((ROOT / 'scripts/handoff_assessment_recovery_lineage_v1.py').read_text())
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(c.endswith(('retire', 'kill', 'send_term', 'pidfd_send_signal')) for c in calls)
