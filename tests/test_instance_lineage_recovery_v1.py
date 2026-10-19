"""Verify the recovery copies preserve every scientific function and scope."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_all_ten_unstarted_controllers_only_change_declared_queue_literals_and_empty_strata_destination():
    mapping=json.loads((ROOT/'reports/instance-lineage-recovery-v1-file-map.json').read_text())
    assert len(mapping['stages'])==10
    assert len({row['queue'] for row in mapping['stages']})==10
    for row in mapping['stages']:
        old=(ROOT/'scripts'/row['original_script']).read_text()
        expected=old
        for a,b in mapping['permitted_literal_replacements'].items():expected=expected.replace(a,b)
        if 'strata' in row['original_script']:
            expected=expected.replace("OUT = ROOT / 'reports/AI2D-fixed-exposure-strata-v5'", "OUT = ROOT / 'reports/AI2D-fixed-exposure-strata-recovery-v1'")
        actual=(ROOT/'scripts'/row['recovery_script']).read_text()
        assert actual==expected
        before=[ast.dump(n) for n in ast.parse(old).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        after=[ast.dump(n) for n in ast.parse(actual).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        assert before==after
        assert row['queue'] in actual
