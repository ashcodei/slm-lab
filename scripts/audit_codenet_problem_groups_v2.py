"""Repeat group diagnosis with the inspected original Python truncation qualified."""
from datetime import datetime, timezone

from audit_codenet_problem_groups_v1 import ROOT, main
from slm_lab.diverse_images import atomic_json


if __name__ == '__main__':
    if (ROOT / 'reports/codenet-problem-groups-v2').exists():
        raise ValueError('Preserve earlier qualified grouping evidence')
    try:
        main('codenet-problem-groups-v2', qualify_python_tail=True,
             extra_bound_paths=[ROOT / 'scripts/audit_codenet_problem_groups_v2.py'])
    except Exception as error:
        atomic_json(ROOT / 'reports/codenet-problem-groups-v2/status.json', {
            'status': 'qualified problem grouping stopped; requires inspection',
            'updated_at': datetime.now(timezone.utc).isoformat(), 'error': str(error),
            'split_assignment_completed': False, 'model_training_started': False,
            'goal_achieved': False})
        raise
