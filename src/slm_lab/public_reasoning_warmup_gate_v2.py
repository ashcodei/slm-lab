"""Exact predecessor completion gate for reusing the released CPU allocation."""
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal
from .function_repair_GPU_contract_v2 import public_ready


def ready(root,edge_owner,main_owner):
    root=Path(root);folder=root/'reports/public-edge-preference-comparison-CPU-v2'
    # An existing receipt never substitutes for an actual terminal birth.
    if not terminal(edge_owner):return False
    path=folder/'complete.json'
    if not path.exists() or (folder/'failure.json').exists():
        raise ValueError('The exact preceding CPU comparison ended without complete original training and assessment')
    done=json.loads(path.read_text())
    if (not done.get('complete') or done.get('owner')!=edge_owner
            or done.get('all77machine_pairs_three_epochs_all_original_contexts_and_EOS_preserved') is not True
            or done.get('all_baseline_and_final15source48code_and_broad_local_suites_complete') is not True):
        raise ValueError('The exact full predecessor source and native-assessment scope must finish first')
    if not terminal(main_owner):return False
    return public_ready(root,main_owner)
