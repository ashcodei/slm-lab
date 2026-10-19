"""Seal internal source roles before any new CodeNet model training."""
from datetime import datetime, timezone
import json
from pathlib import Path

from slm_lab.codenet_splits_v1 import SALT, component_role, partition
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'data/codenet-internal-splits-v1'
    if folder.exists():
        raise ValueError('Preserve earlier sealed roles')
    summary_path = ROOT / 'reports/codenet-complete-contexts-v1/summary.json'
    context_path = summary_path.with_name('records.jsonl')
    group_path = ROOT / 'reports/codenet-problem-groups-v2/groups.json'
    summary = json.loads(summary_path.read_text())
    grouping = json.loads(group_path.read_text())
    if file_hash(context_path) != summary['records_sha256'] or summary['encoder_truncated_sequences']:
        raise ValueError('Complete source-context audit differs')
    bound = {p: file_hash(p) for p in [summary_path, context_path, group_path, Path(__file__),
        ROOT / 'src/slm_lab/codenet_splits_v1.py']}
    records, counts = partition((json.loads(line) for line in context_path.open()),
                               grouping['problem_to_proposed_group'])
    if len(records) != summary['candidates_checked']:
        raise ValueError('Source candidate population differs')
    for path, digest in bound.items():
        if file_hash(path) != digest:
            raise ValueError('Source changed while partitioning')
    folder.mkdir()
    atomic_json(folder / 'plan.json', {
        'recorded_at': datetime.now(timezone.utc).isoformat(), 'salt': SALT,
        'role_rule': 'First64 SHA256 bits; train below90%, validation90–95%, test95–100%; hash only canonical component ID, never model outcomes.',
        'input_files_sha256': {str(p): digest for p, digest in bound.items()},
        'groups': 'Known original identical problems, exact selected program copies and available selected publisher clone relations. Python clone source is partial.',
        'source_programmes_or_model_weights_changed': False,
        'dataset_admitted_to_training': False, 'goal_achieved': False})
    path = folder / 'members.jsonl'
    with path.open('x') as output:
        for row in records:
            output.write(json.dumps(row) + '\n')
    roles = {problem: {'group_id': group, 'split': component_role(group)}
             for problem, group in grouping['problem_to_proposed_group'].items()}
    atomic_json(folder / 'problem-roles.json', roles)
    atomic_json(folder / 'manifest.json', {
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'source_candidates': len(records), 'counts': counts,
        'members_sha256': file_hash(path), 'problem_roles_sha256': file_hash(folder / 'problem-roles.json'),
        'plan_sha256': file_hash(folder / 'plan.json'),
        'dataset_admitted_to_training': False, 'new_model_training_started': False,
        'goal_achieved': False,
        'scope': 'Internal source roles only. Known linked problems and full token/mask copies cannot cross roles. Partial clone coverage, earlier-corpus exposure, semantic equivalence and judge correctness remain qualified. Execution checks and durable admitted training artifacts remain pending.'})
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
