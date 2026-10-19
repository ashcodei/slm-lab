"""Account for every original reference in the completed fixed GNU cohort."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    folder = ROOT / 'reports/codenet-gnu-original-references-v1'
    output = folder / 'independent-population-check.json'
    if output.exists():
        raise ValueError('Preserve earlier completed-cohort evidence')
    summary = json.loads((folder / 'summary.json').read_text())
    plan = json.loads((folder / 'plan.json').read_text())
    for key, name in [('plan_sha256', 'plan.json'), ('selection_sha256', 'selection.jsonl'), ('records_sha256', 'records.jsonl')]:
        if file_hash(folder / name) != summary[key]:
            raise ValueError('Completed cohort artifacts differ')
    for name, digest in plan['source_files_sha256'].items():
        if file_hash(name) != digest:
            raise ValueError('Original GNU source/runtime input differs: ' + name)
    selection = [json.loads(line) for line in (folder / 'selection.jsonl').open()]
    selected = {r['original_archive_member']: r for r in selection}
    if len(selection) != len(selected) or len(selected) != summary['references_checked'] or len(selected) != 10265:
        raise ValueError('Complete fixed unique GNU source selection required')
    paths = [Path(__file__), folder / 'summary.json', folder / 'plan.json',
             folder / 'selection.jsonl', folder / 'records.jsonl']
    bound = {str(p): file_hash(p) for p in paths}
    source_seen, result_seen = set(), set()
    pack = ROOT / 'data/codenet-durable-pack-v1'
    with gzip.open(pack / 'original-rows.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line); member = row['original_archive_member']
            if member not in selected:
                continue
            if member in source_seen or any(row[k] != v for k, v in selected[member].items()):
                raise ValueError('Original source identity/roles differ')
            if hashlib.sha256(row['response'].encode()).hexdigest() != row['original_source_body_sha256']:
                raise ValueError('Original programme bytes differ')
            source_seen.add(member)
    counts, by_role, sample_statuses, components = Counter(), defaultdict(Counter), Counter(), set()
    with (folder / 'records.jsonl').open() as f:
        for line in f:
            r = json.loads(line); member = r['original_archive_member']
            if member in result_seen or member not in selected or any(r[k] != v for k, v in selected[member].items()):
                raise ValueError('Reference result selection/identity/role differs')
            result_seen.add(member); components.add(r['group_id'])
            qualification = r['qualification']; samples = r['samples']; comp = r['compilation']
            if qualification == 'compiler_or_original_environment_review_needed':
                if samples or (comp['status'] == 'completed' and comp['returncode'] == 0):
                    raise ValueError('Compiler qualification inconsistent with recorded execution')
            else:
                if not samples or comp['status'] != 'completed' or comp['returncode'] != 0:
                    raise ValueError('Recorded usable compilation and nonempty samples required')
                if [s['number'] for s in samples] != list(range(1, len(samples)+1)):
                    raise ValueError('Original sample ordering differs')
                expected = ('runtime_or_original_environment_review_needed'
                    if any(s['status'] != 'completed' for s in samples)
                    else 'all_recognized_samples_agree_under_available_source_rule'
                    if all(s['sample_agrees'] for s in samples)
                    else 'sample_or_original_oracle_review_needed')
                if qualification != expected:
                    raise ValueError('Qualification differs from recorded sample outcomes')
            counts[qualification] += 1; by_role[r['split']][qualification] += 1
            sample_statuses.update(s['status'] for s in samples)
    if result_seen != set(selected) or source_seen != set(selected) or dict(counts) != summary['qualification_counts']:
        raise ValueError('Entire original reference population and recorded counts required')
    for name, digest in {**plan['source_files_sha256'], **bound}.items():
        if file_hash(name) != digest:
            raise ValueError('Source/runtime/report changed during independent check')
    atomic_json(output, dict(completed_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound, original_runtime_inputs_sha256=plan['source_files_sha256'],
        complete_predeclared_references=len(result_seen), original_sealed_components=len(components),
        qualification_counts=dict(counts), qualifications_by_role={k: dict(v) for k, v in by_role.items()},
        recorded_sample_status_counts=dict(sample_statuses),
        exact_original_source_and_roles_preserved=True, every_predeclared_reference_accounted_for=True,
        recorded_qualification_arithmetic_checked=True, source_programmes_reexecuted=False,
        qualified_references_automatically_labelled_wrong=False, dataset_admitted_to_training=False,
        new_model_training_started=False, goal_achieved=False,
        scope='Complete source/sample-compatibility accounting under one fixed GNU profile. This rechecks recorded outcomes and source identity, not unrecorded full outputs, hidden-case correctness, algorithmic efficiency or model capability.'))
    print('completed_reference_cohort_verified', len(result_seen), json.dumps(dict(counts)), flush=True)


if __name__ == '__main__':
    main()
