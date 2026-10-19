"""Reconcile every completed reference check with unchanged original v2 source bytes."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, ROLES

ROOT = Path(__file__).resolve().parents[1]
COMPATIBLE = 'all_recognized_samples_agree_under_available_source_rule'


def main():
    reports = ROOT / 'reports/codenet-complete-reference-readback-v1'
    if reports.exists():
        raise ValueError('Preserve any earlier readback evidence')
    if Path('/proc/202962').exists():
        fields = Path('/proc/202962/stat').read_text().rsplit(')', 1)[1].split()
        if fields[19] == '21523923' and fields[0] not in {'Z', 'X'}:
            raise ValueError('Original exact reference checker must be terminal before complete readback')
    source = ROOT / 'reports/codenet-complete-reference-checks-v1'
    summary = json.loads((source / 'summary.json').read_text())
    plan = json.loads((source / 'plan.json').read_text())
    for name, key in [('plan.json', 'plan_sha256'), ('records.jsonl', 'records_sha256'),
            ('selection.jsonl', 'selection_sha256')]:
        if file_hash(source / name) != summary[key]:
            raise ValueError('Exact complete reference artifacts required')
    if plan['explicit_smoke_limit_per_language'] != 0 or summary['references_checked'] != 17729:
        raise ValueError('Entire predeclared17729-source cohort required, not a smoke subset')
    selected = [json.loads(line) for line in (source / 'selection.jsonl').open()]
    records = [json.loads(line) for line in (source / 'records.jsonl').open()]
    selected_by_member = {r['original_archive_member']: r for r in selected}
    results_by_member = {r['original_archive_member']: r for r in records}
    if (len(selected_by_member) != len(selected) or len(results_by_member) != len(records)
            or set(selected_by_member) != set(results_by_member) or len(records) != 17729):
        raise ValueError('Every exact selected original source must occur once in execution records')
    counts = Counter()
    for member, result in results_by_member.items():
        expected = selected_by_member[member]
        for key in ['source_problem_id', 'group_id', 'split', 'language', 'original_source_body_sha256']:
            if result[key] != expected[key]:
                raise ValueError('Original execution source/role identity differs')
        samples = result['samples']
        if not samples:
            qualification = 'compiler_or_original_environment_review_needed'
        elif any(s['status'] != 'completed' for s in samples):
            qualification = 'runtime_or_original_environment_review_needed'
        elif all(s['sample_agrees'] for s in samples):
            qualification = COMPATIBLE
        else:
            qualification = 'sample_or_original_oracle_review_needed'
        if result['qualification'] != qualification:
            raise ValueError('Qualification does not rederive from the recorded execution evidence')
        for sample in samples:
            oracle = sample['source_specific_oracle']
            if oracle is not None and bool(sample['sample_agrees']) != bool(oracle['passed']):
                raise ValueError('Recorded source-specific oracle and agreement disagree')
        counts[qualification] += 1
    if dict(counts) != summary['qualification_counts']:
        raise ValueError('Every original reference qualification count must rederive')
    pack = CompleteCodePack(ROOT / 'data/codenet-durable-pack-v2', 'train', purpose='training')
    matched, local, qualified_train, by_role = set(), Counter(), [], defaultdict(Counter)
    with gzip.open(pack.folder / 'original-rows.jsonl.gz', 'rt') as stream:
        for position, line in enumerate(stream):
            row = json.loads(line)
            role = row['split']
            local_position = local[role]
            local[role] += 1
            member = row['original_archive_member']
            if member not in results_by_member:
                continue
            record = results_by_member[member]
            if (member in matched or row['token_pack_index'] != position
                    or int(pack.index[position]['role']) != ROLES[role]
                    or any(row[k] != record[k] for k in ['source_problem_id', 'group_id', 'split', 'language', 'original_source_body_sha256'])
                    or hashlib.sha256(row['response'].encode()).hexdigest() != record['original_source_body_sha256']):
                raise ValueError('Every executed reference must match its complete original v2 program bytes/role')
            matched.add(member)
            by_role[role][record['qualification']] += 1
            if role == 'train' and record['qualification'] == COMPATIBLE:
                qualified_train.append(dict(id=row['id'], original_archive_member=member,
                    original_source_body_sha256=record['original_source_body_sha256'],
                    global_position=position, local_position=local_position, role=role,
                    source_problem_id=row['source_problem_id'], group_id=row['group_id'],
                    language=row['language'], source_reference_check=record['qualification'],
                    complete_original_input_ids_sha256=row['complete_input_ids_sha256']))
    if matched != set(results_by_member) or len(qualified_train) != 14112:
        raise ValueError('Complete cross-version source readback and exact training-only subset required')
    if any(r['role'] != 'train' for r in qualified_train):
        raise ValueError('No validation/test source may enter a training-only index')
    reports.mkdir()
    index_path = reports / 'sample-compatible-original-training-index.jsonl'
    with index_path.open('x') as output:
        for row in qualified_train:
            output.write(json.dumps(row) + '\n')
    paths = [Path(__file__), source / 'summary.json', source / 'plan.json', source / 'records.jsonl',
        source / 'selection.jsonl', pack.folder / 'manifest.json', pack.folder / 'admission.json',
        ROOT / 'src/slm_lab/codenet_token_pack_v1.py']
    bound = {str(p): file_hash(p) for p in paths}
    bound.update({str(pack.folder / name): h for name, h in pack.manifest['files_sha256'].items()})
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=bound,
        complete_reference_actor_terminal=True, references_reconciled=len(matched),
        all_original_program_bytes_roles_and_groups_rechecked=True,
        recorded_execution_qualification_counts_recomputed=dict(counts),
        qualifications_by_original_role={role: dict(values) for role, values in by_role.items()},
        sample_compatible_training_only_rows=len(qualified_train),
        sample_compatible_training_only_by_language=dict(Counter(r['language'] for r in qualified_train)),
        training_only_index=str(index_path), training_only_index_sha256=file_hash(index_path),
        validation_or_test_rows_in_training_index=0, existing_admission_or_corpus_modified=False,
        source_reviews_used_as_negative_preferences=False, new_neural_training_started=False,
        model_capability_assessed=False, goal_achieved=False,
        scope='Completed eligible original-source Python3/C++ cohort under the original available runtime/sample rules.14112 training-source programs agree with recognized published samples; this does not prove hidden-case correctness or efficiency, certify all54 languages, turn review outcomes into wrong-code labels, or establish generated model performance. Broad historical Accepted-source admission stays unchanged; no automatic new training or promotion.')
    atomic_json(reports / 'results.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_files_sha256'}, indent=2))


if __name__ == '__main__':
    main()
