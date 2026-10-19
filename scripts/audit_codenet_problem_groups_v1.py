"""Verify original body joins and propose problem groups before assigning splits."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from slm_lab.codenet_problem_groups_v1 import ProblemGroups, canonical_member, identical_groups, selected_clone_groups
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def group_summary(mapping):
    sizes = Counter(mapping.values())
    return {'original_problems': len(mapping), 'components': len(sizes),
            'largest_component_problems': max(sizes.values()),
            'component_size_distribution': dict(Counter(sizes.values()))}


def main(folder_name='codenet-problem-groups-v1', qualify_python_tail=False, extra_bound_paths=()):
    folder = ROOT / 'reports' / folder_name
    if folder.exists():
        raise ValueError('Preserve earlier problem-group diagnosis')
    folder.mkdir()
    def status(message, **fields):
        atomic_json(folder / 'status.json', {'status': message,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'split_assignment_completed': False, 'model_training_started': False,
            'goal_achieved': False, **fields})
    bodies_path = ROOT / 'reports/codenet-candidate-bodies-v1/bodies.json'
    inspection_path = ROOT / 'reports/codenet-original-inspection-v1/inspection.json'
    selection_path = ROOT / 'data/codenet-candidate-selection-v1/manifest.json'
    bodies = json.loads(bodies_path.read_text())
    inspection = json.loads(inspection_path.read_text())
    selection = json.loads(selection_path.read_text())
    cache = Path(bodies['candidate_cache'])
    source = cache / 'source-bodies.bin'
    index = cache / 'source-index.jsonl'
    metadata = selection_path.with_name('selected-metadata.jsonl')
    original_cache = Path(inspection['metadata_cache'])
    original_index = original_cache / 'retained-members.jsonl'
    if not bodies['whole_gzip_stream_drained_crc_checked'] or bodies['source_archive_sha256'] != inspection['archive_sha256']:
        raise ValueError('Complete matching original source body copy required')
    expected = {source: bodies['source_body_store_sha256'], index: bodies['source_index_sha256'],
                metadata: selection['selected_metadata_sha256'], original_index: inspection['retained_member_index_sha256']}
    for name, record in bodies['auxiliary_members'].items():
        expected[cache / record['cache_relative_path']] = record['sha256']
    for path, digest in expected.items():
        if file_hash(path) != digest:
            raise ValueError('Exact original group/source input changed')
    bound = {**expected, **{p: file_hash(p) for p in [Path(__file__), bodies_path, inspection_path,
        selection_path, ROOT / 'src/slm_lab/codenet_problem_groups_v1.py', *extra_bound_paths]}}
    if qualify_python_tail:
        from slm_lab.codenet_publisher_qualification_v1 import QualifiedPublisherLines
        p = ROOT / 'src/slm_lab/codenet_publisher_qualification_v1.py'
        bound[p] = file_hash(p)
    problem_list = original_cache / 'metadata/problem_list.csv'
    record = next(r for r in (json.loads(x) for x in original_index.read_text().splitlines())
                  if r['cache_relative_path'] == 'metadata/problem_list.csv')
    if file_hash(problem_list) != record['sha256']:
        raise ValueError('Original problem list differs')
    bound[problem_list] = record['sha256']
    with problem_list.open() as stream:
        problems = [row['id'] for row in csv.DictReader(stream)]
    if len(problems) != len(set(problems)) or len(problems) != 4053:
        raise ValueError('Original complete problem universe differs')
    selected = {row['archive_member']: row['metadata'] for row in (json.loads(x) for x in metadata.read_text().splitlines())}
    if len(selected) != bodies['source_bodies']:
        raise ValueError('Exact metadata/body cohort differs')
    status('checking every stored original source body and index offset')
    eligible = {}
    seen = set()
    offset = 0
    with source.open('rb') as stream, index.open() as entries:
        for line in entries:
            row = json.loads(line)
            member = row['archive_member']
            if member not in selected or member in seen or row['body_offset'] != offset:
                raise ValueError('Missing/duplicate/overlapping original body join')
            raw = stream.read(row['body_bytes'])
            if len(raw) != row['body_bytes'] or hashlib.sha256(raw).hexdigest() != row['body_sha256']:
                raise ValueError('Indexed original body bytes differ')
            m = selected[member]
            if int(m['code_size']) != row['publisher_declared_code_bytes']:
                raise ValueError('Original source metadata association differs')
            try:
                raw.decode('utf-8')
                valid = True
            except UnicodeDecodeError:
                valid = False
            if valid != row['utf8_valid'] or (b'\0' in raw) != row['contains_nul']:
                raise ValueError('Actual source encoding qualification differs')
            if m['language'] != 'Text' and valid and raw.strip() and b'\0' not in raw:
                eligible[member] = {'problem_id': m['problem_id'], 'language': m['language'],
                                    'sha256': row['body_sha256']}
            seen.add(member)
            offset += len(raw)
        if seen != set(selected) or stream.read(1) or offset != source.stat().st_size:
            raise ValueError('Complete nonoverlapping source-store population required')
    groups = ProblemGroups(problems)
    auxiliary = cache / 'auxiliary/duplicates'
    aliases = identical_groups((auxiliary / 'identical_problem_clusters').read_text())
    alias_merges = sum(groups.merge(members) for members in aliases)
    views = {'publisher_identical_problems': group_summary(groups.mapping())}
    fingerprints = {}
    exact_cross_problem_pairs = []
    for member, row in eligible.items():
        prior = fingerprints.setdefault(row['sha256'], member)
        other = eligible[prior]
        if row['problem_id'] != other['problem_id']:
            groups.merge([row['problem_id'], other['problem_id']])
            exact_cross_problem_pairs.append({'first_member': prior, 'second_member': member,
                                            'source_sha256': row['sha256']})
    views['with_exact_selected_programs'] = group_summary(groups.mapping())
    clone_coverage = {}
    for language in ['C', 'C++', 'Java', 'Python']:
        status('grouping selected original publisher clone relations', language=language)
        matched = set()
        relation_groups = cross_problem_groups = 0
        qualification = None
        with (auxiliary / language / f'Project_CodeNet-{language}.clusters').open() as stream:
            lines = stream
            if qualify_python_tail and language == 'Python':
                qualification = QualifiedPublisherLines(stream, '/Volume1/AI4CODE/CodeNe')
                lines = qualification
            for members in selected_clone_groups(lines, eligible, language):
                relation_groups += 1
                matched.update(members)
                ids = {eligible[member]['problem_id'] for member in members}
                if len(ids) > 1:
                    cross_problem_groups += 1
                    groups.merge(ids)
        expected_language = {member for member, row in eligible.items() if row['language'] == language}
        if expected_language and not matched:
            raise ValueError('No selected source identities found in publisher clone data')
        with (auxiliary / language / f'{language}_accepted_outliers.txt').open() as stream:
            outliers = {canonical_member(line) for line in stream if line.strip()}
        clone_coverage[language] = {'selected_programs': len(expected_language),
            'selected_members_in_publisher_clusters': len(matched),
            'selected_members_missing_from_publisher_clusters': sorted(expected_language - matched),
            'publisher_cluster_groups_with_selected_members': relation_groups,
            'publisher_cluster_groups_linking_selected_problems': cross_problem_groups,
            'selected_publisher_size_outliers': sorted(expected_language & outliers),
            'publisher_incomplete_terminal_record': qualification.incomplete_terminal_record if qualification else None}
    mapping = groups.mapping()
    views['with_selected_publisher_clones'] = group_summary(mapping)
    if any(file_hash(path) != digest for path, digest in bound.items()):
        raise ValueError('Bound grouping implementation/source changed')
    atomic_json(folder / 'groups.json', {'completed_at': datetime.now(timezone.utc).isoformat(),
        'source_archive_sha256': bodies['source_archive_sha256'],
        'bound_files_sha256': {str(p): d for p, d in bound.items()},
        'all_candidate_body_joins_verified': len(seen), 'structurally_eligible_programs': len(eligible),
        'structural_exclusions': len(seen) - len(eligible),
        'publisher_identical_problem_clusters': len(aliases), 'publisher_identical_problem_merges': alias_merges,
        'views': views, 'exact_selected_cross_problem_pairs': exact_cross_problem_pairs,
        'clone_coverage': clone_coverage, 'problem_to_proposed_group': mapping,
        'split_assignment_completed': False, 'dataset_admitted_to_training': False,
        'model_training_started': False, 'goal_achieved': False,
        'scope': 'Conservative internal grouping proposal only. Text/output-only, empty, invalid-encoding and NUL bodies do not define program-copy links. All known identical-problem aliases are included; published selected clone links cover four languages with explicit incomplete Python source coverage, not every near duplicate or earlier pretraining contamination. Whole-context/target review, final splits and useful model capability remain pending.'})
    status('original source joins and proposed problem groups checked; splits pending',
           views=views, checked_body_joins=len(seen))
    print(json.dumps(views, indent=2))


if __name__ == '__main__':
    main()
