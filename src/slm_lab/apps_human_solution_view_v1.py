"""Complete public human-solution CE view with named exact screens.

This is neither execution-verified feedback nor human preference RLHF. The
original candidate and every exclusion remain retained separately. The named
exact screens do not certify semantic/final-benchmark independence.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path

import numpy as np

from .codenet_checkpoint_v1 import FileBindings
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash
from .diverse_images import atomic_json

FORMAT = 'own-APPS-human-CE-named-exact-source-view-v1'


def prepare(pack_folder, output):
    pack_folder, output = Path(pack_folder).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('Preserve prior source views; no overwrite')
    audit_path = pack_folder / 'independent-source-audit.json'
    audit = json.loads(audit_path.read_text())
    if (audit['all_actual_complete_pack_tokens_masks_and_source_rows_reencoded'] is not True
            or audit['all_originals_accounted_for_exactly_once'] is not True
            or audit['original_problem_URL_question_components_crossing_roles']
            or audit['publisher_test_rows_encoded']):
        raise ValueError('Complete independently reconciled original source roles/tokens required')
    FileBindings(audit['files_sha256']).check(full=True)
    packs = {role: CompleteCodePack(pack_folder, role, purpose='audit') for role in ['train', 'validation']}
    review_path = pack_folder / 'exact-normalized-duplicate-review.json'
    review = json.loads(review_path.read_text()); exclusions = defaultdict(set)
    for group in review['internal_train_validation']:
        for identity in group['train_ids']: exclusions[identity].add('normalized_program_occurs_in_internal_validation')
    for row in review['named_held_source_matches']:
        for name in row['held_sources']: exclusions[row['id']].add('normalized_program_matches_named_held_source:' + name)
    output.mkdir()
    selections = {'train': [], 'validation': []}; records = {'train': [], 'validation': []}; excluded = []
    cursors = Counter(); targets = Counter()
    with gzip.open(pack_folder / 'original-rows.jsonl.gz', 'rt') as stream:
        for row in map(json.loads, stream):
            role = row['split']; local = cursors[role]; cursors[role] += 1
            if row['id'] in exclusions:
                excluded.append(dict(id=row['id'], role=role, original_local_position=local,
                    reasons=sorted(exclusions[row['id']]), original_source_bytes_preserved=True))
                continue
            selections[role].append(local); targets[role] += row['target_tokens']
            records[role].append({k: row[k] for k in ['id', 'group_id', 'split', 'language', 'source_problem_id',
                'original_solution_index', 'target_tokens', 'complete_input_ids_sha256']})
    if set(exclusions) != {r['id'] for r in excluded}: raise ValueError('Every exact source review must refer to an actual candidate row')
    artifacts = []
    for role in packs:
        if cursors[role] != len(packs[role]) or not selections[role]: raise ValueError('All actual candidate source rows must be accounted for')
        path = output / (role + '-local-positions.npy'); np.save(path, np.asarray(selections[role], dtype='<i4'), allow_pickle=False); artifacts.append(path)
        path = output / (role + '-records.jsonl')
        path.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in records[role])); artifacts.append(path)
    path = output / 'excluded-rows.jsonl'
    path.write_text(''.join(json.dumps(r, separators=(',', ':')) + '\n' for r in excluded)); artifacts.append(path)
    manifest = packs['train'].manifest
    files = {str(pack_folder / n): h for n, h in manifest['files_sha256'].items()}
    files.update({str(p): file_hash(p) for p in [pack_folder / 'manifest.json', audit_path, review_path, Path(__file__).resolve(), *artifacts]})
    result = dict(format=FORMAT, prepared_at=datetime.now(timezone.utc).isoformat(), complete_pack=str(pack_folder),
        tokenizer_sha256=file_hash(pack_folder / 'tokenizer.json'), source_counts={r: len(v) for r, v in selections.items()},
        complete_response_targets_by_role=dict(targets), excluded_original_rows=len(excluded),
        excluded_by_role=dict(Counter(r['role'] for r in excluded)), admitted_for_original_human_solution_CE=True,
        execution_verified=False, human_preferences=False, machine_execution_preference_labels=False,
        only_publisher_train_sources=True, whole_original_questions_programs_and_EOS=True, clipped_programs=0,
        exact_screen_scope=audit['held_screen_scope'], semantic_or_prior_corpus_contamination_certified_clean=False,
        named_normalized_train_validation_and_held_matches_retained_outside_view=True, files_sha256=files,
        original_missing195test_files_not_fabricated=True, model_updates=0, goal_achieved=False,
        scope='Named exact-screened original human-solution SFT input view only. No finite execution correctness, independent final benchmark, model improvement or full earlier-source production claim.')
    atomic_json(output / 'admission.json', result)
    return result


class AppsHumanSolutions:
    def __init__(self, folder, role='train', purpose='training'):
        if role not in ['train', 'validation'] or purpose not in ['training', 'audit'] or (purpose == 'training' and role != 'train'):
            raise ValueError('Only exact original train roles may enter human-solution optimization')
        self.folder = Path(folder).resolve(); self.role = role
        admission_path = self.folder / 'admission.json'
        self.manifest = json.loads(admission_path.read_text())
        if (self.manifest.get('format') != FORMAT or self.manifest.get('admitted_for_original_human_solution_CE') is not True
                or self.manifest.get('execution_verified') is not False or self.manifest.get('human_preferences') is not False
                or self.manifest.get('only_publisher_train_sources') is not True or self.manifest.get('clipped_programs') != 0):
            raise ValueError('Exact original complete named-screened human CE source admission required')
        self.guard = FileBindings({**self.manifest['files_sha256'], str(admission_path): file_hash(admission_path)})
        self.guard.check(full=True)
        self.pack = CompleteCodePack(self.manifest['complete_pack'], role, purpose='audit')
        self.positions = np.load(self.folder / (role + '-local-positions.npy'), mmap_mode='r', allow_pickle=False)
        self.records = list(map(json.loads, (self.folder / (role + '-records.jsonl')).read_text().splitlines()))
        if (self.positions.dtype != np.dtype('<i4') or self.positions.ndim != 1
                or len(self.positions) != self.manifest['source_counts'][role] or len(self.records) != len(self.positions)
                or np.any(self.positions < 0) or np.any(self.positions >= len(self.pack))
                or np.any(np.diff(self.positions) <= 0) or any(r['split'] != role for r in self.records)):
            raise ValueError('Complete unique increasing exact-role original source indices required')

    def __len__(self): return len(self.positions)

    def example(self, position):
        self.guard.check()
        if type(position) is not int or not 0 <= position < len(self): raise ValueError('Exact local human-solution source index required')
        ids, labels = self.pack.sequence(int(self.positions[position]))
        return dict(id=self.records[position]['id'], role=self.role, language='Python',
            group_id=self.records[position]['group_id'], positive=(ids, labels),
            original_human_solution=True, execution_verified=False, human_preferences=False)
