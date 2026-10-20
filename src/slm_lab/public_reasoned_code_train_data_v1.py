"""Admitted whole public R1 code/reasoning SFT, train role only.

Full historical evidence is checked when opening and during explicit final
audits. Hot guards bind the sealed numeric package and its admission receipts;
raw historical test outputs are not reread for every optimizer microbatch.
This data object owns no compute resources and supplies no reward oracle.
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct

import torch

from .codenet_checkpoint_v1 import FileBindings
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash

REVISION = 'eadf535931451525f3e5621d0f960c240bc62fd9'
VIEWS = ('both', 'code_only', 'reasoning')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def terminal(owner):
    path = Path('/proc') / str(owner['pid']) / 'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return True
    return fields[19] != str(owner['birth']) or fields[0] in ('Z', 'X')


class PublicReasonedCodeSFTData:
    role = 'train'
    source_is_public_synthetic = True
    use_as_RL_reward_oracle_forbidden = True
    human_preferences = False
    own_model_on_policy = False

    def __init__(self, root, *, purpose='optimization', views='both'):
        self.root = Path(root).resolve()
        self.folder = self.root / 'data/public-reasoned-code-SFT-v1'
        reports = self.root / 'reports/public-reasoned-code-pack-admission-v1'
        if (purpose not in ('optimization', 'capacity') or views not in VIEWS
                or not (reports / 'complete.json').is_file() or (reports / 'failure.json').exists()
                or not (self.folder / 'admission.json').is_file()):
            raise ValueError('Complete successful native/source admission and an explicit train-view policy required')
        complete = json.loads((reports / 'complete.json').read_text())
        admission = json.loads((self.folder / 'admission.json').read_text())
        plan = json.loads((reports / 'owner-plan.json').read_text())
        owner = plan['owner']
        manifest_path = self.folder / 'manifest.json'
        if (not terminal(owner) or complete.get('owner') != owner or admission.get('owner') != owner
                or complete.get('complete') is not True or complete.get('data_only_weak_SFT_admitted') is not True
                or complete.get('full_native_text_arrays_and_roles_rebuilt_independently') is not True
                or complete.get('admission_sha256') != file_hash(self.folder / 'admission.json')
                or complete.get('source_dataset_manifest_sha256') != file_hash(manifest_path)
                or complete.get('original_source_positions') != 20000
                or complete.get('original_fixed_role_cross_intersections') != 0
                or complete.get('reserved_optimization_forbidden') is not True
                or complete.get('use_as_RL_reward_oracle_forbidden') is not True
                or admission.get('format') != 'slm-public-reasoned-code-weak-SFT-admission-v1'
                or admission.get('admitted_to_training') is not True
                or admission.get('dataset_manifest_sha256') != file_hash(manifest_path)
                or admission.get('admitted_use') != 'weak_supervised_coding_only'
                or admission.get('synthetic_teacher_data') is not True
                or admission.get('external_teacher_weights_loaded') is not False
                or admission.get('live_teacher_API_used') is not False
                or admission.get('human_preferences') is not False or admission.get('own_model_on_policy') is not False
                or admission.get('use_as_RL_reward_oracle_forbidden') is not True
                or admission.get('independent_semantic_correctness_certified') is not False
                or admission.get('original_fixed_roles_and_every_whole_native_input_and_EOS_mask_rebuilt') is not True
                or admission.get('complete_original_source_decisions') != 20000
                or admission.get('complete_source_execution_candidates_read_back') != 6116
                or admission.get('declared_duplicate_component_cross_role_intersections') != 0):
            raise ValueError('Terminal complete source/native proof and explicit weak-SFT-only scope required')
        self.provenance_source_files_sha256 = dict(admission['source_files_sha256'])
        self.provenance_guard = FileBindings(self.provenance_source_files_sha256)
        self.provenance_identity = digest(self.provenance_source_files_sha256)
        self.pack = CompleteCodePack(self.folder, 'train', purpose='training')
        manifest = self.pack.manifest
        if (manifest['provenance'].get('dataset') != 'nvidia/OpenCodeReasoning-2'
                or manifest['provenance'].get('revision') != REVISION
                or manifest['provenance'].get('weak_SFT_only') is not True
                or admission['complete_packed_rows_read_back'] != manifest['source_examples']
                or complete['complete_native_rows'] != manifest['source_examples']
                or admission['positive_train_rows'] != manifest['counts']['train']['examples']
                or admission['separately_reserved_rows'] != manifest['counts']['validation']['examples']
                or admission['actual_native_counts'] != complete['actual_native_counts']):
            raise ValueError('Every actual admitted packet/view population must match its closed evidence')
        with gzip.open(self.folder / 'original-rows.jsonl.gz', 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        if len(rows) != manifest['source_examples']:
            raise ValueError('Entire physical train/reserved row table required')
        groups = {}
        all_counts = {}
        for position, row in enumerate(rows):
            role = row['split']
            fixed = row['original_fixed_role']
            if (row['token_pack_index'] != position or role not in ('train', 'validation')
                    or int(self.pack.index[position]['role']) != (0 if role == 'train' else 1)
                    or row['publisher_split'] != 'train'
                    or fixed['source_role'] != ('train' if role == 'train' else 'reserved')
                    or fixed['optimization_forbidden'] is not (role == 'validation')
                    or fixed['role_fixed_before_fresh_replay'] is not True
                    or row['view'] not in ('code_only', 'reasoning') or row['source_revision'] != REVISION
                    or row['model_generated_examples'] is not True or row['human_preferences'] is not False
                    or row['own_model_on_policy'] is not False or row['use_as_RL_reward_oracle_forbidden'] is not True
                    or row['admitted_use'] != 'weak_supervised_coding_only'
                    or row['external_teacher_weights_loaded_by_this_pipeline'] is not False
                    or row['live_teacher_API_used'] is not False):
                raise ValueError('Every original role/view and public synthetic scope must remain exact')
            if row['group_id'] in groups and groups[row['group_id']] != role:
                raise ValueError('A complete duplicate component cannot cross source roles')
            groups[row['group_id']] = role
            all_counts[role + '_' + row['view'] + '_examples'] = all_counts.get(role + '_' + row['view'] + '_examples', 0) + 1
        for role in ('train', 'validation'):
            for view in ('code_only', 'reasoning'):
                key = role + '_' + view + '_examples'
                if admission['actual_native_counts'].get(key, 0) != all_counts.get(key, 0):
                    raise ValueError('All original code/reasoning view totals must match admission')
            for key in ('examples', 'context_tokens', 'target_tokens'):
                if admission['actual_native_counts'].get(role + '_' + key, 0) != manifest['counts'][role][key]:
                    raise ValueError('Complete native role budgets differ from admission')
        self.purpose, self.views = purpose, views
        self._declared_purpose, self._declared_views = purpose, views
        self._locals = tuple(i for i, position in enumerate(self.pack.positions)
            if views == 'both' or rows[int(position)]['view'] == views)
        if not self._locals:
            raise ValueError('Declared full train view is empty; no synthetic replacement or rounding repeat allowed')
        self._rows = tuple(rows[int(self.pack.positions[i])] for i in self._locals)
        self._information = []
        for position, local in enumerate(self._locals):
            row = self._rows[position]
            if (row['split'] != 'train' or row['original_fixed_role']['source_role'] != 'train'
                    or row['original_fixed_role']['optimization_forbidden'] is not False):
                raise ValueError('Reserved source rows cannot supply optimization or capacity examples')
            ids, labels = self.pack.sequence(local)
            start = next(i for i, value in enumerate(labels) if value != -100)
            actual_sha = hashlib.sha256(struct.pack('<' + 'H' * len(ids), *ids)).hexdigest()
            if (labels != [-100]*start + ids[start:] or ids[-1] != 2 or labels[-1] != 2
                    or any(value < 4 for value in ids[1:-1])
                    or actual_sha != row['complete_input_ids_sha256'] or row['target_tokens'] != len(ids)-start
                    or row['original_native_metadata']['complete_context_tokens'] != len(ids)
                    or row['original_native_metadata']['response_EOS_targets'] != len(ids)-start
                    or row['original_native_metadata']['target_start'] != start):
                raise ValueError('Every admitted full train source and response/EOS mask must remain exact')
            self._information.append(dict(position=position, source_id=row['id'], view=row['view'],
                original_shard_position=row['original_shard_position'], original_group_id=row['group_id'],
                complete_context_tokens=len(ids), response_EOS_targets=len(ids)-start, original_prompt_positions=start,
                complete_native_input_sha256=actual_sha, source_role='train',
                supervision='public_synthetic_weak_SFT_only', use_as_RL_reward_oracle_forbidden=True))
        runtime = {str(self.folder / name): digest_ for name, digest_ in manifest['files_sha256'].items()}
        for path in [manifest_path, self.folder / 'admission.json', reports / 'complete.json', reports / 'owner-plan.json',
                Path(__file__).resolve(), Path(__file__).with_name('codenet_token_pack_v1.py')]:
            runtime[str(path)] = file_hash(path)
        self.source_files_sha256 = runtime
        self.guard = FileBindings(runtime)
        self.manifest_sha256 = file_hash(manifest_path)
        self.admission_sha256 = file_hash(self.folder / 'admission.json')
        self._selection_identity = digest(dict(view_policy=views, original_train_local_positions=self._locals))
        self.slots = tuple(range(len(self._locals)))

    def __len__(self):
        return len(self._locals)

    def _position(self, position):
        if (self.role != 'train' or self.purpose not in ('optimization', 'capacity')
                or self.purpose != self._declared_purpose or self.views != self._declared_views
                or type(position) is not int or not 0 <= position < len(self)):
            raise ValueError('Only an exact admitted whole train-view position may supply optimization data')

    def row(self, position):
        self._position(position)
        return copy.deepcopy(self._rows[position])

    def sequence(self, position):
        self._position(position)
        return self.pack.sequence(self._locals[position])

    def information(self, position):
        self._position(position)
        return copy.deepcopy(self._information[position])

    def batch(self, positions, device='cpu'):
        if not isinstance(positions, list) or not positions or len(positions) != len(set(positions)):
            raise ValueError('Distinct nonempty complete train presentations required')
        self.guard.check()
        sequences = [self.sequence(position) for position in positions]
        length = max(len(ids) for ids, _ in sequences)
        ids = torch.zeros((len(sequences), length), dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (source, target) in enumerate(sequences):
            ids[i, :len(source)] = torch.tensor(source, dtype=torch.long, device=device)
            labels[i, :len(target)] = torch.tensor(target, dtype=torch.long, device=device)
            mask[i, :len(source)] = True
        return dict(input_ids=ids, labels=labels, attention_mask=mask, pixel_values=None)

    def complete_schedule(self, *, epochs=1, width=16, seed=990231):
        self._position(0)
        if self.slots != tuple(range(len(self))):
            raise ValueError('Every original selected train slot must remain in complete source schedules')
        if any(type(value) is not int or value < 1 for value in (epochs, width, seed)):
            raise ValueError('Explicit positive whole epochs, width and source seed required')
        entries = []
        for epoch in range(epochs):
            order = list(self.slots)
            random.Random(seed+epoch).shuffle(order)
            entries.extend(order[i:i+width] for i in range(0, len(order), width))
        return entries

    def budget(self, entries):
        if (not isinstance(entries, list) or not entries
                or any(not isinstance(entry, list) or not entry or len(entry) != len(set(entry)) for entry in entries)):
            raise ValueError('Complete nonempty distinct-source global batches required')
        rows = [self.information(position) for entry in entries for position in entry]
        return dict(updates=len(entries), source_presentations=len(rows),
            complete_context_tokens=sum(row['complete_context_tokens'] for row in rows),
            response_EOS_targets=sum(row['response_EOS_targets'] for row in rows),
            view_presentations={view: sum(row['view']==view for row in rows) for view in ('code_only', 'reasoning')})

    def identity(self, entries):
        self.budget(entries)
        if self._selection_identity != digest(dict(view_policy=self.views, original_train_local_positions=self._locals)):
            raise ValueError('The original complete native train-view selection must remain unchanged')
        return digest(dict(manifest=self.manifest_sha256, admission=self.admission_sha256,
            full_provenance=self.provenance_identity, train_view_selection=self._selection_identity,
            use='weak_supervised_coding_only', complete_original_train_batches=entries))

    def audit_provenance(self):
        self.guard.check(full=True)
        self.provenance_guard.check(full=True)

    def score(self, *args, **kwargs):
        raise ValueError('Public synthetic SFT responses are not human preferences or an independent RL reward oracle')
