"""Admitted complete public synthetic coding SFT; no reward-oracle interface."""
import copy
import gzip
import json
from pathlib import Path
import random

import torch

from .codenet_checkpoint_v1 import FileBindings
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash
from .exact_waiting_process_retirement_v1 import terminal
from .parallel_function_repair_stage_v1 import value_digest


class PublicWeakCodeSFTData:
    role = 'train'
    source_is_public_synthetic = True
    use_as_RL_reward_oracle_forbidden = True
    human_preferences = False
    own_model_on_policy = False

    def __init__(self, root, *, purpose='optimization'):
        self.root = Path(root).resolve()
        self.folder = self.root / 'data/public-weak-code-SFT-v1'
        reports = self.root / 'reports/public-weak-code-pack-readback-admission-CPU-v1'
        if (purpose not in ['optimization', 'capacity'] or not (reports / 'complete.json').is_file()
                or (reports / 'failure.json').exists()):
            raise ValueError('The entire separate terminal weak-SFT data admission is required')
        complete = json.loads((reports / 'complete.json').read_text())
        admission = json.loads((self.folder / 'admission.json').read_text())
        owner = json.loads((reports / 'plan.json').read_text())['owner']
        if (not terminal(owner) or complete.get('complete') is not True or complete.get('data_only_weak_SFT_admitted') is not True
                or complete.get('admission_sha256') != file_hash(self.folder / 'admission.json') or admission.get('owner') != owner
                or admission.get('admitted_to_training') is not True or admission.get('admitted_use') != 'weak_supervised_coding_only'
                or admission.get('synthetic_teacher_data') is not True or admission.get('human_preferences') is not False
                or admission.get('own_model_on_policy') is not False or admission.get('use_as_RL_reward_oracle_forbidden') is not True
                or admission.get('independent_semantic_correctness_certified') is not False
                or admission.get('complete_original_source_decisions') != 100000 or admission.get('complete_packed_rows_read_back') != 24248
                or admission.get('positive_train_rows') != 21739 or admission.get('separately_reserved_rows') != 2509):
            raise ValueError('Every actual whole-source role/readback and explicit weak synthetic provenance must match')
        self.purpose = purpose
        self.source_files_sha256 = dict(admission['source_files_sha256'])
        self.source_files_sha256.update({str(p): file_hash(p) for p in [self.folder / 'admission.json',
            reports / 'complete.json', reports / 'plan.json']})
        self.guard = FileBindings(self.source_files_sha256)
        self.guard.check(full=True)
        self.pack = CompleteCodePack(self.folder, 'train', purpose='training')
        if len(self.pack) != 21739: raise ValueError('All original positive train-role rows are required')
        with gzip.open(self.folder / 'original-rows.jsonl.gz', 'rb') as stream:
            all_rows = [json.loads(raw) for raw in stream]
        if len(all_rows) != 24248: raise ValueError('The complete positive train/reserved pack is required')
        self.rows = [all_rows[int(position)] for position in self.pack.positions]
        for row in self.rows:
            if (row['split'] != 'train' or row['original_fixed_role']['source_role'] != 'train'
                    or row['original_fixed_role']['optimization_forbidden'] is not False
                    or row['admitted_use'] != 'weak_supervised_coding_only' or row['use_as_RL_reward_oracle_forbidden'] is not True
                    or row['model_generated_examples'] is not True or row['human_preferences'] is not False
                    or row['own_model_on_policy'] is not False):
                raise ValueError('Reserved, human-preference and on-policy claims cannot enter weak synthetic SFT')
        self.slots = list(range(21739))
        self.manifest_sha256 = file_hash(self.folder / 'manifest.json')
        self.admission_sha256 = file_hash(self.folder / 'admission.json')
        self.guard.check(full=True)

    def __len__(self): return len(self.rows)

    def row(self, position):
        if type(position) is not int or not 0 <= position < len(self): raise ValueError('Only an admitted original weak train source is allowed')
        return copy.deepcopy(self.rows[position])

    def sequence(self, position):
        self.row(position)
        return self.pack.sequence(position)

    def information(self, position):
        row = self.row(position)
        ids, labels = self.sequence(position)
        return dict(position=position, source_id=row['id'], original_shard_position=row['original_shard_position'],
            original_group_id=row['group_id'], complete_context_tokens=len(ids), response_EOS_targets=sum(t != -100 for t in labels),
            source_role='train', supervision='public_synthetic_weak_SFT_only', use_as_RL_reward_oracle_forbidden=True,
            complete_original_input_sha256=row['complete_input_ids_sha256'])

    def batch(self, positions, device='cpu'):
        if not isinstance(positions, list) or not positions or len(positions) != len(set(positions)):
            raise ValueError('Distinct complete original weak-SFT train presentations required')
        sequences = [self.sequence(position) for position in positions]
        length = max(len(ids) for ids, _ in sequences)
        ids = torch.zeros((len(sequences), length), dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (source, target) in enumerate(sequences):
            ids[i, :len(source)] = torch.tensor(source, dtype=torch.long, device=device)
            labels[i, :len(source)] = torch.tensor(target, dtype=torch.long, device=device)
            mask[i, :len(source)] = True
        return dict(input_ids=ids, labels=labels, attention_mask=mask, pixel_values=None)

    def complete_schedule(self, *, epochs=1, width=16, seed=989812):
        if any(type(v) is not int or v < 1 for v in [epochs, width, seed]): raise ValueError('Explicit complete source epochs/batch/seed required')
        result = []
        for epoch in range(epochs):
            order = list(self.slots)
            random.Random(seed + epoch).shuffle(order)
            result.extend(order[i:i+width] for i in range(0, len(order), width))
        return result

    def budget(self, entries):
        rows = [self.information(position) for entry in entries for position in entry]
        return dict(updates=len(entries), source_presentations=len(rows),
            original_context_tokens=sum(r['complete_context_tokens'] for r in rows),
            original_response_EOS_targets=sum(r['response_EOS_targets'] for r in rows),
            synthetic_weak_SFT_only=True, human_preferences=False, own_model_on_policy=False)

    def identity(self, entries):
        self.budget(entries)
        return value_digest(dict(manifest=self.manifest_sha256, admission=self.admission_sha256,
            use='weak_supervised_coding_only', complete_original_train_batches=entries))

    def score(self, *args, **kwargs):
        raise ValueError('These public generated tests are not an independent RL reward oracle; weak SFT only')
