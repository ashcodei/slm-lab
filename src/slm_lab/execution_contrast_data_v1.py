"""Separately admitted public machine preferences, preserving both full programs."""
import json
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .source_jsonl_records_v1 import records


class ExecutionContrastData:
    role = 'train'
    family = 'machine_execution_preferences'

    def __init__(self, root, *, purpose='optimization'):
        if purpose not in ('audit', 'capacity', 'optimization'):
            raise ValueError('This is a train-only machine-preference source, never a held benchmark')
        self.root = Path(root).resolve(); self.purpose = purpose
        self.folder = self.root / 'data/CodeContests-execution-contrast-admitted-v1'
        manifest_path = self.folder / 'manifest.json'; admission_path = self.folder / 'admission.json'
        self.manifest = json.loads(manifest_path.read_text()); admission = json.loads(admission_path.read_text())
        reader_path = Path(admission['independent_readback_file'])
        if (admission['training_admitted'] is not True
                or admission['source_manifest_sha256'] != file_hash(manifest_path)
                or admission['independent_readback_sha256'] != file_hash(reader_path)
                or admission['training_scope'] != 'machine_DPO_and_chosen_anchor_CE_only'
                or admission['human_preferences'] is not False
                or admission['rejected_programs_must_never_be_positive_CE_targets'] is not True
                or self.manifest['format'] != 'own-original-public-execution-contrast-admitted-v1'):
            raise ValueError('Complete separate machine-preference source admission is required')
        self.guard = FileBindings(admission['source_files_sha256'])
        pair_path = self.folder / 'pairs.jsonl'
        if file_hash(pair_path) != self.manifest['pairs_sha256']:
            raise ValueError('Every admitted whole native pair must match')
        self.rows = list(records(pair_path))
        if len(self.rows) != self.manifest['examples'] or not self.rows:
            raise ValueError('Every nonempty original admitted source position must be present')
        for row in self.rows:
            if (row['split'] != 'train' or row['publisher_split'] != 'train'
                    or row['optimization_admitted'] is not True or row['human_preferences'] is not False
                    or row['original_candidate_optimization_admitted'] is not False):
                raise ValueError('Only original train-role machine preferences may provide gradients')
            prefixes = []
            for field in ('chosen', 'rejected'):
                ids, labels = row[field + '_ids'], row[field + '_labels']
                start = next((i for i, v in enumerate(labels) if v != -100), None)
                if (start is None or not 1 <= start < len(ids) - 1 or len(ids) > 4096
                        or ids[0] != 1 or ids[-1] != 2 or len(ids) != len(labels)
                        or any(type(v) is not int or not 4 <= v < 32768 for v in ids[1:-1])
                        or labels != [-100] * start + ids[start:]):
                    raise ValueError('Every full contiguous native response and EOS mask must remain exact')
                prefixes.append(ids[:start])
            if prefixes[0] != prefixes[1]:
                raise ValueError('Both original responses must share exactly the same whole prompt')
        self.tokenizer_path = self.folder / 'tokenizer.json'
        if file_hash(self.tokenizer_path) != self.manifest['tokenizer_sha256']:
            raise ValueError('The exact own tokenizer must be preserved')
        self.provenance = dict(admission_sha256=file_hash(admission_path), original_train_pairs=len(self.rows),
            tokenizer_sha256=file_hash(self.tokenizer_path), human_preferences=False,
            machine_execution_preferences=True, source_semantic_independence_not_certified=True)

    def __len__(self):
        return len(self.rows)

    def sequence(self, position, field):
        if type(position) is not int or not 0 <= position < len(self) or field not in ('chosen', 'rejected'):
            raise ValueError('An admitted original source position and explicit response field are required')
        row = self.rows[position]
        return list(row[field + '_ids']), list(row[field + '_labels'])

    def batch(self, positions, field, device):
        sequences = [self.sequence(p, field) for p in positions]
        if not sequences:
            raise ValueError('A nonempty complete original source batch is required')
        length = max(len(ids) for ids, _ in sequences)
        ids = torch.zeros((len(sequences), length), dtype=torch.long)
        labels = torch.full_like(ids, -100); mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (source, target) in enumerate(sequences):
            ids[i, :len(source)] = torch.tensor(source); labels[i, :len(source)] = torch.tensor(target)
            mask[i, :len(source)] = True
        return dict(input_ids=ids.to(device), labels=labels.to(device), attention_mask=mask.to(device))
