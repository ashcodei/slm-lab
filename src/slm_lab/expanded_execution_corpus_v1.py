"""Original native code targets plus admitted bounded machine-execution rewards.

All 173 documented functions and 3,548 APPS programs are retained. This is a
training source API, not a new training run or a final independent benchmark.
"""
import gzip
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from .apps_execution_reward_v1 import reward
from .apps_published_cases_v2 import contract
from .bounded_code_reward_catalog_v1 import BoundedPublicCodeReward
from .codenet_checkpoint_v1 import FileBindings
from .codenet_token_pack_v1 import CompleteCodePack
from .corpus import file_hash


class ExpandedExecutionCorpus:
    role = 'train'
    purpose = 'execution_reward_and_complete_reference_CE'

    def __init__(self, root):
        self.root = Path(root).resolve()
        folder = self.root / 'data/apps-execution-rewards-v1'
        admission = json.loads((folder / 'admission.json').read_text())
        manifest = json.loads((folder / 'manifest.json').read_text())
        reader_path = self.root / 'reports/apps-execution-rewards-v1/independent-reader.json'
        reader = json.loads(reader_path.read_text())
        if (admission.get('scoring_admitted_to_training') is not True
                or admission['manifest_sha256'] != file_hash(folder / 'manifest.json')
                or admission['independent_reader_sha256'] != file_hash(reader_path)
                or reader.get('all_new_noops_independently_reexecuted') is not True
                or reader.get('every_original_reference_raw_result_recompared') is not True
                or reader.get('every_complete_source_target_mask_and_training_budget_rederived') is not True):
            raise ValueError('Complete original sources and independent execution-reward admission required')
        self.guard = FileBindings(admission['source_files_sha256']); self.guard.check(full=True)
        self.packs = dict(documented=CompleteCodePack(self.root / 'data/documented-doctest-verified-v1', 'train', 'training'),
                          apps=CompleteCodePack(manifest['original_pack'], 'train', 'training'))
        self.documented_reward = BoundedPublicCodeReward(self.root)
        self.tokenizer_path = self.packs['apps'].folder / 'tokenizer.json'
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        tokenizer_sha = file_hash(self.tokenizer_path)
        if (self.tokenizer.get_vocab_size() != 32768 or any(self.tokenizer.token_to_id(k) != v
                for k, v in [('<pad>', 0), ('<bos>', 1), ('<eos>', 2)])
                or file_hash(self.packs['documented'].folder / 'tokenizer.json') != tokenizer_sha):
            raise ValueError('One exact own tokenizer and native boundaries required')
        original_rows = {}
        for kind, pack in self.packs.items():
            with gzip.open(pack.folder / 'original-rows.jsonl.gz', 'rt') as stream:
                original_rows[kind] = list(map(json.loads, stream))
        catalog = list(map(json.loads, (folder / 'catalog.jsonl').read_text().splitlines()))
        controls = list(map(json.loads, (folder / 'controls.jsonl').read_text().splitlines()))
        self.apps_controls = {r['source_id']: r['control'] for r in controls}
        if (len(original_rows['documented']) != 173 or len(original_rows['apps']) != 3548
                or len(catalog) != 3548 or len(self.apps_controls) != 3548):
            raise ValueError('All documented and APPS source populations required')
        self.rows = []
        for kind in ['documented', 'apps']:
            for local, original in enumerate(original_rows[kind]):
                if original['split'] != 'train' or original['publisher_split'] != 'train':
                    raise ValueError('Held source role cannot be exposed to an optimization API')
                ids, labels = self.packs[kind].sequence(local)
                start = next(i for i, value in enumerate(labels) if value != -100)
                targets = len(ids) - start
                source_id = original['parent_source']['id'] if kind == 'documented' else original['id']
                if kind == 'apps' and (catalog[local]['id'] != source_id or catalog[local]['prefix_positions'] != start):
                    raise ValueError('Admitted native source ordering and prefix must match')
                self.rows.append(dict(id=kind + ':' + source_id, source_id=source_id, kind=kind,
                    pack_local_position=local, original=original, complete_context_positions=len(ids),
                    prefix_positions=start, response_EOS_targets=targets,
                    full_training_action_budget=min(4096 - start, max(512, targets + 64))))
        if len({r['id'] for r in self.rows}) != 3721:
            raise ValueError('Every original source presentation requires a distinct declared identity')
        self.by_id = {r['id']: i for i, r in enumerate(self.rows)}
        self.provenance = dict(apps_execution_admission_sha256=file_hash(folder / 'admission.json'),
            documented_probe_admission_sha256=file_hash(self.root / 'reports/bounded-code-probes-v1/admission.json'),
            tokenizer_sha256=tokenizer_sha, original_train_sources=3721,
            original_documented_sources=173, original_APPS_sources=3548,
            source_families_not_certified_semantically_independent=True, human_preferences=False,
            genuine_human_RLHF=False, model_updates=0)

    def __len__(self): return len(self.rows)

    def row(self, position):
        if type(position) is not int or not 0 <= position < len(self):
            raise ValueError('Only an admitted original training source position is permitted')
        return self.rows[position]

    def sequence(self, position):
        row = self.row(position)
        return self.packs[row['kind']].sequence(row['pack_local_position'])

    def batch(self, position, device):
        ids, labels = self.sequence(position)
        return dict(input_ids=torch.tensor([ids], dtype=torch.long, device=device),
            labels=torch.tensor([labels], dtype=torch.long, device=device),
            attention_mask=torch.ones((1, len(ids)), dtype=torch.bool, device=device), pixel_values=None)

    def score(self, position, response, *, stopped_on_eos):
        row = self.row(position)
        if type(stopped_on_eos) is not bool:
            raise ValueError('Actual generated EOS termination must be explicitly supplied')
        if row['kind'] == 'documented':
            actual = self.documented_reward.score(row['source_id'], response)
            value = actual['reward']
            return dict(actual, reward=min(value, .9) if not stopped_on_eos else value,
                complete_delivery_stopped_on_EOS=stopped_on_eos,
                original_reward_before_delivery_gate=value)
        path = self.root / 'data/public-apps-original-inventory-v1/original-train' / row['original']['source_problem_id'] / 'input_output.json'
        cases = contract(path.read_bytes())
        return reward(row['original'], cases, self.apps_controls[row['source_id']], response,
                      stopped_on_eos=stopped_on_eos)
