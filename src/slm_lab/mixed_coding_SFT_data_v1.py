"""Complete weak coding SFT plus original human-code/procedural replay.

Every family remains train-only. Synthetic generated tests are never queried
as rewards; this is a whole-response CE curriculum, not on-policy RL or human
preference feedback. Source balancing changes positive weights, not targets.
"""
import copy
import hashlib
import math
from pathlib import Path
import random
import struct

import torch

from .active_execution_input_bindings_v1 import input_bindings, merge_bindings
from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from .parallel_function_repair_stage_v1 import value_digest
from .public_reasoning_train_data_v1 import PublicReasoningTrainData
from .public_weak_code_train_data_v1 import PublicWeakCodeSFTData

FAMILIES = ('public_weak_synthetic_code', 'original_human_code', 'public_procedural_reasoning')
PROFILES = ('plain_tokens', 'balanced_sources')
EXPECTED = dict(source_presentations=28028, complete_context_tokens=14090525, response_EOS_targets=7105578)


class MixedCodingSFTData:
    role = 'train'
    purpose = 'optimization'
    use_as_RL_reward_oracle_forbidden = True

    def __init__(self, root):
        root = Path(root).resolve()
        self.weak = PublicWeakCodeSFTData(root)
        self.human = ExpandedExecutionCorpus(root)
        self.reasoning = PublicReasoningTrainData(root / 'data/public-reasoning-curriculum-v1',
            root / 'reports/public-reasoning-curriculum-v1-readback-admission/complete.json')
        if len(self.weak) != 21739 or len(self.human) != 3721 or len(self.reasoning) != 2568:
            raise ValueError('All original admitted weak, human-code and procedural train sources are required')
        self.sources = (self.weak, self.human, self.reasoning)
        self.offsets = (0, 21739, 25460, 28028)
        self.slots = list(range(28028))
        files = {}
        for mapping in [self.weak.source_files_sha256, input_bindings(root, self.human), self.reasoning.source_files_sha256]:
            merge_bindings(files, mapping)
        tokenizers = [self.weak.folder / 'tokenizer.json', self.human.tokenizer_path, self.reasoning.folder / 'tokenizer.json']
        if len({file_hash(p) for p in tokenizers}) != 1: raise ValueError('One exact own native vocabulary is required across all sources')
        self.source_files_sha256 = files
        self.guard = FileBindings(files)
        self._information, self.family_targets = [], dict.fromkeys(FAMILIES, 0)
        for position in self.slots:
            family, local, source = self.locate(position)
            ids, labels = source.sequence(local)
            start = next(i for i, t in enumerate(labels) if t != -100)
            row = source.row(local)
            source_id = row['id']
            if (ids[0] != 1 or ids[-1] != 2 or not 3 <= len(ids) <= 4096
                    or labels != [-100] * start + ids[start:]):
                raise ValueError('Every whole original question/response/EOS and target mask must remain intact')
            targets = len(ids) - start
            self.family_targets[family] += targets
            self._information.append(dict(position=position, family=family, original_family_position=local, source_id=source_id,
                complete_context_tokens=len(ids), response_EOS_targets=targets, original_prompt_positions=start,
                complete_native_input_sha256=hashlib.sha256(struct.pack('<' + 'H' * len(ids), *ids)).hexdigest(),
                synthetic_teacher_data=family == FAMILIES[0], human_preferences=False, own_model_on_policy=False,
                source_role='train', use='complete_response_CE_only'))
        if self.family_targets != dict(zip(FAMILIES, [6502280, 584218, 19080])):
            raise ValueError('Every complete original family target total must match source admission')
        self.family_weights = dict(zip(FAMILIES, [1., .8 * 6502280 / 584218, .2 * 6502280 / 19080]))
        self.identity_sha256 = value_digest(dict(source_files=files, family_targets=self.family_targets,
            source_information=self._information, balanced_family_weights=self.family_weights,
            nominal_whole_epoch_weighted_target_mass=dict(zip(FAMILIES, [.5, .4, .1])),
            per_batch_normalization_does_not_guarantee_epoch_loss_fraction=True))
        self.guard.check(full=True)

    def __len__(self): return len(self.slots)

    def locate(self, position):
        if type(position) is not int or not 0 <= position < 28028: raise ValueError('Only an exact original train source is allowed')
        index = 0 if position < 21739 else 1 if position < 25460 else 2
        return FAMILIES[index], position - self.offsets[index], self.sources[index]

    def information(self, position):
        self.locate(position)
        return copy.deepcopy(self._information[position])

    def sequence(self, position):
        _, local, source = self.locate(position)
        return source.sequence(local)

    def weight(self, profile, position):
        if profile not in PROFILES: raise ValueError('One exact declared positive source-weight profile is required')
        family, _, _ = self.locate(position)
        return 1. if profile == 'plain_tokens' else self.family_weights[family]

    def batch(self, positions, device='cpu'):
        if not isinstance(positions, list) or not positions or len(positions) != len(set(positions)):
            raise ValueError('A nonempty distinct original train source microbatch is required')
        sequences = [self.sequence(p) for p in positions]
        length = max(len(ids) for ids, _ in sequences)
        ids = torch.zeros((len(positions), length), dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (original, target) in enumerate(sequences):
            ids[i, :len(original)] = torch.tensor(original, dtype=torch.long, device=device)
            labels[i, :len(original)] = torch.tensor(target, dtype=torch.long, device=device)
            mask[i, :len(original)] = True
        return dict(input_ids=ids, labels=labels, attention_mask=mask, pixel_values=None)

    def complete_schedule(self, *, epochs=3, width=16, seed=989923):
        if any(type(v) is not int or v < 1 for v in [epochs, width, seed]): raise ValueError('Explicit complete source epochs/batch/seed required')
        entries = []
        for epoch in range(epochs):
            order = list(self.slots)
            random.Random(seed + epoch).shuffle(order)
            entries.extend(order[i:i+width] for i in range(0, len(order), width))
        return entries

    def metadata(self, positions, profile):
        rows = [self.information(p) for p in positions]
        return dict(complete_sources=rows, response_EOS_targets=sum(r['response_EOS_targets'] for r in rows),
            complete_context_tokens=sum(r['complete_context_tokens'] for r in rows),
            actual_global_weighted_targets=math.fsum(r['response_EOS_targets'] * self.weight(profile, r['position']) for r in rows),
            actual_source_weights=[self.weight(profile, p) for p in positions])

    def budget(self, entries):
        if (not isinstance(entries, list) or not entries
                or any(not isinstance(e, list) or not e or len(e) != len(set(e)) for e in entries)):
            raise ValueError('Complete nonempty distinct-source global batches are required')
        rows = [self.information(p) for entry in entries for p in entry]
        return dict(updates=len(entries), source_presentations=len(rows),
            complete_context_tokens=sum(r['complete_context_tokens'] for r in rows),
            response_EOS_targets=sum(r['response_EOS_targets'] for r in rows),
            family_presentations={family: sum(r['family'] == family for r in rows) for family in FAMILIES})

    def identity(self, entries):
        self.budget(entries)
        return value_digest(dict(complete_original_source_identity=self.identity_sha256, complete_global_source_batches=entries))

    def score(self, *args, **kwargs):
        raise ValueError('This mixed source supports supervised CE only; generated tests are not an RL reward oracle')
