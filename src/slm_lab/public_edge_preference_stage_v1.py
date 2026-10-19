"""Whole-function CE, sequence DPO and first-divergence preference controls.

The branch objective compares two next-token log odds at one identical prefix.
Every arm still observes both whole programs and that prefix, with the same own
starting reference. Only chosen CE and the declared preference receive gradients.
This is an experimental local credit-assignment hypothesis, not a novelty or
capability claim. Machine mutations are never labelled on-policy or human data.
"""
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from .codenet_checkpoint_v1 import FileBindings
from .contract_source_contrasts_v2 import encode
from .corpus import file_hash
from .dense_human_preference_v1 import sequence_logp
from .frozen_own_actor_reference_v1 import FrozenOwnActorReference, signature
from .own_dense_actor_v1 import OwnDenseActor
from .own_dense_actor_checkpoint_v1 import make_state, restore_state

METHODS = ('chosen_CE', 'sequence_DPO', 'branch_DPO')
FORMAT = 'own-public-edge-CE-sequence-or-first-divergence-preference-v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fork(row):
    chosen = row['chosen_complete_response_EOS_ids']
    rejected = row['rejected_complete_response_EOS_ids']
    position = next((i for i, (a, b) in enumerate(zip(chosen, rejected)) if a != b), None)
    if (position is None or chosen[-1] != 2 or rejected[-1] != 2
            or chosen[position] < 4 or rejected[position] < 4
            or any(v in (0, 1, 2, 3) for v in chosen[:-1] + rejected[:-1])):
        raise ValueError('Two complete native programs must first diverge at actual source-content tokens')
    prefix = row['native_prompt_ids'] + chosen[:position]
    if not prefix or prefix[0] != 1 or len(prefix) >= 4096:
        raise ValueError('The complete original task and exact shared source prefix must fit')
    return dict(response_position=position, context=prefix, chosen=chosen[position], rejected=rejected[position])


class EdgePreferences:
    role = 'train'
    purpose = 'optimization'

    def __init__(self, folder, proof):
        self.folder = Path(folder).resolve()
        proof = Path(proof).resolve()
        manifest = json.loads((self.folder / 'manifest.json').read_text())
        admitted = json.loads(proof.read_text())
        if (manifest.get('format') != 'public-original-lexical-edge-case-machine-preferences-admitted-v1'
                or not admitted.get('complete') or not manifest.get('machine_preference_training_admitted')
                or manifest.get('human_preference_training_admitted') is not False
                or manifest.get('on_policy_RL_training_admitted') is not False
                or admitted['actual_admitted_machine_pairs'] != manifest['candidate_pairs']
                or admitted['source_files_sha256'].get(str(self.folder / 'manifest.json')) != file_hash(self.folder / 'manifest.json')):
            raise ValueError('Exact complete independent machine-preference admission required')
        self.source_files_sha256 = dict(admitted['source_files_sha256'])
        self.source_files_sha256[str(proof)] = file_hash(proof)
        self.guard = FileBindings(self.source_files_sha256)
        for name, expected in manifest['files_sha256'].items():
            if file_hash(self.folder / name) != expected:
                raise ValueError('Admitted complete native pair artifact changed')
        self.tokenizer_path = self.folder / 'tokenizer.json'
        tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        self.rows = [json.loads(line) for line in (self.folder / 'pairs.jsonl').read_text().splitlines()]
        if len(self.rows) != manifest['candidate_pairs'] or len({r['id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Every declared original machine pair exactly once required')
        for row in self.rows:
            if (row.get('source_role') != 'train' or row.get('human_preferences') is not False
                    or row.get('on_policy_RL') is not False
                    or row.get('machine_derived_negative_not_public_original_code') is not True
                    or not 0 <= row['rejected_finite_cases_passed'] < row['finite_cases']):
                raise ValueError('Only finite-counterexample-labelled original source-train derivatives may train')
            prefix = row['native_prompt_ids']
            for field, program in [('chosen', row['chosen']), ('rejected', row['rejected'])]:
                ids, labels = encode(row['prompt'], program, tokenizer)
                targets = row[field + '_complete_response_EOS_ids']
                if ids != prefix + targets or labels != [-100] * len(prefix) + targets:
                    raise ValueError('Every original task/program byte, complete target mask and EOS must reconstruct')
            fork(row)
        self.slots = list(range(len(self.rows)))
        self.lengths = [max(len(r['native_prompt_ids']) + len(r[k]) for k in
                           ['chosen_complete_response_EOS_ids', 'rejected_complete_response_EOS_ids']) for r in self.rows]
        self.data_sha256 = file_hash(self.folder / 'manifest.json')

    def information(self, position):
        if type(position) is not int or position not in self.slots:
            raise ValueError('Exact admitted train-pair position required')
        row = self.rows[position]
        branch = fork(row)
        return dict(pair_position=position, pair_id=row['id'], source_id=row['source_id'], pair_sha256=digest(row),
                    original_prompt_tokens=len(row['native_prompt_ids']),
                    chosen_response_EOS_targets=len(row['chosen_complete_response_EOS_ids']),
                    rejected_response_EOS_targets=len(row['rejected_complete_response_EOS_ids']),
                    fork_response_position=branch['response_position'], fork_context_tokens=len(branch['context']),
                    chosen_fork_token=branch['chosen'], rejected_fork_token=branch['rejected'])

    def native(self, profile, position):
        if profile not in (*METHODS, 'baseline'):
            raise ValueError('Declared matched method required')
        self.information(position)
        row = self.rows[position]
        prefix = row['native_prompt_ids']
        targets = row['chosen_complete_response_EOS_ids']
        return prefix + targets, [-100] * len(prefix) + targets

    def batch(self, position, field, device='cpu'):
        self.information(position)
        row = self.rows[position]
        if field == 'fork':
            ids = fork(row)['context']
            return dict(input_ids=torch.tensor([ids], device=device),
                        attention_mask=torch.ones((1, len(ids)), dtype=torch.bool, device=device))
        if field not in ('chosen', 'rejected'):
            raise ValueError('A complete original pair side or exact shared prefix required')
        prefix = row['native_prompt_ids']
        targets = row[field + '_complete_response_EOS_ids']
        ids = prefix + targets
        return dict(input_ids=torch.tensor([ids], device=device),
                    attention_mask=torch.ones((1, len(ids)), dtype=torch.bool, device=device),
                    labels=torch.tensor([[-100] * len(prefix) + targets], device=device))

    def identity(self, entries):
        return digest(dict(admitted_manifest_sha256=self.data_sha256, complete_pair_batches=entries))

    def budget(self, entries):
        values = [self.information(p) for entry in entries for p in entry]
        return dict(presentations=len(values), chosen_response_EOS_targets=sum(v['chosen_response_EOS_targets'] for v in values),
                    rejected_response_EOS_targets=sum(v['rejected_response_EOS_targets'] for v in values),
                    original_fork_context_tokens=sum(v['fork_context_tokens'] for v in values),
                    observed_forks=len(values), every_method_observes_all_three_contexts=True)


def schedule(count, *, epochs=3, width=4, seed=980113):
    if any(type(v) is not int or v < 1 for v in [count, epochs, width, seed]):
        raise ValueError('Explicit complete source epochs, bounded width and seed required')
    entries = []
    for epoch in range(epochs):
        order = list(range(count))
        random.Random(seed + epoch).shuffle(order)
        entries.extend(order[i:i + width] for i in range(0, count, width))
    return entries


def log_odds(logits, chosen, rejected):
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] < 1 or chosen == rejected:
        raise ValueError('One actual same-prefix next-token distribution and distinct actions required')
    values = logits[0, -1].float()
    return values[chosen] - values[rejected]


def gradients(model, data, positions, profile, options, reference, *, amp):
    if (type(model) is not OwnDenseActor or type(data) is not EdgePreferences
            or type(reference) is not FrozenOwnActorReference or profile not in METHODS
            or not positions or len(positions) != len(set(positions))
            or not math.isfinite(options['beta']) or not 0 < options['beta'] <= 1
            or not math.isfinite(options['preference_weight']) or not 0 < options['preference_weight'] <= 1):
        raise ValueError('An exact own actor, admitted original train pairs and bound own reference required')
    reference.assert_bound(model)
    info = [data.information(p) for p in positions]
    denominator = sum(r['chosen_response_EOS_targets'] for r in info)
    parameters = [p for _, p in model.named_selected(('actor',))]
    model.train()
    model.policy.gradient_checkpointing = True
    records = []
    for row in info:
        scores = {}
        originals = {}
        for field in ('chosen', 'rejected', 'fork'):
            batch = data.batch(row['pair_position'], field)
            inputs = {k: v for k, v in batch.items() if k != 'labels'}
            with torch.no_grad(), amp():
                before = reference(**inputs)['logits']
            if field == 'fork':
                originals[field] = log_odds(before, row['chosen_fork_token'], row['rejected_fork_token'])
            else:
                originals[field], count = sequence_logp(before, batch['labels'])
                if count != row[field + '_response_EOS_targets']:
                    raise ValueError('The own reference must observe every original whole-program target')
            del before
            enabled = field == 'chosen' or field == 'rejected' and profile == 'sequence_DPO' or field == 'fork' and profile == 'branch_DPO'
            with torch.set_grad_enabled(enabled), amp():
                logits = model(**inputs)['logits']
            if field == 'fork':
                scores[field] = log_odds(logits, row['chosen_fork_token'], row['rejected_fork_token'])
            else:
                scores[field], count = sequence_logp(logits, batch['labels'])
                if count != row[field + '_response_EOS_targets']:
                    raise ValueError('Every original positive and complete negative EOS must remain observed')
            del logits
        if profile == 'branch_DPO':
            margin = scores['fork'] - originals['fork']
        else:
            margin = (scores['chosen'] - scores['rejected']) - (originals['chosen'] - originals['rejected'])
        preference = F.softplus(-options['beta'] * margin)
        loss = -scores['chosen'] / denominator
        if profile != 'chosen_CE':
            loss = loss + options['preference_weight'] * preference / len(positions)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Every complete-source objective must be finite')
        loss.backward(inputs=parameters)
        records.append(dict(**row, chosen_logp=float(scores['chosen'].detach()), rejected_logp=float(scores['rejected'].detach()),
                            policy_fork_log_odds=float(scores['fork'].detach()), reference_fork_log_odds=float(originals['fork']),
                            preference_margin=float(margin.detach()), preference=float(preference.detach())))
    reference.assert_bound(model)
    return dict(format=FORMAT, profile=profile, pair_positions=list(positions), complete_pairs=info,
                all_context_observations=records, chosen_response_EOS_targets=denominator,
                rejected_response_EOS_targets=sum(v['rejected_response_EOS_targets'] for v in info),
                chosen_token_mean_NLL=-math.fsum(v['chosen_logp'] for v in records) / denominator,
                pair_mean_preference=math.fsum(v['preference'] for v in records) / len(records),
                beta=options['beta'], preference_weight=options['preference_weight'],
                first_divergence_preference_gradients=profile == 'branch_DPO',
                whole_negative_preference_gradients=profile == 'sequence_DPO',
                all_original_contexts_targets_masks_and_EOS_observed=True,
                negative_programs_never_positive_CE_targets=True, human_preferences=False, on_policy_RL=False,
                all_methods_observe_same_own_starting_reference=True)


class EdgePreferenceStage:
    def __init__(self, model, optimizer, data, entries, profile, options, sources, cache, mode):
        if (profile not in METHODS or mode not in ('capacity', 'train') or not entries
                or options.get('precision') not in ('CPU_FP32', 'CPU_BF16_cached')):
            raise ValueError('Exact declared source schedule, method and stage required')
        self.model, self.optimizer, self.data = model, optimizer, data
        self.entries, self.profile, self.options, self.sources, self.cache, self.mode = entries, profile, dict(options), sources, cache, mode
        self.parameters = [p for _, p in model.named_selected(('actor',))]
        if [id(p) for group in optimizer.param_groups for p in group['params']] != [id(p) for p in self.parameters]:
            raise ValueError('Only every original actor parameter may enter Adam')
        selected = {id(p) for p in self.parameters}
        self.fixed = [p for p in model.parameters() if id(p) not in selected]
        self.fixed_signature = self.signature()
        self.reference = FrozenOwnActorReference(model, sources=sources)
        if cache is not None:
            cache.attach(self.reference.policy)
        self.reference_binding_sha256 = digest(self.reference.binding())
        self.expected = [dict(pair_positions=list(entry), complete_pairs=[data.information(p) for p in entry]) for entry in entries]
        self.completed, self.metrics = 0, []

    def signature(self):
        return signature(self.fixed)

    def update(self):
        if self.completed >= len(self.entries) or self.signature() != self.fixed_signature:
            raise ValueError('Only the exact next original source batch and unchanged fixed values may update')
        self.data.guard.check()
        self.optimizer.zero_grad(set_to_none=True)
        result = gradients(self.model, self.data, self.entries[self.completed], self.profile, self.options, self.reference,
                           amp=lambda: torch.autocast('cpu', dtype=torch.bfloat16) if self.options['precision'] == 'CPU_BF16_cached' else torch.autocast('cpu', enabled=False))
        if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                or any(p.grad is not None for p in self.fixed)):
            raise ValueError('Every selected actor and no original fixed value must have finite gradients')
        norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.options['gradient_clip'], error_if_nonfinite=True))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.completed += 1
        result.update(step=self.completed, learning_rate=self.options['learning_rate'], gradient_norm_before_clip=norm)
        self.metrics.append(result)
        if self.signature() != self.fixed_signature:
            raise ValueError('An original base/head/critic value changed')
        self.reference.assert_bound(self.model)
        if self.cache is not None:
            self.cache.assert_current()
        return result

    def cursor(self, completed):
        return dict(completed=completed, profile=self.profile, mode=self.mode, schedule_identity_sha256=self.data.identity(self.entries),
                    reference_binding_sha256=self.reference_binding_sha256, complete_pair_batches=self.entries[:completed],
                    training_options=self.options)

    def validate(self, rows):
        if not isinstance(rows, list) or not 0 < len(rows) <= len(self.entries):
            raise ValueError('Exact nonempty original source prefix required')
        for i, value in enumerate(rows):
            if (value.get('format') != FORMAT or value.get('profile') != self.profile or value.get('step') != i + 1
                    or value.get('learning_rate') != self.options['learning_rate']
                    or value.get('beta') != self.options['beta'] or value.get('preference_weight') != self.options['preference_weight']
                    or any(not isinstance(value.get(k), (int, float)) or not math.isfinite(value[k])
                           for k in ['chosen_token_mean_NLL', 'pair_mean_preference', 'gradient_norm_before_clip'])
                    or any(value.get(k) != v for k, v in self.expected[i].items())):
                raise ValueError('Every original complete positive/negative/fork mask and source cursor must reconstruct')

    def state(self):
        self.validate(self.metrics)
        return make_state(self.model, self.optimizer, groups=('actor',), sources=self.sources, completed=self.completed,
                          planned=len(self.entries), cursor=self.cursor(self.completed), metrics=self.metrics, device='cpu')

    def resume(self, state, *, allow_completed=False):
        self.validate(state.get('metrics'))
        count, records = restore_state(self.model, self.optimizer, state, groups=('actor',), expected_sources=self.sources,
                                       planned=len(self.entries), expected_cursor=self.cursor(state['completed']), device='cpu', allow_completed=allow_completed)
        self.completed, self.metrics = count, records
        if self.signature() != self.fixed_signature:
            raise ValueError('Recovery changed original fixed values')
        self.reference.assert_bound(self.model)
        return count
