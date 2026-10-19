"""Human reward learning on the actual current native language features.

The existing native reward head learns complete public human rankings. Its
encoder can later be shared with a frozen-base policy adapter, so the reward
adds no second backbone. Small CPU controls require explicit admission; they
cannot establish a completed full-size human epoch or useful own-output RLHF.
"""
import math

import torch
from torch.nn import functional as F

from .codenet_checkpoint_v1 import FileBindings, capture_rng, restore_rng
from .codenet_stage_v1 import EpochOrder
from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng
from .own_dense_growth_v1 import OwnDenseAssistant
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_data_v2 import WholeHumanData
from .whole_human_reward_v1 import feature_batches, source_fingerprint
from .whole_human_sft_evidence_v1 import tensor_digest

FEATURE_FORMAT = 'own-current-native-complete-human-ranking-features-v1'
HEAD_FORMAT = 'own-current-native-complete-human-ranking-head-v1'
STATE_FORMAT = 'own-current-native-human-reward-head-Adam-RNG-v1'
ROUTE = 'current_frozen_native_last_real_EOS_FP32'
POPULATIONS = dict(train=85786, valid=4642)


def signature(values):
    return [(id(p), p.data_ptr(), p._version, p.requires_grad, str(p.dtype), str(p.device)) for p in values]


def sources_ok(sources):
    return (isinstance(sources, dict) and set(sources) == {'native_checkpoint_sha256', 'human_manifest_sha256',
        'human_admission_sha256', 'frozen_native_encoder_sha256'} and all(isinstance(v, str) and len(v) == 64
        and all(c in '0123456789abcdef' for c in v) for v in sources.values()))


def scope(base, data, sources, allow_mechanics):
    if (type(base) is not OwnDenseAssistant or type(data) is not WholeHumanData or data.family != 'preferences'
            or (data.role, data.purpose) not in [('train', 'optimization'), ('valid', 'evaluation')]
            or not sources_ok(sources) or base.training):
        raise ValueError('Exact current native and admitted complete human-ranking source roles are required')
    full = base.counts()['total_unique'] == 984399690 and len(data) == POPULATIONS[data.role]
    small = (allow_mechanics is True and base.counts()['total_unique'] < 10**7
        and next(base.parameters()).device.type == 'cpu')
    if not full and not small:
        raise ValueError('The complete984Mnative and all85786train/4642held pairs, or explicit small CPU mechanics, are required')
    if tensor_digest(base.network.named_parameters()) != sources['frozen_native_encoder_sha256']:
        raise ValueError('Every actual frozen current native encoder value must match its declared feature basis')
    if small: return False, None
    paths = {str(data.root / 'manifest.json'): sources['human_manifest_sha256'],
        str(data.root / 'admission.json'): sources['human_admission_sha256']}
    for suffix in ['.jsonl', '.tokens.u16']:
        path = data.root / (data.family + '.' + data.role + suffix)
        paths[str(path)] = data.manifest['files_sha256'][path.name]
    guard = FileBindings(paths); guard.check(full=True)
    return True, guard


@torch.no_grad()
def native_features(base, ids, mask=None):
    if (type(base) is not OwnDenseAssistant or base.training or ids.dtype != torch.long or ids.ndim != 2
            or not ids.numel() or ids.device != next(base.parameters()).device
            or any(p.dtype != torch.float32 or p.grad is not None for p in base.parameters())):
        raise ValueError('A frozen-gradient FP32 current native and complete native token batch are required')
    mask = torch.ones_like(ids, dtype=torch.bool) if mask is None else mask
    if (mask.dtype != torch.bool or mask.shape != ids.shape or not bool(mask.any(1).all())
            or bool((mask[:, 1:] & ~mask[:, :-1]).any()) or ids.shape[1] > base.config.max_text_length
            or bool((ids < 0).any()) or bool((ids >= base.config.vocab_size).any())
            or not bool((ids[torch.arange(len(ids), device=ids.device), mask.sum(1)-1] == 2).all())):
        raise ValueError('Whole nonempty right-padded native responses with their actual final EOS are required')
    with torch.autocast(device_type=ids.device.type, enabled=False):
        hidden = base.network(ids, attention_mask=mask, return_hidden=True, return_logits=False)['hidden']
    result = hidden[torch.arange(len(ids), device=ids.device), mask.sum(1)-1].float()
    if not bool(torch.isfinite(result).all()): raise ValueError('Every complete original reward feature must be finite')
    return result


@torch.no_grad()
def collect_features(base, data, rank, world, sources, *, token_budget=16384, maximum_sequences=16,
                     checked=lambda: None, progress=lambda row: None, allow_mechanics=False):
    full, guard = scope(base, data, sources, allow_mechanics)
    if (type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world
            or not list(range(rank, len(data), world))):
        raise ValueError('Every actual whole-pair shard must be nonempty and have a distinct original rank')
    original = signature(base.parameters()); positions = list(range(rank, len(data), world))
    slots = {p: i for i, p in enumerate(positions)}; seen = set(); contexts = 0
    values = torch.empty(len(positions), 2, base.config.hidden_size, dtype=torch.float32)
    for batch in feature_batches(data, positions, token_budget=token_budget, max_sequences=maximum_sequences):
        checked()
        if guard is not None: guard.check()
        length = max(item[2] for item in batch); device = next(base.parameters()).device
        ids = torch.zeros(len(batch), length, dtype=torch.long, device=device); mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (p, field, size) in enumerate(batch):
            sequence, _ = data.sequence(p, field)
            if len(sequence) != size or sequence[-1] != 2 or (p, field) in seen:
                raise ValueError('Every unchanged original response/EOS must be encoded exactly once')
            ids[i, :size] = torch.tensor(sequence, device=device); mask[i, :size] = True
        encoded = native_features(base, ids, mask).cpu()
        for (p, field, size), row in zip(batch, encoded):
            values[slots[p], ('chosen', 'rejected').index(field)] = row; contexts += size; seen.add((p, field))
        progress(dict(encoded_complete_responses=len(seen), planned_complete_responses=2*len(positions), complete_context_tokens=contexts))
    checked()
    if guard is not None: guard.check(full=True)
    if (len(seen) != 2*len(positions) or signature(base.parameters()) != original
            or tensor_digest(base.network.named_parameters()) != sources['frozen_native_encoder_sha256']):
        raise ValueError('Every complete original response and fixed current native encoder value must remain intact')
    return dict(format=FEATURE_FORMAT, feature_route=ROUTE, role=data.role, sources=dict(sources),
        rank=rank, world=world, source_population=len(data), positions=positions,
        source_fingerprints=[source_fingerprint(data, p) for p in positions], features=values,
        complete_context_tokens=contexts, actual_native_values=base.counts()['total_unique'],
        full_original_role_population=full, model_updates=0, goal_achieved=False)


class NativeRewardFeatures:
    def __init__(self, base, data, shards, sources, *, allow_mechanics=False):
        full, guard = scope(base, data, sources, allow_mechanics)
        world = len(shards); by_rank = {s['rank']: s for s in shards}
        if not world or len(by_rank) != world or set(by_rank) != set(range(world)):
            raise ValueError('Every distinct complete original current-native feature shard is required')
        self.values = torch.empty(len(data), 2, base.config.hidden_size, dtype=torch.float32)
        for rank, shard in by_rank.items():
            positions = list(range(rank, len(data), world)); values = shard['features']
            contexts = sum(data.rows[p]['native'][f]['length'] for p in positions for f in ('chosen', 'rejected'))
            if (shard.get('format') != FEATURE_FORMAT or shard.get('feature_route') != ROUTE or shard.get('role') != data.role
                    or shard.get('sources') != sources or shard.get('rank') != rank or shard.get('world') != world
                    or shard.get('source_population') != len(data) or shard.get('positions') != positions
                    or shard.get('source_fingerprints') != [source_fingerprint(data, p) for p in positions]
                    or shard.get('complete_context_tokens') != contexts or shard.get('full_original_role_population') is not full
                    or shard.get('actual_native_values') != base.counts()['total_unique'] or shard.get('model_updates') != 0
                    or values.dtype != torch.float32 or values.shape != (len(positions), 2, base.config.hidden_size)
                    or values.requires_grad or not bool(torch.isfinite(values).all())):
                raise ValueError('All complete source/role/feature-basis fingerprints and original neural features must match')
            self.values[positions] = values
        if guard is not None: guard.check(full=True)
        self.data, self.sources, self.full, self.guard = data, dict(sources), full, guard
        self.role, self.purpose = data.role, data.purpose
        self.sources_sha256 = value_digest(self.sources)
        self.signature = (self.values.data_ptr(), self.values._version, tuple(self.values.shape))
        self.sha256 = value_digest(self.values)

    def __len__(self): return len(self.data)

    def check(self):
        if (self.signature != (self.values.data_ptr(), self.values._version, tuple(self.values.shape))
                or (self.data.family, self.data.role, self.data.purpose) != ('preferences', self.role, self.purpose)
                or value_digest(self.sources) != self.sources_sha256):
            raise ValueError('Every admitted complete original reward feature must remain unchanged')
        if self.guard is not None: self.guard.check()


class NativeRewardHeadStage:
    def __init__(self, base, optimizer, features, options):
        keys = {'seed', 'steps', 'batch_size', 'learning_rate', 'betas', 'weight_decay', 'gradient_clip'}
        if (type(base) is not OwnDenseAssistant or type(features) is not NativeRewardFeatures
                or (features.role, features.purpose) != ('train', 'optimization')
                or set(options) != keys or any(type(options[k]) is not int or options[k] < 1 for k in ['seed', 'steps', 'batch_size'])
                or options['steps'] != math.ceil(len(features)/options['batch_size'])
                or any(not math.isfinite(options[k]) or options[k] <= 0 for k in ['learning_rate', 'gradient_clip'])
                or not math.isfinite(options['weight_decay']) or options['weight_decay'] < 0
                or len(options['betas']) != 2 or any(not 0 <= v < 1 for v in options['betas'])):
            raise ValueError('One complete train-only human-ranking head epoch and finite explicit optimizer settings are required')
        named = list(base.heads['reward'].named_parameters()); self.parameters = [p for _, p in named]
        if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1
                or any(p.dtype != torch.float32 or not p.requires_grad or not bool(torch.isfinite(p).all()) for p in self.parameters)
                or [id(p) for p in optimizer.param_groups[0]['params']] != [id(p) for p in self.parameters]
                or optimizer.state or optimizer.param_groups[0].get('foreach') is not False
                or optimizer.param_groups[0].get('fused') not in (None, False)
                or any(optimizer.param_groups[0].get(k, False) for k in ['amsgrad', 'maximize', 'capturable', 'differentiable'])):
            raise ValueError('One fresh ordinary exact-order native reward-head AdamW group is required')
        group = optimizer.param_groups[0]
        if (group['lr'] != options['learning_rate'] or tuple(group['betas']) != tuple(options['betas'])
                or group['weight_decay'] != options['weight_decay']
                or tensor_digest(base.network.named_parameters()) != features.sources['frozen_native_encoder_sha256']):
            raise ValueError('The actual optimizer and complete current native feature basis must agree')
        self.base, self.optimizer, self.features, self.options = base, optimizer, features, dict(options)
        ids = {id(p) for p in self.parameters}; self.fixed = [(n, p) for n, p in base.named_parameters() if id(p) not in ids]
        self.fixed_signature = signature(p for _, p in self.fixed); self.fixed_sha256 = tensor_digest(self.fixed)
        self.initial_head_sha256 = tensor_digest(named); self.completed = 0; self.metrics = []
        self.order = EpochOrder(len(features), options['seed']); self.device = next(base.parameters()).device

    def positions(self, step):
        if type(step) is not int or not 0 <= step < self.options['steps']: raise ValueError('Exact bounded complete reward epoch required')
        start = step*self.options['batch_size']
        return [self.order.at(i) for i in range(start, min(start+self.options['batch_size'], len(self.features)))]

    def check(self, *, full=False):
        self.features.check()
        if (signature(p for _, p in self.fixed) != self.fixed_signature or any(p.grad is not None for _, p in self.fixed)
                or full and tensor_digest(self.fixed) != self.fixed_sha256):
            raise ValueError('Every unselected current native language/vision/critic/auxiliary value must remain fixed')

    def update(self):
        self.check(); positions = self.positions(self.completed); self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=False):
            pair_scores = self.base.heads['reward'](self.features.values[positions].to(self.device)).squeeze(-1)
            loss = F.softplus(pair_scores[:, 1]-pair_scores[:, 0]).mean()
        try:
            loss.backward(inputs=self.parameters)
            if not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters):
                raise ValueError('Every exact current-native reward-head gradient and pair loss must be finite')
            self.check(); norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.options['gradient_clip'], error_if_nonfinite=True))
            self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True); raise
        self.completed += 1
        row = dict(step=self.completed, source_positions=positions, pairs=len(positions),
            pair_mean_nll=float(loss.detach()), gradient_norm_before_clip=norm)
        self.metrics.append(row); return row

    def state(self):
        self.check(full=True); adam = _cpu(self.optimizer.state_dict()); _validate_adam(adam, self.parameters, self.completed)
        rng = capture_rng(self.device); _validate_rng(rng, self.device)
        return dict(format=STATE_FORMAT, options=dict(self.options), sources=dict(self.features.sources),
            feature_route=ROUTE, feature_sha256=self.features.sha256, full_original_role_population=self.features.full,
            completed=self.completed, metrics=_cpu(self.metrics), head=_cpu(self.base.heads['reward'].state_dict()),
            initial_head_sha256=self.initial_head_sha256, fixed_values_sha256=self.fixed_sha256,
            optimizer=adam, rng=rng, candidate_promoted=False, goal_achieved=False)

    def resume(self, saved, *, allow_completed=False):
        completed = saved.get('completed'); self.check(full=True)
        if (saved.get('format') != STATE_FORMAT or saved.get('options') != self.options
                or saved.get('sources') != self.features.sources or saved.get('feature_route') != ROUTE
                or saved.get('feature_sha256') != self.features.sha256 or value_digest(self.features.values) != self.features.sha256
                or saved.get('initial_head_sha256') != self.initial_head_sha256 or saved.get('fixed_values_sha256') != self.fixed_sha256
                or saved.get('full_original_role_population') is not self.features.full
                or type(completed) is not int or not 0 < completed <= self.options['steps']
                or completed == self.options['steps'] and not allow_completed
                or not isinstance(saved.get('metrics'), list) or len(saved['metrics']) != completed
                or saved.get('candidate_promoted') is not False or saved.get('goal_achieved') is not False):
            raise ValueError('Exact current-native reward source/feature/optimizer/cursor recovery is required before mutation')
        for step, row in enumerate(saved['metrics']):
            if (row.get('step') != step+1 or row.get('source_positions') != self.positions(step)
                    or row.get('pairs') != len(self.positions(step)) or any(not isinstance(row.get(k), (int, float))
                        or not math.isfinite(row[k]) or row[k] < 0 for k in ['pair_mean_nll', 'gradient_norm_before_clip'])):
                raise ValueError('Every original human pair draw and reward-head metric must reconstruct')
        head = self.base.heads['reward'].state_dict(); stored = saved.get('head')
        if (not isinstance(stored, dict) or stored.keys() != head.keys() or any(not isinstance(t, torch.Tensor)
                or t.shape != head[n].shape or t.dtype != head[n].dtype or not bool(torch.isfinite(t).all()) for n, t in stored.items())):
            raise ValueError('Every finite original-shape native reward tensor is required before mutation')
        _validate_adam(saved.get('optimizer'), self.parameters, completed); _validate_rng(saved.get('rng'), self.device)
        if saved['optimizer']['param_groups'] != self.optimizer.state_dict()['param_groups']:
            raise ValueError('Exact original reward-head AdamW hyperparameters are required')
        self.base.heads['reward'].load_state_dict(stored, strict=True)
        self.optimizer.load_state_dict(_cpu(saved['optimizer'])); self.optimizer.zero_grad(set_to_none=True)
        restore_rng(saved['rng'], self.device); self.completed = completed; self.metrics = _cpu(saved['metrics'])

    def finish(self):
        self.check(full=True)
        if self.completed != self.options['steps'] or sorted(p for row in self.metrics for p in row['source_positions']) != list(range(len(self.features))):
            raise ValueError('Every complete original train ranking must be consumed exactly once')
        head_sha256 = tensor_digest(self.base.heads['reward'].named_parameters())
        if head_sha256 == self.initial_head_sha256: raise ValueError('The actual current-native human reward head must learn changed finite values')
        self.state()
        return dict(format=HEAD_FORMAT, complete=True, sources=dict(self.features.sources), feature_route=ROUTE,
            actual_native_values=self.base.counts()['total_unique'], actual_train_pairs=len(self.features), actual_updates=self.completed,
            full_original_train_epoch_complete=self.features.full, trained_head_sha256=head_sha256,
            initial_head_sha256=self.initial_head_sha256, frozen_native_encoder_sha256=self.features.sources['frozen_native_encoder_sha256'],
            head_values=sum(p.numel() for p in self.parameters), learned_head=_cpu(self.base.heads['reward'].state_dict()),
            public_human_rankings_not_fresh_own_output_judgments=True, useful_own_policy_RLHF_unproved=True, goal_achieved=False)


class FrozenSharedNativeReward(torch.nn.Module):
    def __init__(self, base, completed_head, *, allow_mechanics=False):
        super().__init__()
        full = type(base) is OwnDenseAssistant and base.counts()['total_unique'] == 984399690
        small = (type(base) is OwnDenseAssistant and allow_mechanics is True and base.counts()['total_unique'] < 10**7
            and next(base.parameters()).device.type == 'cpu')
        if (not full and not small or completed_head.get('format') != HEAD_FORMAT or completed_head.get('complete') is not True
                or completed_head.get('feature_route') != ROUTE or not sources_ok(completed_head.get('sources'))
                or completed_head.get('actual_native_values') != base.counts()['total_unique']
                or full and (completed_head.get('actual_train_pairs') != POPULATIONS['train']
                    or completed_head.get('full_original_train_epoch_complete') is not True)
                or completed_head.get('trained_head_sha256') != tensor_digest(base.heads['reward'].named_parameters())
                or completed_head.get('frozen_native_encoder_sha256') != tensor_digest(base.network.named_parameters())):
            raise ValueError('The exact current-native encoder and actually completed human-ranking head are required')
        self.base = base.eval(); self.provenance = _cpu({k: v for k, v in completed_head.items() if k != 'learned_head'})
        self.provenance_sha256 = value_digest(self.provenance)
        self.eval(); self.fixed_signature = signature(base.parameters())

    @property
    def encoder(self): return self.base.network

    def train(self, mode=True):
        if mode: raise ValueError('The shared current-native reward encoder/head stays in evaluation mode')
        return super().train(False)

    def check(self):
        if (self.training or any(m.training for m in self.base.modules())
                or signature(self.base.parameters()) != self.fixed_signature or any(p.grad is not None for p in self.base.parameters())
                or value_digest(self.provenance) != self.provenance_sha256):
            raise ValueError('Every shared frozen current-native reward value and its metadata must remain fixed')

    def counts(self):
        return dict(shared_registered_native_values=self.base.counts()['total_unique'],
            reward_head_values=sum(p.numel() for p in self.base.heads['reward'].parameters()),
            additional_values_over_same_native_policy=0, included_in_native_policy=True)

    @torch.no_grad()
    def score(self, ids):
        self.check()
        if ids.ndim != 2 or ids.shape[0] != 1: raise ValueError('One complete own prompt/response/scoring-EOS is required')
        features = native_features(self.base, ids)
        with torch.autocast(device_type=ids.device.type, enabled=False): result = self.base.heads['reward'](features).reshape(())
        if not bool(torch.isfinite(result)): raise ValueError('The actual shared current-native human reward must be finite')
        return result.detach()


@torch.no_grad()
def evaluate_reward(base, features, *, batch_size=256):
    if (type(base) is not OwnDenseAssistant or type(features) is not NativeRewardFeatures
            or (features.role, features.purpose) != ('valid', 'evaluation') or type(batch_size) is not int or batch_size < 1
            or base.training or tensor_digest(base.network.named_parameters()) != features.sources['frozen_native_encoder_sha256']):
        raise ValueError('The exact current native basis and complete held-only human-ranking features are required')
    features.check(); before = signature(base.parameters()); records = []; device = next(base.parameters()).device
    for start in range(0, len(features), batch_size):
        with torch.autocast(device_type=device.type, enabled=False):
            scores = base.heads['reward'](features.values[start:start+batch_size].to(device)).squeeze(-1).cpu()
        if not bool(torch.isfinite(scores).all()): raise ValueError('Every actual held human response score must be finite')
        for position, (chosen, rejected) in enumerate(scores.tolist(), start):
            row = features.data.rows[position]; margin = chosen-rejected
            records.append(dict(source_position=position, id=row['id'], source=row['source'],
                chosen_score=chosen, rejected_score=rejected, margin=margin, correct=margin > 0, tied=margin == 0,
                nll=float(F.softplus(torch.tensor(-margin, dtype=torch.float64))),
                chosen_targets=row['native']['chosen']['length']-row['native']['chosen']['target_start'],
                rejected_targets=row['native']['rejected']['length']-row['native']['rejected']['target_start']))
    if before != signature(base.parameters()) or any(p.grad is not None for p in base.parameters()):
        raise ValueError('Held ranking assessment must preserve every native value and produce no gradient')
    features.check()
    def summary(rows):
        return dict(pairs=len(rows), correct=sum(r['correct'] for r in rows), tied=sum(r['tied'] for r in rows),
            accuracy=sum(r['correct'] for r in rows)/len(rows), pair_mean_nll=math.fsum(r['nll'] for r in rows)/len(rows)) if rows else dict(pairs=0)
    return dict(complete=True, feature_route=ROUTE, full_original_held_population=features.full, results=records,
        overall=summary(records), by_source={s: summary([r for r in records if r['source'] == s]) for s in sorted({r['source'] for r in records})},
        response_length_controls={name: summary([r for r in records if predicate(r)]) for name, predicate in [
            ('chosen_longer', lambda r: r['chosen_targets'] > r['rejected_targets']),
            ('chosen_shorter', lambda r: r['chosen_targets'] < r['rejected_targets']),
            ('equal_length', lambda r: r['chosen_targets'] == r['rejected_targets'])]},
        actual_native_values=base.counts()['total_unique'], head_sha256=tensor_digest(base.heads['reward'].named_parameters()),
        sources=dict(features.sources), features_sha256=features.sha256,
        publisher_human_rankings_not_fresh_own_output_judgments=True, optimizer_updates=0, goal_achieved=False)
