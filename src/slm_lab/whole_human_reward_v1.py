"""Human-ranking reward learning on the existing frozen general representation.

This deliberately uses the original general route, not the changing aligned
policy. The existing reward head adds no second backbone or uncounted weights.
Learning publisher human rankings does not establish reward quality on our own
rollouts, factual accuracy, or successful policy alignment.
"""
from contextlib import nullcontext
import hashlib
import json
import math

import torch
from torch.nn import functional as F

from .codenet_stage_v1 import EpochOrder
from .own_aligned_assistant_v2 import OwnAlignedAssistantV2
from .own_alignment_checkpoint_v1 import make_state, restore_state, parameters

FEATURE_FORMAT = 'whole-original-human-ranking-frozen-general-features-v1'
RESUME_FORMAT = 'whole-original-human-ranking-reward-head-resume-v1'
FEATURE_ROUTE = 'original_general_last_real_token_including_EOS'


def source_fingerprint(data, position):
    row = data.rows[position]
    native = {field: data.sequence(position, field)[0] for field in ('chosen', 'rejected')}
    value = dict(id=row['id'], source_record_sha256=row['source_record_sha256'],
                 split=row['split'], native=native)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@torch.no_grad()
def general_features(model, input_ids, attention_mask=None):
    """No policy features or gradients; select the actual last unpadded token."""
    if type(model) is not OwnAlignedAssistantV2 or input_ids.ndim != 2 or not input_ids.numel():
        raise ValueError('Exact own v2 model and complete nonempty native batch required')
    model.assert_word_contract()
    mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask
    if (mask.dtype != torch.bool or mask.shape != input_ids.shape or not bool(mask.any(1).all())
            or bool((mask[:, 1:] & ~mask[:, :-1]).any())):
        raise ValueError('Complete nonempty right-padded boolean attention mask required')
    original = model.route('general')
    if original.training:
        raise ValueError('Frozen reward encoder must be in evaluation mode')
    hidden = original(input_ids, attention_mask=mask, return_hidden=True, return_logits=False)['hidden']
    features = hidden[torch.arange(len(input_ids), device=hidden.device), mask.sum(1) - 1].float()
    if not bool(torch.isfinite(features).all()):
        raise ValueError('Every original general feature must be finite')
    return features


def scores(model, features):
    if (features.ndim < 2 or features.shape[-1] != model.config.hidden_size or features.requires_grad
            or features.dtype != torch.float32 or not bool(torch.isfinite(features).all())):
        raise ValueError('Complete detached finite FP32 frozen-general features required')
    with torch.autocast(device_type=features.device.type, enabled=False):
        return model.parts['reward'](features).squeeze(-1)


def pairwise_loss(pair_scores):
    if pair_scores.ndim != 2 or pair_scores.shape[1] != 2 or not len(pair_scores):
        raise ValueError('One chosen/rejected scalar score per complete human ranking required')
    return F.softplus(pair_scores[:, 1] - pair_scores[:, 0]).mean()


def feature_batches(data, positions, *, token_budget=4096, max_sequences=16):
    if (type(token_budget) is not int or token_budget < data.manifest['config']['max_text_length']
            or type(max_sequences) is not int or max_sequences < 1):
        raise ValueError('Whole maximum source context must fit the explicit feature batch budget')
    items = sorted(((position, field, data.rows[position]['native'][field]['length'])
        for position in positions for field in ('chosen', 'rejected')), key=lambda x: (x[2], x[0], x[1]))
    batch = []
    for item in items:
        if batch and (len(batch) == max_sequences or (len(batch) + 1) * item[2] > token_budget):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def collect_features(model, data, rank, world, sources, *, device='cpu', amp=None,
                     checked=lambda: None, progress=lambda value: None, token_budget=4096, max_sequences=16):
    if (data.family != 'preferences' or (data.role, data.purpose) not in
            (('train', 'optimization'), ('valid', 'evaluation'))
            or type(rank) is not int or type(world) is not int or not 0 <= rank < world):
        raise ValueError('Exact complete admitted train or evaluation-only held ranking shard required')
    positions = list(range(rank, len(data), world))
    if not positions:
        raise ValueError('Every feature shard must contain original human rankings')
    slots = {position: slot for slot, position in enumerate(positions)}
    features = torch.empty(len(positions), 2, model.config.hidden_size, dtype=torch.float32)
    seen = set(); contexts = 0
    model.eval(); amp = amp or nullcontext
    for batch in feature_batches(data, positions, token_budget=token_budget, max_sequences=max_sequences):
        checked()
        length = max(item[2] for item in batch)
        ids = torch.zeros(len(batch), length, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (position, field, size) in enumerate(batch):
            sequence, _ = data.sequence(position, field)
            if len(sequence) != size or sequence[-1] != 2:
                raise ValueError('Every complete original response and EOS must reach its encoder')
            ids[i, :size] = torch.tensor(sequence, device=device)
            mask[i, :size] = True
        with amp():
            values = general_features(model, ids, mask).cpu()
        for item, value in zip(batch, values):
            position, field, size = item
            if (position, field) in seen:
                raise ValueError('Each original chosen and rejected response must be encoded exactly once')
            features[slots[position], ('chosen', 'rejected').index(field)] = value
            seen.add((position, field)); contexts += size
        progress(dict(encoded_complete_responses=len(seen), planned_complete_responses=2*len(positions),
                      complete_context_tokens=contexts))
    if len(seen) != 2*len(positions):
        raise ValueError('Every complete original ranking response must be covered')
    return dict(format=FEATURE_FORMAT, role=data.role, feature_route=FEATURE_ROUTE,
        sources=dict(sources), rank=rank, world=world, source_population=len(data),
        positions=positions, source_fingerprints=[source_fingerprint(data, p) for p in positions],
        features=features, complete_context_tokens=contexts, actual_counts=model.counts(),
        original_encoder_frozen=True, model_updates=0, goal_achieved=False)


class RewardFeatures:
    """Validate complete source-position and whole-token coverage before Adam."""
    def __init__(self, data, shards, sources):
        if (data.family != 'preferences' or (data.role, data.purpose) not in
                (('train', 'optimization'), ('valid', 'evaluation')) or not shards):
            raise ValueError('Original human ranking train or held evaluation role required')
        world = len(shards); by_rank = {s['rank']: s for s in shards}
        if set(by_rank) != set(range(world)) or len(by_rank) != len(shards):
            raise ValueError('Every distinct complete feature shard required')
        first = shards[0]; hidden = first['features'].shape[-1]
        self.values = torch.empty(len(data), 2, hidden, dtype=torch.float32)
        for rank, shard in by_rank.items():
            positions = list(range(rank, len(data), world)); values = shard['features']
            contexts = sum(data.rows[p]['native'][f]['length'] for p in positions for f in ('chosen', 'rejected'))
            if (shard.get('format') != FEATURE_FORMAT or shard.get('role') != data.role
                    or shard.get('feature_route') != FEATURE_ROUTE or shard.get('sources') != sources
                    or shard.get('world') != world or shard.get('source_population') != len(data)
                    or shard.get('positions') != positions or shard.get('source_fingerprints') != [source_fingerprint(data, p) for p in positions]
                    or shard.get('original_encoder_frozen') is not True or shard.get('model_updates') != 0
                    or shard.get('actual_counts') != first['actual_counts'] or values.dtype != torch.float32
                    or values.shape != (len(positions), 2, hidden) or values.requires_grad
                    or not bool(torch.isfinite(values).all()) or shard.get('complete_context_tokens') != contexts):
                raise ValueError('All full original token/source fingerprints and frozen-feature shard contracts must match')
            self.values[positions] = values
        self.data = data; self.role = data.role; self.purpose = data.purpose
        self.sources = dict(sources); self.actual_counts = dict(first['actual_counts'])

    def __len__(self):
        return len(self.data)

    def batch(self, positions, device):
        return self.values[positions].to(device)


class _RewardRestoreView:
    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.model, name)

    def validate_export(self, payload, *, expected_sources):
        self.model.validate_export(payload, expected_sources=expected_sources)
        if any(not torch.equal(t.detach().cpu(), payload['extensions'][name].detach().cpu())
               for name, t in self.model.parts.state_dict().items() if not name.startswith('reward.')):
            raise ValueError('Every unused language/vision/critic/word extension must remain fixed before reward resume')

    def import_export(self, payload, *, expected_sources):
        self.validate_export(payload, expected_sources=expected_sources)
        with torch.no_grad():
            for name, parameter in self.model.parts.named_parameters():
                if name.startswith('reward.'):
                    parameter.copy_(payload['extensions'][name])


class RewardHeadStage:
    def __init__(self, model, optimizer, features, options, sources, device='cpu'):
        if (type(model) is not OwnAlignedAssistantV2 or (features.role, features.purpose) != ('train', 'optimization')
                or features.actual_counts != model.counts()):
            raise ValueError('Exact own model and complete original train-only human-ranking features required')
        keys = {'seed', 'steps', 'batch_size', 'learning_rate', 'betas', 'weight_decay', 'gradient_clip'}
        if (set(options) != keys or any(type(options[k]) is not int or options[k] < 1 for k in ('seed', 'steps', 'batch_size'))
                or options['steps'] != math.ceil(len(features)/options['batch_size'])
                or any(not isinstance(options[k], (float, int)) or not math.isfinite(options[k]) or options[k] <= 0
                       for k in ('learning_rate', 'gradient_clip')) or not 0 <= options['weight_decay']
                or not math.isfinite(options['weight_decay']) or len(options['betas']) != 2
                or any(not 0 <= v < 1 for v in options['betas'])):
            raise ValueError('Exactly one complete source pass and explicit finite ordinary reward Adam options required')
        self.model, self.optimizer, self.features = model, optimizer, features
        self.options, self.sources, self.device = dict(options), dict(sources), torch.device(device)
        self.parameters = parameters(model, optimizer, ('reward',))
        group = optimizer.param_groups[0]
        if (group['lr'] != options['learning_rate'] or tuple(group['betas']) != tuple(options['betas'])
                or group['weight_decay'] != options['weight_decay']):
            raise ValueError('Exact predeclared reward-only ordinary AdamW hyperparameters required')
        selected = {id(p) for p in self.parameters}
        self.fixed = [p for p in model.parameters() if id(p) not in selected]
        self.order = EpochOrder(len(features), options['seed']); self.completed = 0; self.metrics = []

    def positions(self, step):
        if type(step) is not int or not 0 <= step < self.options['steps']:
            raise ValueError('Exact bounded reward source update required')
        start = step*self.options['batch_size']
        return [self.order.at(i) for i in range(start, min(start+self.options['batch_size'], len(self.features)))]

    def cursor(self, completed):
        return dict(completed_updates=completed, completed_draws=min(completed*self.options['batch_size'], len(self.features)),
                    source_population=len(self.features), seed=self.options['seed'], final_batch_not_repeated=True)

    def update(self):
        positions = self.positions(self.completed); self.optimizer.zero_grad(set_to_none=True)
        try:
            pair_scores = scores(self.model, self.features.batch(positions, self.device))
            loss = pairwise_loss(pair_scores); loss.backward()
            if (not bool(torch.isfinite(loss)) or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for p in self.fixed)):
                raise ValueError('Finite full pair loss and gradients only in the existing reward head required')
            norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.options['gradient_clip'], error_if_nonfinite=True))
            self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True); raise
        self.completed += 1
        record = dict(step=self.completed, source_positions=positions, pairs=len(positions),
            pair_mean_nll=float(loss.detach()), learning_rate=self.options['learning_rate'], gradient_norm_before_clip=norm)
        self.metrics.append(record); return record

    def state(self):
        core = make_state(self.model, self.optimizer, groups=('reward',), sources=self.sources,
            completed_steps=self.completed, planned_steps=self.options['steps'], sampler_state=self.cursor(self.completed),
            metrics=self.metrics, device=self.device)
        return dict(format=RESUME_FORMAT, core=core, options=self.options, feature_route=FEATURE_ROUTE,
                    candidate_promoted=False, goal_achieved=False)

    def resume(self, state, *, allow_completed=False):
        core = state.get('core', {}); completed = core.get('steps')
        if (state.get('format') != RESUME_FORMAT or state.get('options') != self.options
                or state.get('feature_route') != FEATURE_ROUTE or state.get('candidate_promoted') is not False
                or state.get('goal_achieved') is not False or type(completed) is not int
                or not 0 < completed <= self.options['steps'] or len(core.get('metrics', [])) != completed):
            raise ValueError('Exact full human reward source/options/update contract required before mutation')
        for step, metric in enumerate(core['metrics']):
            positions = self.positions(step)
            if (metric.get('step') != step+1 or metric.get('source_positions') != positions or metric.get('pairs') != len(positions)
                    or metric.get('learning_rate') != self.options['learning_rate']
                    or any(not isinstance(metric.get(k), (float, int)) or not math.isfinite(metric[k]) or metric[k] < 0
                           for k in ('pair_mean_nll', 'gradient_norm_before_clip'))):
                raise ValueError('Every complete original human reward pair draw must reconstruct before resume')
        self.completed, self.metrics = restore_state(_RewardRestoreView(self.model), self.optimizer, core,
            groups=('reward',), expected_sources=self.sources, planned_steps=self.options['steps'],
            expected_sampler_state=self.cursor(completed), device=self.device, allow_completed=allow_completed)


@torch.no_grad()
def evaluate_reward(model, features, batch_size=256):
    if (features.role, features.purpose) != ('valid', 'evaluation'):
        raise ValueError('Complete original held human rankings required for reward evaluation')
    records = []; device = next(model.parts['reward'].parameters()).device
    for start in range(0, len(features), batch_size):
        positions = list(range(start, min(start+batch_size, len(features))))
        values = scores(model, features.batch(positions, device)).cpu()
        for position, (chosen, rejected) in zip(positions, values.tolist()):
            row = features.data.rows[position]; margin = chosen-rejected
            records.append(dict(source_position=position, id=row['id'], source=row['source'],
                chosen_score=chosen, rejected_score=rejected, margin=margin,
                nll=float(F.softplus(torch.tensor(-margin, dtype=torch.float64))),
                correct=margin > 0, tied=margin == 0,
                chosen_targets=row['native']['chosen']['length']-row['native']['chosen']['target_start'],
                rejected_targets=row['native']['rejected']['length']-row['native']['rejected']['target_start']))
    def summary(items):
        return dict(pairs=len(items), correct=sum(r['correct'] for r in items), tied=sum(r['tied'] for r in items),
            accuracy=sum(r['correct'] for r in items)/len(items), pair_mean_nll=math.fsum(r['nll'] for r in items)/len(items))
    controls = {}
    for label, predicate in (('chosen_longer', lambda r:r['chosen_targets'] > r['rejected_targets']),
                             ('chosen_shorter', lambda r:r['chosen_targets'] < r['rejected_targets']),
                             ('equal_length', lambda r:r['chosen_targets'] == r['rejected_targets'])):
        items = [r for r in records if predicate(r)]
        controls[label] = summary(items) if items else dict(pairs=0)
    return dict(overall=summary(records), by_source={name:summary([r for r in records if r['source'] == name])
        for name in sorted({r['source'] for r in records})}, response_length_controls=controls, results=records,
        feature_route=FEATURE_ROUTE, publisher_human_rankings_not_fresh_own_rollout_judgments=True,
        policy_updates=0, candidate_promoted=False, goal_achieved=False)
