"""Explicit task adapters sharing one own dense backbone and one critic.

This is a custom budgeted experiment using established low-rank adapters, not
evidence of a new capability. Routing is explicit: the existing three-class
router is counted but has not been trained for these labels. All experts use
the same frozen vision encoder; image learning remains the foundation's job.
"""
from dataclasses import asdict

import torch
from torch import nn

from .own_dense_actor_v1 import OwnDenseActor, budget
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-shared-dense-explicit-task-adapters-v1'
EXPERTS = ('code', 'assistant', 'reasoning')
DEFAULT_RANKS = dict(code=8, assistant=4, reasoning=4)


def inventory(base, ranks):
    if not isinstance(ranks, dict) or set(ranks) != set(EXPERTS):
        raise ValueError('Exactly code, assistant and reasoning ranks required')
    actors = {name: budget(base, ranks[name])['actor'] for name in EXPERTS}
    dense = base.counts()['total_unique']
    critic = base.config.hidden_size
    return dict(shared_dense_and_all_heads=dense, task_adapters=actors,
                single_shared_critic=critic, total_unique=dense + sum(actors.values()) + critic)


class TaskNativeView(nn.Module):
    """Expose one selected computation while registering every exported expert."""
    def __init__(self, owner, expert):
        super().__init__()
        owner._require_expert(expert)
        self.owner, self.expert = owner, expert

    @property
    def config(self): return self.owner.config

    @property
    def vision(self): return self.owner.experts[self.expert].policy.vision

    def forward(self, *args, **kwargs):
        return self.owner(*args, expert=self.expert, **kwargs)

    def generate(self, *args, **kwargs):
        return self.owner.generate(*args, expert=self.expert, **kwargs)

    def prefill(self, *args, **kwargs):
        return self.owner.experts[self.expert].policy.prefill(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.owner.experts[self.expert].policy.decode(*args, **kwargs)


class OwnTaskExperts(nn.Module):
    def __init__(self, base, *, ranks=None, seed=960117, parameter_limit=10**9):
        super().__init__()
        ranks = dict(DEFAULT_RANKS if ranks is None else ranks)
        expected = inventory(base, ranks)
        if (type(parameter_limit) is not int or not expected['total_unique'] < parameter_limit <= 10**9
                or type(seed) is not int or seed < 1):
            raise ValueError('Every shared and expert parameter must fit strictly below 1B')
        self.base = base
        self.settings = dict(ranks=ranks, seed=seed, parameter_limit=parameter_limit)
        self.expected_counts = expected
        self.experts = nn.ModuleDict()
        for i, name in enumerate(EXPERTS):
            actor = OwnDenseActor(base, rank=ranks[name], seed=seed + i, parameter_limit=parameter_limit)
            if i == 0:
                self.critic = actor.parts['critic']
            else:
                actor.parts['critic'] = self.critic
                actor.policy.value_head = self.critic
            self.experts[name] = actor
        self.assert_shared()

    @property
    def config(self): return self.base.config

    def counts(self):
        return {**self.expected_counts, 'actual_unique': sum(p.numel() for p in self.parameters())}

    def assert_shared(self):
        self.base.assert_inventory()
        seen = {id(p) for p in self.base.parameters()} | {id(p) for p in self.critic.parameters()}
        for name in EXPERTS:
            actor = self.experts[name]
            actor.assert_shared()
            added = {id(p) for _, p in actor.named_selected(('actor',))}
            if (actor.base is not self.base or actor.parts['critic'] is not self.critic
                    or actor.policy.value_head is not self.critic or added & seen):
                raise ValueError('One shared backbone/critic and disjoint task adapters required')
            seen |= added
        if (self.counts()['actual_unique'] != self.expected_counts['total_unique']
                or seen != {id(p) for p in self.parameters()}
                or self.counts()['actual_unique'] >= self.settings['parameter_limit']):
            raise ValueError('Every actual expert, router, auxiliary head and critic value must be counted')

    def _require_expert(self, expert):
        if expert not in EXPERTS:
            raise ValueError('Explicit trained task selection required; automatic routing is not available')

    def named_selected(self, expert):
        self._require_expert(expert)
        return [(f'{expert}.{name}', p) for name, p in self.experts[expert].named_selected(('actor',))]

    def forward(self, *args, expert, **kwargs):
        self._require_expert(expert)
        return self.experts[expert](*args, **kwargs)

    def generate(self, *args, expert, **kwargs):
        self._require_expert(expert)
        return self.experts[expert].generate(*args, **kwargs)

    def native_view(self, expert):
        return TaskNativeView(self, expert)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def export(self, sources):
        self.assert_shared()
        if (not sources or any(not isinstance(k, str) or not isinstance(v, str)
                              or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                              for k, v in sources.items())):
            raise ValueError('Explicit own backbone and training source SHA bindings required')
        return dict(format=FORMAT, settings=self.settings, config=asdict(self.config),
            actual_counts=self.counts(), sources=dict(sources),
            frozen_dense_sha256=tensor_digest(self.base.named_parameters()),
            adapters={name: {key: value.detach().to(device='cpu', copy=True)
                            for key, value in self.experts[name].parts.state_dict().items()
                            if not key.startswith('critic.')} for name in EXPERTS},
            critic=self.critic.weight.detach().to(device='cpu', copy=True),
            automatic_routing_trained=False, candidate_promoted=False, goal_achieved=False)

    def validate_export(self, payload, *, expected_sources):
        self.assert_shared()
        if (payload.get('format') != FORMAT or payload.get('settings') != self.settings
                or payload.get('config') != asdict(self.config) or payload.get('actual_counts') != self.counts()
                or not expected_sources or payload.get('sources') != expected_sources
                or payload.get('frozen_dense_sha256') != tensor_digest(self.base.named_parameters())
                or payload.get('automatic_routing_trained') is not False
                or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False
                or not isinstance(payload.get('adapters'), dict) or set(payload['adapters']) != set(EXPERTS)):
            raise ValueError('Exact counted own backbone, task graph and source bindings required before mutation')
        for name in EXPERTS:
            live = {key: t for key, t in self.experts[name].parts.state_dict().items()
                    if not key.startswith('critic.')}
            saved = payload['adapters'][name]
            if (not isinstance(saved, dict) or saved.keys() != live.keys()
                    or any(not isinstance(t, torch.Tensor) or t.shape != live[key].shape
                           or t.dtype != live[key].dtype or not bool(torch.isfinite(t).all())
                           for key, t in saved.items())):
                raise ValueError('Every finite task tensor with exact shape and dtype required before mutation')
        critic = payload.get('critic')
        if (not isinstance(critic, torch.Tensor) or critic.shape != self.critic.weight.shape
                or critic.dtype != self.critic.weight.dtype or not bool(torch.isfinite(critic).all())):
            raise ValueError('Exact single finite critic required before mutation')

    @torch.no_grad()
    def import_export(self, payload, *, expected_sources):
        self.validate_export(payload, expected_sources=expected_sources)
        for name in EXPERTS:
            live = self.experts[name].parts.state_dict()
            for key, value in payload['adapters'][name].items():
                live[key].copy_(value)
        self.critic.weight.copy_(payload['critic'])
        self.assert_shared()
