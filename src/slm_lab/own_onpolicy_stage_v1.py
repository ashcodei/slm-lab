"""Matched complete-code CE +/- actual own-execution DPO, one GPU per arm.

The original general/vision/critic stays frozen. Reference probabilities are
precomputed from the same own initial policy, never an external teacher. DPO
uses finite machine-execution labels rather than human preference labels.
"""
from contextlib import nullcontext
from dataclasses import asdict
import math

import torch
from torch import distributed as dist

from .codenet_checkpoint_v1 import atomic_checkpoint, capture_rng
from .codenet_runner_v1 import FixedReductionModel
from .codenet_stage_v1 import EpochOrder, Objective, objective_gradients
from .execution_preference_v1 import FORMAT as REFERENCE_FORMAT, fingerprint, preference_gradients, validate
from .own_code_delta_v1 import OwnCodeDelta
from .own_code_training_stage_v1 import OwnCodeTrainingStage, LanguageCycleOrder, cpu, FORMAT as PRIOR_FORMAT

FORMAT = 'own-onpolicy-code-CE-execution-DPO-resume-v1'


def validate_reference(pairs, reference, policy_sha256):
    counts = validate(pairs, True)
    if (reference.get('format') != REFERENCE_FORMAT or reference.get('policy_sha256') != policy_sha256
            or reference.get('human_preferences') is not False or set(reference['records']) != {p['id'] for p in pairs}):
        raise ValueError('Exact complete frozen own-policy reference records required')
    for pair, n in zip(pairs, counts):
        if (pair.get('publisher_split') != 'train' or pair.get('negative_is_actual_complete_own_policy_response') is not True
                or pair.get('original_human_target_bytes_unchanged') is not True
                or pair.get('reference_policy_sha256') != policy_sha256):
            raise ValueError('Actual complete own response and unchanged original human train source required')
        ref = reference['records'][pair['id']]
        if (ref['chosen_targets'] != n[0] or ref['rejected_targets'] != n[1]
                or ref['chosen_fingerprint'] != fingerprint(pair['chosen'])
                or ref['rejected_fingerprint'] != fingerprint(pair['rejected'])
                or any(type(ref[k]) not in {int, float} or not math.isfinite(ref[k]) for k in ['chosen_logp', 'rejected_logp'])):
            raise ValueError('Every exact original chosen/rejected token, mask and finite frozen reference score required')


class OwnOnPolicyStage(OwnCodeTrainingStage):
    """Reuse audited reference-only component/Adam/RNG recovery invariants."""
    def __init__(self, model, optimizer, streams, pairs, reference, options, sources, device, amp=None):
        if (type(model) is not OwnCodeDelta or tuple(streams) != ('generation', 'variety')
                or any(s.name != name or s.role != 'train' or s.purpose != 'optimization' or not len(s)
                       for name, s in streams.items())):
            raise ValueError('Exact own model and two complete original optimization streams required')
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError('Independent world-one matched arms required')
        if (any(type(options[k]) is not int or options[k] < 1 for k in ['seed', 'steps', 'world_size',
                'microbatch_per_rank', 'accumulation', 'global_examples_per_stream'])
                or any(options[k] != 1 for k in ['world_size', 'microbatch_per_rank', 'accumulation', 'global_examples_per_stream'])):
            raise ValueError('Explicit positive budget and exactly one whole source/pair per arm update required')
        if (any(type(options[k]) not in {int, float} or not math.isfinite(options[k]) or options[k] <= 0
                    for k in ['learning_rate', 'weight_decay', 'gradient_norm_clip', 'warmup_fraction', 'cosine_floor_fraction', 'DPO_beta'])
                or not 0 < options['warmup_fraction'] < 1 or options['cosine_floor_fraction'] > 1
                or options['DPO_beta'] > 1 or type(options['DPO_alpha']) not in {int, float}
                or not math.isfinite(options['DPO_alpha']) or not 0 <= options['DPO_alpha'] <= 1
                or options['coefficients'] != dict(generation=1., variety=.25)
                or len(options['betas']) != 2 or any(not 0 <= v < 1 for v in options['betas'])):
            raise ValueError('Finite sealed learning schedule, CE weights and DPO coefficients required')
        validate_reference(pairs, reference, sources['initial_policy_manifest_sha256'])
        model.assert_partition(); parameters = model.code_parameters()
        if [id(p) for g in optimizer.param_groups for p in g['params']] != [id(p) for p in parameters]:
            raise ValueError('All and only ordered own code components belong in Adam')
        languages = streams['variety'].language_positions
        if sorted(p for group in languages.values() for p in group) != list(range(len(streams['variety']))):
            raise ValueError('Every original variety source needs exactly one original language')
        self.model = model; self.optimizer = optimizer; self.parameters = parameters; self.streams = streams
        self.pairs = pairs; self.reference = reference; self.options = dict(options); self.sources = dict(sources)
        self.device = torch.device(device); self.amp = amp if amp is not None else nullcontext
        self.world = 1; self.rank = 0; self.steps = options['steps']; self.completed = 0; self.metrics = []
        self.orders = dict(generation=EpochOrder(len(streams['generation']), options['seed']),
            variety=LanguageCycleOrder(languages, options['seed'] + 1009), pairs=EpochOrder(len(pairs), options['seed'] + 2018))
        self.network = FixedReductionModel(model.code)
        self.initial_optimizer_groups = cpu(optimizer.state_dict()['param_groups'])

    def positions(self, step, rank=0):
        if type(step) is not int or not 0 <= step < self.steps or rank != 0:
            raise ValueError('Exact world-one original source update position required')
        return {name: [[order.at(step)]] for name, order in self.orders.items()}

    def source_cursors(self, exposures):
        result = {}
        for index, name in enumerate(['generation', 'variety', 'pairs']):
            population = len(self.pairs) if name == 'pairs' else len(self.streams[name])
            result[name] = (self.orders[name].cursor(exposures) if name == 'variety' else dict(population=population,
                seed=self.options['seed'] + index * 1009, actual_exposures=exposures,
                complete_passes=exposures // population, next_pass_remainder=exposures % population))
        return result

    def update(self):
        if self.completed >= self.steps:
            raise ValueError('Original bounded own-policy update budget already complete')
        validate_reference(self.pairs, self.reference, self.sources['initial_policy_manifest_sha256'])
        self.model.assert_partition(); self.model.code.train(); self.model.base.eval()
        positions = self.positions(self.completed); pair = self.pairs[positions['pairs'][0][0]]
        lr = self.options['learning_rate'] * self.schedule(self.completed)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            dpo = preference_gradients(self.model.code, [pair], self.reference,
                expected_policy_sha256=self.sources['initial_policy_manifest_sha256'], alpha=self.options['DPO_alpha'],
                beta=self.options['DPO_beta'], device=self.device, amp=self.amp)
            objectives = [Objective(name, stream.encoder,
                [[dict(packed_code_index=p) for p in positions[name][0]]], 'response', self.options['coefficients'][name])
                for name, stream in self.streams.items()]
            ce = objective_gradients(self.network, objectives, self.device, self.amp, world=1)
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for p in self.model.base.parameters())):
                raise ValueError('Every and only own code component needs finite gradients')
            norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.options['gradient_norm_clip'], error_if_nonfinite=True))
            for group in self.optimizer.param_groups:
                group['lr'] = lr
            self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True); raise
        self.completed += 1
        record = dict(step=self.completed, learning_rate=lr, gradient_norm_before_clipping=norm,
            CE=ce, DPO=dpo, summed_objective=ce['summed_objective'] + dpo['weighted_DPO'],
            worker_draws=[dict(rank=0, positions=positions)], actual_pair_id=pair['id'])
        self.metrics.append(record); return record

    def save(self, path, maximum_bytes, minimum_free_bytes):
        self.validate_Adam()
        state = dict(format=FORMAT, parts=cpu(self.model.parts.state_dict()), optimizer=cpu(self.optimizer.state_dict()),
            config=asdict(self.model.config), actual_counts=self.model.counts(), reserved_parameters=self.model.reserved_parameters,
            steps=self.completed, planned_steps=self.steps, world_size=1, options=self.options, sources=self.sources,
            sampler_state=self.sampler_state(), worker_rng_states=[capture_rng(self.device)], metrics=self.metrics,
            own_base_is_immutable_reference=True, own_reference_scores_are_source_bound=True,
            model_promoted=False, goal_achieved=False)
        atomic_checkpoint(path, state, maximum_bytes, minimum_free_bytes)

    def resume(self, state):
        validate_reference(self.pairs, self.reference, self.sources['initial_policy_manifest_sha256'])
        if state.get('format') != FORMAT or state.get('own_reference_scores_are_source_bound') is not True:
            raise ValueError('Exact own execution-feedback snapshot and reference binding required')
        # The inherited validator checks every part, Adam moment/options, source
        # cursor and real private-device RNG before changing any live state.
        OwnCodeTrainingStage.resume(self, {**state, 'format': PRIOR_FORMAT})
