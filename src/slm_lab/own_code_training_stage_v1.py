"""Complete original generation/repair CE, with sampled multilingual replay.

Only own conditional-code components train. Immutable own general weights
remain references in collective snapshots. This kernel acquires no GPUs and
does not select/promote a policy. Each objective has its own global target
token normalizer; fixed-order worker averaging permits exact CPU recovery.
"""
from contextlib import nullcontext
from dataclasses import asdict
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import distributed as dist
from tokenizers import Tokenizer

from .codenet_checkpoint_v1 import restore_rng, save_collective
from .codenet_runner_v1 import FixedReductionModel, average_gradients
from .codenet_stage_v1 import EpochOrder, Objective, objective_gradients
from .codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from .corpus import file_hash
from .own_code_delta_v1 import OwnCodeDelta, matrix_names, norm_names

STREAMS = ('generation', 'repair', 'variety')
FORMAT = 'own-code-generation-repair-resume-v1'


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu(v) for v in value)
    return value


def collective_check(function):
    value = None; error = None
    try:
        value = function()
    except Exception as problem:
        error = type(problem).__name__ + ': ' + str(problem)
    if dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
    else:
        errors = [error]
    if any(errors):
        raise ValueError('Collective own-code preflight failed: ' + repr(errors))
    return value


class PreparedCodeStream:
    role = 'train'
    purpose = 'optimization'

    def __init__(self, name, folder, tokenizer, config):
        if name not in STREAMS:
            raise ValueError('Explicit generation, repair or variety stream required')
        self.name = name
        self.folder = Path(folder)
        self.pack = CompleteCodePack(self.folder, 'train', purpose='training')
        admission = json.loads((self.folder / 'admission.json').read_text())
        expected = {'generation': 'own-complete-human-code-verified-generation-admission-v1',
                    'repair': 'own-complete-human-target-execution-repair-CE-admission-v1'}
        if ((name in expected and admission.get('format') != expected[name])
                or admission.get('human_preferences', False)
                or file_hash(self.folder / 'tokenizer.json') != file_hash(tokenizer)):
            raise ValueError('Exact own-tokenizer train-only objective admission required')
        if name == 'variety' and admission.get('admitted_role') != 'train':
            raise ValueError('Only the admitted original CodeNet train role may supply sampled variety')
        self.encoder = packed_encoder(self.pack, Tokenizer.from_file(str(tokenizer)), config)
        self.manifest_sha256 = file_hash(self.folder / 'manifest.json')
        self.admission_sha256 = file_hash(self.folder / 'admission.json')
        if name == 'variety':
            languages = defaultdict(list)
            cursor = 0
            with gzip.open(self.folder / 'original-rows.jsonl.gz', 'rt') as stream:
                for row in map(json.loads, stream):
                    if row['split'] == 'train':
                        languages[row['language']].append(cursor)
                        cursor += 1
            if cursor != len(self.pack):
                raise ValueError('Every actual original train position needs its original language')
            self.language_positions = dict(languages)

    def __len__(self):
        return len(self.pack)


def make_optimizer(model, options):
    return torch.optim.AdamW(model.code_parameters(), lr=options['learning_rate'],
        betas=tuple(options['betas']), weight_decay=options['weight_decay'], foreach=False)


def components(model):
    """Export all trainable pieces in the original own-code inference format."""
    return dict(matrices={name: dict(down=cpu(parts.down), up=cpu(parts.up))
                for name, parts in zip(matrix_names(model.config.num_layers), model.parts['linears'])},
        embedding=cpu(model.parts['embedding'].weight),
        norms={name: cpu(module.weight) for name, module in zip(norm_names(model.config.num_layers), model.parts['norms'])})


class LanguageCycleOrder:
    """Visit every admitted source language once per shuffled language cycle.

    Each language has its own complete permutation of original train rows.
    Rare languages can repeat; this is sampled replay, not a full corpus pass.
    """
    rank_groups = EpochOrder.rank_groups

    def __init__(self, language_positions, seed):
        self.languages = sorted(language_positions)
        if not self.languages or any(not language_positions[name] for name in self.languages):
            raise ValueError('Nonempty original train-language populations required')
        self.positions = {name: list(language_positions[name]) for name in self.languages}
        self.language_order = EpochOrder(len(self.languages), seed)
        self.orders = {name: EpochOrder(len(self.positions[name]), seed + 100003 + i * 1009)
                       for i, name in enumerate(self.languages)}

    def at(self, position):
        if type(position) is not int or position < 0:
            raise ValueError('Nonnegative absolute language-balanced exposure required')
        language = self.languages[self.language_order.at(position)]
        occurrence = position // len(self.languages)
        return self.positions[language][self.orders[language].at(occurrence)]

    def cursor(self, exposures):
        cycles, remainder = divmod(exposures, len(self.languages))
        extra = {self.languages[self.language_order.at(cycles * len(self.languages) + i)] for i in range(remainder)}
        return dict(sampling='balanced_original_language_cycles_v1', actual_exposures=exposures,
            full_original_source_pass=all(cycles >= len(self.positions[name]) for name in self.languages),
            by_language={name: dict(population=len(self.positions[name]), actual_draws=cycles + (name in extra),
                unique_original_positions=min(cycles + (name in extra), len(self.positions[name]))) for name in self.languages})


class OwnCodeTrainingStage:
    def __init__(self, model, optimizer, streams, options, sources, device, amp=None):
        if type(model) is not OwnCodeDelta or tuple(streams) != STREAMS:
            raise ValueError('Exact own conditional-code model and fixed three-stream order required')
        if any(stream.name != name or stream.role != 'train' or stream.purpose != 'optimization' or not len(stream)
               for name, stream in streams.items()):
            raise ValueError('Every complete stream must have an admitted original optimization role')
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        integer = ['seed', 'world_size', 'microbatch_per_rank', 'accumulation', 'global_examples_per_stream', 'steps']
        if (any(type(options[key]) is not int or options[key] < 1 for key in integer)
                or options['world_size'] != self.world
                or options['global_examples_per_stream'] != self.world * options['microbatch_per_rank'] * options['accumulation']
                or options['steps'] * options['global_examples_per_stream'] != len(streams['repair'])):
            raise ValueError('Exactly one complete repair pass and explicit global stream batch required')
        numeric = ['learning_rate', 'weight_decay', 'gradient_norm_clip', 'warmup_fraction', 'cosine_floor_fraction']
        if (any(not math.isfinite(options[key]) or options[key] <= 0 for key in numeric)
                or not options['warmup_fraction'] < 1 or not options['cosine_floor_fraction'] <= 1
                or len(options['betas']) != 2 or any(not 0 <= b < 1 for b in options['betas'])
                or set(options['coefficients']) != set(STREAMS)
                or any(not math.isfinite(v) or v <= 0 for v in options['coefficients'].values())):
            raise ValueError('Finite explicit Adam, schedule and separately normalized objective weights required')
        model.assert_partition()
        self.parameters = model.code_parameters()
        actual = [p for group in optimizer.param_groups for p in group['params']]
        if len(actual) != len(self.parameters) or [id(p) for p in actual] != [id(p) for p in self.parameters]:
            raise ValueError('Exactly the ordered own code components belong in Adam')
        self.model = model; self.optimizer = optimizer; self.streams = streams
        self.options = dict(options); self.sources = dict(sources); self.device = torch.device(device)
        self.amp = amp if amp is not None else nullcontext
        language_positions = streams['variety'].language_positions
        flattened = [p for group in language_positions.values() for p in group]
        if sorted(flattened) != list(range(len(streams['variety']))):
            raise ValueError('Every original variety train position belongs to exactly one source language')
        self.orders = {name: (LanguageCycleOrder(language_positions, options['seed'] + 1009 * i)
                             if name == 'variety' else EpochOrder(len(stream), options['seed'] + 1009 * i))
                       for i, (name, stream) in enumerate(streams.items())}
        self.steps = options['steps']; self.completed = 0; self.metrics = []
        self.network = FixedReductionModel(model.code)
        self.initial_optimizer_groups = cpu(optimizer.state_dict()['param_groups'])

    def positions(self, step, rank):
        if type(step) is not int or not 0 <= step < self.steps:
            raise ValueError('Actual sealed original stream update position required')
        return {name: order.rank_groups(step, rank, self.world, self.options['microbatch_per_rank'], self.options['accumulation'])
                for name, order in self.orders.items()}

    def schedule(self, step):
        warm = max(1, math.ceil(self.steps * self.options['warmup_fraction']))
        if step < warm:
            return (step + 1) / warm
        fraction = (step - warm) / max(1, self.steps - warm)
        floor = self.options['cosine_floor_fraction']
        return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * fraction))

    def update(self):
        if self.completed >= self.steps:
            raise ValueError('Complete original repair epoch already finished')
        self.model.assert_partition(); self.model.code.train(); self.model.base.eval()
        self.optimizer.zero_grad(set_to_none=True)
        lr = self.options['learning_rate'] * self.schedule(self.completed)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        positions = self.positions(self.completed, self.rank)
        def build():
            objectives = []
            for name, stream in self.streams.items():
                groups = [[{'packed_code_index': position} for position in micro] for micro in positions[name]]
                objectives.append(Objective(name, stream.encoder, groups, 'response', self.options['coefficients'][name]))
            return objectives
        objectives = collective_check(build)
        observation = objective_gradients(self.network, objectives, self.device, self.amp, world=self.world)
        def present():
            if (any(p.grad is None for p in self.parameters)
                    or any(p.grad is not None for p in self.model.base.parameters())):
                raise ValueError('All own-code gradients and no frozen-general gradients required')
        collective_check(present)
        average_gradients(self.model.code, self.device, self.world)
        norm = collective_check(lambda: float(torch.nn.utils.clip_grad_norm_(self.parameters,
                                    self.options['gradient_norm_clip'], error_if_nonfinite=True)))
        self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        self.model.assert_partition()
        draw = dict(rank=self.rank, positions=positions)
        if self.world > 1:
            draws = [None] * self.world
            dist.all_gather_object(draws, draw)
        else:
            draws = [draw]
        self.completed += 1
        record = dict(step=self.completed, learning_rate=lr, gradient_norm_before_clipping=norm,
                      **observation, worker_draws=draws)
        self.metrics.append(record)
        return record

    def sampler_state(self):
        exposures = self.completed * self.options['global_examples_per_stream']
        return self.source_cursors(exposures)

    def source_cursors(self, exposures):
        return {name: (self.orders[name].cursor(exposures) if name == 'variety' else
                      dict(population=len(stream), seed=self.options['seed'] + i * 1009,
                           actual_exposures=exposures, complete_passes=exposures // len(stream),
                           next_pass_remainder=exposures % len(stream)))
                for i, (name, stream) in enumerate(self.streams.items())}

    def validate_Adam(self):
        if len(self.optimizer.state) != len(self.parameters):
            raise ValueError('Every own-code component needs complete Adam state')
        for parameter in self.parameters:
            state = self.optimizer.state[parameter]
            if (float(state['step']) != self.completed or not bool(torch.isfinite(parameter).all())
                    or any(key not in state or state[key].shape != parameter.shape
                           or not bool(torch.isfinite(state[key]).all()) for key in ['exp_avg', 'exp_avg_sq'])):
                raise ValueError('Every own-code component/Adam moment and exact step must be finite')

    def save(self, path, maximum_bytes, minimum_free_bytes):
        collective_check(self.validate_Adam)
        def bundle(states):
            return dict(format=FORMAT, parts=cpu(self.model.parts.state_dict()),
                optimizer=cpu(self.optimizer.state_dict()), config=asdict(self.model.config),
                actual_counts=self.model.counts(), reserved_parameters=self.model.reserved_parameters,
                steps=self.completed, planned_steps=self.steps, world_size=self.world,
                options=self.options, sources=self.sources, sampler_state=self.sampler_state(),
                worker_rng_states=states, metrics=self.metrics, own_base_is_immutable_reference=True,
                model_promoted=False, goal_achieved=False)
        save_collective(path, bundle, self.device, self.world, self.rank, maximum_bytes, minimum_free_bytes)

    def resume(self, state):
        if (state.get('format') != FORMAT or state['sources'] != self.sources or state['options'] != self.options
                or state['world_size'] != self.world or state['planned_steps'] != self.steps
                or state['config'] != asdict(self.model.config) or state['actual_counts'] != self.model.counts()
                or state['reserved_parameters'] != self.model.reserved_parameters
                or type(state['steps']) is not int or not 0 < state['steps'] < self.steps
                or len(state['metrics']) != state['steps'] or len(state['worker_rng_states']) != self.world):
            raise ValueError('Exact own-code source/model/optimizer/worker/budget recovery contract required')
        exposures = state['steps'] * self.options['global_examples_per_stream']
        expected = self.source_cursors(exposures)
        if state['sampler_state'] != expected:
            raise ValueError('Exact original generation/repair/replay source cursors required')
        parts = self.model.parts.state_dict()
        if state['parts'].keys() != parts.keys() or any(t.shape != parts[n].shape or t.dtype != parts[n].dtype
                or not bool(torch.isfinite(t).all()) for n, t in state['parts'].items()):
            raise ValueError('Every finite actual own-code component tensor required')
        saved = state['optimizer']; groups = saved['param_groups']
        if len(groups) != 1 or len(self.initial_optimizer_groups) != 1:
            raise ValueError('One exact ordered own-code Adam group required')
        initial = self.initial_optimizer_groups[0]
        if ({k: v for k, v in groups[0].items() if k != 'lr'} != {k: v for k, v in initial.items() if k != 'lr'}
                or groups[0]['lr'] != self.options['learning_rate'] * self.schedule(state['steps'] - 1)
                or set(saved['state']) != set(initial['params'])):
            raise ValueError('Exact ordered own-code Adam options/positions/schedule required')
        for identity, parameter in zip(initial['params'], self.parameters):
            moments = saved['state'][identity]
            if (float(moments['step']) != state['steps'] or any(key not in moments or moments[key].shape != parameter.shape
                    or moments[key].dtype != parameter.dtype or not bool(torch.isfinite(moments[key]).all())
                    for key in ['exp_avg', 'exp_avg_sq'])):
                raise ValueError('Complete finite own-code Adam moments and exact step required')
        for rng in state['worker_rng_states']:
            try:
                random.Random().setstate(rng['python'])
                torch.Generator(device='cpu').set_state(rng['torch'])
                n = rng['numpy']
                np.random.RandomState().set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
                if (rng['cuda'] is None) != (self.device.type != 'cuda'):
                    raise ValueError('Original CPU/CUDA worker RNG contract required')
                if rng['cuda'] is not None and (rng['cuda'].dtype != torch.uint8 or rng['cuda'].ndim != 1 or not rng['cuda'].numel()):
                    raise ValueError('Actual complete CUDA worker RNG bytes required')
                if rng['cuda'] is not None:
                    # Validate the actual device-specific byte format on a
                    # private generator. A merely nonempty uint8 tensor can
                    # still fail CUDA set_state; detect that before changing
                    # live parts, Adam, cursors or any default RNG state.
                    torch.Generator(device=self.device).set_state(rng['cuda'])
            except (KeyError, TypeError, RuntimeError, ValueError) as error:
                raise ValueError('Valid complete original per-worker RNG states required') from error
        # Every check above is nonmutating. Only a fully validated snapshot
        # can change live component weights, Adam, cursors or global RNG.
        self.model.parts.load_state_dict(state['parts'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.completed = state['steps']; self.metrics = list(state['metrics'])
        restore_rng(state['worker_rng_states'][self.rank], self.device)
