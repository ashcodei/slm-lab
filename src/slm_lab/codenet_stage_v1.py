"""Complete-code exposure order and token-mean objectives with explicit rehearsal.

This module does not admit candidate data, choose a checkpoint, update weights,
or alter the live curriculum. A caller updating weights must use an admitted
training view; gradient-only audits can inspect the still-unadmitted candidate.
"""
from dataclasses import dataclass
import math
import random

from .training import supervised_gradients


class EpochOrder:
    """Deterministic full permutations addressed by absolute draw position."""
    def __init__(self, size, seed):
        if type(size) is not int or size < 1 or type(seed) is not int:
            raise ValueError('Positive source population and integer seed required')
        self.size, self.seed, self.epoch, self.order = size, seed, None, None

    def at(self, position):
        if type(position) is not int or position < 0:
            raise ValueError('Nonnegative absolute exposure position required')
        epoch, offset = divmod(position, self.size)
        if epoch != self.epoch:
            self.order = list(range(self.size))
            random.Random(self.seed + epoch).shuffle(self.order)
            self.epoch = epoch
        return self.order[offset]

    def rank_groups(self, step, rank, world, batch_size, accumulation):
        if (any(type(v) is not int for v in [step, rank, world, batch_size, accumulation])
            or step < 0 or world < 1 or not 0 <= rank < world or batch_size < 1 or accumulation < 1):
            raise ValueError('Valid explicit distributed exposure layout required')
        base = step * world * batch_size * accumulation
        return [[self.at(base + micro * world * batch_size + rank * batch_size + item)
                 for item in range(batch_size)] for micro in range(accumulation)]


def exposure_budget(population, requested_passes, world, batch_size, accumulation):
    if any(type(v) is not int or v < 1 for v in [population, requested_passes, world, batch_size, accumulation]):
        raise ValueError('Positive integer exposure budget required')
    width = world * batch_size * accumulation
    steps = (population * requested_passes + width - 1) // width
    exposures = steps * width
    return {'optimizer_steps': steps, 'global_examples_per_update': width,
            'actual_source_exposures': exposures, 'complete_passes': exposures // population,
            'next_pass_remainder': exposures % population,
            'rounding_exposures_above_requested_passes': exposures - population * requested_passes}


def rehearsal_selection(step, family_names):
    if type(step) is not int or step < 0 or not family_names or len(set(family_names)) != len(family_names):
        raise ValueError('Unique nonempty rehearsal families and nonnegative step required')
    return family_names[step % len(family_names)], step // len(family_names)


@dataclass
class Objective:
    name: str
    encoder: object
    groups: list
    field: str | None
    coefficient: float


class _ScaledLoss:
    """Scale the objective before backward, retaining DDP's existing no_sync."""
    def __init__(self, model, coefficient):
        self.model, self.coefficient = model, coefficient

    def __call__(self, **batch):
        return {'loss': self.model(**batch)['loss'] * self.coefficient}

    def no_sync(self):
        return self.model.no_sync()


def objective_gradients(model, objectives, device, amp, world=1):
    """Sum separately normalised objective gradients, without taking an update.

    Each coefficient multiplies that stream's global target-token mean. All
    ranks must use the same stream order and coefficients. Different streams
    can use complete packed code or ordinary text/image rehearsal encoders.
    """
    if type(world) is not int or world < 1:
        raise ValueError('Positive explicit distributed worker count required')
    if not objectives or len({item.name for item in objectives}) != len(objectives):
        raise ValueError('Distinct nonempty objectives required')
    for item in objectives:
        if (not item.name or not math.isfinite(item.coefficient) or item.coefficient <= 0
            or not item.groups or any(not group for group in item.groups)):
            raise ValueError('Finite positive coefficients and nonempty microbatches required')
    result, total, targets = {}, 0., 0
    for item in objectives:
        scaled, count = supervised_gradients(_ScaledLoss(model, item.coefficient), item.encoder,
            item.groups, item.field, device, amp, world=world)
        weighted = float(scaled.detach())
        result[item.name] = {'global_target_tokens': count, 'coefficient': item.coefficient,
                             'token_mean_nll': weighted / item.coefficient, 'weighted_objective': weighted}
        total += weighted
        targets += count
    return {'objectives': result, 'summed_objective': total, 'total_target_tokens': targets}


def freeze_for_code_stage(model):
    """Preserve own vision weights and critic while updating the language model."""
    if not hasattr(model, 'vision') or not hasattr(model, 'value_head'):
        raise ValueError('Own multimodal model with explicit vision and critic required')
    trainable, frozen = [], []
    for name, parameter in model.named_parameters():
        enabled = not name.startswith(('vision.', 'value_head.'))
        parameter.requires_grad_(enabled)
        (trainable if enabled else frozen).append(parameter)
    if not trainable or not frozen:
        raise ValueError('Both language and frozen multimodal/critic parameters required')
    return trainable, frozen
