"""Deterministic modality-safe replay quotas with an explicit full-pass floor."""
from fractions import Fraction
from functools import lru_cache


class CoverageReplaySchedule:
    def __init__(self, populations, steps, global_examples_per_update):
        if (not populations or len({name for name,size in populations}) != len(populations)
            or any(not isinstance(name,str) or not name or type(size) is not int or size < 1 for name,size in populations)
            or type(steps) is not int or steps < 1
            or type(global_examples_per_update) is not int or global_examples_per_update < 1):
            raise ValueError('Unique positive replay populations and explicit integer update layout required')
        self.populations = tuple(populations)
        self.width, self.steps = global_examples_per_update, steps
        minimum = [(size + self.width - 1) // self.width for name,size in self.populations]
        self.minimum_required_updates = sum(minimum)
        if self.minimum_required_updates > steps:
            raise ValueError('Declared replay capacity cannot reach every complete source example')
        extra, remainder = divmod(steps - self.minimum_required_updates, len(populations))
        self.quotas = tuple(required + extra + (i < remainder) for i,required in enumerate(minimum))
        # Midpoints evenly interleave each family's fixed number of updates.
        # Exact rational comparisons make the ordering independent of float rounding.
        entries = [(Fraction(2*occurrence+1,2*quota),i,name,occurrence)
                   for i,((name,size),quota) in enumerate(zip(self.populations,self.quotas))
                   for occurrence in range(quota)]
        self.entries = tuple((name,occurrence) for _,i,name,occurrence in sorted(entries))
        if len(self.entries) != steps:
            raise ValueError('Exact replay update accounting required')

    def at(self, step):
        if type(step) is not int or not 0 <= step < self.steps:
            raise IndexError('Replay update outside its exact declared budget')
        return self.entries[step]


@lru_cache(maxsize=8)
def coverage_replay_schedule(populations, steps, global_examples_per_update):
    return CoverageReplaySchedule(populations,steps,global_examples_per_update)
