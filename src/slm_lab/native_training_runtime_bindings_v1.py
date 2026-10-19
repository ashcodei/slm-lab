"""Additional immutable runtime code bindings, separate from neural data scope.

Keeping policy/data bindings stable permits a later controller/export reader
to reuse qualified capacity. Every added runtime source remains independently
closed and checked; no original policy or data source may be replaced.
"""
from pathlib import Path

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .parallel_function_repair_stage_v1 import value_digest

PROTOCOL = 'own-native-additional-immutable-controller-export-code-v1'


class RuntimeBindings:
    def __init__(self, sources, *, required):
        if (type(sources) is not dict or not sources
                or any(type(k) is not str or type(v) is not str or len(v) != 64 for k, v in sources.items())):
            raise ValueError('The entire additional runtime source map and every exact SHA are required')
        self.sources = dict(sources)
        for path in [Path(__file__).resolve(), *map(lambda p: Path(p).resolve(), required)]:
            if self.sources.get(str(path)) != file_hash(path):
                raise ValueError('Each actual runtime implementation must belong to the independently closed source map')
        self.identity = value_digest(self.sources)
        self.guard = FileBindings(self.sources)

    def check(self):
        if value_digest(self.sources) != self.identity:
            raise ValueError('The closed additional runtime map changed')
        self.guard.check()

    def combine(self, policy_sources):
        self.check()
        combined = dict(policy_sources)
        for path, digest in self.sources.items():
            if path in combined and combined[path] != digest:
                raise ValueError('Additional runtime sources cannot replace any original policy/data binding')
            combined[path] = digest
        return combined
