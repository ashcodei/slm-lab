"""Complete hash validation around source-only recovery, with cheap live checks.

No result may be published until verify_final succeeds. This guard is only
for reconstructing immutable observations; it never authorizes model updates.
"""
import threading

from .codenet_checkpoint_v1 import FileBindings


class SourceOnlyRecoveryGuard:
    def __init__(self, files, resource_check, *, check_every=64):
        if not callable(resource_check) or type(check_every) is not int or not 1 <= check_every <= 64:
            raise ValueError('Explicit source-only resources and bounded metadata-check interval required')
        resource_check()
        self.binding = FileBindings(files)
        self.resource_check, self.check_every = resource_check, check_every
        self.thread, self.calls, self.verified = threading.get_ident(), 0, False
        resource_check()

    def check(self):
        if self.verified or threading.get_ident() != self.thread:
            raise ValueError('Only the owning unverified source-only recovery context may continue')
        self.resource_check()
        if self.calls % self.check_every == 0:
            for path, original in self.binding.states.items():
                if self.binding._state(path) != original:
                    raise ValueError('An immutable recovery input changed: ' + str(path))
        self.calls += 1

    def verify_final(self):
        self.check()
        self.binding.check(full=True)
        self.resource_check()
        self.verified = True
