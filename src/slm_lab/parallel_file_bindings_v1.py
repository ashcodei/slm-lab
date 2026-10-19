"""The same complete SHA256 admission, parallelized across independent files."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .corpus import file_hash


class ParallelFileBindings:
    def __init__(self,files,workers=4):
        if not files or type(workers) is not int or not 1<=workers<=8:raise ValueError('Explicit nonempty source bindings and bounded hash workers required')
        self.files={Path(p):d for p,d in files.items()};self.states={};self.workers=workers
        def verify(item):
            p,digest=item;before=self._state(p);observed=file_hash(p);after=self._state(p)
            if before!=after or observed!=digest:raise ValueError('Bound source changed or differs: '+str(p))
            return p,after
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for p,state in pool.map(verify,self.files.items()):self.states[p]=state

    @staticmethod
    def _state(path):
        s=path.stat();return s.st_ino,s.st_size,s.st_mtime_ns

    def check(self,full=False):
        def verify(item):
            p,digest=item
            if self._state(p)!=self.states[p] or ((full or self.states[p][1]<=8*1024**2) and file_hash(p)!=digest):
                raise ValueError('Bound source changed during run: '+str(p))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:list(pool.map(verify,self.files.items()))
