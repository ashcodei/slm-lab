"""Observe every cached input in one confined interpreter per source/seed.

Each case gets a newly executed module namespace and the declared random
seed. Interpreter/module caches are shared across cases, unlike the retained
V1 fresh-process observations. Expected outputs never enter this process.
"""
import contextlib
import io
import json
from pathlib import Path
import random
import sys
import time


def main():
    settings=json.loads(Path(sys.argv[1]).read_text())
    source=Path(settings['program']).read_text()
    compiled=compile(source,'<original-CodeIO-function>','exec')
    results=[]
    for position,kwargs in enumerate(settings['inputs']):
        started=time.perf_counter();random.seed(settings['seed'])
        namespace={'__name__':'__main__','__file__':'<original-CodeIO-function>'}
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            exec(compiled,namespace)
            value=namespace[settings['function']](**kwargs)
            value=json.loads(json.dumps(value,allow_nan=False))
        results.append(dict(case_position=position,value=value,seconds=time.perf_counter()-started))
    print(json.dumps(dict(format='own-CodeIO-whole-return-values-v2',records=results),allow_nan=False))


if __name__=='__main__':main()
