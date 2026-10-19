"""Return actual ordinary-Python function values; no expected outputs are read.

This trusted harness runs only through the existing OS-confined interpreter.
The function module uses an import-style name, so its ordinary main guard is
inactive. Program bytes are compiled unchanged; no AST repair/subsetting.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import time


def main():
    settings = json.loads(Path(sys.argv[1]).read_text())
    if set(settings) != {'program', 'function', 'inputs'}: raise ValueError('Only actual program/interface/input arguments may enter confinement')
    source = Path(settings['program']).read_text(); compiled = compile(source, '<candidate-module>', 'exec')
    namespace = {'__name__': 'candidate_module', '__file__': '<candidate-module>'}
    clock = time.perf_counter
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()): exec(compiled, namespace)
    fn = namespace[settings['function']]
    if not callable(fn): raise TypeError('The requested actual module function must be callable')
    rows = []
    for args in settings['inputs']:
        started = clock()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()): value = fn(*args)
        # Preserve the explicit JSON-compatible local benchmark result domain.
        value = json.loads(json.dumps(value, allow_nan=False))
        rows.append(dict(value=value, seconds=clock() - started))
    print(json.dumps(dict(format='own-ordinary-Python-function-raw-values-v1', results=rows), allow_nan=False))


if __name__ == '__main__': main()
