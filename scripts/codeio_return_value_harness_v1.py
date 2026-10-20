"""Return one original function value inside the existing confined runtime.

Only a source file, one keyword-input object and declared seed are readable.
Expected values never enter this interpreter. A fresh process handles each
case, so module state from another case cannot become an invisible input.
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
    started=time.perf_counter();random.seed(settings['seed'])
    namespace={'__name__':'__main__','__file__':'<original-CodeIO-function>'}
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        exec(compile(source,'<original-CodeIO-function>','exec'),namespace)
        value=namespace[settings['function']](**settings['kwargs'])
        # Match the publisher's JSON conversion (including tuple -> list),
        # while qualifying nonfinite/unserializable values explicitly.
        value=json.loads(json.dumps(value,allow_nan=False))
    print(json.dumps(dict(format='own-CodeIO-single-return-value-v1',case_position=settings['case_position'],
        value=value,seconds=time.perf_counter()-started),allow_nan=False))


if __name__=='__main__':main()
