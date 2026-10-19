"""Trusted bounded harness; gold outputs never enter this process.

Candidate source executes only inside the existing pre-exec OS confinement.
Literal observations preserve Python tuples, sets and dictionary key types.
"""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys


def main():
    settings = json.loads(Path(sys.argv[1]).read_text())
    if set(settings) != {'program','function','inputs'}:
        raise ValueError('Only program, function and original literal inputs allowed')
    program = Path(settings['program']).read_text()
    namespace = {'__name__':'candidate_module', '__file__':'<candidate-module>'}
    inputs = [(ast.literal_eval(r['args_literal']),ast.literal_eval(r['kwargs_literal'])) for r in settings['inputs']]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(program,'<candidate-module>','exec'),namespace)
    function = namespace[settings['function']]
    values = []
    for args,kwargs in inputs:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            answer = function(*args,**kwargs)
            represented = repr(answer)
        if len(represented.encode()) > 256*1024:
            raise ValueError('Bounded literal output required')
        values.append(represented)
    print(json.dumps(dict(format='public-code-literal-values-v1',actual_value_literals=values)))


if __name__=='__main__':main()
