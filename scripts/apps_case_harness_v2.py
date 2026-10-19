"""Emit program results, never pass/fail labels, inside the existing sandbox.

Expected answers remain in the parent process and are never written into this
worker's readable filesystem. This is a local standard-library APPS runtime,
not the original pyext/NumPy publisher runtime.
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import time


def main():
    settings = json.loads(Path(sys.argv[1]).read_text())
    source = Path(settings['program']).read_text()
    compiled = compile(source, '<original-program>', 'exec')
    prelude = compile(Path(settings['prelude']).read_text(), '<declared-stdlib-prelude>', 'exec')
    inputs = settings['inputs']
    mode = settings['mode']
    if mode not in {'function', 'stdin'}:
        raise ValueError('Unknown explicit invocation mode')
    results = []
    namespace = {'__name__': '__main__', '__file__': '<original-program>'}
    if mode == 'function':
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(prelude, namespace)
            exec(compiled, namespace)
            # Only an actual class declaration selects the publisher's common
            # Solution interface; comments/string literals cannot select it.
            owner = namespace.get('Solution')
            owner = owner() if isinstance(owner, type) else namespace
            method = owner[settings['function']] if isinstance(owner, dict) else getattr(owner, settings['function'])
        for arguments in inputs:
            started = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                value = method(*arguments)
            # JSON converts tuples to lists as in the declared local contract.
            value = json.loads(json.dumps(value, allow_nan=False))
            results.append({'value': value, 'seconds': time.perf_counter() - started})
    else:
        original_stdin = sys.stdin
        for text in inputs:
            namespace = {'__name__': '__main__', '__file__': '<original-program>'}
            sys.stdin = io.TextIOWrapper(io.BytesIO(text.encode()), encoding='utf-8')
            output = io.StringIO()
            started = time.perf_counter()
            try:
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
                    exec(prelude, namespace)
                    try:
                        exec(compiled, namespace)
                    except SystemExit as error:
                        if error.code not in (None, 0):
                            raise
                results.append({'value': output.getvalue(), 'seconds': time.perf_counter() - started})
            finally:
                sys.stdin = original_stdin
    # No expected values, comparisons or asserted success can be supplied by
    # this process. The parent validates all returned values itself.
    print(json.dumps({'format': 'own-APPS-raw-results-v2', 'results': results}, allow_nan=False))


if __name__ == '__main__':
    main()
