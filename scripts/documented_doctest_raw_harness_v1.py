"""Execute source example commands, returning raw output without expected text."""
import contextlib
import io
import json
from pathlib import Path
import sys
import traceback


def main():
    settings=json.loads(Path(sys.argv[1]).read_text());namespace={'__name__':'documented_source','__file__':'<documented-source>'}
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        exec(compile(Path(settings['prelude']).read_text(),'<declared-stdlib-context>','exec'),namespace)
        exec(compile(Path(settings['program']).read_text(),'<unchanged-retained-function>','exec'),namespace)
    rows=[]
    for source in settings['examples']:
        output=io.StringIO();exception=None;calls=[0]
        def profile(frame,event,arg):
            if event=='call' and frame.f_code.co_filename=='<unchanged-retained-function>' and frame.f_code.co_name==settings['function']:
                calls[0]+=1
        try:
            with contextlib.redirect_stdout(output),contextlib.redirect_stderr(io.StringIO()):
                sys.setprofile(profile)
                exec(compile(source,'<original-source-example>','single'),namespace)
        except BaseException as error:
            exception=''.join(traceback.format_exception_only(type(error),error))
        finally:sys.setprofile(None)
        rows.append(dict(stdout=output.getvalue(),exception_only=exception,entry_body_calls=calls[0]))
    print(json.dumps(dict(format='own-documented-doctest-raw-output-v1',results=rows),allow_nan=False))


if __name__=='__main__':main()
