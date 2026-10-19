"""Confined worker: execute source and original LHS values, never RHS golds."""
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time


class Sink:
    def write(self,text):return len(text)
    def flush(self):pass


def main():
    path=Path(sys.argv[1]);settings=json.loads(path.read_text())
    spec=importlib.util.spec_from_file_location('trusted_bounded_values',path.with_name('values.py'))
    values=importlib.util.module_from_spec(spec);spec.loader.exec_module(values)
    namespace=dict(__name__='__candidate__',__file__=settings['program'])
    rows=[];failure=None;source=Path(settings['program']).read_text()
    with contextlib.redirect_stdout(Sink()),contextlib.redirect_stderr(Sink()):
        try:
            exec(compile(source,settings['program'],'exec'),namespace)
            exec(compile(settings['setup'],'<original-input-setup>','exec'),namespace)
        except BaseException as error:failure=dict(status='module_or_input_setup_error',error_type=type(error).__name__,diagnostic=str(error)[:300])
        if failure is None:
            for expression in settings['expressions']:
                start=time.perf_counter()
                try:
                    value=eval(compile(expression,'<original-test-left-expression>','eval'),namespace)
                    row=dict(status='value',value=values.encode(value))
                except BaseException as error:row=dict(status='expression_error',error_type=type(error).__name__,diagnostic=str(error)[:300])
                row['seconds']=time.perf_counter()-start;rows.append(row)
    print(json.dumps(dict(format='own-original-MBPP-raw-left-values-v1',results=rows,module_failure=failure),allow_nan=False),flush=True)


if __name__=='__main__':main()
