"""Observe generated Python against every original public docstring example.

This is machine feedback on source-train functions, not human preference,
held benchmark accuracy, or admission of the rollouts to RL optimization.
The model input contains the original public examples and expected answers.
"""
import ast
import doctest
import json
import math
from pathlib import Path
import tempfile

from .documented_doctest_v1 import PRELUDE,HARNESS,compare,sha
from .standalone_program_v1 import python_policy,run_bounded


def observe(row,program,*,seconds=3):
    source=row['source_candidate']
    if (row['split']!='train' or source['role']!='train' or row['id']!=source['id']
            or type(seconds) not in {int,float} or not math.isfinite(seconds) or not 0<seconds<=30):
        raise ValueError('Only the exact source-train role and bounded actual public examples are eligible')
    text=source['original_literal_docstring'];tests=doctest.DocTestParser().get_examples(text)
    if (len(tests)!=source['whole_original_example_count'] or not 2<=len(tests)<=128
            or any(t.options.get(doctest.SKIP,False) for t in tests)):
        raise ValueError('Every original published example must remain, without filtering or skipped tests')
    base=dict(original_source_id=row['id'],source_role='train',function=source['function'],
        actual_generated_program_sha256=sha(program),original_public_docstring_sha256=sha(text),
        declared_prelude_sha256=sha(PRELUDE),original_public_example_count=len(tests),
        syntax_valid=False,all_original_examples_observed=False,all_original_examples_passed=False,
        original_public_example_answers_visible_in_model_prompt=True,
        expected_outputs_not_sent_as_separate_comparison_artifacts=True,
        generated_program_not_repaired=True,execution_reward_training_admitted=False,
        human_preferences=False,held_benchmark=False,legacy_runtime_code_filename_label_retained=True)
    if not program or len(program.encode())>50000 or '\0' in program:
        return dict(base,status='empty_or_unbounded_generated_program',passed_examples=0,entry_body_calls=0)
    try:ast.parse(program)
    except (SyntaxError,ValueError,RecursionError) as error:
        return dict(base,status='generated_syntax_error',diagnostic=str(error)[:500],passed_examples=0,entry_body_calls=0)
    base['syntax_valid']=True
    with tempfile.TemporaryDirectory(prefix='slm-generated-public-function-v1-') as tmp:
        work=Path(tmp);(work/'program.py').write_text(program);(work/'prelude.py').write_text(PRELUDE)
        (work/'harness.py').write_bytes(HARNESS.read_bytes())
        settings=dict(program=str(work/'program.py'),prelude=str(work/'prelude.py'),function=source['function'],examples=[t.source for t in tests])
        (work/'settings.json').write_text(json.dumps(settings))
        policy=python_policy(work);policy['command'][-1]=str(work/'harness.py');policy['command'].append(str(work/'settings.json'))
        policy['memory_bytes']=512*1024**2;policy['cpu_seconds']=max(1,math.ceil(seconds))
        observed=run_bounded(policy,b'',seconds=seconds,output_bytes=1024**2,python=True)
    execution={k:v for k,v in observed.items() if k not in {'stdout','stderr'}}
    execution.update(stdout_sha256=__import__('hashlib').sha256(observed['stdout']).hexdigest(),stderr=observed['stderr'][-2000:].decode(errors='replace'))
    base['execution']=execution
    if observed['status']!='completed':return dict(base,status='generated_runtime_or_resource_failure',passed_examples=0,entry_body_calls=0)
    try:
        raw=json.loads(observed['stdout']);results=raw['results']
        if (set(raw)!={'format','results'} or raw['format']!='own-documented-doctest-raw-output-v1' or len(results)!=len(tests)
                or any(set(r)!={'stdout','exception_only','entry_body_calls'} or not isinstance(r['stdout'],str)
                    or r['exception_only'] is not None and not isinstance(r['exception_only'],str)
                    or type(r['entry_body_calls']) is not int or r['entry_body_calls']<0 for r in results)):
            raise ValueError('Every original example raw stdout/exception/body-call observation is required')
    except (ValueError,TypeError,KeyError):return dict(base,status='incomplete_generated_raw_observations',passed_examples=0,entry_body_calls=0)
    checks=[compare(t,r) for t,r in zip(tests,results)];calls=sum(r['entry_body_calls'] for r in results);passed=all(checks) and calls>0
    base.update(all_original_examples_observed=True,all_original_examples_passed=passed)
    return dict(base,status='all_original_examples_passed' if passed else 'generated_public_example_failure',
        raw_results=results,example_passes=checks,passed_examples=sum(checks),entry_body_calls=calls)
