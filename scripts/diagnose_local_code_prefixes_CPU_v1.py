"""Post-hoc syntax-only prefix diagnosis of all24 completed initial programs.

This does not change native scoring or stopping, generate new model actions,
train on held tasks, or claim prefix-selected results as pass@1. Every
transformation is an exact contiguous prefix, selected without case outcomes.
The original finite functional cases remain complete; stress timing is outside
this diagnosis. Python also runs in the existing ordinary stdlib OS confinement.
"""
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from tokenizers import Tokenizer

from slm_lab.code_tasks import TASKS,request
from slm_lab.code_quality import execute as native_execute
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.ordinary_python_function_cases_v1 import execute as ordinary_execute

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/public-pretraining-interim-CPU-v2/initial'
REPORT=SOURCE/'code-completion.json'
JOURNAL=SOURCE/'code-completion-native-actions.jsonl'
TOKENIZER=ROOT/'checkpoints/public-pretraining-v1/initial/tokenizer.json'
OUT=ROOT/'reports/initial-local-code-prefix-diagnosis-CPU-v1'
MAX_SYNTAX_CUTS=64


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('x') as stream:stream.write(json.dumps(value,indent=2,ensure_ascii=True)+'\n')
    temporary.replace(path)


def syntax_cuts(program,prompt,language):
    if not program.startswith(prompt):raise ValueError('Exact original whole decoder prompt must remain the prefix')
    markers='\n' if language=='Python' else '}'
    cuts=sorted({i+1 for i,c in enumerate(program) if c in markers and i+1>len(prompt)}|{len(program)})
    # The bounded population is declared, not chosen using correctness scores.
    selected=cuts if len(cuts)<=MAX_SYNTAX_CUTS else cuts[:MAX_SYNTAX_CUTS//2]+cuts[-MAX_SYNTAX_CUTS//2:]
    return cuts,selected


def js_syntax(program,node):
    checked=subprocess.run([node,'--max-old-space-size=64','--check','-'],input=program,text=True,
        capture_output=True,timeout=6,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})
    if checked.returncode not in {0,1}:raise ValueError('JavaScript syntax checker did not terminate normally')
    return checked.returncode==0


def variants(program,prompt,language,name,node):
    all_cuts,cuts=syntax_cuts(program,prompt,language);valid=[];trees={}
    for cut in cuts:
        candidate=program[:cut]
        if language=='Python':
            try:tree=ast.parse(candidate)
            except (SyntaxError,ValueError,RecursionError):continue
            if not any(isinstance(n,ast.FunctionDef) and n.name==name for n in tree.body):continue
            trees[cut]=tree;valid.append(cut)
        elif js_syntax(candidate,node):valid.append(cut)
    selected={'original_native_program':program}
    if valid:
        longest=max(valid);selected['longest_checked_syntax_prefix']=program[:longest]
        if language=='Python':
            target=next(n for n in trees[longest].body if isinstance(n,ast.FunctionDef) and n.name==name)
            ends=[i+1 for i,c in enumerate(program) if c=='\n']+[len(program)]
            end=ends[target.end_lineno-1]
            if end<=longest:selected['first_target_function_prefix']=program[:end]
        else:selected['first_target_function_prefix']=program[:min(valid)]
    if any(not program.startswith(value) for value in selected.values()):raise ValueError('No generated character may be inserted, replaced or reordered')
    return selected,dict(all_candidate_cut_offsets=all_cuts,checked_cut_offsets=cuts,
        syntax_valid_checked_offsets=valid,maximum_syntax_checks=MAX_SYNTAX_CUTS,
        selection_uses_syntax_only=True,not_exhaustive_of_all_stopping_policies=True)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(k)!='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh bounded low-priority CPU-only post-hoc diagnosis required')
    source=json.loads(REPORT.read_text())
    with JOURNAL.open() as stream:events=list(map(json.loads,stream))
    if (source['task_count']!=24 or len(events)!=24 or len(source['results'])!=24
            or source['prompt_mode']!='completion' or source['inference_precision']!='bf16'
            or source['functional_pass_at_1']!=0 or source['generation_budget']!=512):
        raise ValueError('Every24complete native initial completion programs and unchanged512action budget required')
    node=shutil.which('node')
    if not node:raise ValueError('The same original JavaScript syntax checker must be available')
    paths=[REPORT,JOURNAL,TOKENIZER,Path(node).resolve(),ROOT/'scripts/code_worker.py',
        ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json',
        *python_sources(ROOT,[Path(__file__)])]
    files={str(p):sha(p) for p in paths};OUT.mkdir()
    atomic(OUT/'plan.json',dict(source_files_sha256=files,source_report=str(REPORT),
        source_checkpoint_sha256=source['checkpoint_sha256'],endpoint='initial_before_current_public_pretraining',
        original_programs=24,languages=['Python','JavaScript'],maximum_syntax_checks_per_program=64,
        maximum_distinct_execution_variants_per_program=3,all_original_finite_functional_cases_kept=True,
        original_stress_suite_not_used_for_this_posthoc_diagnostic=True,ordinary_Python_module_protocol_also_checked=True,
        selections_use_only_source_text_syntax_not_test_outcomes=True,
        derived_programs_are_not_native_model_outputs_or_benchmark_improvement=True,
        all_source_and_derived_records_forbidden_for_optimization=True,
        neural_forwards=0,new_production_optimizer_updates=0,goal_achieved=False))
    try:
        # Actual selector controls exercise junk tails and braces inside strings.
        py='def f(x):\n    return x + 1\n\nint main() {\n'
        got,_=variants(py,'def f(x):\n','Python','f',node)
        if got['first_target_function_prefix']!='def f(x):\n    return x + 1\n':raise ValueError('Actual Python syntax-only prefix control failed')
        js='function f(x) {\n return "}";\n}\nint main() {\n'
        got,_=variants(js,'function f(x) {\n','JavaScript','f',node)
        if got['first_target_function_prefix']!='function f(x) {\n return "}";\n}':raise ValueError('Actual JS string-brace syntax control failed')
        tokenizer=Tokenizer.from_file(str(TOKENIZER));tasks={t[0]:t for t in TASKS};results=[];totals=Counter()
        references=[];example=None
        with (OUT/'all24-program-diagnoses.jsonl').open('x') as stream:
            for position,(row,event) in enumerate(zip(source['results'],events)):
                if event['original_input_token_ids'][0]!=1:raise ValueError('Original source BOS required')
                prompt=tokenizer.decode(event['original_input_token_ids'][1:],skip_special_tokens=False)
                if tokenizer.decode(event['actual_generated_token_ids'])!=row['raw_response']:
                    raise ValueError('Every original actual neural output must reconstruct')
                task=tasks[row['task']];language=row['language'];program=row['code']
                payload=request(task,language,program);payload['stress']=[]
                if len(payload['cases'])!=row['candidate']['total_cases']:raise ValueError('Every unchanged original functional case required')
                reference_program=task[4] if language=='Python' else task[5]
                ref=native_execute(dict(payload,code=reference_program,reference_mode=True))
                if not ref.get('functional_passed'):raise ValueError('Every original reference must still pass every original functional case')
                control=dict(position=position,language=language,task=row['task'],native=ref)
                if language=='Python':
                    ordinary_ref=ordinary_execute(reference_program,row['task'],payload['cases'])
                    if not ordinary_ref['functional_passed']:raise ValueError('Every original Python reference must pass ordinary stdlib execution')
                    control['ordinary_Python']=ordinary_ref
                references.append(control)
                chosen,selection=variants(program,prompt,language,row['task'],node)
                seen={};observed={}
                for label,value in chosen.items():
                    digest=hashlib.sha256(value.encode()).hexdigest()
                    if digest not in seen:
                        native=native_execute(dict(payload,code=value));ordinary=None
                        if language=='Python':ordinary=ordinary_execute(value,row['task'],payload['cases'])
                        seen[digest]=dict(exact_code=value,exact_code_sha256=digest,native_finite_cases=native,ordinary_Python=ordinary,
                            original_program_prefix_characters=len(value),no_character_inserted_or_replaced=True)
                        totals['actual_distinct_native_program_runs']+=1
                        totals['actual_distinct_ordinary_Python_runs']+=ordinary is not None
                    actual=seen[digest];observed[label]=actual
                    totals[label+':native_syntax']+=bool(actual['native_finite_cases'].get('syntax_valid'))
                    totals[label+':native_functional']+=bool(actual['native_finite_cases'].get('functional_passed'))
                    if actual['ordinary_Python'] is not None:
                        totals[label+':ordinary_Python_functional']+=actual['ordinary_Python']['functional_passed']
                totals['original_programs']+=1
                if any(v['native_finite_cases'].get('functional_passed') for v in observed.values()):totals['any_checked_variant_native_functional']+=1
                if language=='Python' and any(v['ordinary_Python']['functional_passed'] for v in observed.values()):totals['any_checked_variant_ordinary_Python_functional']+=1
                result=dict(position=position,language=language,task=row['task'],original_code_sha256=hashlib.sha256(program.encode()).hexdigest(),
                    original_native_event=event,syntax_only_selection=selection,variants=observed,
                    finite_case_count=len(payload['cases']),stress_efficiency_not_assessed=True,
                    posthoc_diagnostic_only_not_native_score=True,optimization_forbidden=True)
                results.append(result);stream.write(json.dumps(result,ensure_ascii=True)+'\n');stream.flush()
                if language=='Python' and row['task']=='sum_even':
                    cropped=chosen.get('first_target_function_prefix',program)
                    example=dict(original_code=program,prefix_code=cropped,explicit_public_spec_input=[2,4],expected_even_sum=6,
                        actual_prefix_execution=ordinary_execute(cropped,'sum_even',[dict(args=[[2,4]],expected=6)]),
                        from_public_spec_not_admitted_to_training=True)
                atomic(OUT/'progress.json',dict(original_programs_complete=position+1,actual_counts=dict(totals),neural_forwards=0,goal_achieved=False))
        if totals['original_programs']!=24 or any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every original program and exact source must remain complete and unchanged')
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            endpoint='initial_before_current_public_pretraining',source_checkpoint_sha256=source['checkpoint_sha256'],
            actual_counts=dict(totals),all24_original_reference_controls=references,actual_syntax_selector_controls_passed=2,
            concrete_public_spec_example=example,source_native_code_report_and_journals_unchanged=True,
            all_original_finite_cases_used=True,stress_efficiency_not_assessed=True,
            exact_contiguous_prefix_variants_not_native_model_results=True,
            not_all_prefixes_or_decoding_strategies_exhaustively_ruled_out=True,
            no_change_to_stopping_prompts_budgets_sources_or_scores=True,optimization_forbidden=True,
            neural_forwards=0,new_production_optimizer_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=True,actual_counts=dict(totals),neural_forwards=0,original_benchmark_changed=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),automatic_retry=False,goal_achieved=False));raise


if __name__=='__main__':main()
