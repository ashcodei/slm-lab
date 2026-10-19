"""Collect execution-backed repair candidates from original APPS train sources.

Never changes the live epoch or admits a training/preference corpus. Complete
runtime failures and unchanged outputs stay in the observation journal only.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

from tokenizers import Tokenizer

from slm_lab.apps_execution_repair_v1 import mutations,counterexample,repair_example
from slm_lab.apps_published_cases_v2 import contract,execute,implementation_sources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CodePackWriter
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'reports/APPS-original-complete-cases-v2'
SOURCE=ROOT/'data/public-apps-original-inventory-v1'
VIEW=ROOT/'data/public-apps-human-solution-view-v1'
OUTPUT=ROOT/'data/APPS-execution-repairs-v1'
REPORTS=ROOT/'reports/APPS-execution-repairs-v1'
TOKEN=ROOT/'checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json'
SEED=932021


def original_program(row):
    path=SOURCE/'original-train'/row['problem']/'solutions.json'
    program=json.loads(path.read_bytes())[row['original_solution_index']]
    if hashlib.sha256(program.encode()).hexdigest()!=row['original_program_sha256']:
        raise ValueError('Complete original human program differs from verified passing source')
    return program


def check(original,selection):
    program=original_program(original)
    folder=SOURCE/'original-train'/original['problem']
    cases=contract((folder/'input_output.json').read_bytes())
    if cases['original_tests_sha256']!=original['original_tests_sha256']:
        raise ValueError('Complete original source cases differ')
    changes=mutations(program,original['id'],count=4,seed=SEED)
    actual=[{k:c[k] for k in ['ordinal','patch','program_sha256']} for c in changes]
    if actual!=selection['mutations']:
        raise ValueError('Exact predeclared deterministic source edits required')
    results=[]
    for change in changes:
        observed=execute(change['program'],cases,seconds=20)
        if observed['original_program_sha256']!=change['program_sha256']:
            raise ValueError('Actual executed mutation bytes differ from original edit')
        witness=counterexample(original,observed,cases)
        results.append(dict(original_id=original['id'],problem=original['problem'],publisher_split='train',
            original_source_body_sha256=original['original_program_sha256'],
            original_tests_sha256=cases['original_tests_sha256'],
            mutation=change,execution=observed,witness=witness,
            source_counterexample_observed=witness is not None,human_preferences=False,
            model_generated=False,external_model_teacher_used=False))
    return original,program,results


def main():
    if OUTPUT.exists() or REPORTS.exists():
        raise ValueError('Preserve previous repair candidate/evidence; no implicit retry')
    source_audit=json.loads((PARENT/'independent-readback.json').read_text())
    source_summary=json.loads((PARENT/'summary.json').read_text())
    if (not source_audit['all_selected_original_IDs_program_hashes_and_test_hashes_reconciled']
            or not source_audit['every_retained_raw_case_result_independently_compared_to_original_expected']
            or source_summary['observations_sha256']!=file_hash(PARENT/'observations.jsonl')):
        raise ValueError('Actual independent full original execution evidence required')
    source_plan=json.loads((PARENT/'plan.json').read_text())
    files=dict(source_plan['source_files_sha256'])
    files.update(implementation_sources())
    for p in [Path(__file__).resolve(),ROOT/'src/slm_lab/apps_execution_repair_v1.py',
            ROOT/'tests/test_apps_execution_repair_v1.py',ROOT/'reports/APPS-execution-repair-v1-tests.log',
            PARENT/'plan.json',PARENT/'summary.json',PARENT/'observations.jsonl',PARENT/'independent-readback.json',TOKEN,
            ROOT/'configs/model.json',ROOT/'data/public-apps-publisher-source-v1/publisher-LICENSE',
            ROOT/'data/public-apps-publisher-source-v1/publisher-README.md',ROOT/'src/slm_lab/codenet_token_pack_v1.py']:
        files[str(p)]=file_hash(p)
    if '2 passed' not in (ROOT/'reports/APPS-execution-repair-v1-tests.log').read_text():
        raise ValueError('Actual edit/full-case/mask/held-role checks required')
    if file_hash(TOKEN)!='0072bb6de61da87bd9ba5f4e6739b594cc3e60c62c75a59d25d499629a70088f':
        raise ValueError('Original own scratch-trained tokenizer required')
    guard=FileBindings(files)
    if shutil.disk_usage(ROOT).free<28*1024**3:
        raise ValueError('Fund maximum2GiB candidate above26GiB persistent reserve')
    passing=[r for r in map(json.loads,(PARENT/'observations.jsonl').read_text().splitlines())
        if r['status']=='all_finite_published_cases_pass']
    passing.sort(key=lambda r:r['id'])
    if len(passing)!=source_audit['passing_original_programs']:
        raise ValueError('Every actual independently verified passing human source required')
    admitted={r['id']:r for r in map(json.loads,(VIEW/'train-records.jsonl').read_text().splitlines())}
    inventory=json.loads((SOURCE/'train-problems.json').read_text())
    selections=[]
    for row in passing:
        original=admitted[row['id']]
        if original['split']!='train' or original['source_problem_id']!=row['problem']:
            raise ValueError('Original admitted optimization role and source group must match')
        changes=mutations(original_program(row),row['id'],count=4,seed=SEED)
        selections.append(dict(id=row['id'],original_source_body_sha256=row['original_program_sha256'],
            mutations=[{k:c[k] for k in ['ordinal','patch','program_sha256']} for c in changes]))
    guard.check(full=True)
    OUTPUT.mkdir();REPORTS.mkdir()
    selection_path=OUTPUT/'selected-mutations.jsonl'
    selection_path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in selections))
    shutil.copy2(TOKEN,OUTPUT/'tokenizer.json')
    shutil.copy2(ROOT/'configs/model.json',OUTPUT/'model-config.json')
    shutil.copy2(ROOT/'data/public-apps-publisher-source-v1/publisher-LICENSE',OUTPUT/'DATASET-LICENSE.txt')
    shutil.copy2(ROOT/'data/public-apps-publisher-source-v1/publisher-README.md',OUTPUT/'publisher-README.md')
    plan=dict(created_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        source_passing_original_programs=len(passing),source_selected_mutations=sum(len(r['mutations']) for r in selections),
        selection_sha256=file_hash(selection_path),deterministic_mutation_seed=SEED,
        maximum_mutations_per_original=4,maximum_concurrent_CPU_workers=4,maximum_seconds_per_program=20,
        maximum_complete_context=4096,no_program_prompt_or_counterexample_clipping=True,
        source_original_roles_only_train=True,optimization_or_generation_on_held_sources=False,
        target='Unmodified original human-written program that passes every original published training case',
        admission_rule='Only compiled, complete-execution mutants with actual wrong returned values on original cases',
        runtime_failures_or_unobserved_cases_supply_no_repair_labels=True,
        feedback_origin='Machine execution of deterministic source mutations; not genuine human feedback',
        external_model_teacher_used=False,model_generated=False,human_preferences=False,
        current_live_training_changed=False,training_admitted=False,model_updates=0,goal_achieved=False)
    atomic_json(REPORTS/'plan.json',plan)
    tokenizer=Tokenizer.from_file(str(TOKEN))
    writer=CodePackWriter(OUTPUT,maximum_bytes=2*1024**3,minimum_free_gib=26)
    outcomes=Counter();counts=Counter();families=Counter();problems=set();maximum=0
    with ThreadPoolExecutor(max_workers=4) as pool,(OUTPUT/'observations.jsonl').open('x') as journal,\
            (OUTPUT/'reviewed-repairs.jsonl').open('x') as reviews:
        futures=[pool.submit(check,row,selected) for row,selected in zip(passing,selections)]
        for future in as_completed(futures):
            original,program,records=future.result()
            problem=original['problem'];folder=SOURCE/'original-train'/problem;entry=inventory[problem]
            question=(folder/'question.txt').read_bytes().decode()
            starter=(folder/'starter_code.py').read_bytes().decode() if 'starter_code.py' in entry['files'] else None
            source=dict(admitted[original['id']],publisher_split='train',
                original_source_body_sha256=original['original_program_sha256'],
                original_question_sha256=entry['files']['question.txt']['sha256'],
                original_tests_sha256=original['original_tests_sha256'])
            for record in records:
                journal.write(json.dumps(record,separators=(',',':'),allow_nan=False)+'\n')
                outcomes[record['execution']['status']]+=1
                counts['executed_mutations']+=1
                if record['witness'] is None:
                    continue
                counts['complete_finite_counterexamples']+=1
                row,ids,labels=repair_example(source,question,starter,program,record['mutation'],record['witness'],tokenizer)
                maximum=max(maximum,len(ids))
                if row['review_reasons']:
                    counts['reviewed_whole_repair_context']+=1
                    reviews.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                    continue
                writer.add(row,ids,labels)
                counts['complete_fit_repair_examples']+=1
                families[record['mutation']['patch']['family']]+=1
                problems.add(problem)
            journal.flush();reviews.flush()
            counts['original_programs_processed']+=1
            if counts['original_programs_processed']%50==0:
                guard.check()
                progress=dict(observed_at=datetime.now(timezone.utc).isoformat(),counts=dict(counts),
                    execution_outcomes=dict(outcomes),training_admitted=False,model_updates=0,goal_achieved=False)
                atomic_json(REPORTS/'progress.json',progress);print(json.dumps(progress),flush=True)
    guard.check(full=True)
    if (counts['original_programs_processed']!=len(passing)
            or counts['executed_mutations']!=plan['source_selected_mutations']):
        raise ValueError('Every original source and preselected mutation must complete or remain explicitly qualified')
    manifest=writer.finish(dict(source='Original public APPS publisher-train human programs with machine-executed single-token counterexamples',
        tokenizer_sha256=file_hash(TOKEN),source_plan_sha256=file_hash(REPORTS/'plan.json'),
        observations_sha256=file_hash(OUTPUT/'observations.jsonl'),reviewed_repairs_sha256=file_hash(OUTPUT/'reviewed-repairs.jsonl'),
        selected_mutations_sha256=file_hash(selection_path),human_preferences=False,
        original_target_programs_byte_unchanged=True,held_roles_used=False,
        finite_published_cases_not_correctness_or_efficiency_proof=True))
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),counts=dict(counts),execution_outcomes=dict(outcomes),
        complete_fit_repair_problems=len(problems),mutation_families=dict(families),
        complete_original_response_targets=manifest['counts']['train']['target_tokens'],
        maximum_observed_full_repair_context=maximum,stored_pack_bytes=manifest['stored_bytes'],
        candidate_manifest_sha256=file_hash(OUTPUT/'manifest.json'),source_plan_sha256=file_hash(REPORTS/'plan.json'),
        source_files_sha256=files,selected_mutations_sha256=file_hash(selection_path),
        observations_sha256=file_hash(OUTPUT/'observations.jsonl'),reviewed_repairs_sha256=file_hash(OUTPUT/'reviewed-repairs.jsonl'),
        actual_GPU_contexts=0,model_generated_examples=0,external_model_teacher_used=False,
        human_preferences=False,training_admitted=False,model_updates=0,current_live_training_changed=False,
        original_bytes_clipped=0,goal_achieved=False,
        remaining=['Independently reconcile every token edit, original-case result, witness, whole BPE span and target mask.',
            'Retain all source-role, named held-source and runtime qualifications before any separate training admission.',
            'Require complete own-model baseline/actual capacity/fixed objective and matched evaluations for any neural experiment.'])
    atomic_json(REPORTS/'candidate-complete.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='source_files_sha256'},indent=2),flush=True)


if __name__=='__main__':
    main()
