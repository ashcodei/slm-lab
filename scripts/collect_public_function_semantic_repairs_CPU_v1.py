"""Inspect every826admitted train function for verified logical repair examples.

Keeps all source functions in the original pack. This optional derived view
retains a full original positive target, a deterministic single-site AST bug,
and real feedback from every original public example. It is not admitted to
training until a separate complete readback. No neural model is constructed.
"""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CodePackWriter,CompleteCodePack,FILES
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.documented_doctest_v1 import HARNESS
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_function_rollout_observation_v1 import observe
from slm_lab.public_function_semantic_repair_v1 import FORMAT,candidates,compact,counterexample,view,sha
from slm_lab.standalone_program_v1 import PYTHON_LAUNCHER

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
DATA=ROOT/'data/public-function-semantic-repair-candidate-v1'
OUT=ROOT/'reports/public-function-semantic-repair-candidate-CPU-v1'
MAXIMUM_MUTATIONS=16


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or DATA.exists():raise ValueError('Fresh CPU-only complete source derivation required')
    tests=ROOT/'reports/public-function-semantic-repair-v1-final-tests.log'
    if '4 passed' not in tests.read_text():raise ValueError('Actual mutation/source/mask/real-execution controls required')
    pack=CompleteCodePack(ORIGINAL,'train','training');original=list(map(json.loads,gzip.open(ORIGINAL/'original-rows.jsonl.gz','rt')))
    if len(pack)!=826 or len(original)!=841:raise ValueError('Every826original train function and all15held functions must remain in their original pack')
    admission=json.loads((ORIGINAL/'admission.json').read_text());files=dict(admission['source_files_sha256'])
    for p in [ORIGINAL/name for name in [*FILES,'manifest.json','admission.json']]+[tests,HARNESS,PYTHON_LAUNCHER,ROOT/'scripts/native_sandbox.py',
            *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_function_semantic_repair_v1.py'])]:files[str(p)]=file_hash(p)
    evidence=ROOT/'reports/own-function-training-results-CPU-v1-readback/chosen_CE-complete.json'
    if not json.loads(evidence.read_text()).get('complete'):raise ValueError('The separate actual source-train learning diagnostic must be verified before this extension')
    files[str(evidence)]=file_hash(evidence);guard=FileBindings(files)
    tokenizer=Tokenizer.from_file(str(ORIGINAL/'tokenizer.json'));OUT.mkdir();DATA.mkdir()
    for name in ['tokenizer.json','model-config.json','DATASET-LICENSE.txt','publisher-README.md']:shutil.copyfile(ORIGINAL/name,DATA/name)
    source_rows=[(position,int(index),original[int(index)]) for position,index in enumerate(pack.positions)]
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]);started=time.monotonic()
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=me,source_files_sha256=files,
        original_source_train_functions=826,original_source_valid_functions_excluded=15,maximum_mutations_per_function=MAXIMUM_MUTATIONS,
        source_function_order='original admitted train index order',threads=8,public_runtime_seconds_per_program=3,
        no_original_function_or_benchmark_population_replaced=True,source_candidate_training_admitted=False,
        related_work=['https://arxiv.org/abs/2106.06600','https://arxiv.org/abs/2207.01780'],
        not_a_replication_or_global_novelty_claim=True,neural_forwards=0,neural_updates=0,goal_achieved=False))
    def work(item):
        position,index,row=item
        result=dict(source_train_position=position,original_global_index=index,source_function_id=row['id'],source_group=row['group_id'],
            original_complete_function_sha256=sha(row['response']),original_source_role=row['split'],actual_reference_runs=[],attempts=[])
        gold=[observe(row,row['response']) for _ in range(2)];result['actual_reference_runs']=[compact(r) for r in gold]
        if any(not r['all_original_examples_passed'] for r in gold):
            result['disposition']='current_reference_public_checks_failed';return result,None
        for mutation in candidates(row,maximum=MAXIMUM_MUTATIONS):
            first=observe(row,mutation['program']);trial=dict(mutation={k:v for k,v in mutation.items() if k!='program'},actual_mutant_runs=[compact(first)])
            feedback=counterexample(row,first)
            if feedback is None:
                trial['disposition']='not_a_bounded_complete_executable_public_counterexample';result['attempts'].append(trial);continue
            second=observe(row,mutation['program']);trial['actual_mutant_runs'].append(compact(second))
            if compact(first)!=compact(second):
                trial['disposition']='public_counterexample_not_stable_across_two_fresh_runs';result['attempts'].append(trial);continue
            candidate,budget=view(row,mutation,feedback,tokenizer);trial.update(budget);result['attempts'].append(trial)
            if candidate is None:continue
            result.update(disposition='complete_semantic_repair_candidate',selected_repair_id=candidate[0]['id'],selected_mutation_position=mutation['mutation_position'])
            candidate[0].update(original_native_train_position=position,original_global_native_index=index,
                original_source_pack_manifest_sha256=files[str(ORIGINAL/'manifest.json')],
                actual_reference_public_checks=result['actual_reference_runs'],actual_mutant_public_checks=trial['actual_mutant_runs'])
            return result,candidate
        result['disposition']='no_eligible_stable_complete_repair_in_declared_first16mutations';return result,None
    writer=CodePackWriter(DATA,maximum_bytes=256*1024**2,minimum_free_gib=26);dispositions=Counter();kinds=Counter();done=0;selected=0
    try:
        with (OUT/'every-original-source-disposition-and-attempt.jsonl').open('x') as journal,ThreadPoolExecutor(max_workers=8) as pool:
            for result,candidate in pool.map(work,source_rows):
                journal.write(json.dumps(result,ensure_ascii=False)+'\n');journal.flush();done+=1;dispositions[result['disposition']]+=1
                if candidate is not None:
                    row,ids,labels=candidate;writer.add(row,ids,labels);selected+=1;kinds[row['mutation']['kind']]+=1
                if done%25==0 or done==826:
                    atomic_json(OUT/'progress.json',dict(original_train_functions_inspected=done,planned=826,complete_repair_candidates=selected,
                        source_dispositions=dict(dispositions),actual_mutation_kinds=dict(kinds),seconds=time.monotonic()-started,
                        candidate_training_admitted=False,new_neural_updates=0,goal_achieved=False))
        guard.check(full=True)
        manifest=writer.finish(dict(format=FORMAT,original_source_pack=str(ORIGINAL),original_source_manifest_sha256=file_hash(ORIGINAL/'manifest.json'),
            all826source_train_functions_inspected=True,original15validation_functions_excluded=True,
            first_stable_full_context_semantic_counterexample_in_first16deterministic_body_mutations=True,
            complete_original_positive_function_and_true_EOS_unchanged=True,all_published_examples_run_without_favorable_case_selection=True,
            full_native_context_limit=4096,overbudget_candidates_classified_never_truncated=True,
            original_publisher_README_and_MIT_license_notices_retained=True,machine_derived_prompt_not_publisher_authored_instruction=True,
            no_neural_teacher_or_model_generated_targets=True,source_files_sha256=files))
        paths=[OUT/'plan.json',OUT/'every-original-source-disposition-and-attempt.jsonl',DATA/'manifest.json',*[DATA/name for name in FILES]]
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,original_source_train_functions_inspected=done,
            complete_repair_candidates=selected,source_dispositions=dict(dispositions),actual_mutation_kinds=dict(kinds),
            native_pack_counts=manifest['counts'],full_independent_source_mutation_execution_and_native_readback_still_required=True,
            original_full826function_pack_and_every_existing_training_and_benchmark_job_unchanged=True,
            source_files_sha256={str(p):file_hash(p) for p in paths},candidate_training_admitted=False,neural_forwards=0,neural_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),complete_repair_candidates=selected,training_admitted=False)),flush=True)
    except BaseException as error:
        writer.close();atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise

if __name__=='__main__':main()
