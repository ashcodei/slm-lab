"""Fix all15source-pack validation inputs and verify every public reference.

This is read-only source/runtime preparation, not neural model evaluation or
certification that these public programs were never seen in any older corpus.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip,json,os
from pathlib import Path

from tokenizers import Tokenizer
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack,FILES
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.documented_doctest_v1 import HARNESS
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_function_validation_observation_v1 import observe
from slm_lab.standalone_program_v1 import PYTHON_LAUNCHER

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'data/direct-algorithm-function-contracts-admitted-v1'
OUT=ROOT/'reports/source-function-validation-runtime-CPU-v1'

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only validation preparation required')
    tests=ROOT/'reports/public-function-validation-observation-v1-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('Actual preserved-role public validation runtime controls required')
    pack=CompleteCodePack(SOURCE,'validation','audit');rows=list(map(json.loads,gzip.open(SOURCE/'original-rows.jsonl.gz','rt')))
    if len(pack)!=15:raise ValueError('Every15original source-pack validation function required')
    tokenizer=Tokenizer.from_file(str(SOURCE/'tokenizer.json'))
    files={str(p):file_hash(p) for p in [*[SOURCE/name for name in [*FILES,'manifest.json','admission.json']],tests,HARNESS,PYTHON_LAUNCHER,
        ROOT/'scripts/native_sandbox.py',*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_function_validation_observation_v1.py'])]}
    guard=FileBindings(files);cases=[];train_groups={row['group_id'] for row in rows if row['split']=='train'}
    for i,index in enumerate(pack.positions):
        source=rows[int(index)];ids,labels=pack.sequence(i);start=next(j for j,v in enumerate(labels) if v!=-100)
        prefix=ids[:start];targets=ids[start:]
        if (source['split']!='validation' or source['source_candidate']['role']!='valid' or source['group_id'] in train_groups
                or tokenizer.decode(targets[:-1],skip_special_tokens=False)!=source['response']
                or len(prefix)+1024>4096 or len(targets)>1024):
            raise ValueError('Every original validation role, whole positive code and fixed1024action budget must fit without truncation')
        cases.append(dict(position=i,original_global_native_index=int(index),original_source_id=source['id'],
            original_source_group=source['group_id'],source_role='validation',original_prompt_ids=prefix,
            whole_original_response_EOS_ids=targets,native_generation_budget=1024,
            original_complete_context_ids_sha256=source['complete_input_ids_sha256'],original_source_row=source,
            optimization_forbidden=True,all_history_independence_not_certified=True))
    OUT.mkdir();atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,all15source_pack_validation_functions_required=True,native_generation_budget_each=1024,
        original_public_example_answers_visible_in_model_prompt=True,no_neural_model_or_optimizer=True,neural_updates=0,goal_achieved=False))
    def check(case):
        row=case['original_source_row'];actual=[observe(row,row['response']) for _ in range(2)]
        if any(not r['all_original_examples_passed'] for r in actual):raise ValueError('Every original validation public reference must pass every case twice')
        return dict(original_source_id=row['id'],actual_two_fresh_validation_reference_runs=actual)
    with ThreadPoolExecutor(max_workers=2) as pool:proofs=list(pool.map(check,cases))
    atomic_json(OUT/'all15-fixed-native-validation-cases.json',cases);atomic_json(OUT/'all15-fresh-reference-executions.json',proofs)
    guard.check(full=True)
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,original_source_pack_validation_functions=15,
        original_public_examples=sum(r['original_source_row']['original_published_example_count'] for r in cases),
        actual_fresh_reference_runs=30,fixed_native_action_budget_each=1024,
        all_original_prompt_and_positive_EOS_bytes_reconstructed=True,whole_source_groups_disjoint_from826training_functions=True,
        original_source_roles_not_relabelled_to_train=True,source_pack_validation_not_all_history_independent_benchmark=True,
        source_files_sha256={**files,**{str(OUT/name):file_hash(OUT/name) for name in ['plan.json','all15-fixed-native-validation-cases.json','all15-fresh-reference-executions.json']}},
        new_neural_forwards=0,new_neural_updates=0,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),source_pack_validation_functions=15,new_neural_forwards=0)),flush=True)

if __name__=='__main__':main()
