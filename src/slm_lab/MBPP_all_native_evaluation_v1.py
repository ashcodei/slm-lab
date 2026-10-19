"""Full original600native deliveries and original finite Python executions."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .MBPP_original_population_v2 import BUDGET,FullOriginalMBPP,prompt,role
from .code_quality import instruction_code
from .corpus import file_hash
from .diverse_images import atomic_json
from .finite_Python_values_v2 import decode
from .mbpp_original_function_cases_v2 import cases,execute

FORMAT='own-all600-original-MBPP-native-deliveries-v1'


def grade(code,row):
    if not code.strip() or '\0' in code or len(code.encode())>32768:
        return dict(status='empty_NUL_or_whole_program_runtime_bound',syntax_valid=False,functional_passed=False,
            original_cases=len(cases(row)),actual_observed_cases=0,all_cases_observed=False,passed_cases=0,
            program_sha256=hashlib.sha256(code.encode()).hexdigest(),original_program_not_clipped_or_repaired=True)
    return execute(code,row,seconds=4.)


@torch.inference_mode()
def evaluate_case(model,tokenizer,row,expected_input):
    if model.training or any(p.grad is not None for p in model.parameters()):raise ValueError('Exact evaluation-only native model required')
    native_prompt=prompt(row);ids=[1]+tokenizer.encode(native_prompt).ids
    digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
    if (ids!=expected_input['native_input_ids'] or digest!=expected_input['native_input_ids_sha256']
            or expected_input['generation_budget']!=BUDGET or len(ids)+BUDGET>model.config.max_text_length):
        raise ValueError('Every full original already-preflighted input and unchanged512budget required')
    device=next(model.parameters()).device;input_tensor=torch.tensor([ids],device=device)
    amp=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with amp:generated=model.generate(input_tensor,max_new_tokens=BUDGET,eos_token_id=2,use_cache=True)
    if generated.ndim!=2 or len(generated)!=1 or not torch.equal(generated[0,:len(ids)],input_tensor[0]):
        raise ValueError('Actual native generation must retain every original prompt token')
    actions=generated[0,len(ids):].tolist()
    if (not 1<=len(actions)<=BUDGET or 2 in actions[:-1]
            or (len(actions)<BUDGET and actions[-1]!=2)
            or any(type(v) is not int or not 0<=v<model.config.vocab_size for v in actions)):
        raise ValueError('Every actual own-vocabulary greedy first-EOS/full-budget action required')
    raw=tokenizer.decode(actions);code=instruction_code(raw);execution=grade(code,row)
    return dict(original_task_id=row['task_id'],publisher_role=role(row['task_id']),
        original_native_prompt=native_prompt,original_native_prompt_ids=ids,original_native_prompt_ids_sha256=digest,
        actual_generated_ids=actions,actual_generated_response=raw,actual_generated_code=code,
        genuine_generated_EOS=actions[-1]==2,generation_budget=BUDGET,
        original_reference_full_case_pass=expected_input['original_reference_full_case_pass'],execution=execution,
        inference_device_type=device.type,inference_precision='BF16_autocast' if device.type=='cuda' else 'FP32',
        actual_action_log_probabilities_not_retained=True,optimization_forbidden=True,model_updates=0)


def verify_record(tokenizer,row,expected_input,record):
    ids=[1]+tokenizer.encode(prompt(row)).ids;actions=record['actual_generated_ids']
    if (record['original_task_id']!=row['task_id'] or record['publisher_role']!=role(row['task_id'])
            or record['original_native_prompt']!=prompt(row) or record['original_native_prompt_ids']!=ids
            or ids!=expected_input['native_input_ids'] or record['original_native_prompt_ids_sha256']!=expected_input['native_input_ids_sha256']
            or record['generation_budget']!=BUDGET or not 1<=len(actions)<=BUDGET or 2 in actions[:-1]
            or (len(actions)<BUDGET and actions[-1]!=2)
            or any(type(v) is not int or not 0<=v<tokenizer.get_vocab_size() for v in actions)
            or record['actual_generated_response']!=tokenizer.decode(actions)
            or record['actual_generated_code']!=instruction_code(record['actual_generated_response'])
            or record['genuine_generated_EOS']!=(actions[-1]==2)
            or record['original_reference_full_case_pass']!=expected_input['original_reference_full_case_pass']
            or record['optimization_forbidden'] is not True or record['model_updates']!=0):
        raise ValueError('Every original input, actual native action, unmodified extracted code and source qualification must reconstruct')
    execution=record['execution'];code=record['actual_generated_code'];original=cases(row)
    if (execution['program_sha256']!=hashlib.sha256(code.encode()).hexdigest() or execution['original_cases']!=len(original)):
        raise ValueError('Actual source code and every original case denominator required')
    observations=execution.get('raw_observations')
    if observations is not None and observations.get('module_failure') is None:
        if len(observations['results'])!=len(original):raise ValueError('Every actual returned original LHS required')
        flags=[r['status']=='value' and decode(r['value'])==c['expected'] for r,c in zip(observations['results'],original)]
        verdicts=execution['original_case_verdicts']
        if (len(verdicts)!=len(original) or execution['actual_observed_cases']!=len(original)
                or execution['passed_cases']!=sum(flags) or execution['functional_passed']!=all(flags)
                or execution['all_cases_observed'] is not True):raise ValueError('Every original exact-value verdict must rederive')
        for c,v,flag in zip(original,verdicts,flags):
            if (v['kind']!=c['kind'] or v['original_position']!=c['original_position']
                    or v['original_test_sha256']!=c['original_test_sha256'] or v['passed']!=flag):
                raise ValueError('Every original case identity and reconstructed value comparison required')
    elif execution['functional_passed'] or execution['all_cases_observed'] or execution['passed_cases']:
        raise ValueError('A failed/incomplete source execution cannot become a full original-case pass')


def summary(rows):
    def group(values):
        return dict(problems=len(values),syntax_valid=sum(bool(r['execution']['syntax_valid']) for r in values),
            functionally_correct=sum(bool(r['execution']['functional_passed']) for r in values),
            original_cases=sum(r['execution']['original_cases'] for r in values),
            actual_observed_cases=sum(r['execution']['actual_observed_cases'] for r in values),
            passed_original_cases=sum(r['execution']['passed_cases'] for r in values))
    groups={'all_reserved':group(rows)}
    for name in ('publisher_test','publisher_validation','publisher_prompting'):
        values=[r for r in rows if r['publisher_role']==name];groups[name]=group(values)
        groups[name+':original_reference_qualified_secondary']=group([r for r in values if r['original_reference_full_case_pass']])
    return groups


@torch.inference_mode()
def evaluate_shard(model,population,path,*,rank,world,checkpoint_sha256,checked=lambda:None,record=lambda row:None):
    if type(population) is not FullOriginalMBPP or not 0<=rank<world<=600:raise ValueError('Whole original evaluation-only population and exact nonempty shard required')
    selected=list(range(rank,600,world));results=[];path=Path(path)
    if path.exists():raise ValueError('Preserve all original full/partial generation evidence')
    model.eval();signature=[(id(p),p._version,p.data_ptr()) for p in model.parameters()]
    for position in selected:
        checked();row=evaluate_case(model,population.tokenizer,population.rows[position],population.inputs[position])
        verify_record(population.tokenizer,population.rows[position],population.inputs[position],row)
        record(row);results.append(row)
    if signature!=[(id(p),p._version,p.data_ptr()) for p in model.parameters()] or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Every actual model value must remain unchanged during held generation/execution')
    report=dict(format=FORMAT,rank=rank,world=world,checkpoint_sha256=checkpoint_sha256,
        original_population_sha256=population.identity_sha256,own_tokenizer_sha256=population.tokenizer_sha256,
        original_selected_positions=selected,generation_budget=BUDGET,
        results=results,summary=summary(results),all_original_problems_retained=True,
        original_tests_not_asymptotic_efficiency_proof=True,not_official_few_shot_MBPP_protocol=True,model_updates=0,goal_achieved=False)
    atomic_json(path,report);return report


def reconcile(population,paths,output):
    rows={};seen=set();shared=None;bindings={}
    for path in map(Path,paths):
        shard=json.loads(path.read_text());rank=shard['rank'];world=shard['world']
        if rank in seen or world!=len(paths) or shard['original_selected_positions']!=list(range(rank,600,world)):
            raise ValueError('Every original shard and problem position required exactly once')
        profile={k:shard[k] for k in ['checkpoint_sha256','original_population_sha256','own_tokenizer_sha256','generation_budget']}
        if shared is None:shared=profile
        if profile!=shared or profile['original_population_sha256']!=population.identity_sha256 or profile['generation_budget']!=BUDGET:
            raise ValueError('Every endpoint must use its exact checkpoint and full original native protocol')
        if [r['original_task_id'] for r in shard['results']]!=[p+1 for p in shard['original_selected_positions']]:
            raise ValueError('Every original model delivery required without qualification filtering')
        for row in shard['results']:
            position=row['original_task_id']-1
            verify_record(population.tokenizer,population.rows[position],population.inputs[position],row);rows[position]=row
        if shard['summary']!=summary(shard['results']):raise ValueError('Every shard score must reconstruct from original cases')
        seen.add(rank);bindings[str(path)]=file_hash(path)
    if seen!=set(range(len(paths))) or set(rows)!=set(range(600)):raise ValueError('Every original600problem must retain an outcome')
    results=[rows[p] for p in range(600)]
    report=dict(format=FORMAT,**shared,results=results,summary=summary(results),
        original_publisher_test_denominator=500,all_reserved_original_problems=600,
        reference_qualifications_not_used_to_drop_primary_problems=True,
        every_actual_raw_value_and_original_case_verdict_rederived=True,
        no_new_neural_generation_or_program_execution_by_reconciliation=True,
        not_official_few_shot_or_general_efficiency_proof=True,
        source_files_sha256=bindings,model_updates=0,candidate_promoted=False,goal_achieved=False)
    atomic_json(output,report);return report
