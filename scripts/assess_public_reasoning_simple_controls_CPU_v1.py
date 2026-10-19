"""Actual non-neural train-only controls on every new source-reserved case."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_token_pack_v1 import CompleteCodePack
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_reasoning_simple_controls_v1 import fit,predict
from slm_lab.reasoning_source_oracles_v1 import expected,strict_reward

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/public-reasoning-curriculum-v1'
PROOF=ROOT/'reports/public-reasoning-curriculum-v1-readback-admission/complete.json'
OUT=ROOT/'reports/public-reasoning-simple-controls-CPU-v1'
TESTS=ROOT/'reports/public-reasoning-simple-controls-v1-tests.log'


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(name)!='1' for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh low-priority single-thread CPU control assessment required')
    if '3 passed' not in TESTS.read_text():raise ValueError('Actual train-only fit, input-only prediction and no-solver controls required')
    proof=json.loads(PROOF.read_text());files=dict(proof['source_files_sha256'])
    for path in [PROOF,TESTS,DATA/'admission.json',*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_reasoning_simple_controls_v1.py'])]:
        files[str(path)]=file_hash(path)
    guard=FileBindings(files);train_pack=CompleteCodePack(DATA,'train',purpose='training');reserved_pack=CompleteCodePack(DATA,'validation',purpose='audit')
    with gzip.open(DATA/'original-rows.jsonl.gz','rt') as stream:all_rows=list(map(json.loads,stream))
    train=[all_rows[int(position)] for position in train_pack.positions]
    reserved=[all_rows[int(position)] for position in reserved_pack.positions]
    if len(train)!=2568 or len(reserved)!=289 or {r['group_id'] for r in train}&{r['group_id'] for r in reserved}:
        raise ValueError('Every original train/reserved example and disjoint declared input group is required')
    for row in all_rows:
        if expected(row['original_entry'])!=row['response'] or row['response']!=row['original_entry']['answer']:
            raise ValueError('Every source target must independently reconstruct before controls are scored')
    fitted=fit(train);tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'));OUT.mkdir()
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        actual_train_examples=2568,actual_reserved_examples=289,prediction_inputs_only=['task','original question'],
        predictors=['per-task training-majority answer','literal original matrix copy','selection between those using training scores only'],
        tie_rule='lexicographic smallest majority answer; choose majority on selection ties',
        controls_are_handwritten_programs_not_neural_model_outputs=True,new_neural_forwards=0,new_model_updates=0,
        no_original_training_queue_data_or_native_evaluator_modified=True,goal_achieved=False))
    atomic_json(OUT/'train-fitted-controls.json',fitted)
    results=[]
    with (OUT/'all289-actual-control-predictions.jsonl').open('x') as stream:
        for position,row in enumerate(reserved):
            values={};task=row['task'];question=row['original_entry']['question']
            for method in ['training_majority','copy_input','train_selected']:
                if method=='copy_input' and not fitted[task]['copy_input_available']:continue
                text=predict(fitted,task,question,method=method)
                ids=tokenizer.encode(text,add_special_tokens=False).ids+[2]
                horizon=256 if task in ['binary_matrix','rotate_matrix'] else 64
                if (len(ids)>horizon or any(not 4<=value<32768 for value in ids[:-1])
                        or tokenizer.decode(ids[:-1],skip_special_tokens=False)!=text):
                    raise ValueError('The complete programmatic control output must fit its original native budget without repair')
                correct=bool(strict_reward(text,row['original_entry']))
                values[method]=dict(prediction=text,complete_programmatic_response_EOS_ids=ids,correct=correct,
                    EOS_provided_by_program_not_generated_by_neural_model=True)
            result=dict(position=position,source_id=row['id'],source_group=row['group_id'],task=task,
                original_question_sha256=hashlib.sha256(question.encode()).hexdigest(),source_role='validation',
                prediction_never_receives_reserved_answer=True,actual_controls=values)
            results.append(result);stream.write(json.dumps(result)+'\n')
    # Rebuild the written case/label totals from actual observations. No neural
    # model or production eval result is present in this artifact.
    with (OUT/'all289-actual-control-predictions.jsonl').open() as stream:stored=list(map(json.loads,stream))
    if stored!=results or len(stored)!=289 or len({row['source_id'] for row in stored})!=289:
        raise ValueError('Every actual source-reserved control delivery must reconstruct')
    by_task={}
    for task in sorted(fitted):
        selected=[row for row in stored if row['task']==task]
        counts={method:dict(examples=sum(method in row['actual_controls'] for row in selected),
            correct=sum(row['actual_controls'].get(method,{}).get('correct',False) for row in selected))
            for method in ['training_majority','copy_input','train_selected']}
        by_task[task]=dict(train=fitted[task],reserved_examples=len(selected),actual_reserved_controls=counts)
    overall={method:dict(examples=sum(method in row['actual_controls'] for row in stored),
        correct=sum(row['actual_controls'].get(method,{}).get('correct',False) for row in stored))
        for method in ['training_majority','copy_input','train_selected']}
    guard.check(full=True)
    complete=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        fitted_controls_sha256=file_hash(OUT/'train-fitted-controls.json'),actual_predictions_sha256=file_hash(OUT/'all289-actual-control-predictions.jsonl'),
        all2568train_sources_fit_without_reserved_labels=True,all289original_reserved_cases_scored=True,
        actual_overall=overall,actual_per_task=by_task,training_and_reserved_roles_or_native_scoring_not_modified=True,
        direct_controls_supply_own_EOS_and_are_not_neural_generations=True,
        source_group_reserved_only_not_all_history_or_semantic_independence=True,
        new_neural_forwards=0,new_production_model_updates=0,no_model_quality_gain_claim=True,goal_achieved=False)
    atomic_json(OUT/'complete.json',complete)
    lines=['# Non-neural controls for source-reserved reasoning','',
        'Every predictor is fit on all2,568training examples only. The prediction API receives only a task name and original question. '
        'The reserved289answers never enter fitting or prediction. The three controls are per-task training-majority output, '
        'literal input copying for the two matrix-output tasks, and a fixed choice between these using their training accuracy only.', '',
        '| Task | Train answers: distinct | Reserved cases | Majority correct | Copy correct | Train-selected correct |',
        '|---|---:|---:|---:|---:|---:|']
    for task,row in by_task.items():
        c=row['actual_reserved_controls'];copy_score=str(c['copy_input']['correct']) if c['copy_input']['examples'] else 'n/a'
        lines.append(f'| {task} | {row["train"]["train_distinct_canonical_answers"]} | {row["reserved_examples"]} | {c["training_majority"]["correct"]} | {copy_score} | {c["train_selected"]["correct"]} |')
    lines += ['',f'Across all289cases, the training-majority program gets **{overall["training_majority"]["correct"]}/289** and the train-selected program gets **{overall["train_selected"]["correct"]}/289**. '
        f'Literal grid copying is applicable to **{overall["copy_input"]["examples"]}** cases and gets **{overall["copy_input"]["correct"]}** correct. '
        'Copying performs no rotation, distance calculation or other task solver. It preserves the actual input cells and ignores the requested transformation.', '',
        '**These are handwritten program controls, not neural outputs or model improvements.** The programs provide their own final EOS token by construction. '
        'They calibrate interpretation of the upcoming native baseline/warmup experiment; they do not replace model deliveries or modify scoring. '
        'A later model gain must be compared with these per-task priors/copyable cases, together with the original coding and broad-domain controls. '
        'The source-reserved set is not a certificate of independence from every historical model input.', '',
        'No original data, queue, training stage or native evaluator is changed. No model is loaded or run, and no new optimizer update occurs. '
        'The original full model goal and every final capability requirement remain incomplete.','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    print(json.dumps(dict(report=str(OUT),actual_overall=overall,new_neural_forwards=0,new_production_updates=0)),flush=True)


if __name__=='__main__':main()
