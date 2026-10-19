"""Fresh-process full readback of public procedural-source qualification.

This does not admit training or certify historical benchmark disjointness.
The complete original population, including duplicates, must reconstruct.
"""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import traceback

from tokenizers import Tokenizer

from slm_lab.reasoning_source_loader_v1 import REVISION,load,verify
from slm_lab.reasoning_source_oracles_v1 import CONFIGS,expected,format_instruction,strict_reward

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/reasoning-gym-pinned-source-v1'
PRODUCER=ROOT/'reports/public-reasoning-generator-qualification-CPU-v2'
DATA=ROOT/'data/public-reasoning-generator-qualified-candidates-v2'
OUT=ROOT/'reports/public-reasoning-generator-qualification-v2-readback'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists()
            or any(os.environ.get(name)!='1' for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread full readback required')
    receipt=read(PRODUCER/'complete.json');plan=read(PRODUCER/'plan.json');manifest=read(DATA/'manifest.json')
    if (not receipt.get('complete') or receipt['actual_counts']['entries']!=3072 or receipt['training_admitted']
            or plan['seeds']!=[981113,1981113] or plan['size_per_seed_per_task']!=128 or plan['selected_configs']!=CONFIGS
            or (PRODUCER/'failure.json').exists()):raise ValueError('Complete exact original full-population qualification required')
    files=dict(receipt['source_files_sha256']);files.update(verify(SOURCE))
    for path in [Path(__file__),PRODUCER/'complete.json',PRODUCER/'plan.json',PRODUCER/'actual-controls.jsonl',
                 PRODUCER/'question-duplicates.json',DATA/'manifest.json']:
        files[str(path)]=sha(path)
    for name,digest in receipt['data_files_sha256'].items():
        if sha(DATA/name)!=digest:raise ValueError('Every retained data byte must match the producer')
        files[str(DATA/name)]=digest
    if (manifest['data_files_sha256']!=receipt['data_files_sha256']
            or manifest['actual_counts']!=receipt['actual_counts'] or manifest['training_admitted']):
        raise ValueError('Candidate manifest must preserve all counters and non-admission')
    def checked():
        if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Bound producer/input bytes changed during readback')
    checked();OUT.mkdir()
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        exact_original_entries=3072,all_original_seeds_tasks_indices_required=True,
        all_native_BPE_targets_independently_reencoded=True,source_reexecution=True,
        no_new_neural_work=True,training_admitted=False,goal_achieved=False))
    try:
        rows=list(map(json.loads,(DATA/'entries.jsonl').read_text().splitlines()))
        controls=list(map(json.loads,(PRODUCER/'actual-controls.jsonl').read_text().splitlines()))
        if len(rows)!=3072:raise ValueError('Full original population, without omissions or padding, required')
        tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'));by_task={};total=Counter();questions=defaultdict(list)
        position=0;control_position=0;rng=random.getstate()
        with load(SOURCE) as factory:
            for task,options in CONFIGS.items():
                checked();counts=Counter()
                for seed in [981113,1981113]:
                    dataset=factory.create_dataset(task,seed=seed,size=128,**options)
                    for index in range(128):
                        row=rows[position];entry=dataset[index];answer=expected(entry)
                        if (row['id']!=task+':'+str(seed)+':'+str(index) or row['original_entry']!=entry
                                or row['task']!=task or row['seed']!=seed or row['index']!=index
                                or row['original_public_revision']!=REVISION or row['training_admitted']
                                or row['role']!='source_qualification_only' or row['held_model_benchmark']
                                or row['independent_answer']!=answer or answer!=entry['answer']):
                            raise ValueError('All source draws, original entries, independent answers and roles must reproduce')
                        text=entry['question']+'\n'+format_instruction(task)
                        prefix=[1]+tokenizer.encode('User: '+text+'\nAssistant:',add_special_tokens=False).ids
                        target=tokenizer.encode(answer,add_special_tokens=False).ids+[2]
                        boundary=len(prefix);ids=row['input_ids'];labels=row['labels']
                        if (ids!=prefix+target or labels!=[-100]*boundary+target or row['prompt']!=text
                                or not 0<boundary<len(ids)<=4096 or ids[-1]!=2
                                or tokenizer.decode(ids[boundary:-1],skip_special_tokens=False)!=answer
                                or tokenizer.decode(ids[1:boundary],skip_special_tokens=False)!='User: '+text+'\nAssistant:'
                                or any(not 4<=value<32768 for value in ids[1:-1])
                                or row['response_EOS_targets']!=len(target) or row['complete_context_tokens']!=len(ids)):
                            raise ValueError('Native complete BPE source/target/EOS/masks must independently reconstruct')
                        key=hashlib.sha256(json.dumps([task,entry['question']],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                        if key!=row['exact_question_sha256']:raise ValueError('Question identity must ignore generated solution metadata')
                        questions[key].append(dict(task=task,seed=seed,index=index))
                        if strict_reward(answer,entry)!=1. or strict_reward(answer+' unrelated answer',entry)!=0.:
                            raise ValueError('Strict positive and format-negative controls must reproduce')
                        counts.update(entries=1,response_EOS_targets=len(target),complete_context_tokens=len(ids))
                        expected_controls=[]
                        if task not in ['binary_matrix','bitwise_arithmetic']:
                            extra=answer+' unrelated answer';score=float(dataset.score_answer(extra,entry))
                            expected_controls.append(dict(task=task,seed=seed,index=index,kind='fixed_extra_text',answer=extra,
                                upstream_fractional_score=score,strict_score=0.,full_correctness=False))
                            counts['upstream_extra_text_fractional_controls']+=int(0.<score<1.)
                        if task in ['count_bits','gcd','lcm','chain_sum','products','rotten_oranges']:
                            wrong='1'+answer if not answer.startswith('-') else '-1'+answer[1:]
                            if int(wrong)==int(answer) or strict_reward(wrong,entry)!=0.:raise ValueError('A fixed wrong integer must truly be wrong')
                            score=float(dataset.score_answer(wrong,entry))
                            expected_controls.append(dict(task=task,seed=seed,index=index,kind='different_integer',answer=wrong,
                                upstream_fractional_score=score,strict_score=0.,full_correctness=False))
                            counts['wrong_integer_upstream_partial_credit']+=int(0.<score<1.)
                        if controls[control_position:control_position+len(expected_controls)]!=expected_controls:
                            raise ValueError('Every originally recorded reward control must reproduce exactly')
                        control_position+=len(expected_controls);position+=1
                by_task[task]=dict(counts);total.update(counts)
        if position!=len(rows) or control_position!=len(controls) or random.getstate()!=rng:
            raise ValueError('All rows and controls must reconstruct without hidden global RNG changes')
        duplicate_groups={key:value for key,value in questions.items() if len(value)>1}
        cross_seed={key:value for key,value in duplicate_groups.items() if len({v['seed'] for v in value})>1}
        duplicates=read(PRODUCER/'question-duplicates.json')
        if (duplicates['exact_question_duplicates']!=duplicate_groups or duplicates['cross_seed_question_duplicates']!=cross_seed
                or dict(total)!=receipt['actual_counts'] or by_task!=receipt['actual_by_task']
                or receipt['actual_unique_questions']!=len(questions)
                or receipt['actual_question_duplicates_beyond_first']!=3072-len(questions)
                or receipt['actual_cross_seed_duplicate_groups']!=len(cross_seed)):
            raise ValueError('Every reported aggregate and duplicate disposition must reconstruct')
        checked()
        atomic(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            actual_entries_regenerated=position,actual_reward_controls_reexecuted=control_position,
            actual_native_complete_masks_rebuilt=position,actual_counts=dict(total),actual_by_task=by_task,
            actual_unique_questions=len(questions),actual_duplicate_draws=3072-len(questions),actual_cross_seed_duplicate_groups=len(cross_seed),
            native_generation_or_held_model_quality_assessed=False,source_training_readiness_only=True,
            training_admitted=False,model_has_consumed_rows=False,historical_benchmark_overlap_not_certified=True,
            new_neural_forwards=0,new_neural_updates=0,no_existing_sources_queues_or_workers_changed=True,goal_achieved=False))
        print(json.dumps(dict(report=str(OUT),actual_entries_regenerated=position,actual_controls_reexecuted=control_position,
            training_admitted=False,goal_achieved=False)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,training_admitted=False,goal_achieved=False))
        raise


if __name__=='__main__':main()
