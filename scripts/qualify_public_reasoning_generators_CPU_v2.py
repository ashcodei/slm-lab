"""Replay a fixed public-source qualification population, without neural work.

Every generated entry is retained, including repeats. This is not a held model
benchmark or training admission. The original public answer and a separately
implemented algorithm must agree before any native target can be retained.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from tokenizers import Tokenizer

from slm_lab.reasoning_source_loader_v1 import REVISION, load, verify
from slm_lab.reasoning_source_oracles_v1 import CONFIGS, expected, prompt, strict_reward

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/reasoning-gym-pinned-source-v1'
TOKENIZER=ROOT/'data/direct-algorithm-function-contracts-admitted-v1/tokenizer.json'
OUT=ROOT/'reports/public-reasoning-generator-qualification-CPU-v2'
DATA=ROOT/'data/public-reasoning-generator-qualified-candidates-v2'
SEEDS=(981113,1981113)
SIZE=128
MAXIMUM=64*1024**2
RESERVE=40*1024**3


def digest(value):return hashlib.sha256(value).hexdigest()


def canonical(value):return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def native(entry,tokenizer):
    text=prompt(entry);answer=expected(entry)
    prefix=tokenizer.encode('User: '+text+'\nAssistant:',add_special_tokens=False).ids
    response=tokenizer.encode(answer,add_special_tokens=False).ids
    if (len(prefix)+len(response)+2>4096 or not response
            or any(not 4<=v<32768 for v in prefix+response)
            or tokenizer.decode(prefix,skip_special_tokens=False)!='User: '+text+'\nAssistant:'
            or tokenizer.decode(response,skip_special_tokens=False)!=answer):
        raise ValueError('Whole original question, canonical answer and true EOS must fit exactly')
    return [1]+prefix+response+[2],[-100]*(len(prefix)+1)+response+[2]


def input_identity(entry):
    """Question-input key, ignoring seed, published answers and solved metadata.

    A conservative question key is enough for this qualification-only artifact.
    It is not a certificate of equivalence-free or historically held-out data.
    """
    return digest(canonical([entry['metadata']['source_dataset'],entry['question']]).encode())


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists() or DATA.exists()
            or any(os.environ.get(name)!='1' for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('A fresh CUDA-hidden, single-thread, nice19 source qualification process is required')
    tests=ROOT/'reports/reasoning-source-oracles-v1-tests.log'
    if '16 passed' not in tests.read_text():raise ValueError('All actual independent-oracle tests must pass')
    files=verify(SOURCE)
    for path in [Path(__file__),TOKENIZER,tests,ROOT/'src/slm_lab/reasoning_source_loader_v1.py',
                 ROOT/'src/slm_lab/reasoning_source_oracles_v1.py',ROOT/'tests/test_reasoning_source_oracles_v1.py',
                 SOURCE/'commit.json',SOURCE/'git-tree.json']:
        files[str(path)]=digest(path.read_bytes())
    def checked():
        if any(digest(Path(path).read_bytes())!=sha for path,sha in files.items()):
            raise ValueError('A bound source changed during qualification')
        if shutil.disk_usage(ROOT).free<RESERVE+MAXIMUM:raise ValueError('Original persistent writer reserve must remain available')
    checked();tokenizer=Tokenizer.from_file(str(TOKENIZER));OUT.mkdir();DATA.mkdir()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        repository='https://github.com/open-thought/reasoning-gym',revision=REVISION,
        selected_configs=CONFIGS,seeds=list(SEEDS),size_per_seed_per_task=SIZE,planned_actual_entries=len(CONFIGS)*len(SEEDS)*SIZE,
        original_question_and_public_answer_preserved=True,independent_oracle_required_for_every_entry=True,
        protocol='canonical string answer, whole question plus explicit serialization instruction; no output execution',
        upstream_fractional_scoring_is_not_our_exact_correctness=True,
        no_GSM_symbolic_or_teacher_CoT=True,model_generated_answers=False,external_neural_teacher=False,
        human_preferences=False,on_policy_RL=False,new_neural_forwards=0,new_neural_updates=0,
        qualification_population_not_held_model_benchmark=True,training_admitted=False,
        existing_sources_queues_and_workers_unchanged=True,maximum_new_bytes=MAXIMUM,persistent_free_reserve=RESERVE,
        goal_achieved=False))
    counts=Counter();by_task={};questions=defaultdict(list);answers=Counter();byte_count=0;started=time.monotonic()
    def append(stream,value):
        nonlocal byte_count
        raw=(canonical(value)+'\n').encode();byte_count+=len(raw)
        if byte_count>MAXIMUM:raise ValueError('Complete qualification artifacts exceed the declared bound')
        stream.write(raw)
    try:
        with (DATA/'entries.jsonl').open('xb') as output,(OUT/'actual-controls.jsonl').open('xb') as controls,load(SOURCE) as factory:
            for task,options in CONFIGS.items():
                checked();task_counts=Counter()
                for seed in SEEDS:
                    dataset=factory.create_dataset(task,seed=seed,size=SIZE,**options)
                    for index in range(SIZE):
                        entry=dataset[index];answer=expected(entry)
                        if entry['answer']!=answer or strict_reward(answer,entry)!=1.:
                            append(controls,dict(task=task,seed=seed,index=index,entry=entry,independent_answer=answer,disposition='mismatch_fail_closed'))
                            raise ValueError('Public and independent answers disagree; no row can be silently omitted')
                        ids,labels=native(entry,tokenizer);key=input_identity(entry)
                        record=dict(id=task+':'+str(seed)+':'+str(index),task=task,seed=seed,index=index,
                            original_entry=entry,independent_answer=answer,prompt=prompt(entry),
                            exact_question_sha256=key,input_ids=ids,labels=labels,
                            response_EOS_targets=sum(value!=-100 for value in labels),complete_context_tokens=len(ids),
                            original_public_revision=REVISION,role='source_qualification_only',
                            training_admitted=False,held_model_benchmark=False)
                        append(output,record);questions[key].append(dict(task=task,seed=seed,index=index));answers[task+'\0'+answer]+=1
                        task_counts['entries']+=1;task_counts['response_EOS_targets']+=record['response_EOS_targets']
                        task_counts['complete_context_tokens']+=len(ids)
                        # Fixed safe strings only. In particular, binary_matrix's upstream
                        # eval-based scorer never receives arbitrary model-generated text.
                        extra=answer+' unrelated answer'
                        if strict_reward(extra,entry)!=0.:raise ValueError('Extra output must not receive full correctness')
                        if task not in ['binary_matrix','bitwise_arithmetic']:
                            upstream=float(dataset.score_answer(extra,entry))
                            append(controls,dict(task=task,seed=seed,index=index,kind='fixed_extra_text',answer=extra,
                                upstream_fractional_score=upstream,strict_score=0.,full_correctness=False))
                            task_counts['upstream_extra_text_fractional_controls']+=int(0.<upstream<1.)
                        if task in ['count_bits','gcd','lcm','chain_sum','products','rotten_oranges']:
                            wrong=('1'+answer) if not answer.startswith('-') else '-1'+answer[1:]
                            if int(wrong)==int(answer) or strict_reward(wrong,entry)!=0.:raise ValueError('Wrong integer control must actually be incorrect')
                            upstream=float(dataset.score_answer(wrong,entry))
                            append(controls,dict(task=task,seed=seed,index=index,kind='different_integer',answer=wrong,
                                upstream_fractional_score=upstream,strict_score=0.,full_correctness=False))
                            task_counts['wrong_integer_upstream_partial_credit']+=int(0.<upstream<1.)
                    output.flush();controls.flush()
                by_task[task]=dict(task_counts);counts.update(task_counts)
                atomic(OUT/'progress.json',dict(completed_tasks=list(by_task),actual_counts=dict(counts),goal_achieved=False))
        checked()
        duplicates={key:value for key,value in questions.items() if len(value)>1}
        cross_seed={key:value for key,value in duplicates.items() if len({v['seed'] for v in value})>1}
        atomic(OUT/'question-duplicates.json',dict(exact_question_duplicates=duplicates,cross_seed_question_duplicates=cross_seed,
            seeds_alone_do_not_establish_question_independence=True,semantic_equivalence_or_historical_benchmark_overlap_not_certified=True))
        for src,dest in [(TOKENIZER,'tokenizer.json'),(SOURCE/'upstream/LICENSE','LICENSE'),(SOURCE/'upstream/NOTICE.txt','NOTICE.txt')]:
            shutil.copyfile(src,DATA/dest)
        data_files={p.name:digest(p.read_bytes()) for p in DATA.iterdir() if p.is_file()}
        summary=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,actual_counts=dict(counts),
            actual_by_task=by_task,actual_unique_questions=len(questions),actual_question_duplicates_beyond_first=counts['entries']-len(questions),
            actual_cross_seed_duplicate_groups=len(cross_seed),actual_source_answer_mismatches=0,all_actual_source_draws_retained=True,
            actual_native_targets_have_true_EOS_and_exact_whole_question=True,actual_fixed_wrong_integer_controls=sum(v['entries'] for k,v in by_task.items() if k in ['count_bits','gcd','lcm','chain_sum','products','rotten_oranges']),
            training_admitted=False,held_benchmark_independence_certified=False,new_neural_updates=0,elapsed_seconds=time.monotonic()-started,
            data_files_sha256=data_files,goal_achieved=False)
        atomic(DATA/'manifest.json',dict(summary,format='pinned-public-procedural-source-qualified-candidates-v1',
            source_report=str(OUT),independent_readback_required=True))
        atomic(OUT/'complete.json',summary)
        print(json.dumps(dict(report=str(OUT),actual_counts=dict(counts),training_admitted=False,new_neural_updates=0)),flush=True)
    except BaseException as error:
        atomic(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,training_admitted=False,goal_achieved=False))
        raise


if __name__=='__main__':main()
