"""Regenerate every public draw and admit only exact source-grouped text data."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_token_pack_v1 import CompleteCodePack,ROLES
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_reasoning_protected_text_v1 import overlaps,population
from slm_lab.reasoning_curriculum_roles_v1 import assignments
from slm_lab.reasoning_source_loader_v1 import load
from slm_lab.reasoning_source_oracles_v1 import CONFIGS,expected,format_instruction

ROOT=Path(__file__).resolve().parents[1]
PRODUCER=ROOT/'reports/public-reasoning-curriculum-CPU-v1'
SOURCE=ROOT/'data/public-reasoning-generator-qualified-candidates-v3'
DATA=ROOT/'data/public-reasoning-curriculum-v1'
OUT=ROOT/'reports/public-reasoning-curriculum-v1-readback-admission'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def jsonl(path):
    with Path(path).open() as stream:return list(map(json.loads,stream))


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists() or (DATA/'admission.json').exists()
            or os.environ.get('PYTHONHASHSEED')!='0'
            or any(os.environ.get(name)!='1' for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh single-thread CUDA-hidden source curriculum admission required')
    done=read(PRODUCER/'complete.json');plan=read(PRODUCER/'plan.json')
    if (not done.get('complete') or done['original_draws']!=3072 or done['complete_protected_input_stems']!=31349
            or done['training_admitted'] or plan['python_hash_seed']!='0' or plan['python_version']!=sys.version
            or (PRODUCER/'failure.json').exists()):raise ValueError('The exact complete original candidate and source runtime are required')
    files=dict(done['source_files_sha256']);files.update(done['artifact_files_sha256'])
    paths=[PRODUCER/'complete.json',PRODUCER/'plan.json',PRODUCER/'all3072-original-dispositions.jsonl',
        PRODUCER/'protected-input-registry.json',
        *python_sources(ROOT,[Path(__file__)])]
    files.update({str(path):sha(path) for path in paths})
    def checked():
        if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('A bound producer/input/artifact changed during admission')
    checked();OUT.mkdir()
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
        original_draws=3072,complete_protected_input_stems=31349,
        reexecute_every_original_public_draw_and_independent_oracle=True,rebuild_all_actual_native_masks=True,
        replay_every_overlap_disposition_and_original_group_before_admission=True,
        no_new_neural_forwards_or_updates=True,goal_achieved=False))
    try:
        original=jsonl(SOURCE/'entries.jsonl');tokenizer=Tokenizer.from_file(str(DATA/'tokenizer.json'))
        if len(original)!=3072:raise ValueError('Every original qualified draw required')
        with load(ROOT/'data/reasoning-gym-pinned-source-v1') as factory:
            datasets={};native={}
            for row in original:
                task,seed,index=row['task'],row['seed'],row['index'];key=(task,seed)
                if key not in datasets:datasets[key]=factory.create_dataset(task,seed=seed,size=128,**CONFIGS[task])
                entry=datasets[key][index];answer=expected(entry)
                if json.loads(json.dumps(entry))!=row['original_entry'] or answer!=entry['answer'] or row['independent_answer']!=answer:
                    raise ValueError('Every public source entry and independent target must freshly reproduce')
                text=entry['question']+'\n'+format_instruction(task)
                prefix=[1]+tokenizer.encode('User: '+text+'\nAssistant:',add_special_tokens=False).ids
                target=tokenizer.encode(answer,add_special_tokens=False).ids+[2]
                ids=prefix+target;labels=[-100]*len(prefix)+target
                if (ids!=row['input_ids'] or labels!=row['labels'] or text!=row['prompt']
                        or len(ids)>4096 or tokenizer.decode(target[:-1],skip_special_tokens=False)!=answer):
                    raise ValueError('Each complete native prompt/answer/EOS/mask must independently reconstruct')
                native[row['id']]=(ids,labels)
        protected=population(ROOT);stored=read(PRODUCER/'protected-input-registry.json')
        if stored['complete_original_input_stems']!=protected:raise ValueError('Every original protected input stem must stay exact')
        hits,settings=overlaps(original,protected,checked=checked)
        roles=assignments(original,hits)
        if roles!=jsonl(PRODUCER/'all3072-original-dispositions.jsonl') or settings!=stored['configuration']:
            raise ValueError('Every original source overlap, equivalent input and declared role must reconstruct')
        selected=[(row,assignment) for row,assignment in zip(original,roles)
            if assignment['representative'] and assignment['split'] in ['train','validation']]
        with gzip.open(DATA/'original-rows.jsonl.gz','rt') as stream:packed=list(map(json.loads,stream))
        if len(packed)!=len(selected):raise ValueError('Every exact representative, without extra rows or omissions, required')
        packs={split:CompleteCodePack(DATA,split,purpose='audit') for split in ['train','validation','test']}
        groups={split:set() for split in ['train','validation','test']};local=Counter();counts={split:Counter() for split in groups}
        per_task={split:Counter() for split in ['train','validation']}
        for position,((row,assignment),record) in enumerate(zip(selected,packed)):
            split=assignment['split'];ids,labels=native[row['id']];pack=packs[split]
            if (record['id']!='public-reasoning:'+row['id'] or record['token_pack_index']!=position
                    or record['original_qualified_id']!=row['id'] or record['original_entry']!=row['original_entry']
                    or record['prompt']!=row['prompt'] or record['response']!=row['independent_answer']
                    or record['split']!=split or record['group_id']!=assignment['group_id'] or record['task']!=row['task']
                    or record['license']!='Apache-2.0' or record['original_public_revision']!=row['original_public_revision']
                    or record['attribution_notice_sha256']!=sha(DATA/'NOTICE.txt')
                    or record['original_seed']!=row['seed'] or record['original_index']!=row['index']
                    or int(pack.positions[local[split]])!=position or pack.sequence(local[split])!=(ids,labels)
                    or record['complete_input_ids_sha256']!=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
                    or record['target_tokens']!=sum(value!=-100 for value in labels)):
                raise ValueError('Every packed source identity, declared role, whole original target and native bytes must reconstruct')
            local[split]+=1;groups[split].add(record['group_id']);per_task[split][row['task']]+=1
            counts[split].update(examples=1,context_tokens=len(ids),target_tokens=sum(v!=-100 for v in labels))
        actual={split:{key:counts[split][key] for key in ['examples','context_tokens','target_tokens']} for split in groups}
        if (actual!=done['actual_packed_counts'] or actual!=packs['train'].manifest['counts']
                or {split:dict(values) for split,values in per_task.items()}!=done['actual_per_task']
                or any(local[split]!=len(pack) for split,pack in packs.items()) or groups['train']&groups['validation']
                or len(set.union(*groups.values()))!=len(selected)
                or Counter(r['disposition'] for r in roles)!=done['actual_original_dispositions']):
            raise ValueError('All source/target budgets and disjoint original input groups must reconcile')
        try:CompleteCodePack(DATA,'train',purpose='training')
        except ValueError as error:
            if 'not been admitted' not in str(error):raise
        else:raise ValueError('Unadmitted candidate must reject optimization access')
        for split in ['validation','test']:
            try:CompleteCodePack(DATA,split,purpose='training')
            except ValueError as error:
                if 'Held source roles' not in str(error):raise
            else:raise ValueError('Every held source role must reject optimization access')
        checked()
        attestation=dict(at=datetime.now(timezone.utc).isoformat(),dataset_manifest_sha256=sha(DATA/'manifest.json'),
            admitted_to_training=True,admitted_roles=['train'],held_roles_forbidden_for_optimization=['validation','test'],
            source_files_sha256=files,actual_counts=actual,complete_original_draws_regenerated=3072,
            complete_protected_input_stems_rechecked=31349,actual_packed_sequences_rebuilt=len(packed),
            every_native_whole_question_answer_and_true_EOS_mask_rebuilt=True,source_group_train_validation_intersection=0,
            strict_finite_public_generator_targets_not_human_feedback_or_on_policy_RL=True,
            held_role_means_only_reserved_from_this_new_curriculum=True,semantic_or_historical_ancestry_independence_not_certified=True,
            new_neural_forwards=0,new_neural_updates=0,model_has_consumed_rows=False,goal_achieved=False)
        atomic(DATA/'admission.json',attestation)
        admitted=CompleteCodePack(DATA,'train',purpose='training')
        if len(admitted)!=2568:raise ValueError('Exact actually admitted whole training population required')
        atomic(OUT/'complete.json',dict(attestation,complete=True,
            actual_unadmitted_training_rejection_observed=True,actual_both_held_training_rejections_observed=True,
            actual_admitted_training_access_verified=True,admission_file_sha256=sha(DATA/'admission.json'),
            existing_neural_jobs_sources_and_full_benchmark_populations_unchanged=True))
        print(json.dumps(dict(report=str(OUT),actual_counts=actual,training_admitted=True,new_neural_updates=0,goal_achieved=False)),flush=True)
    except BaseException as error:
        if (DATA/'admission.json').exists() and not (OUT/'complete.json').exists():
            (DATA/'admission.json').rename(OUT/'rejected-admission.json')
        atomic(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,training_admitted=False,goal_achieved=False))
        raise


if __name__=='__main__':main()
