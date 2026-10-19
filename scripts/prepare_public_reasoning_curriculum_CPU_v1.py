"""Screen every fixed public draw and build source-grouped complete text packs."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

from slm_lab.codenet_token_pack_v1 import CodePackWriter
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.public_reasoning_protected_text_v1 import overlaps,population,source_paths
from slm_lab.reasoning_curriculum_roles_v1 import assignments

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/public-reasoning-generator-qualified-candidates-v3'
READBACK=ROOT/'reports/public-reasoning-generator-qualification-v3-readback/complete.json'
OUT=ROOT/'reports/public-reasoning-curriculum-CPU-v1'
DATA=ROOT/'data/public-reasoning-curriculum-v1'
RESERVE=40*1024**3
MAXIMUM=128*1024**2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.pending')
    with temporary.open('x') as stream:
        stream.write(json.dumps(value,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def jsonl(path):
    with Path(path).open() as stream:return list(map(json.loads,stream))


def main():
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists() or DATA.exists()
            or os.environ.get('PYTHONHASHSEED')!='0'
            or any(os.environ.get(name)!='1' for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh low-priority single-thread CPU source screening required')
    reader=read(READBACK)
    if (not reader.get('complete') or reader['actual_entries_regenerated']!=3072
            or reader['actual_reward_controls_reexecuted']!=4096 or reader['training_admitted']):
        raise ValueError('The full independent source/reward qualification must be complete')
    tests=[(ROOT/'reports/reasoning-curriculum-roles-v1-tests.log','7 passed'),
           (ROOT/'reports/public-reasoning-protected-text-v1-tests.log','3 passed')]
    if any(expected not in path.read_text() for path,expected in tests):raise ValueError('Actual source grouping and complete protected text tests must pass')
    files=dict(reader['source_files_sha256'])
    paths=[READBACK,*source_paths(ROOT),SOURCE/'entries.jsonl',SOURCE/'tokenizer.json',
        ROOT/'data/direct-algorithm-function-contracts-admitted-v1/model-config.json',
        ROOT/'data/reasoning-gym-pinned-source-v1/upstream/LICENSE',ROOT/'data/reasoning-gym-pinned-source-v1/upstream/NOTICE.txt',
        ROOT/'data/reasoning-gym-pinned-source-v1/upstream/README.md',*[path for path,_ in tests],
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_reasoning_curriculum_roles_v1.py',ROOT/'tests/test_public_reasoning_protected_text_v1.py'])]
    files.update({str(path):sha(path) for path in paths})
    def checked():
        if any(sha(path)!=digest for path,digest in files.items()):raise ValueError('Every fixed source/input must remain exact during screening')
        if shutil.disk_usage(ROOT).free<RESERVE+MAXIMUM:raise ValueError('Original persistent writer reserve must stay available')
    checked();OUT.mkdir();DATA.mkdir()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        original_fixed_draws=3072,complete_protected_input_stems=31349,python_hash_seed='0',python_version=sys.version,
        grouping='Declared input-only arithmetic/string/grid equivalences before deterministic90/10hash roles',
        any_protected_overlap_excludes_whole_input_group=True,first_original_representative_per_group=True,
        every_original_draw_disposition_retained=True,protected_gold_labels_never_used=True,
        fixed_scans=['both normalized literal directions','rare informative word-shingle copy check'],
        semantic_or_historical_ancestry_independence_not_certified=True,
        original_full_source_and_human_RL_benchmark_scopes_unchanged=True,maximum_new_artifact_bytes=MAXIMUM,
        candidate_training_admission_requires_fresh_full_readback=True,new_neural_updates=0,goal_achieved=False))
    writer=None
    try:
        rows=jsonl(SOURCE/'entries.jsonl');protected=population(ROOT)
        if len(rows)!=3072:raise ValueError('Every originally qualified source draw is required')
        atomic(OUT/'progress.json',dict(phase='screening all original draws against every protected input stem',new_neural_updates=0,goal_achieved=False))
        hits,settings=overlaps(rows,protected,checked=checked);roles=assignments(rows,hits)
        with (OUT/'all3072-original-dispositions.jsonl').open('x') as stream:
            for value in roles:stream.write(json.dumps(value,sort_keys=True)+'\n')
        atomic(OUT/'protected-input-registry.json',dict(configuration=settings,complete_original_input_stems=protected))
        copies=[(SOURCE/'tokenizer.json','tokenizer.json'),
            (ROOT/'data/direct-algorithm-function-contracts-admitted-v1/model-config.json','model-config.json'),
            (ROOT/'data/reasoning-gym-pinned-source-v1/upstream/LICENSE','DATASET-LICENSE.txt'),
            (ROOT/'data/reasoning-gym-pinned-source-v1/upstream/NOTICE.txt','NOTICE.txt'),
            (ROOT/'data/reasoning-gym-pinned-source-v1/upstream/README.md','publisher-README.md')]
        for path,name in copies:shutil.copyfile(path,DATA/name)
        writer=CodePackWriter(DATA,maximum_bytes=MAXIMUM,minimum_free_gib=40)
        per_task=Counter();all_counts=Counter()
        for row,assigned in zip(rows,roles):
            if not assigned['representative'] or assigned['split'] not in ['train','validation']:continue
            split=assigned['split'];task=row['task']
            category='arithmetic' if task in ['count_bits','gcd','lcm','chain_sum','bitwise_arithmetic','products'] else 'algorithmic'
            record=dict(id='public-reasoning:'+row['id'],group_id=assigned['group_id'],split=split,task=task,
                prompt=row['prompt'],response=row['independent_answer'],original_qualified_id=row['id'],
                original_entry=row['original_entry'],original_seed=row['seed'],original_index=row['index'],
                original_public_revision=row['original_public_revision'],source='Reasoning Gym public procedural generator',
                source_url='https://github.com/open-thought/reasoning-gym/blob/'+row['original_public_revision']+'/reasoning_gym/'+category+'/'+task+'.py',
                license='Apache-2.0',attribution_notice_sha256=sha(DATA/'NOTICE.txt'),
                teacher_generated_or_human_preference=False,whole_original_question_and_public_answer_preserved=True,
                format_instruction_added=True,historical_model_exposure_or_semantic_benchmark_independence_not_certified=True)
            writer.add(record,row['input_ids'],row['labels']);per_task[(split,task)]+=1
            all_counts[split+'_examples']+=1;all_counts[split+'_targets']+=row['response_EOS_targets']
        tasks={row['task'] for row in rows}
        if any(per_task[('train',task)]==0 or per_task[('validation',task)]==0 for task in tasks):
            raise ValueError('Every original task requires both nonempty future training and reserved original-input groups')
        manifest=writer.finish(dict(source_files_sha256=files,source_readback=str(READBACK),
            original_source_draws=3072,complete_protected_stems=31349,overlap_configuration=settings,
            all_original_dispositions_sha256=sha(OUT/'all3072-original-dispositions.jsonl'),
            additional_NOTICE_sha256=sha(DATA/'NOTICE.txt'),native_4096_context_and_true_EOS_no_truncation=True,
            validation_is_only_new_source_reserved_not_historical_independence_certificate=True,
            machine_public_generator_targets_not_human_feedback_or_on_policy_RL=True))
        writer=None;checked()
        artifact_files={str(path):sha(path) for path in DATA.iterdir() if path.is_file()}
        result=dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=files,
            original_draws=3072,complete_protected_input_stems=31349,protected_input_counts=settings['complete_protected_counts'],
            actual_input_groups=len({r['group_id'] for r in roles}),actual_original_dispositions=dict(Counter(r['disposition'] for r in roles)),
            actual_original_draws_with_text_overlap=sum(bool(v) for v in hits.values()),
            actual_whole_groups_excluded=len({r['group_id'] for r in roles if r['split']=='excluded_protected_text_overlap'}),
            actual_packed_counts=manifest['counts'],actual_per_task={split:{task:per_task[(split,task)] for task in sorted(tasks)} for split in ['train','validation']},
            actual_native_target_counts=dict(all_counts),artifact_files_sha256=artifact_files,
            actual_source_group_train_validation_intersection=0,training_admitted=False,
            semantic_or_historical_independence_not_certified=True,new_neural_forwards=0,new_neural_updates=0,
            existing_training_evaluation_queues_unchanged=True,goal_achieved=False)
        atomic(OUT/'complete.json',result)
        atomic(OUT/'progress.json',dict(phase='complete candidate; fresh full readback required',counts=manifest['counts'],new_neural_updates=0,goal_achieved=False))
        print(json.dumps({k:v for k,v in result.items() if k not in ['source_files_sha256','artifact_files_sha256']}),flush=True)
    except BaseException as error:
        if writer is not None:writer.close()
        atomic(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False,training_admitted=False,goal_achieved=False))
        raise


if __name__=='__main__':main()
