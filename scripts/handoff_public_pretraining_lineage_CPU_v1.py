"""Replace only verified unstarted CPU queues after complete tested replacements.

The exact live original code trainer and its workers are never signalled. Every
old scientific artifact is retained. All controller pidfds are opened before
any retirement signal; an explicit completion marker releases the new route.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import observe,retire,terminal
from slm_lab.public_pretraining_lineage_tests_v1 import bind_tests
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/public-pretraining-lineage-handoff-v1'
STAGES=[
 ('phase-human-preference-v7','phase-human-preference-v8','train_phase_human_preference_v7.py','train_phase_human_preference_v8.py'),
 ('broad-anchored-code-v6','broad-anchored-code-v7','train_broad_anchored_code_v6.py','train_broad_anchored_code_v7.py'),
 ('code-first-final-standalone-CPU-v3','public-pretraining-final-standalone-CPU-v1','finalize_code_first_student_CPU_v3.py','finalize_public_pretraining_student_CPU_v1.py'),
 ('full-held-QA-pipeline-v4','full-held-QA-pipeline-v5','assess_full_held_QA_pipeline_v4.py','assess_full_held_QA_pipeline_v5.py'),
 ('full-AI2D-pipeline-v4','full-AI2D-pipeline-v5','assess_full_AI2D_pipeline_v4.py','assess_full_AI2D_pipeline_v5.py'),
 ('full-original-MBPP-pipeline-v4','full-original-MBPP-pipeline-v5','assess_full_original_MBPP_pipeline_v4.py','assess_full_original_MBPP_pipeline_v5.py'),
 ('AI2D-fixed-exposure-strata-v4','AI2D-fixed-exposure-strata-v5','assess_AI2D_fixed_exposure_strata_v4.py','assess_AI2D_fixed_exposure_strata_v5.py'),
 ('full-machine-preference-v1','full-machine-preference-v2','train_full_machine_preference_v1.py','train_full_machine_preference_v2.py'),
 ('machine-preference-final-standalone-CPU-v1','machine-preference-final-standalone-CPU-v2','finalize_machine_preference_student_CPU_v1.py','finalize_machine_preference_student_CPU_v2.py'),
 ('machine-standalone-full-assessment-v1','machine-standalone-full-assessment-v2','assess_machine_standalone_full_v1.py','assess_machine_standalone_full_v2.py')]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden complete lineage handoff required')
    core=ROOT/'reports/code-first-continuation-v3'
    original=json.loads((core/'plan.json').read_text());owner.REPORTS=core;owner.resources.REPORTS=core
    owner.owned(original)
    if (core/'complete.json').exists() or (core/'restored.json').exists() or (core/'failure.json').exists():
        raise ValueError('Only unstarted successors of the still-live intact original trainer may be replaced')
    workers=[r for r in json.loads((core/'children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(workers)!=4 or any(terminal(r) for r in workers):raise ValueError('All four unchanged original training ranks must remain live')
    queue_path=ROOT/'reports/public-pretraining-v1-queue.json';queue=json.loads(queue_path.read_text())
    next_owner=observe(queue['owner'],'scripts/train_public_pretraining_v1.py')
    files=dict(queue['source_files_sha256']);files[str(queue_path)]=file_hash(queue_path)
    def merge(mapping):
        for path,sha in mapping.items():
            if path in files and files[path]!=sha:raise ValueError('Conflicting immutable original/new source bindings: '+path)
            files[path]=sha
    old=[];new=[];queue_observations={}

    for old_name,new_name,old_driver,new_driver in STAGES:
        old_path=ROOT/'reports'/(old_name+'-queue.json');new_path=ROOT/'reports'/(new_name+'-queue.json')
        before=json.loads(old_path.read_text());after=json.loads(new_path.read_text())
        a=observe(before['owner'],old_driver);b=observe(after['owner'],new_driver)
        folder=ROOT/'reports'/old_name
        if any((folder/name).exists() for name in ['training-complete.json','complete.json','restored.json','failure.json','children.json']):
            raise ValueError('Only an unstarted childless old CPU controller may be retired: '+old_name)
        old.append(dict(stage=old_name,**a));new.append(dict(stage=new_name,**b))
        merge(before['source_files_sha256']);merge(after['source_files_sha256'])
        for p in [old_path,new_path]:queue_observations[str(p)]=dict(sha256=file_hash(p),value=json.loads(p.read_text()))
        for p in [ROOT/'scripts'/old_driver,ROOT/'scripts'/new_driver]:files[str(p)]=file_hash(p)
    bind_tests(files,ROOT)
    tests=ROOT/'reports/exact-waiting-process-retirement-v1-final-tests.log'
    if '1 passed' not in tests.read_text():raise ValueError('Actual pidfd retirement and unrelated-process preservation must pass')
    for p in [tests,Path(__file__),ROOT/'src/slm_lab/exact_waiting_process_retirement_v1.py',core/'plan.json',
        ROOT/'reports/public-pretraining-lineage-file-map-v1.json']:files[str(p)]=file_hash(p)
    guard=FileBindings(files);owner.owned(original)
    for row in old:observe(row,row['command'])
    for row in new:observe(row,row['command'])
    me=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    OUT.mkdir()
    for index,(name,captured) in enumerate(queue_observations.items()):
        if file_hash(name)!=captured['sha256']:raise ValueError('An observed actual queue changed content before handoff')
        path=OUT/('original-queue-'+str(index)+'.json');atomic_json(path,dict(path=name,**captured));files[str(path)]=file_hash(path)
    guard=FileBindings(files)
    plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=me,original_code_first_owner=original['owner'],
        original_four_training_workers=workers,new_public_pretraining_owner=queue['owner'],old_unstarted_controllers=old,
        actual_ten_new_replacement_controllers=new,source_files_sha256=files,
        every_original_training_example_and_benchmark_population_preserved=True,
        new_pre_alignment_public_endpoint_added=True,no_original_training_worker_signal_authorized=True,
        goal_achieved=False)
    atomic_json(OUT/'plan.json',plan)
    try:
        owner.owned(original)
        retire(old)
        if any(not terminal(row) for row in old) or any(terminal(row) for row in new):
            raise ValueError('Every retired unstarted birth must end and every exact replacement must remain live')
        owner.owned(original)
        if any(terminal(row) for row in workers):raise ValueError('Original training ranks must remain untouched and live')
        guard.check(full=True)
        record=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,old_unstarted_controllers_all_terminal=True,
            retired_unstarted_controllers=old,actual_ten_new_replacement_controllers=new,
            replacement_full_pipeline_tested_and_queued=True,new_public_pretraining_owner=queue['owner'],
            original_code_first_owner=original['owner'],original_four_training_workers=workers,
            original_training_parent_and_all_workers_not_signalled=True,all_original_queues_and_scientific_files_preserved=True,
            all_original_benchmark_populations_preserved_at_nine_predeclared_endpoints=True,
            full_human85786_code3721_machine1620_original_sources_retained=True,
            complete_full_public_admission_and_four_L40_capacity_still_required_before_new_updates=True,
            source_files_sha256={**files,str(OUT/'plan.json'):file_hash(OUT/'plan.json')},model_updates=0,goal_achieved=False)
        atomic_json(OUT/'complete.json',record)
        print(json.dumps(dict(complete=str(OUT/'complete.json'),retired_unstarted_controllers=len(old),
            new_replacement_controllers=len(new),original_GPU_workers_untouched=True,new_corpus_neural_updates=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise


if __name__=='__main__':main()
