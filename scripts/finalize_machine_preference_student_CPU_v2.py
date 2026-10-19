"""Fold the fixed complete machine-DPO candidate into one own dense student.

This CPU-only owner may overlap the predecessor's GPU assessments after both
full source epochs and all eight private worker states are committed. Choosing
this fixed export endpoint does not select a quality winner or promote it.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.public_pretraining_lineage_tests_v1 import bind_tests
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/full-machine-preference-v2'
SOURCE_QUEUE=ROOT/'reports/full-machine-preference-v2-queue.json'
CHECKPOINT=ROOT/'checkpoints/full-machine-preference-v2/machine_DPO-actor.pt'
OUT=ROOT/'reports/machine-preference-final-standalone-CPU-v2'
OUTPUT=ROOT/'checkpoints/machine-preference-final-standalone-v2'
QUEUE=ROOT/'reports/machine-preference-final-standalone-CPU-v2-queue.json'


def same(identity):
    path=Path('/proc')/str(identity['pid'])/'stat'
    if not path.exists():return False
    fields=path.read_text().split(') ',1)[1].split()
    return fields[19]==identity['ticks'] and fields[0] not in {'Z','X'}


def source_ready(identity):
    if (SOURCE/'failure.json').exists():raise ValueError('Exact machine comparison failed; no older or partial policy substitution')
    paths=[SOURCE/'plan.json',SOURCE/'training-complete.json',CHECKPOINT]
    for arm in ['chosen_CE','machine_DPO']:
        paths.extend([SOURCE/(arm+'-training-complete.json'),SOURCE/('train-'+arm+'-step102-distributed-state.json')])
    if any(not p.is_file() for p in paths):
        if not same(identity):raise ValueError('Exact machine owner ended before committing both complete source epochs')
        return False
    plan=json.loads(paths[0].read_text());trained=json.loads(paths[1].read_text())
    if (plan.get('owner')!=identity or plan.get('original_machine_pairs')!=1620 or plan.get('planned_steps')!=102
            or plan.get('all_original_machine_pairs_once_each_arm') is not True or plan.get('human_preferences') is not False
            or trained.get('all1620original_machine_pairs_once_each_arm') is not True
            or trained.get('actual_updates_each_arm')!=102 or trained.get('human_preferences') is not False):
        raise ValueError('The exact owner, complete1620pair source epochs and explicit machine-feedback identity are required')
    for arm in ['chosen_CE','machine_DPO']:
        done=json.loads((SOURCE/(arm+'-training-complete.json')).read_text())
        receipt=json.loads((SOURCE/('train-'+arm+'-step102-distributed-state.json')).read_text())
        workers=receipt.get('all_four_private_states',[])
        if (done.get('actual_updates')!=102 or done.get('original_machine_pairs')!=1620
                or done.get('all_original_machine_pairs_once') is not True or done.get('rounded_repeats')!=0
                or done.get('human_preferences') is not False or receipt.get('completed')!=102
                or receipt.get('complete_distributed_state_requires_every_private_file') is not True
                or len(workers)!=4 or {w.get('rank') for w in workers}!=set(range(4))):
            raise ValueError('Both complete matched machine epochs and all four private states per arm must reconcile')
        for worker in workers:
            path=Path(worker['path'])
            if worker.get('actual_completed')!=102 or not path.is_file() or file_hash(path)!=worker.get('sha256'):
                raise ValueError('Every exact committed final private worker file must exist unchanged')
    return True


def main():
    if '16 passed' not in (ROOT/'reports/public-pretraining-lineage-drivers-v1-final2-tests.log').read_text():
        raise ValueError('Actual complete revised public-pretraining lineage and every original benchmark endpoint must pass')

    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(p.exists() for p in [OUT,OUTPUT,QUEUE]):
        raise ValueError('A fresh GPU-hidden finalizer must preserve every earlier attempt')
    from slm_lab.local_source_closure_v2 import python_sources
    previous=json.loads(SOURCE_QUEUE.read_text());tests=ROOT/'reports/own-dense-finalizer-v1-tests.log'
    driver_tests=ROOT/'reports/machine-preference-finalizer-v1-driver-tests.log'
    if '1 passed' not in tests.read_text() or '4 passed' not in driver_tests.read_text():
        raise ValueError('Actual dense fold/cold-native and exact complete machine source controls must pass')
    files=dict(previous['source_files_sha256'])
    files.update({str(p):file_hash(p) for p in [SOURCE_QUEUE,tests,driver_tests,ROOT/'tests/test_own_dense_finalizer_v1.py',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_machine_preference_finalizer_v1_driver.py'])]})
    bind_tests(files,ROOT);guard=FileBindings(files);expires=time.time()+22*86400
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(QUEUE,dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,exact_machine_owner=previous['owner'],
        source_files_sha256=files,fixed_source_arm='machine_DPO',fixed_full_machine_updates=102,original_machine_pairs=1620,
        both_matched_source_epochs_required=True,all_eight_final_private_worker_files_required=True,
        no_GPU_ownership_or_initialization=True,source_training_may_finish_before_original_GPU_assessments=True,
        fixed_export_not_a_quality_winner_selection=True,expires_at_unix=expires,automatic_retry=False,
        model_updates=0,candidate_promoted=False,goal_achieved=False))
    while not source_ready(previous['owner']):
        if time.time()>expires:raise ValueError('CPU observation lifetime ended; original jobs left intact')
        guard.check();time.sleep(5)
    import torch
    from slm_lab.machine_preference_policy_v2 import load
    from slm_lab.focused_code_continuation_data_v3 import CODE,FocusedCodeContinuationData
    from slm_lab.own_dense_finalizer_v1 import finalize
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    plan_path=SOURCE/'plan.json';source_plan=json.loads(plan_path.read_text())
    inputs=dict(checkpoint_sha256=file_hash(CHECKPOINT),plan_sha256=file_hash(plan_path))
    actor,metadata=load(CHECKPOINT,plan_path,'machine_DPO',expected_checkpoint_sha256=inputs['checkpoint_sha256'],
        expected_plan_sha256=inputs['plan_sha256'],device='cpu')
    if actor.counts()['actual_unique']!=999706442 or metadata['machine_preference_updates']!=102 or torch.cuda.is_initialized():
        raise ValueError('Exact complete own999706442-value machine-DPO actor on CPU required')
    files.update({str(CHECKPOINT):inputs['checkpoint_sha256'],str(plan_path):inputs['plan_sha256']})
    files[str(SOURCE/'training-complete.json')]=file_hash(SOURCE/'training-complete.json')
    for arm in ['chosen_CE','machine_DPO']:
        receipt_path=SOURCE/('train-'+arm+'-step102-distributed-state.json')
        files[str(receipt_path)]=file_hash(receipt_path)
        files[str(SOURCE/(arm+'-training-complete.json'))]=file_hash(SOURCE/(arm+'-training-complete.json'))
        for row in json.loads(receipt_path.read_text())['all_four_private_states']:files[row['path']]=row['sha256']
    data=FocusedCodeContinuationData(ROOT,actor.config,purpose='source_preparation');files.update(data.source_files_sha256)
    position=max(range(data.populations[CODE]),key=lambda p:data.lengths[CODE][p])
    photo=max(range(data.populations['images:additional_photo_captions']),key=lambda p:data.lengths['images:additional_photo_captions'][p])
    word_position=next(i for i,r in enumerate(data.replay.words.rows) if r['byte_family']=='real_word')
    probes=[('complete4096original_code',data.batch(CODE,[position],'cpu')),
        ('whole_real_native448photo',data.batch('images:additional_photo_captions',[photo],'cpu'))]
    word=data.batch('words',[word_position],'cpu')
    if (probes[0][1]['input_ids'].shape[1]!=4096 or tuple(probes[1][1]['pixel_values'].shape)!=(1,3,448,448)
            or file_hash(data.code.tokenizer_path)!=source_plan['tokenizer_sha256']):
        raise ValueError('Whole4096original code, actual448photo and the exact own tokenizer are required')
    bind_tests(files,ROOT);guard=FileBindings(files);OUT.mkdir()
    sources=dict(own_complete_machine_DPO_actor_sha256=inputs['checkpoint_sha256'],own_complete_machine_DPO_plan_sha256=inputs['plan_sha256'])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        fixed_source_arm='machine_DPO',actual_complete_machine_updates=102,original_machine_pairs=1620,
        actual_code_view_position=position,actual_photo_position=photo,actual_real_word_position=word_position,
        maximum_FP32_fold_logit_error=.002,source_policy_values=999706442,expected_standalone_values=984399690,
        no_GPU_or_optimizer_or_reference_for_native_inference=True,
        fixed_export_not_a_quality_winner_selection=True,model_updates=0,goal_achieved=False))
    try:
        result=finalize(actor,OUTPUT,data.code.tokenizer_path,sources=sources,completed_updates=102,
            probes=probes,word_pixels=word['word_pixels'])
        if result['actual_counts']['total_unique']!=984399690 or torch.cuda.is_initialized():
            raise ValueError('Every complete own dense/vision/word/head value must fit the actual sub1B budget on CPU')
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,**result,
            fixed_machine_DPO_export_not_quality_winner=True,all_original_machine_source_updates_complete=True,
            source_files_sha256={str(OUT/'plan.json'):file_hash(OUT/'plan.json'),str(OUTPUT/'model.pt'):file_hash(OUTPUT/'model.pt'),
                str(OUTPUT/'tokenizer.json'):file_hash(OUTPUT/'tokenizer.json')}))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_unique_student_values=984399690,final_quality_still_unproved=True)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
