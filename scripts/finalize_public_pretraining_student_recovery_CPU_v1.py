"""Export the fixed completed code-first human/execution policy on CPU.

May overlap the predecessor's GPU assessments after its immutable full epoch
checkpoint exists. It never owns, pauses or initializes a GPU.
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
SOURCE=ROOT/'reports/broad-anchored-code-v7'
SOURCE_QUEUE=ROOT/'reports/broad-anchored-code-v7-recovery-v1-queue.json'
CHECKPOINT=ROOT/'checkpoints/broad-anchored-code-v7/anchor_execution-actor.pt'
OUT=ROOT/'reports/public-pretraining-final-standalone-CPU-v1'
OUTPUT=ROOT/'checkpoints/public-pretraining-final-standalone-v1'
QUEUE=ROOT/'reports/public-pretraining-final-standalone-CPU-v1-recovery-v1-queue.json'


def same(identity):
    path=Path('/proc')/str(identity['pid'])/'stat'
    if not path.exists():return False
    fields=path.read_text().split(') ',1)[1].split()
    return fields[19]==identity['ticks'] and fields[0] not in {'Z','X'}


def source_ready(identity):
    if (SOURCE/'failure.json').exists():raise ValueError('Exact original code source failed; no older actor substitution')
    paths=[SOURCE/'plan.json',SOURCE/'anchor_execution-training-complete.json',CHECKPOINT,
        SOURCE/'train-anchor_execution-resume-distributed-state.json']
    if any(not p.exists() for p in paths):
        if not same(identity):raise ValueError('Exact code owner ended before its complete immutable source export')
        return False
    plan=json.loads(paths[0].read_text());trained=json.loads(paths[1].read_text());private=json.loads(paths[3].read_text())
    if (plan.get('planned_steps')!=3721 or plan.get('original_coding_sources')!=3721
            or trained.get('actual_updates')!=3721 or trained.get('all_3721_original_coding_sources_once') is not True
            or private.get('completed')!=3721 or {r['rank'] for r in private.get('all_four_private_states',[])}!=set(range(4))):
        raise ValueError('Every original complete code-source update and all four committed private states required')
    return True


def main():
    if '16 passed' not in (ROOT/'reports/public-pretraining-lineage-drivers-v1-final2-tests.log').read_text():
        raise ValueError('Actual complete revised public-pretraining lineage and every original benchmark endpoint must pass')

    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or any(p.exists() for p in [OUT,OUTPUT,QUEUE]):raise ValueError('Fresh CPU-only finalizer queue required')
    from slm_lab.local_source_closure_v2 import python_sources
    previous=json.loads(SOURCE_QUEUE.read_text());tests=ROOT/'reports/own-dense-finalizer-v1-tests.log'
    driver_tests=ROOT/'reports/state-code-first-training-drivers-v1-final-tests.log'
    if '1 passed' not in tests.read_text() or '11 passed' not in driver_tests.read_text():raise ValueError('Actual trained-fold/native/RNG and source-owner controls required')
    files=dict(previous['source_files_sha256'])
    files.update({str(p):file_hash(p) for p in [SOURCE_QUEUE,tests,driver_tests,ROOT/'tests/test_own_dense_finalizer_v1.py',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_code_first_finalizer_driver_v3.py'])]})
    bind_tests(files,ROOT);guard=FileBindings(files);expires=time.time()+14*86400
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(QUEUE,dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,exact_code_owner=previous['owner'],source_files_sha256=files,
        fixed_source_arm='anchor_execution',fixed_full_code_updates=3721,no_GPU_ownership_or_initialization=True,
        source_training_may_finish_before_original_GPU_assessments=True,automatic_retry=False,model_updates=0,goal_achieved=False))
    while not source_ready(previous['owner']):
        if time.time()>expires:raise ValueError('CPU observation deadline; original source job left intact')
        guard.check();time.sleep(5)
    import torch
    from slm_lab.broad_anchored_policy_v3 import load
    from slm_lab.focused_code_continuation_data_v3 import CODE,FocusedCodeContinuationData
    from slm_lab.own_dense_finalizer_v1 import finalize
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    plan_path=SOURCE/'plan.json';inputs=dict(checkpoint_sha256=file_hash(CHECKPOINT),plan_sha256=file_hash(plan_path))
    actor,metadata=load(CHECKPOINT,plan_path,'anchor_execution',expected_checkpoint_sha256=inputs['checkpoint_sha256'],expected_plan_sha256=inputs['plan_sha256'])
    if actor.counts()['actual_unique']!=999706442 or metadata['native_code_updates']!=3721 or torch.cuda.is_initialized():
        raise ValueError('Exact own completed999706442-value human/execution actor on CPU required')
    files.update({str(CHECKPOINT):inputs['checkpoint_sha256'],str(plan_path):inputs['plan_sha256']})
    data=FocusedCodeContinuationData(ROOT,actor.config,purpose='source_preparation');files.update(data.source_files_sha256)
    position=max(range(data.populations[CODE]),key=lambda p:data.lengths[CODE][p])
    photo=max(range(data.populations['images:additional_photo_captions']),key=lambda p:data.lengths['images:additional_photo_captions'][p])
    word_position=next(i for i,r in enumerate(data.replay.words.rows) if r['byte_family']=='real_word')
    probes=[('complete4096original_code',data.batch(CODE,[position],'cpu')),
        ('whole_real_native448photo',data.batch('images:additional_photo_captions',[photo],'cpu'))]
    word=data.batch('words',[word_position],'cpu')
    if probes[0][1]['input_ids'].shape[1]!=4096 or tuple(probes[1][1]['pixel_values'].shape)!=(1,3,448,448):
        raise ValueError('Whole original4096code and actual448photo input required')
    bind_tests(files,ROOT);guard=FileBindings(files);OUT.mkdir()
    sources=dict(own_complete_code_actor_sha256=inputs['checkpoint_sha256'],own_complete_code_plan_sha256=inputs['plan_sha256'])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        fixed_source_arm='anchor_execution',actual_complete_code_updates=3721,
        actual_code_view_position=position,actual_photo_position=photo,actual_real_word_position=word_position,
        maximum_FP32_fold_logit_error=.002,source_policy_values=999706442,expected_standalone_values=984399690,
        no_GPU_or_optimizer_or_reference_for_native_inference=True,model_updates=0,goal_achieved=False))
    try:
        result=finalize(actor,OUTPUT,data.code.tokenizer_path,sources=sources,completed_updates=3721,
            probes=probes,word_pixels=word['word_pixels'])
        if result['actual_counts']['total_unique']!=984399690:raise ValueError('Every final dense/vision/head value must fit the actual declared budget')
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,**result,
            source_files_sha256={str(OUT/'plan.json'):file_hash(OUT/'plan.json'),str(OUTPUT/'model.pt'):file_hash(OUTPUT/'model.pt'),
                str(OUTPUT/'tokenizer.json'):file_hash(OUTPUT/'tokenizer.json')}))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_unique_student_values=984399690,final_quality_still_unproved=True)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
