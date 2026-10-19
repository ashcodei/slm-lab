"""Explicit post-public source, ownership, lineage and full-study contracts."""
import hashlib
import json
from pathlib import Path

from .corpus import file_hash
from .exact_waiting_process_retirement_v1 import terminal

PROFILES=['plain_repeat','semantic_repair','weighted_semantic_repair']
REPORT='reports/full-function-repair-comparison-GPU-v1'
QUEUE='reports/full-function-repair-comparison-GPU-v1-queue.json'
ADMISSION='reports/function-repair-GPU-lineage-v1/admission.json'
OUTPUT='checkpoints/full-function-repair-comparison-GPU-v1'
PUBLIC='reports/public-pretraining-v1'
RUNTIME='reports/public-pretraining-recovery-v1'
MODEL='checkpoints/public-pretraining-v1/trained/model.pt'
NATIVE_COHORT='reports/source-function-validation-runtime-CPU-v1/all15-fixed-native-validation-cases.json'


def read(path):return json.loads(Path(path).read_text())


def public_ready(root,identity):
    root=Path(root)
    if not terminal(identity):return False
    runtime=root/RUNTIME;public=root/PUBLIC
    if (runtime/'failure.json').exists() or (public/'failure.json').exists():raise ValueError('The exact full public source predecessor failed')
    if any(not (folder/name).exists() for folder,name in [(runtime,'plan.json'),(runtime,'complete.json'),(runtime,'restored.json'),
            (runtime,'children.json'),(public,'plan.json'),(public,'training-complete.json'),(public,'complete.json'),(public,'restored.json')]):
        raise ValueError('Every actual complete public epoch, native assessment and GPU restoration receipt required')
    operational=read(runtime/'plan.json');done=read(runtime/'complete.json');restored=read(runtime/'restored.json')
    if operational['owner']!=identity or any(not terminal(child) for child in read(runtime/'children.json')):return False
    plan=read(public/'plan.json');trained=read(public/'training-complete.json');final=read(public/'complete.json')
    if (plan.get('planned_steps')!=4447 or plan.get('all_original_raw_train_windows_once') is not True
            or plan.get('all826verified_original_train_functions_once') is not True
            or plan.get('all_previous_selected_source_presentations_preserved') is not True
            or done.get('original_scientific_job_complete') is not True
            or done.get('original_neural_plan_sha256')!=file_hash(public/'plan.json')
            or restored.get('all_registered_point_actors_terminal') is not True
            or read(public/'restored.json').get('actual_recovery_owner')!=identity
            or trained.get('actual_updates')!=4447 or trained.get('all_new_admitted_raw_train_windows_once') is not True
            or trained.get('all826verified_train_functions_once') is not True
            or final.get('full_public_pretraining_source_epoch_complete') is not True
            or final.get('original_local_code_and_broad_assessments_complete') is not True or not (root/MODEL).is_file()):
        raise ValueError('The complete unchanged4447step public raw/function/retention epoch and native controls are required')
    return True


def handoff_admitted(root,identity):
    root=Path(root);path=root/ADMISSION
    if not path.exists():return False
    done=read(path)
    if (done.get('complete') is not True or done.get('GPU_study_owner')!=identity
            or done.get('old_ten_waiting_owners_retired') is not True
            or done.get('all_original_neural_functions_and_benchmark_populations_preserved') is not True
            or len(done.get('new_controllers',[]))!=10):raise ValueError('Exact tested complete ten-controller lineage admission required')
    for item in done['new_controllers']:
        if terminal(item['owner']):raise ValueError('A gated replacement controller ended before GPU study admission')
        if read(root/item['queue'])['owner']!=item['owner']:raise ValueError('Every replacement queue must name its actual current process birth')
    for path,digest in done.get('source_files_sha256',{}).items():
        if file_hash(path)!=digest:raise ValueError('A tested handoff source changed')
    return True


def study_released(root):
    """Optional research may fail, but cannot release followers with live GPUs."""
    root=Path(root);queue=root/QUEUE
    if not queue.exists():return False
    identity=read(queue)['owner']
    if not terminal(identity):return False
    folder=root/REPORT
    if not (folder/'plan.json').exists():
        # The trusted controller never creates or initializes a GPU child
        # before publishing its owned plan/registration list. Kernel flock
        # ownership also ends with this exact parent birth.
        if (folder/'children.json').exists() and read(folder/'children.json'):raise ValueError('Unplanned owned GPU workers require explicit recovery')
        return True
    if read(folder/'plan.json')['owner']!=identity:raise ValueError('Exact optional-study owner must match its actual queue')
    if not (folder/'restored.json').exists():raise ValueError('Optional GPU study ended without verified resource restoration')
    if any(not terminal(child) for child in read(folder/'children.json')):return False
    if read(folder/'restored.json').get('all_registered_point_actors_terminal') is not True:
        raise ValueError('All optional GPU actors must be terminal before unchanged human training proceeds')
    return True


def capacity_entries(entries):
    if len(entries)!=306 or any(not row or len(row)!=len(set(row)) for row in entries):
        raise ValueError('All306full original source batches with no repeated presentation slot inside a step required')
    tail=next((list(row) for row in entries if len(row)==1),None)
    if tail is None:raise ValueError('The real one-source global tail must enter actual capacity/recovery')
    return [list(entries[0]),list(entries[1]),tail]


def sources(plan,plan_sha256,profile):
    if profile not in ['baseline',*PROFILES]:raise ValueError('Exact fixed matched profile required')
    return dict(comparison_plan_sha256=plan_sha256,own_complete_public_model_sha256=plan['initializer']['checkpoint_sha256'],
        matched_profile_sha256=hashlib.sha256(profile.encode()).hexdigest())


def training_options():
    return dict(learning_rate=.0001,betas=[.9,.95],weight_decay=.01,gradient_clip=1.,maximum_microbatch=4,
        maximum_microbatch_tokens=4096,seed=979113,checkpoint_every=64,epochs=3,width=16,precision='bf16')
