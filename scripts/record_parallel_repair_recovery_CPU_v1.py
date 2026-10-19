"""Archive actual small four-worker replay/commit proofs; keep full scope open."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.parallel_function_repair_stage_v1 import manifest,value_digest
from slm_lab.parallel_repair_durable_state_v1 import read_latest
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/parallel-repair-recovery-CPU-v1'


def read(path):return json.loads(Path(path).read_text())


def main(stage_folder,IO_folder):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only recovery evidence required')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    stage_folder=Path(stage_folder).resolve();IO_folder=Path(IO_folder).resolve()
    tests={ROOT/'reports/parallel-function-repair-v1-final-tests.log':7,
        ROOT/'reports/parallel-function-repair-stage-v1-final-tests.log':4,ROOT/'reports/parallel-repair-durable-state-v1-tests.log':5}
    for path,count in tests.items():
        if str(count)+' passed' not in path.read_text():raise ValueError('Every actual source/parallel/recovery/durable-publication test must pass')
    stage_proofs=[];IO_proofs=[]
    for rank in range(4):
        rows=read(stage_folder/('rank-'+str(rank)+'.json'));io=read(IO_folder/('rank-'+str(rank)+'.json'))
        if (len(rows)!=6 or any(r['rank']!=rank or r['logical_updates']!=3 or r['physical_optimizer_calls']!=4
                or not r['exact_actual_next_draws_source_update_actor_values_Adam_and_private_RNG']
                or not r['peer_only_wrong_valid_rng_and_resealed_malformed_peer_rng_cursor_and_Adam_rejected_before_mutation']
                or not r['single_unhealthy_peer_rejected_before_different_collectives_or_any_mutation'] for r in rows)
                or io['rank']!=rank or not io['root_only_IO_failure_coordinated_to_all_four_replicas']
                or not io['actual_actor_Adam_private_RNG_recovered_from_completed_shared_durable_file']):
            raise ValueError('Actual four-replica replay, corruption, collective-order and root-I/O proofs required')
        stage_proofs.append(rows);IO_proofs.append(io)
    current=ROOT/'reports/public-pretraining-recovery-v1';plan=read(current/'plan.json')
    owner.REPORTS=current;owner.resources.REPORTS=current;owner.owned(plan)
    ranks=[r for r in read(current/'children.json') if r['subject'].startswith('train-rank-') and not terminal(r)]
    handoff=read(ROOT/'reports/instance-lineage-recovery-v2/complete.json')
    if len(ranks)!=4 or any(terminal(r['actual_owner']) for r in handoff['actual_ten_new_controllers']):raise ValueError('All original GPU actors and ten full downstream queues must remain live')
    CPU=ROOT/'reports/full-function-repair-comparison-CPU-v1';CPU_plan=read(CPU/'plan.json')
    if terminal(CPU_plan['owner']) or (CPU/'failure.json').exists():raise ValueError('The unchanged full-size CPU comparison must remain live')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous)['requirements']);targets=[master_path,ROOT/'README.md',ROOT/'RESEARCH.md']
    args=['rg','-l','-F','-g','*.json']
    for path in targets:args.extend(['-e','"'+str(path)+'"','-e','"'+str(path.relative_to(ROOT))+'"'])
    scan=subprocess.run([*args,'reports'],cwd=ROOT,text=True,capture_output=True)
    if scan.returncode not in [0,1]:raise ValueError(scan.stderr)
    for name in scan.stdout.splitlines():
        value=json.loads((ROOT/name).read_text(),parse_int=str)
        if isinstance(value,dict) and any(str(Path(p).resolve()) in map(str,targets) for p in value.get('source_files_sha256',{})):
            raise ValueError('Source-bound root summaries cannot be edited')
    OUT.mkdir();at=datetime.now(timezone.utc).isoformat();artifacts=[]
    for rank in range(4):
        for source,label in [(stage_folder,'stage'),(IO_folder,'IO')]:
            target=OUT/(label+'-actual-rank-'+str(rank)+'.json');target.write_bytes((source/('rank-'+str(rank)+'.json')).read_bytes());artifacts.append(target)
    for profile in ['plain_repeat','semantic_repair','weighted_semantic_repair']:
        for precision in ['fp32','bf16']:
            original=stage_folder/(profile+'-'+precision+'.pt');target=OUT/(profile+'-'+precision+'-final-small-shared.pt')
            shutil.copyfile(original,target);state=torch.load(target,map_location='cpu',weights_only=True,mmap=True)
            if (state['completed']!=3 or state['world_size']!=4 or state['profile']!=profile or state['precision']!=precision
                    or len({value_digest(w['rng']) for w in state['private_workers']})!=4):
                raise ValueError('Each actual final small shared state must retain every distinct private worker')
            observed=manifest(state);path=OUT/(profile+'-'+precision+'-reader-derived-final-manifest.json');atomic_json(path,observed)
            artifacts.extend([target,path]);del state
    state,receipt,pointer_sha=read_latest(IO_folder/'durable')
    if (state['completed']!=3 or any(p['final_checkpoint_sha256']!=receipt['checkpoint_sha256'] for p in IO_proofs)):
        raise ValueError('The actual durable root checkpoint must match all four independent worker observations')
    saved=OUT/'actual-durable-final-small-shared.pt';shutil.copyfile(receipt['checkpoint'],saved)
    durable_proof=OUT/'actual-external-durable-receipt.json';atomic_json(durable_proof,dict(receipt=receipt,observed_pointer_sha256=pointer_sha))
    artifacts.extend([saved,durable_proof]);del state
    prefix=[]
    for line in (current/'training-metrics.jsonl').read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):break
        prefix.append(line)
    rows=[json.loads(line) for line in prefix]
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('Actual complete main source prefix required')
    path=OUT/'captured-main-training-prefix.jsonl';path.write_bytes(b''.join(prefix));artifacts.append(path)
    snapshot=read(CPU/'progress.json');path=OUT/'captured-CPU-study-progress.json';atomic_json(path,snapshot);artifacts.append(path)
    pending=dict(GPU_controller_implemented=False,GPU_study_queue_published=False,lineage_handoff_executed=False,
        initializer=str(ROOT/'checkpoints/public-pretraining-v1/trained/model.pt'),loader='slm_lab.public_pretraining_native_v1.load',
        full4447source_epoch_original_native_assessments_and_actual_GPU_restoration_required=True,
        human_controller_uses_nonblocking_GPU_flock=True,explicit_new_human_predecessor_gate_required_before_new_GPU_job=True,
        original_ten_waiting_controller_births_and_source_maps_still_active_and_unmodified=True,
        fresh_queued_controller_versions_and_safe_ten_owner_handoff_still_to_implement=True,
        full_QA_waits_broad_code_and_its_CPU_finalizer_AI2D_waits_full_QA=True,
        all_original_neural_training_and_assessment_functions_source_populations_and_nine_endpoints_must_remain=True,
        new_optional_GPU_study_must_release_verified_resources_before_human_even_on_optional_study_failure=True,
        new_GPU_queue_must_not_claim_resources_before_verified_handoff_admission=True,
        actual_four_L40_capacity_long_context_photo_word_tail_recovery_and_cold_proofs_required_before_production=True,
        actual_CPU_study_not_stopped_or_migrated=True,new_model_promoted=False,goal_achieved=False)
    path=OUT/'pending-GPU-controller-and-handoff.json';atomic_json(path,pending);artifacts.append(path)
    sources=[Path(__file__),previous,current/'plan.json',CPU/'plan.json',ROOT/'reports/instance-lineage-recovery-v2/complete.json',
        *tests,*python_sources(ROOT,[ROOT/'tests/test_parallel_function_repair_stage_v1.py',ROOT/'tests/test_parallel_repair_durable_state_v1.py']),*artifacts]
    files={str(p):file_hash(p) for p in sources}
    atomic_json(OUT/'complete.json',dict(at=at,complete=True,scope='Actual small four-CPU neural shared recovery and durable publication only',
        actual_workers=4,matched_profiles=3,precisions=['fp32','bf16'],actual_logical_updates_per_profile_precision=3,
        actual_physical_replica_optimizer_calls_for_recovery_run=96,all_private_RNGs_distinct_and_retained=True,
        exact_next_private_draws_source_update_actor_values_Adam_and_private_RNG_proved=True,
        wrong_peer_valid_RNG_and_resealed_bad_RNG_cursor_Adam_and_masks_rejected_before_mutation=True,
        one_unhealthy_peer_rejected_without_collective_sequence_mismatch=True,
        actual_root_IO_failures_coordinated_previous_commit_survives_and_cold_shared_recovery_passes=True,
        atomic_slot_replacement_preserves_old_mmap_readers=True,kernel_tests=7,shared_recovery_tests=4,durable_IO_tests=5,
        full_size_GPU_controller_and_queue_not_implemented=True,new_full_size_model_updates=0,new_GPU_model_allocations=0,
        no_neural_quality_metric_added=True,source_files_sha256=files,goal_achieved=False))
    requirements['serious_full_model_training']['evidence']=(f'The unchanged public-pretraining run has {len(rows)}/4447 complete actual updates and '
        f'{int(sum(r["response_EOS_targets"] for r in rows))} response/EOS targets in this captured prefix. '
        f'The unchanged full-size CPU comparison is at {snapshot.get("actual_complete_updates")}/306 in its current {snapshot.get("profile")}arm. '
        'The additional parallel kernel/shared-stage/durable-checkpoint implementation passes7+4+5tests. Actual small4CPUworkers prove all3profiles inFP32/BF16, '
        'exact next private draws/source update/actor/Adam/RNG replay, all-private rank cursor validation, coordinated invalid-peer rejection, '
        'root-only write-failure preservation, two-slot durable publication and cold shared-file recovery. These are numerical/recovery mechanics. '
        'The actual full-size GPU controller, queue/handoff and fresh full four-L40 capacity still remain to implement/run. No new quality result or promotion exists.')
    requirements['gpu_runtime']['evidence']+=' New shared actor/Adam checkpointing retains every worker RNG/cursor and independently retained manifests; every replica must prove equal shared values before root serializes one copy. Two-slot tensor/receipt/LATEST publication fsyncs files and directories. The human predecessor takes a nonblocking GPU flock, so explicit queue gating and a verified handoff are still required before adding a GPU study; all current actors/queues remain intact.'
    requirements['any_code_correctness_and_efficiency']['evidence']+=' The new recovery tests do not establish generated-code correctness or change the last0/48functional result. The complete826function/791repair/3epoch source study and all predeclared full final benchmarks remain in scope.'
    for item in requirements.values():item['status']='incomplete_for_requested_final_model'
    if len(requirements)!=10:raise ValueError('All ten original final requirements must remain in scope')
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,source_files_sha256={**files,str(OUT/'complete.json'):file_hash(OUT/'complete.json')},
        previous_full_scope_review=str(previous.relative_to(ROOT)),previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,original_full_scope_preserved=True,candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_research_progress=snapshot,latest_parallel_recovery_proof=str((OUT/'complete.json').relative_to(ROOT)),
        all_ten_final_requirements_incomplete=True,current_goal_turn_classification='progress',candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Parallel shared recovery and durable state, **{at}**: all current GPU/CPU training and ten original downstream owners remain live. '
        f'The main four-L40run has {len(rows)}/4,447complete updates; the current CPU comparison arm has {snapshot.get("actual_complete_updates")}/306. '
        'New shared-state and two-slot durable I/O implementations pass **4and5tests**, in addition to7parallel-kernel tests. '
        'Actual small4CPUreplicas exercise all3profiles inFP32/BF16, exact next-update weights/Adam/private-RNG/source replay, '
        'corrupt peer rejection before mutation, and injected root write failures with the previous committed state preserved. '
        '**These are recovery mechanics, not actual full-size GPU capacity or model-quality gains.** '
        'The GPU controller and explicit nonblocking-lease queue handoff are still pending; no original neural exposure or benchmark population is changed. '
        '**Last completed coding accuracy remains0/48functional; all ten final requirements remain incomplete.** '
        '[Actual recovery artifacts](reports/parallel-repair-recovery-CPU-v1/complete.json), '
        '[remaining GPU integration](reports/parallel-repair-recovery-CPU-v1/pending-GPU-controller-and-handoff.json), '
        '[full review](reports/parallel-repair-recovery-CPU-v1/review.json). Earlier entries are historical snapshots.\n\n')
    for path in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),main_updates=len(rows),CPU_study_updates=snapshot.get('actual_complete_updates'),
        tests_total=16,GPU_controller_queued=False,all_ten_incomplete=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage-output',required=True);parser.add_argument('--IO-output',required=True);args=parser.parse_args();main(args.stage_output,args.IO_output)
