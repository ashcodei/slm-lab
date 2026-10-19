"""Publish actual four-CPU numerical evidence and preserve the full goal scope."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.local_source_closure_v2 import python_sources
import own_training_gpu_lease_v2 as owner

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/parallel-function-repair-CPU-mechanics-v1'


def read(path):return json.loads(Path(path).read_text())


def main(unit_output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only prototype evidence required')
    tests=ROOT/'reports/parallel-function-repair-v1-final-tests.log'
    if '7 passed' not in tests.read_text():raise ValueError('All actual serial/replica/precision/source-role controls required')
    unit_output=Path(unit_output).resolve();proofs=[]
    for rank in range(4):
        path=unit_output/(str(rank)+'.json');raw=read(path)
        if not 0<=tests.stat().st_mtime-path.stat().st_mtime<180:raise ValueError('Raw replica evidence must be from the actual just-completed unit run')
        if len(raw)!=6 or {(p['profile'],p['precision']) for p in raw}!={(p,q) for p in ['plain_repeat','semantic_repair','weighted_semantic_repair'] for q in ['fp32','bf16']}:
            raise ValueError('Every real matched profile and precision must be tested')
        for row in raw:
            if (row['actual_rank']!=rank or row['actual_global_updates']!=2 or row['actual_neural_forwards']!=[2,1,1,0][rank]
                    or row['serial_max_absolute_actor_parameter_error']>=5e-6 or not row['all_replicas_actor_values_identical']
                    or not row['invalid_single_replica_source_role_rejected_before_mutation']):
                raise ValueError('Actual target-normalized replica/serial/tail/source guards must hold')
            for positions,record in zip([[0,1,2],[1]],row['records']):
                if (record['actual_world_size']!=4 or record['actual_device_type']!='cpu'
                        or record['actual_rank_source_shards']!=[positions[r::4] for r in range(4)]
                        or record['response_EOS_targets']!=sum(record['actual_rank_observations'][r]['actual_response_EOS_targets'] for r in range(4))):
                    raise ValueError('Raw original shard and complete target observations must rebuild')
        proofs.append(raw)
    for i in range(6):
        if any(proofs[r][i]['records']!=proofs[0][i]['records'] for r in range(4)):
            raise ValueError('Every actual global source/objective record must agree across replicas')
    runtime=ROOT/'reports/public-pretraining-recovery-v1';plan=read(runtime/'plan.json')
    owner.REPORTS=runtime;owner.resources.REPORTS=runtime;owner.owned(plan)
    ranks=[r for r in read(runtime/'children.json') if r['subject'].startswith('train-rank-') and not terminal(r)]
    handoff=read(ROOT/'reports/instance-lineage-recovery-v2/complete.json')
    if len(ranks)!=4 or any(terminal(r['actual_owner']) for r in handoff['actual_ten_new_controllers']):raise ValueError('All original GPU ranks and ten original queues must remain live')
    research=ROOT/'reports/full-function-repair-comparison-CPU-v1';research_plan=read(research/'plan.json')
    if terminal(research_plan['owner']) or (research/'failure.json').exists():raise ValueError('The original matched CPU study must continue unchanged')
    cache_path=ROOT/'reports/unused-GPU-wheel-cache-v2/complete.json';cache=read(cache_path)
    if not cache['complete'] or not cache['installed_package_versions_unchanged']:raise ValueError('Only completed safe cache maintenance may be reported')
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
    OUT.mkdir();at=datetime.now(timezone.utc).isoformat()
    for rank in range(4):(OUT/('actual-rank-'+str(rank)+'.json')).write_bytes((unit_output/(str(rank)+'.json')).read_bytes())
    prefix=[]
    for line in (runtime/'training-metrics.jsonl').read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):break
        prefix.append(line)
    main_rows=[json.loads(line) for line in prefix]
    if [r['step'] for r in main_rows]!=list(range(1,len(main_rows)+1)):raise ValueError('Actual main source prefix must be complete')
    (OUT/'captured-main-training-prefix.jsonl').write_bytes(b''.join(prefix))
    snapshot=read(research/'progress.json');atomic_json(OUT/'captured-research-progress.json',snapshot)
    maximum=max(r['serial_max_absolute_actor_parameter_error'] for rows in proofs for r in rows)
    sources=[Path(__file__),tests,previous,cache_path,runtime/'plan.json',research/'plan.json',
        *python_sources(ROOT,[ROOT/'src/slm_lab/parallel_function_repair_v1.py',ROOT/'tests/test_parallel_function_repair_v1.py']),
        *[OUT/('actual-rank-'+str(rank)+'.json') for rank in range(4)],OUT/'captured-main-training-prefix.jsonl',OUT/'captured-research-progress.json']
    files={str(p):file_hash(p) for p in sources}
    pending=dict(intended_own_initializer=str(ROOT/'checkpoints/public-pretraining-v1/trained/model.pt'),
        intended_initializer_loader='slm_lab.public_pretraining_native_v1.load',
        requires_full4447source_updates_original_native_assessments_and_GPU_restoration=True,
        intended_original_matched_source_schedule=dict(epochs=3,width=16,seed=979113,updates_per_profile=306,
            original_train_functions=826,verified_repair_views=791,positive_presentations_per_profile=4851,targets_per_profile=2287110),
        no_live_queue_handoff_executed=True,existing_CPU_comparison_not_migrated_or_stopped=True,
        GPU_controller_and_checkpoint_recovery_still_to_implement=True,
        all_original_human85786_code3721_machine1620_sources_and_nine_full_benchmark_endpoints_must_remain=True,
        important_existing_finalizer_name_is_not_the_public_pretraining_initializer='finalize_public_pretraining_student_recovery_CPU_v1.py exports broad-code after human/RL; it is not the new GPU study predecessor')
    atomic_json(OUT/'pending-GPU-integration.json',pending)
    atomic_json(OUT/'complete.json',dict(at=at,complete=True,scope='Small actual neural four-CPU replica and objective mechanics only',
        actual_workers=4,profiles=3,precisions=['fp32','bf16'],actual_global_updates_per_profile_precision=2,
        actual_distributed_replica_optimizer_calls=48,serial_reference_optimizer_calls=48,
        observed_max_absolute_serial_actor_parameter_error=maximum,
        actual_global_records_identical_across_replicas=True,empty_rank_neural_forwards=0,
        actual_source_role_corruption_on_one_rank_rejected_before_any_optimizer_mutation=True,
        CPU_fixture_not_full_dataset_or_actual984M_model=True,new_full_model_updates=0,
        actual_GPU_capacity_and_recovery_not_run=True,new_GPU_controller_not_implemented_or_queued=True,
        original_frozen_GPU_CPU_and_ten_queue_jobs_unmodified=True,
        reclaimed_unused_dependency_cache_bytes=cache['removed_allocated_cache_bytes'],source_files_sha256=files,goal_achieved=False))
    requirements['serious_full_model_training']['evidence']=(f'The original four-L40 public epoch has {len(main_rows)}/4447 complete updates and '
        f'{int(sum(r["response_EOS_targets"] for r in main_rows))} actual response/EOS targets in this captured prefix. '
        f'The unchanged full-size CPU comparison is at {snapshot}. All three full-size CPU capacity/recovery/cold gates passed before production. '
        'A new device-aware parallel CE kernel passes7tests, including4actual small CPU replicas under FP32/BF16 and disjoint/empty tails. '
        f'Observed maximum replica-versus-serial actor error is {maximum}. This is numerical mechanics, not full-size GPU training or recovery. '
        'The faster GPU study controller/queue integration remains to implement; no original learned job or full evaluation scope is replaced.')
    requirements['gpu_runtime']['evidence']+=' All four current original L40 actors remain live. The proposed additional GPU repair study has a tested numerical kernel only; no GPU lease, new GPU model or production controller has been started. It must wait for the full current public epoch, original native controls and restoration. Cache maintenance reclaims1935958016allocated bytes from7unused public dependency download copies, with installed versions and all16original owners preserved.'
    requirements['any_code_correctness_and_efficiency']['evidence']+=' The parallel kernel preserves all original gold targets/EOS, already-global normalization and exact source-rank coverage; an empty rank performs zero neural forwards. Seven numerical tests do not change the last0/48functional result or establish algorithmic improvement.'
    for value in requirements.values():value['status']='incomplete_for_requested_final_model'
    if len(requirements)!=10:raise ValueError('The full original ten final requirements must remain')
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,source_files_sha256={**files,str(OUT/'complete.json'):file_hash(OUT/'complete.json'),
        str(OUT/'pending-GPU-integration.json'):file_hash(OUT/'pending-GPU-integration.json')},previous_full_scope_review=str(previous.relative_to(ROOT)),
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,
        original_full_scope_preserved=True,candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_research_progress=snapshot,latest_storage_maintenance=str(cache_path.relative_to(ROOT)),
        all_ten_final_requirements_incomplete=True,current_goal_turn_classification='progress',candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'First full-source comparison updates and parallel-kernel mechanics, **{at}**: the same four L40s have {len(main_rows)}/4,447complete updates. '
        f'The unchanged CPU repair study has {snapshot.get("actual_complete_updates",0)}/306updates in its current {snapshot.get("profile")}arm. '
        'All three actual full-size CPU capacity/recovery/cold gates passed. Seven new tests verify a device-aware replica kernel with4actual small CPU workers inFP32andBF16, '
        'already-global target normalization and zero-forward empty tails. '
        '**The GPU controller, GPU checkpoint recovery, actual four-L40 capacity and queue integration remain unimplemented.** '
        'The intended additional GPU study will use the complete public-pretraining model; all current human/RL/full-benchmark stages must remain. '
        'Unused dependency-download cache cleanup recovers **1.80GiB**. '
        '**No new coding-accuracy gain is established; the last complete broader result remains0/48functional. All ten final requirements remain incomplete.** '
        '[Actual numerical evidence](reports/parallel-function-repair-CPU-mechanics-v1/complete.json), '
        '[pending integration](reports/parallel-function-repair-CPU-mechanics-v1/pending-GPU-integration.json), '
        '[full review](reports/parallel-function-repair-CPU-mechanics-v1/review.json). Earlier entries are historical snapshots.\n\n')
    for path in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),main_updates=len(main_rows),CPU_research_updates=snapshot.get('actual_complete_updates'),
        observed_serial_error=maximum,new_GPU_controller_queued=False,all_ten_incomplete=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--unit-output',required=True);args=parser.parse_args();main(args.unit_output)
