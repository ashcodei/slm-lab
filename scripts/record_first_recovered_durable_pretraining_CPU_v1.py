"""Observe the first committed recovered state; publish the actual scope audit.

This does not restore or modify model weights and does not certify a complete
independent semantic readback of every tensor. Full-profile next-step recovery
is separately exercised by the actual four-GPU capacity run.
"""
from datetime import datetime,timezone
import json,os,subprocess,time,traceback
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/recovered-public-first-durable-state-CPU-v1'
RUNTIME=ROOT/'reports/public-pretraining-recovery-v1'
CHECKPOINT=ROOT/'checkpoints/public-pretraining-v1/resume.pt'
AUDIT=ROOT/'scripts/record_instance_recovery_and_matched_results_v1.py'

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only durable-state observation required')
    OUT.mkdir();audit_sha=file_hash(AUDIT);started=time.monotonic()
    atomic_json(OUT/'plan.json',dict(owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256={str(Path(__file__)):file_hash(__file__),str(AUDIT):audit_sha},neural_updates=0,goal_achieved=False))
    try:
        while not CHECKPOINT.exists():
            if (RUNTIME/'failure.json').exists():raise ValueError('Actual recovered GPU controller failed; no durable success can be inferred')
            atomic_json(OUT/'progress.json',dict(waiting_for_actual_committed_resume=True,seconds=time.monotonic()-started,goal_achieved=False))
            time.sleep(25)
        import torch
        torch.set_num_threads(1)
        before=CHECKPOINT.stat();raw=torch.load(CHECKPOINT,weights_only=True,map_location='cpu',mmap=True);core=raw['core']
        plan=ROOT/'reports/public-pretraining-v1/plan.json';capacity=json.loads((RUNTIME/'capacity.json').read_text())
        completed=core['completed'];metrics=core['metrics']
        if (not capacity['passed'] or core['mode']!='train' or core['run_plan_sha256']!=file_hash(plan)
                or len(core['rngs'])!=4 or core['world_size']!=4 or len(metrics)!=completed
                or core['model']['completed_updates']!=completed or completed<1
                or [row['step'] for row in metrics]!=list(range(1,completed+1))):
            raise ValueError('Actual complete committed training prefix and all-worker state envelope required')
        digest=file_hash(CHECKPOINT);after=CHECKPOINT.stat()
        if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
            raise ValueError('A later atomic generation arrived during observation; preserve this incomplete observation')
        atomic_json(OUT/'stored-state-observation.json',dict(at=datetime.now(timezone.utc).isoformat(),
            actual_completed_training_updates=completed,actual_checkpoint=str(CHECKPOINT),checkpoint_sha256=digest,bytes=after.st_size,
            actual_original_plan_sha256=file_hash(plan),all_four_private_RNG_states_present=True,
            observed_actual_Adam_leaves=len(core['optimizer']['state']),stored_metric_prefix_steps_exact=True,
            every_parameter_value_or_semantic_source_mask_not_independently_recomputed_here=True,
            producer_four_GPU_next_update_recovery_proof=str(RUNTIME/'capacity.json'),goal_achieved=False))
        del raw,core
        if file_hash(AUDIT)!=audit_sha:raise ValueError('Bound status writer changed')
        subprocess.run([str(ROOT/'.venv/bin/python'),str(AUDIT)],cwd=ROOT,env=dict(os.environ),check=True)
        atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),
            actual_committed_recovered_training_updates=completed,durable_state_bytes_observed_and_hashed=True,
            full_scope_audit_updated=True,neural_updates=0,goal_achieved=False))
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),goal_achieved=False));raise

if __name__=='__main__':main()
