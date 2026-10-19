"""Reconstruct the real prefix and stop before the offending step commits."""
import json
from pathlib import Path

import torch
from torch import distributed as dist

from .codenet_checkpoint_v1 import atomic_checkpoint, capture_rng, restore_rng
from .code_probability_numerical_reference_v1 import compare
from .corpus import file_hash
from .diverse_images import atomic_json
from .execution_group_checked_host_v7 import CoordinatedLocalUpdateError
from .execution_recovery_semantics_v1 import same_executed_update
from .whole_human_sft_evidence_v1 import tensor_digest


class DiagnosticCommitBoundary(Exception):pass


def run(stage, plan, rank, device, reports, output, save, step):
    arm = stage.method
    root = Path(plan['origin_plan']).parents[2]
    original_folder = Path(plan['origin_plan']).parent
    original = [json.loads(line) for line in (original_folder/(arm+'-metrics.jsonl')).open()]
    completed_records = []
    stop = None
    while stage.completed < 139:
        before_rng = capture_rng(device)
        original_step = stage.optimizer.step
        if stage.completed == 138:
            stage.capture_parity_logits = True
            def boundary(*args, **kwargs):
                raise DiagnosticCommitBoundary('Diagnostic step139 must stop before optimizer commit')
            stage.optimizer.step = boundary
        try:
            actual = step()
        except (CoordinatedLocalUpdateError, DiagnosticCommitBoundary) as error:
            stop = dict(type=type(error).__name__, message=str(error),
                        actual_completed=stage.completed, failed_attempt=stage.completed+1)
            restore_rng(before_rng, device)
            break
        finally:
            stage.optimizer.step = original_step
        prior = original[actual['step']-1]
        if (actual['source_id'] != prior['source_id'] or actual['global_rewards'] != prior['global_rewards']
                or any(x[k] != y[k] for x,y in zip(actual['trajectories'],prior['trajectories'])
                       for k in ['response_ids','actual_cached_behavior_logps','own_response'])):
            raise ValueError('Actual original prefix actions/probabilities/rewards must reproduce before the offending attempt')
        completed_records.append(actual)
        atomic_json(reports/f'diagnostic-progress-{rank}.json',dict(actual_completed=stage.completed,
            inherited_updates=128,disposable_reconstructed_updates=stage.completed-128,
            old_recorded_behavior_and_rewards_reproduced=True,production_updates=0,goal_achieved=False))
    if stop is None or stage.completed >= 139:
        raise ValueError('The diagnostic must stop without any step139 commit')
    observations = stage.parity_observations
    if not observations or not all('actual_full_FP32_logits' in r for r in observations):
        # A nonoffending peer may have completed earlier local rows before
        # discovering its peer's error. Only the stopped step139 requests all
        # logits; an earlier unexpected failure remains explicit.
        if stage.completed == 138:
            raise ValueError('Every local step139 parity observation and original full FP32 logits required')
    raw_path = output/f'raw-parity-{rank}.pt'
    atomic_checkpoint(raw_path,dict(stop=stop,observations=observations),256*1024**2,26*1024**3)
    saved_path = save('diagnostic-stop')
    frozen = tensor_digest(stage.model.named_parameters())
    RNG = capture_rng(device)
    worst = max(observations,key=lambda r:r['max_absolute_action_logp_difference'])
    reference = compare(stage.model,worst)
    from .execution_group_host_capacity_v6 import equal
    if frozen != tensor_digest(stage.model.named_parameters()) or not equal(RNG,capture_rng(device)):
        raise ValueError('Numerical reference must not change the student or its restored private RNG')
    result = dict(physical_GPU=rank,arm=arm,actual_counts=stage.model.counts(),stop=stop,
        inherited_updates=128,disposable_reconstructed_updates=stage.completed-128,
        old_recorded_prefix_behavior_and_rewards_reproduced=True,
        actor_Adam_and_RNG_checkpoint_before_rejected_attempt=str(saved_path),
        actor_Adam_and_RNG_checkpoint_sha256=file_hash(saved_path),
        raw_original_FP32_comparisons=str(raw_path),raw_comparison_sha256=file_hash(raw_path),
        local_parity_observations=[{k:v for k,v in r.items() if k!='actual_full_FP32_logits'} for r in observations],
        same_student_weight_numerical_reference=reference,
        diagnostic_step139_not_committed=True,source_files_sha256=plan['source_files_sha256'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        full_profile_training_capacity_and_quality_not_claimed=True,production_updates=0,goal_achieved=False)
    atomic_json(reports/f'diagnostic-{rank}.json',result)
