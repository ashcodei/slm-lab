"""Read every learned tensor and metric of the completed original systems stage."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2)
    folder = ROOT / 'checkpoints/dialogue-retention-v1/systems-variety-code'
    reports = ROOT / 'reports/dialogue-retention-v1'
    model = folder / 'model.pt'; base = ROOT / 'checkpoints/dialogue-retention-v1/broad-code/model.pt'
    names = ['systems-variety-code-plan.json', 'systems-variety-actual-target-receipt.json',
        'systems-variety-vision-preservation.json', 'systems-variety-completed-state-cleanup.json']
    paths = [model, base, folder / 'training.json', folder / 'progress.json', folder / 'tokenizer.json',
        ROOT / 'data/systems-variety-recovery-v2/prepared-code/packed/corpus.json', Path(__file__).resolve()]
    paths += [reports / n for n in names]
    bound = {str(p): file_hash(p) for p in paths}
    plan, receipt, frozen, retired = [json.loads((reports / n).read_text()) for n in names]
    state = torch.load(model, map_location='cpu', weights_only=True, mmap=True)
    previous = torch.load(base, map_location='cpu', weights_only=True, mmap=True)
    training = json.loads((folder / 'training.json').read_text()); progress = json.loads((folder / 'progress.json').read_text())
    if (state['steps'] != 1138 or state['planned_steps'] != 1138 or state['world_size'] != 4
            or state['stage'] != 'pretrain' or state['trainable'] != 'language' or state['truncated_sequences'] != 0
            or training['metrics'] != state['metrics'] or state['config'] != previous['config']
            or state['tokenizer_sha256'] != previous['tokenizer_sha256'] or state['tokenizer_sha256'] != file_hash(folder / 'tokenizer.json')
            or state['data_sha256'] != plan['data_sha256'] or state['data_sha256'] != bound[str(paths[5])]
            or bound[str(base)] != plan['base_checkpoint_sha256']
            or bound[str(base)] != '743363d2792c2726352addfe9057649ffa3afe557707e8d82fed0f1bc69f9347'):
        raise ValueError('Exact complete original1138-update/four-worker systems stage required')
    metrics = state['metrics']
    if len(metrics) != 1138: raise ValueError('Every actual update must be retained')
    for i, m in enumerate(metrics):
        if m['step'] != i + 1 or m['global_target_tokens'] != 262080 or any(not math.isfinite(m[k]) for k in ['loss', 'learning_rate', 'step_seconds', 'global_target_tokens_per_second']):
            raise ValueError('Complete finite ordered target/throughput/loss metrics required')
    targets = sum(m['global_target_tokens'] for m in metrics)
    if targets != 298247040 or progress['processed_target_tokens'] != targets or receipt['actual_target_exposures'] != targets:
        raise ValueError('Actual complete systems target exposures differ')
    if set(state['model']) != set(previous['model']): raise ValueError('Every original own tensor must remain')
    changed = []
    for name, t in state['model'].items():
        old = previous['model'][name]
        if t.shape != old.shape or t.dtype != old.dtype or not bool(torch.isfinite(t).all()):
            raise ValueError('Complete finite exact tensor layout required')
        if not torch.equal(t, old): changed.append(name)
    if not changed or any(n.startswith('vision.') or n == 'value_head.weight' for n in changed):
        raise ValueError('Only original language tensors may change')
    count = sum(t.numel() for n, t in state['model'].items() if n != 'lm_head.weight')
    if count != 706012672 or not torch.equal(state['model']['embedding.weight'], state['model']['lm_head.weight']):
        raise ValueError('Exact original own706M and tied embedding/output required')
    if (frozen['candidate_checkpoint_sha256'] != bound[str(model)] or frozen['base_checkpoint_sha256'] != bound[str(base)]
            or retired['retained_model_sha256'] != bound[str(model)] or retired['all_learned_tensors_verified_equal'] is not True
            or retired['removed'] is not True or plan['packed_tail_token_positions_not_sampled'] != 2003
            or plan['final_batch_repeated_windows'] != 38):
        raise ValueError('Exact retained-weight receipts and declared legacy sampling gaps required')
    torch.Generator().set_state(state['torch_rng_state'])
    for path, digest in bound.items():
        if file_hash(path) != digest: raise ValueError('Completed stage bytes changed during readback')
    result = dict(observed_at=datetime.now(timezone.utc).isoformat(), passed=True, completed_updates=1138,
        world_size=4, actual_target_exposures=targets, unique_model_parameters=count,
        all_learned_tensors_finite=len(state['model']), changed_language_tensor_names=changed,
        every_original_vision_and_critic_tensor_identical=True, tied_embeddings_exact=True,
        all1138_metrics_match_checkpoint_and_report=True, actual_retained_checkpoint_sha256=bound[str(model)],
        base_checkpoint_sha256=bound[str(base)], new_systems_source_tokens=52970300,
        older_code_rehearsal_tokens=245123134, combined_source_tokens=298093434,
        legacy_unsampled_tail_targets=2003, repeated_final_batch_windows=38, truncated_sequences=0,
        completed_optimizer_retired_by_original_controller=True, complete_four_rank_resume_certificate=False,
        new_execution_feedback_used=False, new_capability_gain_measured=False, goal_achieved=False, files_sha256=bound,
        scope='Actual completed own706M OS/network/security continuation and all saved weights/metrics. The packed tail gap remains explicit; later complete replay, instruction/image/human-feedback continuation and native capability evaluation remain required.')
    atomic_json(ROOT / 'reports/completed-systems-code-state-readback-v1.json', result)
    print(json.dumps({k: result[k] for k in ['passed', 'completed_updates', 'actual_target_exposures', 'actual_retained_checkpoint_sha256', 'goal_achieved']}, indent=2))


if __name__ == '__main__': main()
