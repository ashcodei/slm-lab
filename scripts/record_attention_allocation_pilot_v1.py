"""Read real complete code-allocation checkpoints on CPU; no model updates."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

from tokenizers import Tokenizer
import torch

from slm_lab.code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_feedback_union_v1 import ExecutionFeedbackUnion
from slm_lab.execution_preference_v1 import fingerprint as input_fingerprint
from slm_lab.recurrent_ablation import state_digest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/attention-allocation-pilot-v1'


def inventory(attention):
    result = {}
    for i in range(24):
        prefix = f'{i}.ffn' if attention else str(i)
        width = 1728 if attention else 2048
        result[prefix + '.down.weight'] = (width, 1536)
        result[prefix + '.up.weight'] = (1536, width)
        if attention:
            for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                result[f'{i}.attention.{name}.down.weight'] = (96, 1536)
                result[f'{i}.attention.{name}.up.weight'] = (512 if name in ['k_proj', 'v_proj'] else 1536, 96)
    return result


def main():
    torch.set_num_threads(2)
    if torch.cuda.is_initialized(): raise ValueError('Independent CPU-only readback required')
    plan = json.loads((REPORTS / 'plan.json').read_text())
    baseline = ROOT / 'checkpoints/extended-curriculum-v1/joint/model.pt'
    if file_hash(baseline) != plan['base_checkpoint_sha256']: raise ValueError('Exact original own base required')
    original = torch.load(baseline, map_location='cpu', weights_only=True, mmap=True)
    base_digest = state_digest(original['model'])
    base_count = sum(t.numel() for n, t in original['model'].items() if n != 'lm_head.weight')
    token = baseline.parent / 'tokenizer.json'
    feedback = ExecutionFeedbackUnion(ROOT / 'data/execution-feedback-union-v1', Tokenizer.from_file(str(token)), token)
    examples = [feedback.example(i) for i in range(374)]
    previous_metrics = json.loads((ROOT / 'reports/execution-code-pilot-v1-observed-checkpoint-metrics.json').read_text())['verified_CE_control']
    records, references, before = {}, {}, {}
    with tempfile.TemporaryDirectory(prefix='.attention-readback-', dir=ROOT / 'reports') as temp:
        for profile in plan['profiles']:
            name = profile['name']; source = ROOT / 'checkpoints/attention-allocation-pilot-v1' / name / 'latest.pt'
            stable = Path(temp) / (name + '.pt'); os.link(source, stable)
            digest = file_hash(stable)
            state = torch.load(stable, map_location='cpu', weights_only=True, mmap=True)
            initial = json.loads((REPORTS / name / 'initial-policy.json').read_text())
            ref = json.loads((REPORTS / name / 'reference.json').read_text())
            steps = state['steps']; attention = profile['attention']
            architecture = dict(format='own-code-FFN-attention-allocation-v1', residual_width=1728,
                attention_rank=96, attention_scale=1., attention_projections=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                placement='code-only low-rank projection updates plus post-block SiLU residual') if attention else dict(
                    format='own-code-FFN-only-v1', residual_width=2048)
            if (type(steps) is not int or not 32 <= steps <= 128 or steps % 32 or state['planned_steps'] != 128
                    or state['world_size'] != 1 or state['run_plan_sha256'] != file_hash(REPORTS / 'plan.json')
                    or state['base_checkpoint_sha256'] != plan['base_checkpoint_sha256'] or state['profile'] != profile
                    or state['tokenizer_sha256'] != plan['tokenizer_sha256'] or state['config'] != original['config']
                    or state['architecture'] != architecture or initial['architecture'] != architecture
                    or state['residual_width'] != architecture['residual_width']
                    or state['frozen_backbone_state_sha256'] != base_digest or initial['frozen_backbone_state_sha256'] != base_digest):
                raise ValueError('Actual own complete checkpoint architecture, source, step and frozen-base bindings required')
            shapes = inventory(attention); learned = state['residuals']
            if list(learned) != list(shapes) or state['residual_parameter_names'] != list(shapes):
                raise ValueError('Every exact ordered FFN/attention tensor must be retained')
            if any(t.shape != shapes[k] or t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for k, t in learned.items()):
                raise ValueError('Complete actual finite FP32 architecture tensor shapes required')
            count = sum(t.numel() for t in learned.values())
            if base_count != 706012672 or count != 150994944 or state_digest(learned) == initial['residual_state_sha256']:
                raise ValueError('Actual changed full857M parameter-matched code model required')
            with torch.device('meta'):
                parameters = [torch.nn.Parameter(torch.empty(shape, dtype=torch.float32)) for shape in shapes.values()]
            _validate_adam(state['optimizer'], parameters, steps)
            _validate_rng({**state['rng'], 'cuda': None}, 'cpu')
            cuda_rng = state['rng']['cuda']
            if not isinstance(cuda_rng, torch.Tensor) or cuda_rng.dtype != torch.uint8 or cuda_rng.ndim != 1 or not cuda_rng.numel():
                raise ValueError('Original actual CUDA RNG bytes must be retained; writer validates their CUDA restore')
            cache = state['execution_reference']
            if (cache != ref['cache'] or fingerprint(cache) != state['execution_reference_fingerprint']
                    or state['execution_reference_policy'] != ref['policy'] or ref['policy']['architecture'] != architecture
                    or cache['policy_sha256'] != fingerprint(ref['policy'])
                    or set(cache['records']) != {e['id'] for e in examples}):
                raise ValueError('All374 full frozen reference pairs and exact architecture policy must remain embedded')
            for example in examples:
                r = cache['records'][example['id']]
                for side in ['chosen', 'rejected']:
                    if (r[side + '_fingerprint'] != input_fingerprint(example[side])
                            or r[side + '_targets'] != sum(v != -100 for v in example[side][1][1:])):
                        raise ValueError('Complete original program/input/mask/reference target bytes required')
            if len(state['metrics']) != steps or state['clipped_sequences']:
                raise ValueError('Every actual recorded update and no truncation required')
            for i, (metric, old) in enumerate(zip(state['metrics'], previous_metrics)):
                if metric['step'] != i + 1 or metric['DPO']['alpha'] != 0. or metric['DPO']['weighted_DPO'] != 0.:
                    raise ValueError('Fixed128update CE-only architecture hypothesis required')
                for key in ['code_source_ids', 'verified_source_ids', 'pair_source_ids', 'replay_family', 'replay_position', 'positive_positions', 'pair_positions']:
                    if metric[key] != old[key]: raise ValueError('Exact same source order and exposure budget as original CE control required')
            cursors = state['sampler_state']
            if any(cursors[k] != multiplier * steps for k, multiplier in [('updates', 1), ('code_draws', 4), ('positive_draws', 2), ('pair_draws', 4), ('replay_draws', 1)]):
                raise ValueError('Complete exact source cursors required')
            if file_hash(stable) != digest: raise ValueError('Private complete checkpoint inode changed')
            references[name] = cache['records']
            before[name] = json.loads((REPORTS / name / 'source-conditioning-before.json').read_text())
            records[name] = dict(checkpoint_sha256=digest, observed_checkpoint_steps=steps, architecture=architecture,
                actual_total_unique_parameters=base_count + count, actual_trainable_parameters=count,
                actual_attention_parameters=23592960 if attention else 0,
                complete_finite_adapter_tensors=len(learned), whole_adapter_state_changed=True,
                zero_initialized_up_tensors_nonzero=sum(bool(t.count_nonzero()) for k,t in learned.items() if k.endswith('.up.weight')),
                complete_step_matched_Adam=True,
                CPU_RNG_valid_and_actual_CUDA_RNG_bytes_retained=True, full374reference_inputs_and_masks_reencoded=True,
                identical_original_source_order_verified=True, exact_original_frozen_706M_base_bound=True,
                source_cursors_and_all_metrics_retained=True, actual_GPU_updates=True,
                source=str(source), temporary_readback_inode_not_retained=True)
    names = list(records)
    if references[names[0]] != references[names[1]] or before[names[0]] != before[names[1]]:
        raise ValueError('Both architectures must have identical original initial374reference scores and all128validation-pair scores')
    files = [Path(__file__).resolve(), REPORTS / 'plan.json', baseline, token]
    files += [REPORTS / n / f for n in names for f in ['initial-policy.json', 'reference.json', 'source-conditioning-before.json']]
    output = dict(files_sha256={str(p):file_hash(p) for p in files}, observed_at=datetime.now(timezone.utc).isoformat(), actual_GPU_checkpoint_observations=records,
        exact_parameter_budget_match=True, all374_initial_reference_scores_identical=True,
        all128_initial_validation_source_scores_identical=True, initial_function_preservation_verified_on_original_sources=True,
        different_adapter_layouts_not_identical_initial_random_tensors=True, compute_matched=False,
        no_independent_CUDA_context=not torch.cuda.is_initialized(), candidate_promoted=False,
        new_capability_gain_established=False, goal_achieved=False)
    atomic_json(ROOT / 'reports/attention-allocation-pilot-v1-readback.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__': main()
