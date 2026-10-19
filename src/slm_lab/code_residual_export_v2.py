"""Reference-bound inference export; complete training state remains separate."""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .code_residual_branch_v1 import RoutedCodeResidual
from .code_residual_checkpoint_v1 import _digest, _validate_adam, _validate_rng
from .code_residual_collective_checkpoint_v2 import FORMAT as TRAIN_FORMAT, fingerprint, _metadata
from .codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
from .corpus import file_hash
from .recurrent_ablation import state_digest
from .training import load_checkpoint


FORMAT = 'own-code-residual-inference-v2'
FIELDS = {'format', 'training_checkpoint_sha256', 'base_checkpoint_sha256', 'tokenizer_sha256',
    'run_plan_sha256', 'data_sha256', 'config', 'residual_width', 'model_counts',
    'frozen_backbone_state_sha256', 'residual_parameter_names', 'residuals',
    'residual_state_sha256', 'steps', 'planned_steps', 'training_world_size',
    'optimizer_or_rng_included', 'candidate_promoted', 'goal_achieved'}


def _layout(model, state):
    if (fingerprint(asdict(model.config)) != fingerprint(state.get('config'))
            or fingerprint(model.parameter_counts()) != fingerprint(state.get('model_counts'))
            or [name for name, _ in model.residuals.named_parameters()] != state.get('residual_parameter_names')
            or state_digest(model.base.state_dict()) != state.get('frozen_backbone_state_sha256')):
        raise ValueError('Exact own backbone/config/unique parameter count and residual order required')
    model.load_adapter_state(state['residuals'])
    if state_digest(state['residuals']) != state.get('residual_state_sha256'):
        raise ValueError('Every saved residual byte must match its bound fingerprint')


def export_inference(complete_checkpoint, base_checkpoint, tokenizer, run_plan, source_manifest, output,
                     *, expected_training_checkpoint_sha256, maximum_bytes, minimum_free_bytes):
    """Export a completed four-rank state without dropping any resumable state.

    This writes a separate inference-only artifact. Its exact own base and
    tokenizer are retained references; a portable directory copy is separate.
    """
    complete_checkpoint, base_checkpoint, tokenizer, run_plan, source_manifest, output = map(Path,
        (complete_checkpoint, base_checkpoint, tokenizer, run_plan, source_manifest, output))
    if output.exists() or output.resolve() in {p.resolve() for p in [complete_checkpoint, base_checkpoint, tokenizer, run_plan, source_manifest]}:
        raise ValueError('Use a new distinct inference artifact; preserve complete training/source state')
    if not _digest(expected_training_checkpoint_sha256):
        raise ValueError('Exact externally bound complete training checkpoint required')
    guard = FileBindings({str(complete_checkpoint): expected_training_checkpoint_sha256,
        **{str(p): file_hash(p) for p in [base_checkpoint, tokenizer, run_plan, source_manifest]}})
    saved = torch.load(complete_checkpoint, map_location='cpu', weights_only=True, mmap=True)
    step = saved.get('steps')
    if (saved.get('format') != TRAIN_FORMAT or type(saved.get('world_size')) is not int or saved['world_size'] != 4
            or type(step) is not int or step < 1 or type(saved.get('planned_steps')) is not int or saved['planned_steps'] != step
            or saved.get('candidate_promoted') is not False or saved.get('goal_achieved') is not False):
        raise ValueError('Only an explicitly completed four-rank own training state may export')
    sources = dict(base_checkpoint_sha256=file_hash(base_checkpoint), tokenizer_sha256=file_hash(tokenizer),
        run_plan_sha256=file_hash(run_plan))
    plan = json.loads(run_plan.read_text())
    if (any(saved.get(k) != v or plan.get(k) != v for k, v in sources.items() if k != 'run_plan_sha256')
            or saved.get('run_plan_sha256') != sources['run_plan_sha256']
            or plan.get('code_manifest_sha256') != file_hash(source_manifest)):
        raise ValueError('Exact completed own base/tokenizer/plan/code-manifest bindings required')
    base, prior = load_checkpoint(base_checkpoint, 'cpu', training_state=False)
    if prior.get('tokenizer_sha256') != sources['tokenizer_sha256']:
        raise ValueError('Original own base tokenizer differs')
    model = RoutedCodeResidual(base, width=saved['residual_width'])
    _layout(model, saved)
    _validate_adam(saved.get('optimizer'), model.adapter_parameters(), step)
    rngs = saved.get('worker_rng_states')
    if not isinstance(rngs, list) or len(rngs) != 4:
        raise ValueError('Complete original per-rank training RNG state required')
    for rng in rngs:
        _validate_rng({**rng, 'cuda': None}, 'cpu')
        cuda = rng['cuda']
        if cuda is not None and (not isinstance(cuda, torch.Tensor) or cuda.dtype != torch.uint8 or cuda.ndim != 1 or cuda.numel() == 0):
            raise ValueError('Complete saved CUDA generator bytes required')
    if len({r['cuda'] is None for r in rngs}) != 1:
        raise ValueError('Original CPU/CUDA worker contract must match across ranks')
    meta = _metadata(model, sources, step, step, saved['sampler_state'])
    if (fingerprint(saved['optimizer']) != saved.get('optimizer_state_sha256')
            or fingerprint(rngs) != saved.get('worker_rng_states_sha256')
            or fingerprint(dict(meta=meta, base=saved['frozen_backbone_state_sha256'],
                residuals=saved['residuals'], optimizer=saved['optimizer'])) != saved.get('shared_state_fingerprint')):
        raise ValueError('Every complete residual/Adam/RNG/source/progress byte must match before export')
    if not isinstance(saved.get('metrics'), list) or [m.get('step') for m in saved['metrics']] != list(range(1, step + 1)):
        raise ValueError('Complete ordered original training history required')
    json.dumps(saved['metrics'], allow_nan=False)
    inference = {k: saved[k] for k in ['config', 'residual_width', 'model_counts', 'frozen_backbone_state_sha256',
        'residual_parameter_names', 'residuals', 'residual_state_sha256', 'steps', 'planned_steps']}
    inference.update(format=FORMAT, training_checkpoint_sha256=expected_training_checkpoint_sha256,
        **sources, data_sha256=file_hash(source_manifest), training_world_size=4,
        optimizer_or_rng_included=False, candidate_promoted=False, goal_achieved=False)
    guard.check(full=True)
    atomic_checkpoint(output, inference, maximum_bytes, minimum_free_bytes)
    guard.check(full=True)
    return dict(checkpoint=str(output), checkpoint_sha256=file_hash(output), checkpoint_bytes=output.stat().st_size,
        complete_training_checkpoint_preserved_sha256=file_hash(complete_checkpoint),
        base_checkpoint_sha256=sources['base_checkpoint_sha256'], data_sha256=file_hash(source_manifest),
        run_plan_sha256=sources['run_plan_sha256'], optimizer_or_rng_included=False)


def load_inference(checkpoint, base_checkpoint, tokenizer, *, expected_checkpoint_sha256,
                   expected_base_sha256, expected_plan_sha256,
                   expected_training_checkpoint_sha256, expected_data_sha256, device='cpu'):
    checkpoint, base_checkpoint, tokenizer = map(Path, (checkpoint, base_checkpoint, tokenizer))
    if not all(_digest(v) for v in [expected_checkpoint_sha256, expected_base_sha256, expected_plan_sha256,
            expected_training_checkpoint_sha256, expected_data_sha256]):
        raise ValueError('Exact inference/own-base/run-plan digests required')
    guard = FileBindings({str(checkpoint): expected_checkpoint_sha256, str(base_checkpoint): expected_base_sha256,
        str(tokenizer): file_hash(tokenizer)})
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    step = state.get('steps')
    if (set(state) != FIELDS or state.get('format') != FORMAT or state.get('base_checkpoint_sha256') != expected_base_sha256
            or state.get('tokenizer_sha256') != file_hash(tokenizer) or state.get('run_plan_sha256') != expected_plan_sha256
            or type(step) is not int or step < 1 or type(state.get('planned_steps')) is not int or state['planned_steps'] != step
            or type(state.get('training_world_size')) is not int or state['training_world_size'] != 4
            or state.get('optimizer_or_rng_included') is not False or state.get('candidate_promoted') is not False
            or state.get('goal_achieved') is not False or state.get('training_checkpoint_sha256') != expected_training_checkpoint_sha256
            or state.get('data_sha256') != expected_data_sha256):
        raise ValueError('Exact completed own inference-only export and source identities required')
    base, prior = load_checkpoint(base_checkpoint, 'cpu', training_state=False)
    if prior.get('tokenizer_sha256') != state['tokenizer_sha256']:
        raise ValueError('Own base tokenizer differs')
    routed = RoutedCodeResidual(base, width=state['residual_width'])
    _layout(routed, state)
    guard.check(full=True)
    routed.eval().to(device)
    if sum(p.numel() for p in routed.code.parameters()) != state['model_counts']['total_unique']:
        raise ValueError('Every unique own general/vision/critic/code parameter must belong to the returned route')
    metadata = dict(stage='own_code_residual_inference_v2', steps=step, planned_steps=step,
        tokenizer_sha256=state['tokenizer_sha256'], data_sha256=state['data_sha256'],
        own_base_checkpoint_sha256=expected_base_sha256, run_plan_sha256=expected_plan_sha256,
        training_checkpoint_sha256=state['training_checkpoint_sha256'], model_counts=state['model_counts'],
        world_size=4, route='code', optimizer_loaded_on_GPU=False, candidate_promoted=False)
    return routed.code, metadata
