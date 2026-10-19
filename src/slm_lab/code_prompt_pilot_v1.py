"""Bounded full-model source-matching updates and frozen-policy evaluation.

This experiment uses association labels, never benchmark answers or human
preferences. Its evaluation bridge returns the exact live saved code route;
the complete world-one Adam/RNG checkpoint remains on disk.
"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import torch

from .code_prompt_matching_v1 import matching_gradients
from .code_residual_checkpoint_v1 import FORMAT, _validate_adam, _validate_rng
from .code_residual_pilot_v1 import PROFILES, learning_rate
from .code_residual_stage_v3 import replay_field
from .codenet_checkpoint_v1 import FileBindings
from .codenet_runner_v1 import FixedReductionModel
from .codenet_stage_v1 import Objective, objective_gradients
from .corpus import file_hash
from .recurrent_ablation import state_digest


def validate_selection(selection, sources):
    if set(selection) != {'train', 'validation'} or set(sources) != set(selection):
        raise ValueError('Both exact original sealed roles required')
    for role, number in [('train', 512), ('validation', 128)]:
        rows = selection[role]
        if len(rows) != number or len({r['id'] for r in rows}) != number:
            raise ValueError('Exact original512/128 distinct source selections required')
        for row in rows:
            source = sources[role].records[row['local_position']]
            expected = dict(role=role, id=source['id'], group_id=source['group_id'],
                language=source['language'], global_position=source['global_position'],
                context_tokens=source['prefix_tokens'] + source['target_tokens'],
                target_tokens=source['target_tokens'], full_input_ids_sha256=source['full_input_sha256'])
            if any(row[k] != value for k, value in expected.items()):
                raise ValueError('Every selected full original input/mask/source identity must match')
        if {r['language'] for r in rows} != {r['language'] for r in sources[role].records}:
            raise ValueError('Every original role language must remain represented')
        for field in ['target_tokens', 'context_tokens']:
            values = [(r['target_tokens'] if field == 'target_tokens' else r['prefix_tokens'] + r['target_tokens'])
                for r in sources[role].records]
            if max(r[field] for r in rows) != max(values):
                raise ValueError('Both original complete source size maxima must remain represented')
    if {r['group_id'] for r in selection['train']} & {r['group_id'] for r in selection['validation']}:
        raise ValueError('Training and validation source groups must be disjoint')


def update(model, optimizer, examples, replay_encoder, replay_row, family, step, alpha, device, amp):
    model.assert_partition()
    if any(p.grad is not None for p in model.base.parameters()):
        raise ValueError('Original backbone cannot receive gradients')
    model.train()
    model.code.gradient_checkpointing = True
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(PROFILES[1], step)
    optimizer.param_groups[0]['lr'] = lr
    matched = matching_gradients(model.code, examples, alpha=alpha, beta=10., margin=.1,
        device=device, amp=amp)
    replay = objective_gradients(FixedReductionModel(model.code),
        [Objective(family, replay_encoder, [[replay_row]], replay_field(family), .5)], device, amp)
    if replay_encoder.truncated:
        raise ValueError('Every original replay prompt/response must remain complete')
    if (any(p.grad is not None for p in model.base.parameters())
            or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.adapter_parameters())):
        raise ValueError('All and only residual tensors need finite gradients')
    norm = torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(step=step + 1, source_matching=matched, replay=replay,
        replay_family=family, learning_rate=lr, gradient_norm_before_clipping=float(norm),
        train_source_ids=[e['id'] for e in examples],
        mismatched_source_ids=[e['negative_source_id'] for e in examples])


def verify_saved_state(model, path, *, device, plan_sha256, base_sha256, tokenizer_sha256, completed):
    """Read back the complete artifact, including Adam/RNG, without GPU moments."""
    bundle = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (bundle.get('format') != FORMAT or bundle.get('world_size') != 1
            or bundle.get('steps') != completed or bundle.get('planned_steps') != 128
            or bundle.get('run_plan_sha256') != plan_sha256
            or bundle.get('base_checkpoint_sha256') != base_sha256
            or bundle.get('tokenizer_sha256') != tokenizer_sha256
            or bundle.get('config') != asdict(model.config) or bundle.get('residual_width') != model.width
            or bundle.get('model_counts') != model.parameter_counts()
            or bundle.get('frozen_backbone_state_sha256') != state_digest(model.base.state_dict())
            or bundle.get('residual_parameter_names') != [k for k, _ in model.residuals.named_parameters()]
            or len(bundle.get('metrics', [])) != completed
            or bundle.get('candidate_promoted') is not False or bundle.get('goal_achieved') is not False):
        raise ValueError('Exact full own world-one checkpoint and all source/step bindings required')
    expected = model.residuals.state_dict()
    actual = bundle.get('residuals')
    if (not isinstance(actual, dict) or set(actual) != set(expected)
            or any(actual[k].shape != expected[k].shape or actual[k].dtype != expected[k].dtype
                or not bool(torch.isfinite(actual[k]).all()) for k in expected)
            or state_digest(actual) != state_digest(expected)):
        raise ValueError('Every saved residual tensor must match the frozen live policy')
    _validate_adam(bundle.get('optimizer'), model.adapter_parameters(), completed)
    _validate_rng(bundle.get('rng'), device)
    return dict(checkpoint_sha256=file_hash(path), residual_state_sha256=state_digest(actual),
        frozen_backbone_state_sha256=bundle['frozen_backbone_state_sha256'],
        optimizer_and_RNG_complete=True, optimizer_loaded_on_GPU=False,
        model_counts=bundle['model_counts'], steps=completed)


class FrozenPilotLoader:
    def __init__(self, model, checkpoint, metadata, receipt, device, guard):
        self.model, self.path, self.metadata = model, Path(checkpoint).absolute(), dict(metadata)
        self.receipt, self.device, self.guard = dict(receipt), str(torch.device(device)), guard
        self.calls = 0
        self.files = FileBindings({str(self.path): receipt['checkpoint_sha256'],
            str(self.path.parent / 'tokenizer.json'): metadata['tokenizer_sha256']})

    def check(self):
        self.guard()
        self.files.check(full=True)
        if (state_digest(self.model.residuals.state_dict()) != self.receipt['residual_state_sha256']
                or state_digest(self.model.base.state_dict()) != self.receipt['frozen_backbone_state_sha256']
                or any(p.grad is not None for p in self.model.parameters())):
            raise ValueError('The complete frozen evaluated policy changed')

    def __call__(self, checkpoint, device='cpu', training_state=True):
        if (Path(checkpoint).absolute() != self.path or training_state is not False
                or str(torch.device(device)) != self.device):
            raise ValueError('Only exact saved-policy inference path/device allowed')
        self.check()
        self.calls += 1
        self.model.eval()
        return self.model.code, dict(self.metadata)


@contextmanager
def evaluator_loader(module, loader):
    if not isinstance(loader, FrozenPilotLoader) or not callable(getattr(module, 'load_checkpoint', None)):
        raise ValueError('Exact frozen live-policy loader and unchanged evaluator required')
    original = module.load_checkpoint
    try:
        module.load_checkpoint = loader
        yield
    finally:
        module.load_checkpoint = original
        loader.check()
