"""Run the actual visual kernel through the established bounded lifecycle.

The generic lifecycle owns latest/previous/pending checkpoints. This small
adapter supplies its receipt and progress interfaces without changing either
the previously verified visual kernel or the existing coding controller.
"""
from pathlib import Path

import torch

from .code_residual_checkpoint_v1 import _cpu
from .code_residual_collective_checkpoint_v2 import _digest, _same, _validated, fingerprint
from .code_residual_lifecycle_v4 import checkpoint_layout, rank_zero, run_updates, report_json
from .codenet_checkpoint_v1 import atomic_checkpoint, FileBindings
from .corpus import file_hash
from .paired_visual_training_v1 import PairedVisualKernel, FORMAT as STATE_FORMAT


EXPORT_FORMAT = 'own-paired-visual-inference-v1'


class _LifecycleView:
    def __init__(self, kernel):
        self.kernel = kernel

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def update(self):
        metric = self.kernel.update()
        # The established loop's throughput counter counts original proper CE
        # targets. Extra contrast forwards are not additional original labels.
        return dict(**metric, total_target_tokens=metric['proper_target_tokens'],
            throughput_basis='proper_image_response_EOS_targets_including_rehearsal')

    def snapshot(self, path, **storage):
        receipt = self.kernel.snapshot(path, **storage)
        size = rank_zero(lambda: Path(path).stat().st_size)
        return dict(**receipt, checkpoint_bytes=size)


def run_visual_stage(kernel, output, *, resume_sha256=None, maximum_checkpoint_bytes,
        minimum_free_bytes, checkpoint_every, log_every, maximum_report_bytes, until_step=None):
    """An explicitly bounded stop remains incomplete; recovery requires a hash."""
    def arguments():
        if type(kernel) is not PairedVisualKernel or kernel.completed or kernel.metrics:
            raise ValueError('Begin the lifecycle with a fresh original visual kernel')
        if resume_sha256 is not None and not _digest(resume_sha256):
            raise ValueError('Explicit exact latest-checkpoint digest required')
        return fingerprint(dict(output=str(Path(output).resolve()), resume=resume_sha256,
            maximum_checkpoint_bytes=maximum_checkpoint_bytes, minimum_free_bytes=minimum_free_bytes,
            checkpoint_every=checkpoint_every, log_every=log_every,
            maximum_report_bytes=maximum_report_bytes, until_step=until_step))
    _same(_validated(arguments))
    ledger = rank_zero(lambda: checkpoint_layout(output, kernel.sources['run_plan_sha256']))
    if resume_sha256 is not None:
        if ledger['latest'] is None or ledger['latest']['checkpoint_sha256'] != resume_sha256:
            raise ValueError('Explicit visual resume must match the latest complete journaled state')
        kernel.restore(Path(output) / 'resume.pt', resume_sha256)
    elif ledger['latest'] is not None:
        raise ValueError('Existing visual state requires explicit recovery, never an implicit restart')
    return run_updates(_LifecycleView(kernel), output, maximum_checkpoint_bytes=maximum_checkpoint_bytes,
        minimum_free_bytes=minimum_free_bytes, checkpoint_every=checkpoint_every, log_every=log_every,
        maximum_report_bytes=maximum_report_bytes, until_step=until_step)


def finish_visual_export(kernel, output, result, *, maximum_inference_bytes,
        minimum_free_bytes, maximum_report_bytes):
    """Export a completed, journaled policy; preserve complete state and old files.

All workers validate current shared state. Only rank zero serializes and checks
the independent native weights export. No capability-based selection occurs.
"""
    output = Path(output).resolve()
    def boundary():
        kernel._boundary(full=True)
        if (kernel.completed != kernel.steps or len(kernel.metrics) != kernel.steps
                or result.get('training_complete') is not True
                or result.get('completed_steps') != kernel.steps or result.get('planned_steps') != kernel.steps
                or any(p.grad is not None for p in kernel.model.parameters())
                or any(type(v) is not int or v < 1 for v in [maximum_inference_bytes, maximum_report_bytes])
                or type(minimum_free_bytes) is not int or minimum_free_bytes < 0):
            raise ValueError('Complete original clean visual training and explicit export storage required')
        return fingerprint(dict(identity=kernel.identity, model=kernel.model.state_dict(),
            optimizer=kernel.optimizer.state_dict(), metrics=kernel.metrics,
            output=str(output), result=result, maximum_inference_bytes=maximum_inference_bytes,
            minimum_free_bytes=minimum_free_bytes, maximum_report_bytes=maximum_report_bytes))
    _same(_validated(boundary))
    def finish():
        ledger = checkpoint_layout(output, kernel.sources['run_plan_sha256'])
        latest = ledger['latest']
        if latest != result['latest_checkpoint'] or latest['steps'] != kernel.steps:
            raise ValueError('The final complete visual checkpoint must match the owned journal')
        complete = output / 'resume.pt'
        binding = FileBindings({str(complete): latest['checkpoint_sha256']})
        state = torch.load(complete, map_location='cpu', weights_only=True, mmap=True)
        if (state.get('format') != STATE_FORMAT or state.get('identity') != kernel.identity
                or state.get('steps') != kernel.steps or state.get('planned_steps') != kernel.steps
                or state.get('state_sha256') != fingerprint({k: v for k, v in state.items() if k != 'state_sha256'})
                or fingerprint(state['model']) != fingerprint(kernel.model.state_dict())
                or fingerprint(state['optimizer']) != fingerprint(kernel.optimizer.state_dict())
                or state['metrics'] != kernel.metrics or state['sampler_state'] != kernel.sampler_state(kernel.steps)):
            raise ValueError('Every exported weight must match actual complete visual training')
        exported = dict(format=EXPORT_FORMAT, config=state['config'], model=_cpu(state['model']),
            tokenizer_sha256=kernel.sources['tokenizer_sha256'],
            base_checkpoint_sha256=kernel.sources['base_checkpoint_sha256'],
            run_plan_sha256=kernel.sources['run_plan_sha256'],
            complete_training_checkpoint_sha256=latest['checkpoint_sha256'],
            visual_identity_sha256=kernel.identity_sha256, steps=kernel.steps,
            candidate_promoted=False, goal_achieved=False)
        inference = output / 'inference.pt'
        if inference.exists():
            existing = torch.load(inference, map_location='cpu', weights_only=True, mmap=True)
            if fingerprint(existing) != fingerprint(exported):
                raise ValueError('Existing visual inference differs; inspect without overwriting it')
        else:
            atomic_checkpoint(inference, exported, maximum_inference_bytes, minimum_free_bytes)
        binding.check(full=True)
        receipt = dict(checkpoint=str(inference), checkpoint_sha256=file_hash(inference),
            checkpoint_bytes=inference.stat().st_size, complete_training_checkpoint_preserved_sha256=file_hash(complete),
            optimizer_or_rng_included=False, automatic_candidate_promotion=False)
        completion = {**result, 'inference': receipt, 'run_plan_sha256': kernel.sources['run_plan_sha256'],
            'actual_model_counts': kernel.model.parameter_counts(), 'visual_identity_sha256': kernel.identity_sha256,
            'capability_evaluated': False, 'candidate_promoted': False, 'goal_achieved': False}
        target = output / 'training-completed.json'
        if target.exists():
            import json
            if json.loads(target.read_text()) != completion:
                raise ValueError('Existing visual completion report differs; inspect without overwriting')
        else:
            report_json(target, completion, maximum_report_bytes)
        return receipt
    return rank_zero(finish)
