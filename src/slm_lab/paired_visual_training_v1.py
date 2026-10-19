"""Vision-only paired grounding and complete earlier visual rehearsal.

This is a training kernel, not a launcher. Actual source admission, final own
base/baselines, GPU capacity and capability-based selection remain external.
Full checkpoints retain all weights, Adam, each worker's RNG, metrics and
deterministic source cursors. No policy promotion occurs here.
"""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

import torch
from torch import distributed as dist

from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng
from .code_residual_collective_checkpoint_v2 import _digest, _same, _validated, fingerprint
from .code_residual_stage_v3 import _EncodedBatches
from .codenet_checkpoint_v1 import collect_worker_rng, atomic_checkpoint, FileBindings, restore_rng
from .codenet_replay_schedule_v1 import coverage_replay_schedule
from .codenet_runner_v1 import average_gradients, VerifiedReplayEncoder
from .codenet_stage_v1 import EpochOrder
from .corpus import file_hash
from .execution_training_v5 import global_CE_gradients
from .finalized_visual_replay_v1 import VISUAL_FAMILIES
from .model import TinyMultimodalLM
from .paired_visual_contrast_v1 import gradients
from .recurrent_ablation import state_digest


FORMAT = 'own-paired-visual-training-state-v1'
INTEGERS = {'steps', 'seed', 'pair_batch_size', 'pair_accumulation',
    'replay_batch_size', 'replay_accumulation', 'image_size'}
FLOATS = {'learning_rate', 'weight_decay', 'gradient_clip', 'alpha', 'beta', 'margin', 'replay_coefficient'}
RUNTIME = ('paired_visual_training_v1', 'paired_visual_contrast_v1', 'finalized_visual_replay_v1',
    'finalized_visual_pairs_v1', 'execution_training_v5', 'code_residual_stage_v3',
    'codenet_replay_schedule_v1', 'codenet_runner_v1', 'codenet_stage_v1', 'codenet_checkpoint_v1',
    'code_residual_checkpoint_v1', 'code_residual_collective_checkpoint_v2', 'recurrent_ablation', 'model')


def freeze_for_visual_stage(model):
    if type(model) is not TinyMultimodalLM:
        raise ValueError('Original own multimodal model required')
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('vision.'))
    return list(model.vision.parameters())


class PairedVisualKernel:
    def __init__(self, model, optimizer, pairs, replay, encoder, options, sources, device, amp, *, guard):
        if not dist.is_initialized() or dist.get_world_size() != 4:
            raise ValueError('Four actual initialized workers required')
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.model, self.optimizer, self.pairs, self.replay = model, optimizer, pairs, replay
        self.encoder, self.options, self.sources = encoder, deepcopy(options), deepcopy(sources)
        self.device, self.amp, self.guard = device, amp, guard
        self.completed, self.metrics = 0, []

        def preflight():
            files = [Path(__file__).with_name(name + '.py') for name in RUNTIME]
            runtime = {str(path.resolve()): file_hash(path) for path in files}
            self.runtime_guard = FileBindings(runtime)
            if (set(options) != INTEGERS | FLOATS
                    or any(type(options[k]) is not int or options[k] < (0 if k == 'seed' else 1) for k in INTEGERS)
                    or any(type(options[k]) not in {int, float} or not math.isfinite(options[k]) for k in FLOATS)
                    or any(options[k] <= 0 for k in ['learning_rate', 'gradient_clip', 'beta', 'replay_coefficient'])
                    or options['weight_decay'] < 0 or not 0 <= options['alpha'] <= 1
                    or not 0 < options['beta'] <= 100 or not 0 <= options['margin'] <= 10):
                raise ValueError('Exact finite visual objective and complete exposure options required')
            if (set(sources) != {'base_checkpoint_sha256', 'tokenizer_sha256', 'run_plan_sha256'}
                    or not all(_digest(v) for v in sources.values()) or guard is None):
                raise ValueError('Exact own base/tokenizer/plan and external file guard required')
            if (type(model) is not TinyMultimodalLM or not isinstance(encoder, VerifiedReplayEncoder)
                    or encoder.config != model.config or encoder.image_size != options['image_size'] or encoder.truncated):
                raise ValueError('Matching own model and complete golden-pixel replay encoder required')
            self._partition()
            if any(p.grad is not None for p in model.parameters()) or optimizer.state:
                raise ValueError('Construct with exact own initial weights and a fresh optimizer; restore afterwards')
            if set(replay.families) != set(VISUAL_FAMILIES) or len(replay.families) != 8 or len(pairs) < 1:
                raise ValueError('All original paired questions and eight earlier visual families required')
            self._contracts()
            self._guard(full=True)
            if (any(not bool(torch.isfinite(t).all()) for t in model.state_dict().values())
                    or not torch.equal(model.embedding.weight, model.lm_head.weight)):
                raise ValueError('Complete finite original own weights and tied embeddings required')
            self.populations = tuple((n, replay.size(n)) for n in replay.families)
            self.steps = options['steps']
            self.width = 4 * options['pair_batch_size'] * options['pair_accumulation']
            if self.steps * self.width < len(pairs):
                raise ValueError('Declared stage must cover every original pair')
            self.schedule = coverage_replay_schedule(self.populations, self.steps,
                4 * options['replay_batch_size'] * options['replay_accumulation'])
            self.frozen_sha256 = self._frozen_digest(model.state_dict())
            self.identity = dict(format=FORMAT, sources=self.sources, config=asdict(model.config),
                options=self.options, pair_contract_sha256=pairs.contract_sha256,
                replay_contract_sha256=replay.contract_sha256, populations=self.populations,
                original_pairs=len(pairs), frozen_language_critic_sha256=self.frozen_sha256,
                initial_model_state_sha256=state_digest(model.state_dict()), runtime_files_sha256=runtime,
                trainable_names=[n for n, p in model.named_parameters() if p.requires_grad])
            self.identity_sha256 = fingerprint(self.identity)
            return fingerprint(dict(identity=self.identity, model=model.state_dict(), optimizer=optimizer.state_dict()))
        _same(_validated(preflight))

    @staticmethod
    def _frozen_digest(state):
        return state_digest({n: t for n, t in state.items() if not n.startswith('vision.')})

    def _partition(self):
        if any(p.requires_grad != n.startswith('vision.') for n, p in self.model.named_parameters()):
            raise ValueError('Only the original vision path may train; all language/critic weights stay frozen')
        group = self.optimizer.param_groups
        if (type(self.optimizer) is not torch.optim.AdamW or len(group) != 1
                or [id(p) for p in group[0]['params']] != [id(p) for p in self.model.vision.parameters()]
                or group[0].get('foreach') is not False or group[0].get('fused') not in [None, False]
                or any(group[0].get(k, False) for k in ['amsgrad', 'maximize', 'capturable', 'differentiable'])
                or tuple(group[0]['betas']) != (.9, .999) or group[0]['eps'] != 1e-8
                or group[0]['weight_decay'] != self.options['weight_decay']):
            raise ValueError('Exact ordinary vision-only AdamW parameter/hyperparameter contract required')

    def _contracts(self):
        for source in [self.pairs, self.replay]:
            if not _digest(source.contract_sha256) or fingerprint(source.contract) != source.contract_sha256:
                raise ValueError('Complete original source contracts must remain unchanged')

    def _guard(self, full=False):
        self.runtime_guard.check(full=full)
        self.guard.check(full=full)
        self.pairs.guard.check(full=full)
        self.replay.guard.check(full=full)

    def _boundary(self, full=False):
        self._contracts(); self._guard(full=full); self._partition()
        if (fingerprint(self.identity) != self.identity_sha256 or self.identity['options'] != self.options
                or self.identity['sources'] != self.sources
                or self.identity['pair_contract_sha256'] != self.pairs.contract_sha256
                or self.identity['replay_contract_sha256'] != self.replay.contract_sha256
                or tuple((n, self.replay.size(n)) for n in self.replay.families) != self.populations
                or self.identity['original_pairs'] != len(self.pairs)
                or self.steps != self.options['steps']):
            raise ValueError('Exact original visual policy/source/optimization identity required')
        if full and self._frozen_digest(self.model.state_dict()) != self.frozen_sha256:
            raise ValueError('An original frozen language/critic tensor changed')

    def sampler_state(self, step):
        if type(step) is not int or not 0 <= step <= self.steps:
            raise ValueError('Exact in-budget completed visual update count required')
        pair_draws = step * self.width
        families = {}
        for name, size in self.populations:
            occurrences = sum(self.schedule.at(s)[0] == name for s in range(step))
            draws = occurrences * 4 * self.options['replay_batch_size'] * self.options['replay_accumulation']
            families[name] = dict(source_records=size, occurrences=occurrences, draws=draws,
                epoch_and_cursor=list(divmod(draws, size)))
        return dict(updates=step, paired=dict(source_pairs=len(self.pairs), draws=pair_draws,
            epoch_and_cursor=list(divmod(pair_draws, len(self.pairs)))), earlier_visual=families)

    def update(self):
        def boundary():
            self._boundary()
            if not 0 <= self.completed < self.steps or len(self.metrics) != self.completed:
                raise ValueError('Original complete visual budget finished or inconsistent')
            return fingerprint(dict(identity=self.identity, sampler=self.sampler_state(self.completed)))
        _same(_validated(boundary))
        step, opts = self.completed, self.options
        name, occurrence = self.schedule.at(step)
        def prepare():
            positions = [i for group in EpochOrder(len(self.pairs), opts['seed']).rank_groups(
                step, self.rank, 4, opts['pair_batch_size'], opts['pair_accumulation']) for i in group]
            groups = [[self.replay.row(name, i) for i in group] for group in EpochOrder(self.replay.size(name),
                opts['seed'] + 1000 + self.replay.families.index(name)).rank_groups(
                    occurrence, self.rank, 4, opts['replay_batch_size'], opts['replay_accumulation'])]
            if any(not row.get('image') for group in groups for row in group):
                raise ValueError('Every original visual rehearsal row must retain its image')
            encoded = _EncodedBatches(self.encoder, groups, 'response', self.device)
            if self.encoder.truncated:
                raise ValueError('Every original visual prompt/answer/EOS must stay complete')
            return [self.pairs.example(i) for i in positions], encoded
        examples, encoded = _validated(prepare)
        self.model.train(); self.model.gradient_checkpointing = True
        lr = opts['learning_rate'] * min(1., (step + 1) / max(1, self.steps // 10))
        lr *= .1 + .9 * .5 * (1 + math.cos(math.pi * step / self.steps))
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        self.optimizer.zero_grad(set_to_none=True)
        try:
            paired = gradients(self.model, examples, contract_sha256=self.pairs.contract_sha256,
                alpha=opts['alpha'], beta=opts['beta'], margin=opts['margin'], world=4, device=self.device, amp=self.amp)
            replay = global_CE_gradients(self.model, [(name, encoded, opts['replay_coefficient'])],
                self.model.config, self.device, self.amp)
            def finite():
                if (any(p.grad is not None for n, p in self.model.named_parameters() if not n.startswith('vision.'))
                        or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.model.vision.parameters())):
                    raise ValueError('All and only vision tensors require finite gradients')
            _validated(finite)
            average_gradients(self.model, self.device, 4)
            norm = _validated(lambda: float(torch.nn.utils.clip_grad_norm_(self.model.vision.parameters(),
                opts['gradient_clip'], error_if_nonfinite=True)))
            self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        result = dict(step=step + 1, paired=paired, earlier_visual_replay=replay,
            summed_objective=paired['summed_objective'] + replay['CE_summed_objective'],
            proper_target_tokens=paired['global_proper_target_tokens'] + replay['CE_total_target_tokens'],
            replay_family=name, replay_occurrence=occurrence, learning_rate=lr,
            gradient_norm_before_clipping=norm, human_preference_RLHF=False, capability_claim=False)
        self.completed += 1; self.metrics.append(result)
        return result

    def snapshot(self, path, *, maximum_bytes, minimum_free_bytes):
        def preflight():
            self._boundary(full=True)
            if not 0 < self.completed <= self.steps or any(p.grad is not None for p in self.model.parameters()):
                raise ValueError('Snapshot only at a completed clean optimizer boundary')
            if [m.get('step') for m in self.metrics] != list(range(1, self.completed + 1)):
                raise ValueError('Every completed visual update must retain its metric')
            if any(not bool(torch.isfinite(t).all()) for t in self.model.state_dict().values()):
                raise ValueError('Every learned tensor must be finite before saving')
            _validate_adam(self.optimizer.state_dict(), list(self.model.vision.parameters()), self.completed)
            if type(maximum_bytes) is not int or maximum_bytes < 1 or type(minimum_free_bytes) is not int or minimum_free_bytes < 0:
                raise ValueError('Explicit complete snapshot cap and reserve required')
            return fingerprint(dict(identity=self.identity, model=self.model.state_dict(),
                optimizer=self.optimizer.state_dict(), metrics=self.metrics, sampler=self.sampler_state(self.completed),
                path=str(Path(path).resolve()), maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes))
        _same(_validated(preflight))
        rngs = collect_worker_rng(self.device, 4)
        _validated(lambda: [_validate_rng(r, self.device) for r in rngs])
        outcome = [None]
        if self.rank == 0:
            try:
                state = dict(format=FORMAT, identity=deepcopy(self.identity), identity_sha256=self.identity_sha256,
                    config=asdict(self.model.config), model=_cpu(self.model.state_dict()),
                    tokenizer_sha256=self.sources['tokenizer_sha256'], optimizer=_cpu(self.optimizer.state_dict()),
                    world_size=4, worker_rng_states=rngs, steps=self.completed, planned_steps=self.steps,
                    sampler_state=self.sampler_state(self.completed), metrics=deepcopy(self.metrics),
                    candidate_promoted=False, goal_achieved=False)
                state['state_sha256'] = fingerprint(state)
                atomic_checkpoint(path, state, maximum_bytes, minimum_free_bytes)
                outcome[0] = dict(checkpoint=str(path), checkpoint_sha256=file_hash(path), steps=self.completed)
            except Exception as error:
                outcome[0] = dict(error=type(error).__name__ + ': ' + str(error))
        dist.broadcast_object_list(outcome, src=0)
        if 'error' in outcome[0]:
            raise ValueError('Collective visual snapshot failed: ' + outcome[0]['error'])
        return outcome[0]

    def restore(self, path, expected_checkpoint_sha256):
        def preflight():
            self._boundary(full=True)
            if self.completed or self.metrics or any(p.grad is not None for p in self.model.parameters()):
                raise ValueError('Restore into a fresh exact own-base kernel')
            binding = FileBindings({str(path): expected_checkpoint_sha256})
            state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            content = {k: v for k, v in state.items() if k != 'state_sha256'}
            if (state.get('format') != FORMAT or state.get('identity') != self.identity
                    or state.get('identity_sha256') != self.identity_sha256 or state.get('config') != asdict(self.model.config)
                    or state.get('tokenizer_sha256') != self.sources['tokenizer_sha256']
                    or state.get('world_size') != 4 or state.get('planned_steps') != self.steps
                    or state.get('candidate_promoted') is not False or state.get('goal_achieved') is not False
                    or state.get('state_sha256') != fingerprint(content)):
                raise ValueError('Exact original complete visual checkpoint required')
            step = state['steps']
            if type(step) is not int or not 0 < step <= self.steps or state['sampler_state'] != self.sampler_state(step):
                raise ValueError('Exact original complete visual source cursors required')
            if [m.get('step') for m in state['metrics']] != list(range(1, step + 1)):
                raise ValueError('Complete original visual metric history required')
            current, saved = self.model.state_dict(), state['model']
            if set(current) != set(saved) or any(not isinstance(saved[n], torch.Tensor) or saved[n].shape != t.shape
                    or saved[n].dtype != t.dtype or not bool(torch.isfinite(saved[n]).all()) for n, t in current.items()):
                raise ValueError('Every original finite exact-shape/dtype learned tensor required')
            if self._frozen_digest(saved) != self.frozen_sha256 or not torch.equal(saved['embedding.weight'], saved['lm_head.weight']):
                raise ValueError('Frozen language/critic and tied embeddings must remain exact')
            _validate_adam(state['optimizer'], list(self.model.vision.parameters()), step)
            group = state['optimizer']['param_groups'][0]
            if (tuple(group['betas']) != (.9, .999) or group['eps'] != 1e-8
                    or group['weight_decay'] != self.options['weight_decay'] or group['lr'] != state['metrics'][-1]['learning_rate']):
                raise ValueError('Original vision AdamW hyperparameters required')
            if len(state['worker_rng_states']) != 4:
                raise ValueError('Every worker needs its exact original RNG state')
            for rng in state['worker_rng_states']:
                _validate_rng(rng, self.device)
            binding.check(full=True)
            return state
        state = _validated(preflight)
        _same(state['state_sha256'])
        def commit():
            self.model.load_state_dict(state['model'], strict=True)
            self.optimizer.load_state_dict(state['optimizer'])
            restore_rng(state['worker_rng_states'][self.rank], self.device)
            self.completed, self.metrics = state['steps'], deepcopy(state['metrics'])
        _validated(commit)
        _same(fingerprint(dict(model=self.model.state_dict(), optimizer=self.optimizer.state_dict(), metrics=self.metrics)))
        return self.completed
