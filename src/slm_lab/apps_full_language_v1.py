"""One complete human-program pass with bounded balanced earlier rehearsal.

Updates the own language backbone, freezing vision/critic. Rehearsal samples
every prepared family but is not a full pass over all earlier source rows.
This is a new data experiment, not the final all-source production stage.
"""
from dataclasses import asdict
import math

import torch
from torch import distributed as dist

from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng
from .code_residual_collective_checkpoint_v2 import _same, _validated, fingerprint
from .code_residual_stage_v3 import _EncodedBatches, replay_field
from .codenet_checkpoint_v1 import restore_rng, save_collective
from .codenet_runner_v1 import FixedReductionModel, average_gradients
from .codenet_stage_v1 import EpochOrder, Objective, exposure_budget, freeze_for_code_stage, objective_gradients
from .recurrent_ablation import state_digest
from .training import Encoder

FORMAT = 'own-complete-APPS-full-language-state-v1'


class HumanProgramEncoder(Encoder):
    def __init__(self, source, tokenizer, config):
        super().__init__(tokenizer, config, source.folder)
        self.source = source

    def sequence(self, row, response_field=None):
        if response_field != 'response' or set(row) != {'human_program_index'}:
            raise ValueError('One explicit whole admitted human-program index required')
        original = self.source.example(row['human_program_index'])
        if (original['role'] != 'train' or original.get('original_human_solution') is not True
                or original.get('execution_verified') is not False or original.get('human_preferences') is not False):
            raise ValueError('Only original human-solution train CE examples may enter this objective')
        ids, labels = original['positive']
        start = next((i for i, v in enumerate(labels) if v != -100), len(ids))
        if (not 2 <= len(ids) <= self.config.max_text_length or ids[0] != 1 or ids[-1] != 2
                or not 1 <= start < len(ids) or labels != [-100] * start + ids[start:]
                or any(type(v) is not int or not 0 <= v < self.config.vocab_size for v in ids)):
            raise ValueError('Every original prompt/program/EOS and contiguous full response mask must remain complete')
        return ids, labels


class FullLanguageStage:
    def __init__(self, raw, optimizer, source, code_encoder, replay, replay_encoder, options, sources, device, amp):
        if not dist.is_initialized() or dist.get_world_size() != 4:
            raise ValueError('Four initialized actual workers required')
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.raw, self.optimizer, self.source = raw, optimizer, source
        self.code_encoder, self.replay, self.replay_encoder = code_encoder, replay, replay_encoder
        self.options, self.sources, self.device, self.amp = options.copy(), sources.copy(), device, amp
        self.wrapped = FixedReductionModel(raw)
        self.code_order = EpochOrder(len(source), options['seed'])
        self.replay_orders = {name: EpochOrder(replay.size(name), options['seed'] + 1000 + i) for i, name in enumerate(replay.families)}
        self.completed, self.metrics = 0, []
        def validate():
            if (set(options) != {'seed', 'code_accumulation', 'replay_accumulation', 'image_size', 'learning_rate', 'weight_decay', 'replay_coefficient', 'gradient_clip'}
                    or any(type(options[k]) is not int or options[k] < 1 for k in ['seed', 'code_accumulation', 'replay_accumulation', 'image_size'])
                    or any(type(options[k]) not in [int, float] or not math.isfinite(options[k]) or options[k] <= 0
                        for k in ['learning_rate', 'weight_decay', 'replay_coefficient', 'gradient_clip'])):
                raise ValueError('Exact finite complete-pass options required')
            raw.config.image_token_count(options['image_size'])
            parameters = [p for p in raw.parameters() if p.requires_grad]
            if (type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1
                    or [id(p) for p in optimizer.param_groups[0]['params']] != [id(p) for p in parameters]
                    or optimizer.param_groups[0]['foreach'] is not False
                    or any(p.requires_grad for n, p in raw.named_parameters() if n.startswith(('vision.', 'value_head.')))
                    or any(not p.requires_grad for n, p in raw.named_parameters() if not n.startswith(('vision.', 'value_head.')))):
                raise ValueError('All and only own language parameters in ordinary explicit AdamW required')
            if any(p.grad is not None for p in raw.parameters()): raise ValueError('Clean optimizer boundary required')
            if any(type(v) is not str or len(v) != 64 for v in sources.values()): raise ValueError('Exact source/plan digests required')
            return exposure_budget(len(source), 1, 4, 1, options['code_accumulation'])
        self.budget = _validated(validate); self.steps = self.budget['optimizer_steps']
        self.frozen_digest = state_digest({n: t for n, t in raw.state_dict().items() if n.startswith(('vision.', 'value_head.'))})
        _same(fingerprint(dict(config=asdict(raw.config), model=state_digest(raw.state_dict()), options=options,
            sources=sources, source_size=len(source), families=[(n, replay.size(n)) for n in replay.families], budget=self.budget)))

    def sampler_state(self, completed):
        draws = completed * self.world * self.options['code_accumulation']
        families = {n: sum(i % len(self.replay.families) == j for i in range(completed))
            for j, n in enumerate(self.replay.families)}
        return dict(updates=completed, code_global_draws=draws, code_epoch_and_cursor=list(divmod(draws, len(self.source))),
            source_population=len(self.source), replay={n: dict(updates=k, global_draws=k*self.world*self.options['replay_accumulation'],
                population=self.replay.size(n)) for n, k in families.items()}, full_prior_source_pass=False)

    def update(self):
        step = self.completed; options = self.options
        def prepare():
            if not 0 <= step < self.steps: raise ValueError('Declared one-pass optimizer budget completed')
            if any(p.grad is not None for p in self.raw.parameters()): raise ValueError('Clean update boundary required')
            family = self.replay.families[step % len(self.replay.families)]; occurrence = step // len(self.replay.families)
            positions = self.code_order.rank_groups(step, self.rank, 4, 1, options['code_accumulation'])
            code_groups = [[dict(human_program_index=i) for i in group] for group in positions]
            replay_positions = self.replay_orders[family].rank_groups(occurrence, self.rank, 4, 1, options['replay_accumulation'])
            replay_groups = [[self.replay.row(family, i) for i in group] for group in replay_positions]
            field = replay_field(family)
            code = _EncodedBatches(self.code_encoder, code_groups, 'response', self.device)
            previous = _EncodedBatches(self.replay_encoder, replay_groups, field, self.device)
            if self.code_encoder.truncated or self.replay_encoder.truncated: raise ValueError('No original clipping permitted')
            return family, occurrence, code_groups, replay_groups, field, code, previous, positions, replay_positions
        family, occurrence, code_groups, replay_groups, field, code, previous, positions, replay_positions = _validated(prepare)
        _same(fingerprint(dict(step=step, family=family, occurrence=occurrence, options=options)))
        self.raw.train(); self.raw.gradient_checkpointing = True
        lr = options['learning_rate'] * min(1., (step+1)/max(1, self.steps//10))
        lr *= .1 + .9*.5*(1+math.cos(math.pi*step/self.steps))
        self.optimizer.param_groups[0]['lr'] = lr
        result = objective_gradients(self.wrapped, [Objective('complete_original_human_program', code, code_groups, 'response', 1.),
            Objective(family, previous, replay_groups, field, options['replay_coefficient'])], self.device, self.amp, world=4)
        average_gradients(self.raw, self.device, 4)
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.raw.parameters() if p.requires_grad], options['gradient_clip'], error_if_nonfinite=True)
        self.optimizer.step(); self.optimizer.zero_grad(set_to_none=True)
        local = dict(code_positions=[g[0] for g in positions], replay_positions=[g[0] for g in replay_positions])
        all_positions = [None]*4; dist.all_gather_object(all_positions, local)
        result.update(step=step+1, learning_rate=lr, gradient_norm_before_clipping=float(norm),
            replay_family=family, replay_occurrence=occurrence, replay_response_field=field, worker_source_positions=all_positions)
        self.completed += 1; self.metrics.append(result)
        return result

    def save(self, path, maximum_bytes, minimum_free_bytes):
        def preflight():
            if not 0 < self.completed <= self.steps or len(self.metrics) != self.completed:
                raise ValueError('Every completed update must have its full metric')
            if any(p.grad is not None for p in self.raw.parameters()): raise ValueError('Save only at a clean optimizer boundary')
            if state_digest({n: t for n, t in self.raw.state_dict().items() if n.startswith(('vision.', 'value_head.'))}) != self.frozen_digest:
                raise ValueError('Original own vision and critic must remain unchanged')
            _validate_adam(self.optimizer.state_dict(), [p for p in self.raw.parameters() if p.requires_grad], self.completed)
            return fingerprint(dict(steps=self.completed, sources=self.sources, sampler=self.sampler_state(self.completed),
                model=state_digest(self.raw.state_dict()), optimizer=fingerprint(self.optimizer.state_dict())))
        _same(_validated(preflight))
        def make(rngs):
            return dict(format=FORMAT, model=_cpu(self.raw.state_dict()), optimizer=_cpu(self.optimizer.state_dict()),
                config=asdict(self.raw.config), stage='sft', training_kind='complete_APPS_human_solution_language_v1',
                tokenizer_sha256=self.sources['tokenizer_sha256'], **{k:v for k,v in self.sources.items() if k != 'tokenizer_sha256'},
                steps=self.completed, planned_steps=self.steps, world_size=4, worker_rng_states=rngs,
                trainable_parameter_names=[n for n,p in self.raw.named_parameters() if p.requires_grad],
                training_options=self.options, sampler_state=self.sampler_state(self.completed), metrics=self.metrics,
                frozen_vision_critic_sha256=self.frozen_digest, truncated_sequences=0, candidate_promoted=False, goal_achieved=False)
        save_collective(path, make, self.device, 4, self.rank, maximum_bytes, minimum_free_bytes)

    def restore(self, state):
        def validate():
            if (state.get('format') != FORMAT or state.get('world_size') != 4
                    or state.get('planned_steps') != self.steps or state.get('training_options') != self.options
                    or state.get('config') != asdict(self.raw.config)
                    or any(state.get(k) != v for k, v in self.sources.items())
                    or state.get('frozen_vision_critic_sha256') != self.frozen_digest
                    or state.get('trainable_parameter_names') != [n for n,p in self.raw.named_parameters() if p.requires_grad]
                    or not 0 < state.get('steps',0) <= self.steps or state.get('truncated_sequences') != 0):
                raise ValueError('Exact original model, language optimizer, source plan and progress required')
            expected = self.raw.state_dict(); actual = state['model']
            if (set(actual) != set(expected) or any(actual[n].shape != t.shape or actual[n].dtype != t.dtype
                    or not bool(torch.isfinite(actual[n]).all()) for n,t in expected.items())
                    or not torch.equal(actual['embedding.weight'], actual['lm_head.weight'])):
                raise ValueError('Complete finite tied own-model tensor inventory required')
            if state_digest({n:t for n,t in actual.items() if n.startswith(('vision.', 'value_head.'))}) != self.frozen_digest:
                raise ValueError('Frozen original image and critic tensors differ')
            _validate_adam(state['optimizer'], [p for p in self.raw.parameters() if p.requires_grad], state['steps'])
            if len(state['worker_rng_states']) != 4: raise ValueError('Every original worker RNG required')
            for rng in state['worker_rng_states']: _validate_rng(rng,self.device)
            if ([m['step'] for m in state['metrics']] != list(range(1,state['steps']+1))
                    or state['sampler_state'] != self.sampler_state(state['steps'])):
                raise ValueError('All actual ordered metrics and complete source cursors required')
            return fingerprint(dict(model=state_digest(actual), optimizer=state['optimizer'], steps=state['steps'],
                sources=self.sources, rngs=state['worker_rng_states']))
        _same(_validated(validate))
        self.raw.load_state_dict(state['model'], strict=True); self.optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['worker_rng_states'][self.rank], self.device)
        self.completed, self.metrics = state['steps'], state['metrics'].copy()
