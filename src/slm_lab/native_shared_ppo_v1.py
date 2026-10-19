"""Full unrepeated own-prompt PPO using one counted native reward backbone.

FP32 cached actions retain their actual behavior log probabilities. Only the
adapter/norm policy and separate critic learn; the native KL reference, reward
head, vision and all other native values stay fixed. Full trajectories live
in bounded allocated frames, not a growing list in every private checkpoint.
This implements learning mechanics, not demonstrated useful own-output RLHF.
"""
from contextlib import nullcontext
import copy
import math

import torch

from .codenet_checkpoint_v1 import capture_rng, restore_rng
from .codenet_stage_v1 import EpochOrder
from .code_residual_checkpoint_v1 import _cpu, _validate_adam, _validate_rng, _digest
from .dense_ppo_cached_v2 import DenseCachedPPOStage, SAMPLER, validate_cached
from .dense_ppo_v1 import action_outputs, prompt_source, validate_rollouts
from .native_human_reward_v1 import FrozenSharedNativeReward, scope, signature
from .native_shared_ppo_history_v1 import BINDING, History, read_prefix
from .objectives import ppo_loss
from .own_dense_actor_checkpoint_v1 import selected
from .own_dense_actor_v1 import OwnDenseActor
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_data_v2 import WholeHumanData
from .whole_human_sft_evidence_v1 import tensor_digest

FORMAT = 'own-native-shared-reward-full-prompt-PPO-Adam-RNG-history-tip-v1'
OPTIONS = dict(seed=999171, steps=42893, prompts_per_update=2, max_new_tokens=512,
    learning_rate=1e-5, betas=[.9, .95], weight_decay=.01, gradient_clip=1.,
    KL_coefficient=.05, gamma=1., GAE_lambda=.95, clip=.2, value_coefficient=.5)
OPTION_KEYS = set(OPTIONS)


class NativeSharedPPOStage:
    def __init__(self, model, optimizer, data, reward, options, sources, device='cpu', *, allow_mechanics=False):
        if (type(model) is not OwnDenseActor or type(data) is not WholeHumanData
                or (data.family, data.role, data.purpose) != ('preferences', 'train', 'optimization')
                or type(reward) is not FrozenSharedNativeReward or reward.base is not model.base
                or type(options) is not dict or set(options) != OPTION_KEYS
                or any(type(options[k]) is not int or options[k] < 1 for k in
                    ['seed', 'steps', 'prompts_per_update', 'max_new_tokens'])
                or options['prompts_per_update'] < 2 or options['steps'] != math.ceil(len(data)/options['prompts_per_update'])
                or any(not isinstance(options[k], (float, int)) or not math.isfinite(options[k]) or options[k] <= 0
                    for k in ['learning_rate', 'gradient_clip', 'value_coefficient'])
                or not 0 < options['clip'] < 1 or not 0 <= options['gamma'] <= 1 or not 0 <= options['GAE_lambda'] <= 1
                or not math.isfinite(options['KL_coefficient']) or options['KL_coefficient'] < 0
                or not math.isfinite(options['weight_decay']) or options['weight_decay'] < 0
                or not isinstance(options['betas'], list) or len(options['betas']) != 2
                or any(not 0 <= v < 1 for v in options['betas'])
                or not isinstance(sources, dict) or not sources or any(not _digest(v) for v in sources.values())
                or any(k not in sources for k in ['human_reward_bundle_sha256', 'tokenizer_sha256'])):
            raise ValueError('Exact shared native actor/reward, train-only whole prompts and one complete unrepeated PPO epoch required')
        reward.check()
        if any(sources.get(k) != v for k, v in reward.provenance['sources'].items()):
            raise ValueError('Every human reward source and current native encoder must match the policy experiment')
        full, guard = scope(model.base, data, reward.provenance['sources'], allow_mechanics)
        if full and (options != OPTIONS or model.settings['rank'] != 16 or model.counts()['actual_unique'] != 999706442):
            raise ValueError('The full original85786prompt/512action recipe and complete counted native policy are required')
        self.device = torch.device(device)
        if (any(p.dtype != torch.float32 or p.device != self.device or p.grad is not None for p in model.parameters())
                or reward.base.network is not model.base.network):
            raise ValueError('The actual policy/reference/reward must share one FP32 native on the declared device')
        named = selected(model, optimizer, ('actor', 'critic')); group = optimizer.param_groups[0]
        if (optimizer.state or group['lr'] != options['learning_rate'] or tuple(group['betas']) != tuple(options['betas'])
                or group['weight_decay'] != options['weight_decay']):
            raise ValueError('One fresh exact-order actor/critic AdamW and all fixed optimizer settings are required')
        ids = {id(p) for _, p in named}
        if ids & {id(p) for p in reward.parameters()}:
            raise ValueError('No shared native reward/reference parameter may be optimized by the policy')
        if sum(p.numel() for p in {id(p): p for m in [model, reward] for p in m.parameters()}.values()) != model.counts()['actual_unique']:
            raise ValueError('The actual policy/reward union must add no hidden second encoder or learned value')
        # Scan the whole original source before any neural sampling, not only
        # the easy prefix. No prompt truncation, skipped pair or shorter tail.
        maximum = 0
        for position in range(len(data)):
            row = prompt_source(data, position)
            maximum = max(maximum, len(row['prompt_ids']))
        if maximum+options['max_new_tokens']+1 > model.config.max_text_length:
            raise ValueError('Every complete original prompt plus full action/scoring budget must fit without truncation')
        self.model, self.optimizer, self.data, self.reward = model, optimizer, data, reward
        self.options, self.sources, self.full, self.guard = copy.deepcopy(options), dict(sources), full, guard
        self.parameters = [p for _, p in named]
        self.fixed = [(n, p) for n, p in model.named_parameters() if id(p) not in ids]
        self.fixed_signature = signature(p for _, p in self.fixed)
        self.frozen_digest = tensor_digest(model.base.named_parameters())
        self.order = EpochOrder(len(data), options['seed']); self.completed = 0
        self.amp = nullcontext; self.history = None; self.previous_prefix = None; self.failed = False
        self.binding = dict(format=BINDING, sources=dict(sources), planned_updates=options['steps'],
            options=copy.deepcopy(options), policy_counts=model.counts(), reward_counts=reward.counts(),
            reward_provenance=_cpu(reward.provenance), sampler=dict(SAMPLER),
            frozen_reference=dict(kind='same_counted_frozen_current_native_without_policy_adapters',
                all_native_values_sha256=self.frozen_digest),
            optimizer_settings_sha256=value_digest(optimizer.state_dict()['param_groups']),
            full_original_source_population=len(data), maximum_whole_original_prompt_tokens=maximum,
            full_original_recipe=full, optimized_names=[n for n, _ in named])
        self.binding_sha256 = value_digest(self.binding)

    def check(self, *, full=False):
        if (self.failed or self.options != self.binding['options'] or self.sources != self.binding['sources']
                or value_digest(self.binding) != self.binding_sha256
                or (self.data.family, self.data.role, self.data.purpose) != ('preferences', 'train', 'optimization')
                or len(self.data) != self.binding['full_original_source_population']
                or signature(p for _, p in self.fixed) != self.fixed_signature
                or any(p.grad is not None for _, p in self.fixed)
                or value_digest(self.optimizer.state_dict()['param_groups']) != self.binding['optimizer_settings_sha256']):
            raise ValueError('Every original frozen native, source role and fixed PPO recipe must remain unchanged')
        self.reward.check()
        if self.guard is not None: self.guard.check(full=full)
        if full and tensor_digest(self.model.base.named_parameters()) != self.frozen_digest:
            raise ValueError('Every actual frozen native language/vision/reward/auxiliary value must remain exact')

    def positions(self, step):
        if type(step) is not int or not 0 <= step < self.options['steps']:
            raise ValueError('An exact remaining full-source PPO update is required')
        start = step*self.options['prompts_per_update']
        return [self.order.at(i) for i in range(start, min(start+self.options['prompts_per_update'], len(self.data)))]

    def cursor(self, completed):
        if type(completed) is not int or not 0 <= completed <= self.options['steps']:
            raise ValueError('The exact original source cursor must remain inside one full unrepeated epoch')
        draws = min(completed*self.options['prompts_per_update'], len(self.data))
        return dict(completed_updates=completed, original_prompt_population=len(self.data), actual_prompt_draws=draws,
            complete_prompt_passes=draws//len(self.data), next_pass_prompts=draws%len(self.data),
            seed=self.options['seed'], world_size=1, repeated_rounding_prompts=0)

    @torch.no_grad()
    def collect(self, positions):
        self.check()
        # The existing cached collector accepts this exact shared scorer's
        # score(ids) interface. Its numerical sampling/GAE route is unchanged.
        with torch.autocast(device_type=self.device.type, enabled=False):
            return DenseCachedPPOStage._collect(self, positions)

    def start_history(self, directory, budgets, *, checked=lambda: None):
        self.check(full=True)
        if self.history is not None: raise ValueError('Never reopen or replace an active trajectory writer')
        self.history = History(directory, binding=self.binding, budgets=budgets,
            previous=self.previous_prefix, guard=lambda: (checked(), self.check()))
        return self.history

    def validate_record(self, row):
        step = row.get('step')
        positions = self.positions(step-1) if type(step) is int else None
        if row.get('source_positions') != positions:
            raise ValueError('Every actual whole-source PPO position must reconstruct without replay or omissions')
        rows = row.get('rollouts', [])
        validate_rollouts(self.data, positions, self.options, rows, self.reward.provenance,
            self.model.config.vocab_size, self.model.config.max_text_length)
        validate_cached(rows)
        if (row.get('actual_action_tokens') != sum(len(r['response_ids']) for r in rows)
                or row.get('actual_complete_prompt_tokens') != sum(len(r['prompt_ids']) for r in rows)
                or row.get('learning_rate') != self.options['learning_rate']
                or row.get('policy_unique_parameters') != self.model.counts()['actual_unique']
                or row.get('shared_native_reward_counts') != self.reward.counts()
                or row.get('additional_reward_encoder_parameters') != 0
                or row.get('own_sampled_responses_not_new_human_judgments') is not True
                or row.get('shared_reward_in_native_policy_export') is not True
                or not isinstance(row.get('token_weighted_PPO_loss'), (float, int))
                or not math.isfinite(row['token_weighted_PPO_loss'])
                or not isinstance(row.get('gradient_norm_before_clip'), (float, int))
                or not math.isfinite(row['gradient_norm_before_clip']) or row['gradient_norm_before_clip'] < 0):
            raise ValueError('Every actual shared-native trajectory and actor/critic update metric must reconcile')

    def update(self):
        self.check()
        if (self.history is None or self.history.starting+self.history.writer.records != self.completed
                or self.completed >= self.options['steps']):
            raise ValueError('A funded immutable trajectory writer at the exact remaining source cursor is required')
        positions = self.positions(self.completed)
        # Reserve actual success and separate failure blocks before sampling.
        self.history.reserve_next(); rows = None; adam_returned = False
        self.optimizer.zero_grad(set_to_none=True)
        try:
            rows = self.collect(positions); total = sum(len(r['response_ids']) for r in rows)
            self.model.train(); self.model.policy.gradient_checkpointing = True; losses = []
            with torch.autocast(device_type=self.device.type, enabled=False):
                for row in rows:
                    ids = torch.tensor([row['prompt_ids']+row['response_ids']], device=self.device)
                    logps, values = action_outputs(self.model, ids, len(row['prompt_ids']), self.amp)
                    old, advantage, old_value, returns = [torch.tensor(row[k], device=self.device)
                        for k in ['old_logps', 'advantages', 'old_values', 'returns']]
                    loss = ppo_loss(logps, old, advantage, values, old_value, returns,
                        self.options['clip'], self.options['value_coefficient'])*len(logps)/total
                    if not bool(torch.isfinite(loss)): raise ValueError('The real token-weighted PPO loss must be finite')
                    loss.backward(inputs=self.parameters); losses.append(float(loss.detach()))
            if (any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.parameters)
                    or any(p.grad is not None for _, p in self.fixed)):
                raise ValueError('Every actor/critic and no shared frozen native needs finite gradients')
            self.check()
            norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, self.options['gradient_clip'], error_if_nonfinite=True))
            record = dict(step=self.completed+1, source_positions=positions, rollouts=rows, actual_action_tokens=total,
                actual_complete_prompt_tokens=sum(len(r['prompt_ids']) for r in rows), token_weighted_PPO_loss=math.fsum(losses),
                gradient_norm_before_clip=norm, learning_rate=self.options['learning_rate'],
                policy_unique_parameters=self.model.counts()['actual_unique'], shared_native_reward_counts=self.reward.counts(),
                additional_reward_encoder_parameters=0, shared_reward_in_native_policy_export=True,
                own_sampled_responses_not_new_human_judgments=True)
            self.validate_record(record)
            # Preserve complete pre-Adam actions/targets in allocated storage.
            # A receipt alone never certifies that its Adam call succeeded.
            self.history.append(record)
            self.optimizer.step(); adam_returned = True
            self.optimizer.zero_grad(set_to_none=True); self.check()
            self.history.acknowledge(record['step']); self.completed += 1
            return record
        except BaseException as error:
            self.failed = True; self.optimizer.zero_grad(set_to_none=True)
            try:
                self.history.failure(dict(attempted_step=self.completed+1, source_positions=positions,
                    rollouts=rows, exception_type=type(error).__name__, exception=str(error),
                    optimizer_step_returned=adam_returned, completed_policy_update_not_certified=True))
            except BaseException as evidence_error:
                error.add_note('Independent failure preservation also failed: '+repr(evidence_error))
            raise

    def state(self):
        self.check(full=True)
        if self.completed < 1 or self.history is None:
            raise ValueError('Only an actual completed policy update with its entire trajectory prefix may be saved')
        prefix = self.history.prefix()
        if prefix is None or prefix['completed'] != self.completed:
            raise ValueError('Every completed policy update needs its exact immutable history tip')
        adam = _cpu(self.optimizer.state_dict()); _validate_adam(adam, self.parameters, self.completed)
        rng = capture_rng(self.device); _validate_rng(rng, self.device)
        return dict(format=FORMAT, binding=_cpu(self.binding), binding_sha256=self.binding_sha256,
            completed=self.completed, cursor=self.cursor(self.completed), policy=self.model.export(self.sources),
            optimizer=adam, rng=rng, history=prefix, history_budgets=dict(self.history.budgets),
            no_resident_trajectory_history=True, useful_own_output_alignment_proved=False,
            candidate_promoted=False, goal_achieved=False)

    def resume(self, state, *, allow_completed=False, checked=lambda: None):
        # Full trajectory and private neural/RNG validation precedes every
        # live copy. Resuming later starts a NEW branch, never truncates history.
        self.check(full=True); completed = state.get('completed')
        if (state.get('format') != FORMAT or state.get('binding') != self.binding
                or state.get('binding_sha256') != self.binding_sha256
                or type(completed) is not int or not 0 < completed <= self.options['steps']
                or completed == self.options['steps'] and not allow_completed
                or state.get('cursor') != self.cursor(completed)
                or state.get('history', {}).get('completed') != completed
                or state.get('no_resident_trajectory_history') is not True
                or any(state.get(k) is not False for k in ['candidate_promoted', 'goal_achieved', 'useful_own_output_alignment_proved'])):
            raise ValueError('The exact complete shared-native PPO source/reward/cursor binding is required before mutation')
        self.model.validate_export(state['policy'], expected_sources=self.sources)
        _validate_adam(state.get('optimizer'), self.parameters, completed); _validate_rng(state.get('rng'), self.device)
        if state['optimizer']['param_groups'] != self.optimizer.state_dict()['param_groups']:
            raise ValueError('Every original actor/critic Adam hyperparameter must match before recovery')
        result = read_prefix(state['history'], binding=self.binding, budgets=state['history_budgets'],
            validate=self.validate_record, guard=lambda: (checked(), self.check()))
        if result['original_prompt_draws'] != self.cursor(completed)['actual_prompt_draws']:
            raise ValueError('Every original unrepeated prompt must reconcile with the saved policy source cursor')
        named = selected(self.model, self.optimizer, ('actor', 'critic'))
        with torch.no_grad():
            for name, p in named: p.copy_(state['policy']['parts'][name])
        self.optimizer.load_state_dict(_cpu(state['optimizer'])); self.optimizer.zero_grad(set_to_none=True)
        restore_rng(state['rng'], self.device); self.completed = completed
        self.previous_prefix = copy.deepcopy(state['history']); self.history = None
        return result

    def finish(self):
        state = self.state()
        if self.completed != self.options['steps']:
            raise ValueError('Every original train-only prompt must complete its whole unrepeated PPO epoch')
        result = read_prefix(state['history'], binding=self.binding, budgets=state['history_budgets'],
            validate=self.validate_record, guard=lambda: self.check())
        if result['original_prompt_draws'] != len(self.data):
            raise ValueError('Every original prompt, including the small final tail, must be retained exactly once')
        return dict(complete=True, actual_updates=self.completed, original_train_prompts=len(self.data),
            max_new_tokens=self.options['max_new_tokens'], actual_action_tokens=result['actual_action_tokens'],
            full_original_PPO_recipe_complete=self.full, history=state['history'],
            actual_unique_policy_reward_values=self.model.counts()['actual_unique'],
            additional_reward_encoder_parameters=0, useful_own_output_alignment_proved=False, goal_achieved=False)
