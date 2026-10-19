"""Batch recorded-source validation without rebuilding complete file guards.

Actual code sampling/execution/gradients use the untouched version1 kernel and
qualified episodes. Recorded observations reconstruct the same exact source,
control proof, action/EOS boundary, worker program and reward semantics. The
whole source guard is checked at both ends of each explicit validation batch.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import threading

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .execution_group_controlled_v2 import advantages
from .parallel_native_suffix_stage_v1 import ParallelNativeSuffixStage, CE_ONLY, KERNEL, private_seeds
from .public_token_suffix_curriculum_v1 import view, descriptor
from .qualified_native_token_suffix_episode_v1 import FORMAT as EPISODE, PROOF

PROTOCOL = 'native-suffix-records-exact-source-checked-batch-v2'


class NativeSuffixValidationStageV2(ParallelNativeSuffixStage):
    def __init__(self, *args, **kwargs):
        self._validation_owner = None
        self._extra_source_guards = []
        self._extra_source_files = {}
        super().__init__(*args, **kwargs)

    def binding(self):
        return dict(super().binding(), recorded_source_validation=PROTOCOL)

    def intact(self):
        if self._validation_owner is not None:
            raise ValueError('No source validation batch may leak across a neural/optimizer boundary')
        super().intact()

    @contextmanager
    def checked_validation_sources(self):
        caller = threading.get_ident()
        if self._validation_owner is not None:
            if self._validation_owner != caller:
                raise ValueError('An independent thread cannot borrow the main source-validation batch')
            yield
            return
        self.data.guard.check()
        for guard in self._extra_source_guards: guard.check()
        with self.data.oracle.checked_source_batch():
            self._validation_owner = caller
            try:
                yield
            finally:
                self._validation_owner = None
                for guard in self._extra_source_guards: guard.check()
                self.data.guard.check()

    def validation_source(self, position):
        if self._validation_owner != threading.get_ident():
            raise ValueError('Reconstruction requires an actual checked source-validation batch')
        info = self.data.information(position)
        record = self.data.row(position)
        proof = record['CPU_control_proof'] if self.mode == 'mechanics' else record['producer']['native_profile_control_proof']
        if (proof.get('format') != PROOF or proof.get('complete') is not True
                or proof.get('scope') != ('CPU-mechanics' if self.mode == 'mechanics' else 'complete-source-catalogue')
                or proof.get('source_training_admitted') is not False or proof.get('production_RL_admitted') is not False
                or proof.get('model_generations') != 0 or proof.get('model_updates') != 0):
            raise ValueError('The complete original native controls and their exact scope must remain')
        source, level = info['original_source_position'], info['level']
        original = self.data.corpus.row(source)
        prepared = view(self.data.corpus.tokenizer, original['original'], *self.data.corpus.sequence(source), level)
        source_info = self.data.oracle.source_information(source)
        if self.data._info(position, source, level, descriptor(prepared), info['native_execution_eligible']) != info:
            raise ValueError('Every original in-memory source descriptor must still reconstruct from its actual native bytes')
        if (proof['source_position'] != source or proof['nominal_missing_non_EOS_tokens'] != level
                or proof['native_view'] != descriptor(prepared) or proof['source_information'] != source_info
                or proof['oracle_binding'] != self.data.oracle.binding() or prepared['original_has_embedded_boundary_tokens']
                or any(t in [0, 1, 3] for t in prepared['complete_original_anchor_ids'][1:-1])):
            raise ValueError('Every original native source/format/mask/horizon and exact raw scorer must reconstruct')
        if self.mode == 'mechanics':
            if self.data.catalogue is not None or proof.get('complete_source_population_qualified') is not False:
                raise ValueError('A CPU fixture cannot become complete-source eligible')
            if proof != json.loads(self.data._CPU_path.read_text()):
                raise ValueError('The actual immutable CPU control file and in-memory record differ')
            control_sha = file_hash(self.data._CPU_path)
            catalogue_sha = None
        else:
            if (self.data.catalogue.oracle is not self.data.oracle
                    or proof['complete_legacy_catalogue_sha256'] != self.data.catalogue.digest):
                raise ValueError('The complete original source catalogue and actual native qualification remain required')
            self.data.catalogue.check_view(source, level, descriptor(prepared))
            control_sha = hashlib.sha256((json.dumps(proof, ensure_ascii=True, allow_nan=False) + '\n').encode()).hexdigest()
            catalogue_sha = self.data.catalogue.digest
        controls = proof['actual_controls']
        if len(controls) != 2:
            raise ValueError('Both actual complete native gold/prefix and original legacy prefix controls are required')
        for row in controls:
            gold, prefix, legacy = row['native_gold'], row['native_prefix_then_EOS'], row['legacy_prefix_then_EOS']
            if (gold['reward'] != 1. or gold['actual_response_origin'] != 'original_reference_control'
                    or gold['actual_generated_suffix_ids'] != prepared['original_missing_suffix_and_EOS_ids']
                    or gold['actual_supplied_public_reference_ids'] != prepared['public_reference_prefix_ids']
                    or gold['complete_actual_response'] != original['original']['response'] or not gold['actual_stopped_on_EOS']
                    or gold['rendered_response']['exact_execution_program_sha256'] != source_info['original_execution_program_sha256']
                    or not gold['all_observed_worker_program_bytes_equal_complete_native_program']
                    or prefix['reward'] is None or not 0 <= prefix['reward'] < 1
                    or prefix['actual_generated_suffix_ids'] != [2]
                    or prefix['actual_supplied_public_reference_ids'] != prepared['public_reference_prefix_ids']
                    or prefix['complete_actual_response'] != prepared['public_reference_prefix_text']
                    or legacy.get('reward') is None or not 0 <= legacy['reward'] < 1):
                raise ValueError('Every retained real control and nontrivial prefix rejection remains required')
        # The data guard has already checked every matching source byte. New
        # proof-only files get their own full initial SHA and boundary checks.
        known = {p.resolve(): h for p, h in self.data.guard.files.items()}
        known.update(self._extra_source_files)
        extra = {}
        for name, digest in proof['source_files_sha256'].items():
            path = Path(name).resolve()
            if path in known:
                if known[path] != digest:
                    raise ValueError('An existing complete data/control binding conflicts')
            else: extra[path] = digest
        if extra:
            self._extra_source_guards.append(FileBindings(extra))
            self._extra_source_files.update(extra)
        binding = dict(format=EPISODE, scope=proof['scope'], source_position=source, level=level,
            exact_native_view=descriptor(prepared), native_control_proof_sha256=control_sha,
            oracle_binding=self.data.oracle.binding(), complete_legacy_catalogue_sha256=catalogue_sha,
            unconditional_original_CE_targets=source_info['complete_original_response_EOS_targets'],
            complete_original_action_horizon=prepared['full_native_action_budget'],
            source_training_eligibility_only=True, production_RL_admitted=False, unassisted_capability_score=False)
        return prepared, binding

    def validate_kernel(self, row, position, step):
        with self.checked_validation_sources(): self._validate_kernel(row, position, step)

    def validate_record(self, row, index):
        with self.checked_validation_sources(): super().validate_record(row, index)

    def validate_records(self, rows):
        with self.checked_validation_sources(): super().validate_records(rows)

    def validate_delivery(self, source_position, prepared, actions, outcome, horizon):
        if (not isinstance(actions, list) or not 1 <= len(actions) <= horizon
                or any(type(x) is not int or not 0 <= x < 32768 for x in actions)
                or 2 in actions[:-1] or actions[-1] != 2 and len(actions) != horizon):
            raise ValueError('Every actual native action must end at first EOS or its exact declared horizon')
        prefix = prepared['public_reference_prefix_ids']
        stopped = actions[-1] == 2
        body = actions[:-1] if stopped else actions
        response = self.data.corpus.tokenizer.decode(prefix + body, skip_special_tokens=False)
        raw = self.data.corpus.tokenizer.decode(prefix + actions, skip_special_tokens=False)
        if (outcome.get('actual_generated_suffix_ids') != actions or outcome.get('actual_supplied_public_reference_ids') != prefix
                or outcome.get('actual_stopped_on_EOS') != stopped or outcome.get('actual_delivered_action_budget') != horizon
                or outcome.get('actual_response_origin') != 'own_policy' or outcome.get('complete_actual_response') != response
                or outcome.get('actual_response_with_native_control_tokens_preserved') != raw):
            raise ValueError('Complete native actions, declared code envelope, origin and EOS boundary must reconstruct')
        reward = outcome.get('reward')
        if reward is not None and (type(reward) not in [int, float] or not math.isfinite(reward) or not 0 <= reward <= 1):
            raise ValueError('Only finite actual public execution rewards or explicit abstentions are valid')
        if any(t in [0, 1, 3] for t in actions):
            if reward != 0 or outcome.get('status') != 'invalid_generated_native_control_token':
                raise ValueError('Emitted invalid native controls cannot earn reward')
            return
        rendered = self.data.oracle.render(source_position, response)
        if outcome.get('rendered_response') != rendered:
            raise ValueError('The complete actual worker program must follow its declared exact native source envelope')
        if not rendered['valid']:
            if reward != 0 or outcome.get('status') != rendered['status']:
                raise ValueError('An invalid declared complete code envelope cannot earn reward')
            return
        actual = outcome.get('actual_original_oracle_observation')
        if not isinstance(actual, dict):
            raise ValueError('The actual original bounded oracle observation cannot be omitted')
        original = actual.get('reward')
        if type(original) not in [float, int] or not math.isfinite(original) or not 0 <= original <= 1:
            raise ValueError('Every actual original oracle reward must remain finite and bounded')
        informative = actual.get('informative') is not False and actual.get('supported') is not False
        expected = (original if stopped else min(original, .9)) if informative else None
        if reward != expected or outcome.get('status') != actual.get('status'):
            raise ValueError('Actual finite-source reward, abstention and native EOS credit must reconcile')
        observed = actual.get('actual_execution', actual.get('raw_candidate_execution'))
        if original > 0 and observed is None:
            raise ValueError('Positive original reward requires retained actual worker program/output observations')
        if observed is not None:
            actual_hash = observed.get('actual_program_sha256', observed.get('original_program_sha256', observed.get('program_sha256')))
            if actual_hash != rendered['exact_execution_program_sha256']:
                raise ValueError('The recorded actual worker program differs from the complete generated native program')

    def _validate_kernel(self, row, position, step):
        info = self.data.information(position)
        self.validate_anchor(row['unconditional_original_CE_anchor'], info)
        if (row.get('profile') != self.profile or row.get('own_reference_binding') != self.reference_binding):
            raise ValueError('Every actual source uses this exact own starting policy and matched profile')
        if not info['native_execution_eligible']:
            if (row.get('format') != CE_ONLY or row.get('complete_original_source') != info
                    or row.get('trajectories') != [] or row.get('actual_group_size') != 0
                    or row.get('no_execution_or_policy_reward_claimed') is not True
                    or row.get('every_original_source_target_and_true_EOS_retained') is not True):
                raise ValueError('Unqualified views retain whole CE, without invented execution trajectories or rewards')
            return
        prepared, expected_binding = self.validation_source(position)
        horizon = self.options['CPU_mechanics_horizon'] or info['full_native_action_budget']
        seeds = private_seeds(self.options['seed'], step, position, self.options['group_size'])
        if (row.get('format') != KERNEL or row.get('episode_binding') != expected_binding
                or row.get('actual_private_seeds') != seeds or row.get('actual_group_size') != len(seeds)
                or row.get('actual_delivered_action_horizon') != horizon
                or row.get('full_original_native_horizon') != info['full_native_action_budget']
                or row.get('CPU_mechanics_horizon') != self.options['CPU_mechanics_horizon']
                or row.get('KL_weight') != self.options['KL_weight'] or row.get('anchor_weight') != self.options['anchor_weight']
                or row.get('cache_full_absolute_bound') != self.options['cache_absolute_bound']
                or row.get('execution_workers') != self.options['execution_workers']
                or row.get('resident_training_inventory') != self.reference.resident_inventory(self.model)
                or row.get('actual_private_global_RNG_unchanged') is not True
                or row.get('all_original_reference_CE_targets_unconditional') is not True):
            raise ValueError('Exact source controls, own reference, private seeds and whole native budgets are required')
        items = row['trajectories']
        sampling = row['complete_batched_sampling_observation']
        if len(items) != len(seeds) or len(sampling['trajectories']) != len(seeds):
            raise ValueError('All actual private attempts and their observations must remain')
        rewards, lengths = [], []
        for i, (item, sampled) in enumerate(zip(items, sampling['trajectories'])):
            emitted = item['actual_generated_token_ids']
            self.validate_delivery(info['original_source_position'], prepared, emitted, item['actual_native_execution'], horizon)
            if (item['sample_position'] != i or item['private_seed'] != seeds[i]
                    or item['original_prompt_ids'] != prepared['native_rollout_prefix_ids']
                    or any(item.get(k) != v for k, v in sampled.items())
                    or item['actual_native_action_count'] != len(emitted) or item['actual_stopped_on_EOS'] != (emitted[-1] == 2)):
                raise ValueError('Every native source/sample/seed/delivery must reconstruct without code extraction')
            cached = item['actual_cached_behavior_logps']
            for key in ['actual_full_forward_behavior_logps', 'actual_gradient_forward_behavior_logps']:
                full = item[key]
                if (len(cached) != len(emitted) or len(full) != len(emitted)
                        or any(type(x) not in [float, int] or not math.isfinite(x) or x > 0 for x in cached + full)
                        or max(abs(a - b) for a, b in zip(cached, full)) > self.options['cache_absolute_bound']):
                    raise ValueError('Every sampled/cached/full/gradient native log probability must agree within the original bound')
            rewards.append(item['actual_native_execution']['reward'])
            lengths.append(len(emitted))
        informative = all(r is not None for r in rewards)
        relative = advantages(rewards) if informative else [0.] * len(rewards)
        if (row['actual_rewards'] != rewards or row['actual_advantages'] != relative
                or row['policy_rewards_informative'] != informative
                or row['policy_gradient_abstained_due_to_missing_verdict'] != (not informative)
                or row['actual_sampled_action_tokens'] != sum(lengths) or not sampling['behavior_rescore_passed']
                or sampling['nonfinite_full_forward_behavior_observed'] or sampling['actual_prefill_batch_sizes'] != [len(seeds)]
                or sampling['actual_decode_batch_sizes'] != [sum(n > k for n in lengths) for k in range(1, max(lengths))]):
            raise ValueError('Every genuine reward/abstention, advantage and complete native action denominator must reconcile')

