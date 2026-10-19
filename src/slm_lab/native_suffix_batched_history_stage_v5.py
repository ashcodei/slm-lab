"""Same complete indexed validation inside an actual checked source batch.

Every record is still read, byte-checked and semantically reconstructed.
Source bytes and the own reference are checked at both batch boundaries;
record reads cannot recursively repeat the full source check inside that
same owning-thread batch. Only resolved immutable paths are cached locally.
No historical neural observations or validation verdicts are cached.
"""
from contextlib import contextmanager
from pathlib import Path
import threading

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .native_suffix_indexed_stage_v4 import NativeSuffixIndexedStageV4
from .parallel_function_repair_stage_v1 import value_digest

import hashlib
import json

from .public_token_suffix_curriculum_v1 import view, descriptor
from .qualified_native_token_suffix_episode_v1 import FORMAT as EPISODE, PROOF

PROTOCOL = 'own-native-history-complete-records-checked-source-boundary-batch-v5'


class NativeSuffixBatchedHistoryStageV5(NativeSuffixIndexedStageV4):
    def __init__(self, *args, **kwargs):
        self._shared_history_batch = None
        super().__init__(*args, **kwargs)
        error = None
        try:
            self.batched_history_source = Path(__file__).resolve()
            if self.sources.get(str(self.batched_history_source)) != file_hash(self.batched_history_source):
                raise ValueError('The exact checked-history batch source must be explicitly bound')
        except Exception as exc: error = repr(exc)
        self.coordinated(error, 'checked history source-batch construction')

    def binding(self):
        return dict(super().binding(), source_checked_history_batch_protocol=PROTOCOL)

    def check_history_reference(self):
        super().check_history_reference()
        if (hasattr(self, 'batched_history_source')
                and self.sources.get(str(self.batched_history_source)) != file_hash(self.batched_history_source)):
            raise ValueError('The exact source-batch implementation changed')

    def history_guard(self):
        context = self._shared_history_batch
        if context is not None:
            caller = threading.get_ident()
            if context['thread'] != caller or self._validation_owner != caller:
                raise ValueError('Another thread cannot borrow the actual checked source-validation batch')
            return
        super().history_guard()

    def source_map_identity(self):
        return value_digest([(str(path), digest) for path, digest in self.data.guard.files.items()])

    @contextmanager
    def checked_validation_sources(self):
        if self._validation_owner is not None:
            with super().checked_validation_sources(): yield
            return
        self.check_history_reference()
        try:
            with super().checked_validation_sources():
                if type(self.data.guard) is not FileBindings:
                    raise ValueError('The original complete actual FileBindings source guard is required')
                initial = self.source_map_identity()
                self._shared_history_batch = dict(thread=threading.get_ident(), source_identity=initial,
                    known_sources={p.resolve(): digest for p, digest in self.data.guard.files.items()},
                    proof_paths={})
                try: yield
                finally:
                    if self.source_map_identity() != initial:
                        raise ValueError('The full actual source-binding map changed inside validation')
                    self._shared_history_batch = None
        finally:
            self._shared_history_batch = None
            self.check_history_reference()

    def resolved_proof_paths(self, proof):
        context = self._shared_history_batch
        if (context is None or context['thread'] != threading.get_ident()
                or self._validation_owner != context['thread']):
            raise ValueError('Only an actual owning checked batch may resolve recorded source proofs')
        key = tuple(proof['source_files_sha256'].items())
        if key not in context['proof_paths']:
            context['proof_paths'][key] = [(Path(name).resolve(), digest) for name, digest in key]
        return context['proof_paths'][key]


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
        known = dict(self._shared_history_batch['known_sources'])
        known.update(self._extra_source_files)
        extra = {}
        for path, digest in self.resolved_proof_paths(proof):
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
