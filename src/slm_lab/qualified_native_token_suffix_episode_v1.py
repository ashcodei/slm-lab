"""Exact public-code suffix episodes with closed actual native controls.

Small CPUmechanics proofs never grant full-source or GPUeligibility. A complete
catalogue still requires every original legacy view/fresh control; each native
episode additionally retains its actual gold/prefix controls and full CEanchor.
"""
import copy
import json
from pathlib import Path

from .bounded_gzip_observation_stream_v1 import lines
from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .native_code_execution_oracle_v2 import NativeCodeExecutionOracle
from .parallel_function_repair_stage_v1 import value_digest
from .public_token_suffix_curriculum_v1 import view, descriptor

FORMAT = 'own-qualified-native-token-suffix-episode-v1'
PROOF = 'own-closed-actual-native-token-suffix-controls-v1'


class CompleteLegacySuffixCatalogue:
    """Full original catalogue, not production admission for the new profile."""
    def __init__(self, oracle):
        if type(oracle) is not NativeCodeExecutionOracle:
            raise ValueError('Exact native public-source oracle required')
        self.oracle = oracle
        root = oracle.corpus.root
        source = root / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2/complete.json'
        reader = root / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v2/complete.json'
        if not source.exists() or not reader.exists():
            raise ValueError('The entire original3721source qualifier and all18605fresh controls are still required')
        a, b = json.loads(source.read_text()), json.loads(reader.read_text())
        if (not a['complete'] or not b['complete'] or not b['integrity_readback_passed']
                or a['actual_counts']['complete_original_sources'] != 3721
                or a['actual_counts']['prefix_control_runs'] != 14884
                or b['actual_counts']['complete_original_sources'] != 3721
                or b['actual_counts']['fresh_full_reference_runs'] != 3721
                or b['actual_counts']['fresh_prefix_control_runs'] != 14884
                or b['actual_counts']['all_native_views_rebuilt'] != 14884
                or a['source_training_admitted'] is not False or b['source_training_admitted'] is not False):
            raise ValueError('The exact complete unadmitted legacy source and independent fresh scopes must remain intact')
        for name, digest in b['source_files_sha256'].items():
            if file_hash(name) != digest: raise ValueError('An original full-catalogue source binding changed')
        catalog = Path(b['actual_view_disposition_file'])
        if file_hash(catalog) != b['actual_all3721source14884view_dispositions_sha256']:
            raise ValueError('Every closed original view disposition must retain its bytes')
        self.files = {str(p): file_hash(p) for p in [source, reader, catalog,
            Path(a['complete_control_file']), Path(b['actual_fresh_execution_control_file'])]}
        self.guard = FileBindings(self.files)
        self.rows = {}
        for number, (_, row) in enumerate(lines(catalog)):
            key = (number // 4, [4, 16, 64, 0][number % 4])
            if key != (row['source_position'], row['nominal_missing_non_EOS_tokens']) or key in self.rows:
                raise ValueError('Every original source and level requires one ordered disposition')
            self.rows[key] = row
        if len(self.rows) != 14884: raise ValueError('Every original source/level disposition is required')
        self.digest = value_digest(self.files)

    def check_view(self, position, level, expected):
        self.guard.check()
        row = self.rows[(position, level)]
        if (row['exact_native_descriptor'] != expected or not row['bounded_nontrivial_control_evidence']
                or row['disposition'] != 'nontrivial_in_both_complete_control_runs'):
            raise ValueError('This exact conditional view lacks complete original nontrivial control evidence; retain its CEanchor separately')


class QualifiedNativeTokenSuffixEpisode:
    def __init__(self, oracle, controls_path, *, catalogue=None):
        if type(oracle) is not NativeCodeExecutionOracle:
            raise ValueError('Exact native public-code oracle required')
        controls_path = Path(controls_path).resolve()
        proof = json.loads(controls_path.read_text())
        if (proof.get('format') != PROOF or proof.get('complete') is not True
                or proof.get('scope') not in ['CPU-mechanics', 'complete-source-catalogue']
                or proof.get('source_training_admitted') is not False or proof.get('production_RL_admitted') is not False
                or proof.get('model_generations') != 0 or proof.get('model_updates') != 0):
            raise ValueError('A closed actual-control proof and explicit unadmitted source scope are required')
        self.oracle, self.corpus = oracle, oracle.corpus
        self.position, self.level, self.scope = proof['source_position'], proof['nominal_missing_non_EOS_tokens'], proof['scope']
        self._view = view(self.corpus.tokenizer, self.corpus.row(self.position)['original'], *self.corpus.sequence(self.position), self.level)
        if (proof['native_view'] != descriptor(self._view) or proof['source_information'] != oracle.source_information(self.position)
                or proof['oracle_binding'] != oracle.binding()
                or self._view['original_has_embedded_boundary_tokens']
                or any(t in [0, 1, 3] for t in self._view['complete_original_anchor_ids'][1:-1])):
            raise ValueError('Every original source/format/native mask/horizon and exact raw scorer must match')
        if self.scope == 'complete-source-catalogue':
            if type(catalogue) is not CompleteLegacySuffixCatalogue or catalogue.oracle is not oracle or proof['complete_legacy_catalogue_sha256'] != catalogue.digest:
                raise ValueError('Actual complete original catalogue and this exact native qualification are required for full-source use')
            catalogue.check_view(self.position, self.level, descriptor(self._view))
        elif catalogue is not None or proof.get('complete_source_population_qualified') is not False:
            raise ValueError('A one-source CPUfixture cannot be upgraded into complete-source eligibility')
        controls = proof['actual_controls']
        if len(controls) != 2: raise ValueError('Two actual complete gold/native-prefix/legacy-prefix control observations required')
        source = proof['source_information']
        for row in controls:
            gold, prefix, legacy = row['native_gold'], row['native_prefix_then_EOS'], row['legacy_prefix_then_EOS']
            if (gold['reward'] != 1. or gold['actual_response_origin'] != 'original_reference_control'
                    or gold['actual_generated_suffix_ids'] != self._view['original_missing_suffix_and_EOS_ids']
                    or gold['actual_supplied_public_reference_ids'] != self._view['public_reference_prefix_ids']
                    or gold['complete_actual_response'] != self.corpus.row(self.position)['original']['response']
                    or not gold['actual_stopped_on_EOS']
                    or gold['rendered_response']['exact_execution_program_sha256'] != source['original_execution_program_sha256']
                    or not gold['all_observed_worker_program_bytes_equal_complete_native_program']
                    or prefix['reward'] is None or not 0 <= prefix['reward'] < 1
                    or prefix['actual_generated_suffix_ids'] != [2]
                    or prefix['actual_supplied_public_reference_ids'] != self._view['public_reference_prefix_ids']
                    or prefix['complete_actual_response'] != self._view['public_reference_prefix_text']
                    or legacy.get('reward') is None or not 0 <= legacy['reward'] < 1):
                raise ValueError('Every native gold must pass and both real prefix-only protocols must reject trivial full credit')
        self.catalogue = catalogue
        self._proof = copy.deepcopy(proof)
        files = dict(proof['source_files_sha256'])
        files[str(controls_path)] = file_hash(controls_path)
        self.guard = FileBindings(files)
        self._binding = dict(format=FORMAT, scope=self.scope, source_position=self.position, level=self.level,
            exact_native_view=descriptor(self._view), native_control_proof_sha256=file_hash(controls_path),
            oracle_binding=oracle.binding(), complete_legacy_catalogue_sha256=None if catalogue is None else catalogue.digest,
            unconditional_original_CE_targets=source['complete_original_response_EOS_targets'],
            complete_original_action_horizon=self._view['full_native_action_budget'],
            source_training_eligibility_only=True, production_RL_admitted=False, unassisted_capability_score=False)
        self.check()

    def binding(self): return copy.deepcopy(self._binding)

    def check(self):
        self.oracle.assert_bound()
        self.guard.check()
        actual = view(self.corpus.tokenizer, self.corpus.row(self.position)['original'], *self.corpus.sequence(self.position), self.level)
        if (actual != self._view or self.position != self._binding['source_position'] or self.level != self._binding['level']
                or self.scope != self._binding['scope'] or self.oracle.binding() != self._binding['oracle_binding']):
            raise ValueError('The exact qualified original source/view/oracle/scope changed')
        if self.catalogue is not None: self.catalogue.check_view(self.position, self.level, descriptor(actual))

    def rollout_view(self): return copy.deepcopy(self._view)

    def batch(self, device): return self.corpus.batch(self.position, device)

    def score(self, actions, *, horizon):
        if self.position != self._binding['source_position'] or self.scope != self._binding['scope']:
            raise ValueError('The exact source episode identity changed during sampling')
        return self.oracle.delivery(self.position, self._view['public_reference_prefix_ids'], actions,
            delivered_action_budget=horizon, origin='own_policy')
