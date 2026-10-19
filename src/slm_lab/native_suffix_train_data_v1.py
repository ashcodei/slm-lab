"""Complete human source suffix curriculum with a separate data-only admission.

Unreliable execution views keep all original unconditional reference targets.
The explicit nine-presentation CPU fixture repeats one original source; it
never claims nine independent human sources or grants full GPU eligibility.
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import tempfile

from .active_execution_input_bindings_v1 import input_bindings
from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from .exact_waiting_process_retirement_v1 import terminal
from .indexed_gzip_records_v1 import Records
from .native_code_execution_oracle_v2 import NativeCodeExecutionOracle
from .parallel_function_repair_stage_v1 import value_digest
from .qualified_native_token_suffix_episode_v1 import CompleteLegacySuffixCatalogue, QualifiedNativeTokenSuffixEpisode

FORMAT = 'own-complete-human-native-suffix-training-data-v1'
LEVELS = (4, 16, 64, 0)
SCOPE = 'complete_original_human_native_suffix_data_only'


def schedule_source_views(*, epochs=1, width=4, seed=989711):
    """Source-only schedule selection; never training or source admission."""
    if any(type(v) is not int or v < 1 for v in [epochs, width, seed]) or width > 4:
        raise ValueError('Explicit complete epochs and one-to-four actual source replicas required')
    entries = []
    for epoch in range(epochs):
        for j in range(4):
            order = list(range(3721))
            random.Random(seed + 4 * epoch + j).shuffle(order)
            views = [4 * p + j for p in order]
            entries.extend(views[i:i+width] for i in range(0, len(views), width))
    return entries


class NativeSuffixTrainData:
    role = 'train'

    def __init__(self, root, *, purpose='optimization'):
        self.root = Path(root).resolve()
        self.folder = self.root / 'data/native-human-code-suffix-v1'
        path = self.folder / 'admission.json'
        if purpose not in ['optimization', 'capacity'] or not path.is_file():
            raise ValueError('Separate complete original-source data-only admission is required')
        completed = self.root / 'reports/native-human-code-suffix-data-admission-CPU-v1'
        if (not (completed / 'complete.json').is_file() or (completed / 'failure.json').exists()):
            raise ValueError('The exact complete terminal data-only admission owner is required')
        admitted = json.loads(path.read_text())
        receipt = json.loads((completed / 'complete.json').read_text())
        owner = json.loads((completed / 'plan.json').read_text())['owner']
        if (not terminal(owner) or admitted.get('owner') != owner or receipt.get('complete') is not True
                or receipt.get('data_only_training_admitted') is not True or receipt.get('admission_sha256') != file_hash(path)):
            raise ValueError('Every actual complete data-only receipt and original publisher birth must match')
        manifest = json.loads((self.folder / 'manifest.json').read_text())
        if (admitted.get('training_admitted') is not True or admitted.get('admitted_role') != 'train'
                or admitted.get('scope') != SCOPE or admitted.get('runtime_or_GPU_admission_not_granted') is not True
                or admitted.get('manifest_sha256') != file_hash(self.folder / 'manifest.json')
                or admitted.get('original_sources') != 3721 or admitted.get('all_native_views') != 14884
                or admitted.get('each_level_original_context_tokens') != 2287522
                or admitted.get('each_level_original_response_EOS_targets') != 584218
                or manifest.get('format') != FORMAT or manifest.get('records') != 14884
                or manifest.get('original_levels') != list(LEVELS)):
            raise ValueError('Every original source/view/whole target and separate runtime boundary must remain intact')
        self.source_files_sha256 = dict(admitted['source_files_sha256'])
        self.source_files_sha256[str(path)] = file_hash(path)
        self.source_files_sha256.update({str(p): file_hash(p) for p in [completed / 'complete.json', completed / 'plan.json']})
        self.guard = FileBindings(self.source_files_sha256)
        self.guard.check(full=True)
        self.corpus = ExpandedExecutionCorpus(self.root)
        self.oracle = NativeCodeExecutionOracle(self.corpus)
        self.catalogue = CompleteLegacySuffixCatalogue(self.oracle)
        self.records = Records(self.folder / 'native-views.records.gz', self.folder / 'native-views.index', count=14884)
        self.purpose, self.full_source_data_admitted = purpose, True
        self.admission_sha256 = file_hash(path)
        self.manifest_sha256 = file_hash(self.folder / 'manifest.json')
        self.slots = list(range(14884))
        self._information = []
        with self.oracle.checked_source_batch():
            for position in self.slots:
                _, row = self.records.read(position)
                self._information.append(self._validate(row, position))
        self.guard.check(full=True)
        if (sum(i['native_execution_eligible'] for i in self._information) != admitted['native_execution_eligible_views']
                or any(sum(i['original_response_EOS_targets'] for i in self._information if i['level'] == level) != 584218
                    or sum(i['original_context_tokens'] for i in self._information if i['level'] == level) != 2287522 for level in LEVELS)):
            raise ValueError('Every original level must retain every whole source/CEtarget and independently qualified execution view')

    @classmethod
    def CPU_mechanics(cls, root, controls_path):
        result = cls.__new__(cls)
        result.root = Path(root).resolve()
        result.folder = None
        result.corpus = ExpandedExecutionCorpus(result.root)
        result.oracle = NativeCodeExecutionOracle(result.corpus)
        path = Path(controls_path).resolve()
        episode = QualifiedNativeTokenSuffixEpisode(result.oracle, path)
        if episode.scope != 'CPU-mechanics': raise ValueError('Only the explicit one-source CPU control proof is allowed')
        result.catalogue, result.records = None, None
        result.purpose, result.full_source_data_admitted = 'CPU-mechanics', False
        result._CPU_path = path
        result._CPU_proof = json.loads(path.read_text())
        result.source_files_sha256 = {**input_bindings(result.root, result.corpus), **result.oracle.sources, str(path): file_hash(path)}
        result.guard = FileBindings(result.source_files_sha256)
        result.admission_sha256 = None
        result.manifest_sha256 = file_hash(path)
        result.slots = list(range(9))
        result._information = [result._info(position, episode.position, episode.level, result._CPU_proof['native_view'], True)
            for position in result.slots]
        return result

    def __len__(self): return len(self.slots)

    def _info(self, position, source, level, metadata, eligible):
        original = self.corpus.row(source)
        return dict(position=position, original_source_position=source, original_source_id=original['id'],
            kind=original['kind'], level=level, native_execution_eligible=eligible,
            original_context_tokens=metadata['original_context_tokens'], original_response_EOS_targets=metadata['original_response_EOS_targets'],
            original_complete_input_sha256=metadata['original_complete_input_sha256'],
            original_full_CE_labels_sha256=metadata['original_unconditional_labels_sha256'],
            conditional_prefix_tokens=metadata['conditional_rollout_prefix_tokens'],
            supplied_public_reference_tokens=metadata['retained_reference_tokens'],
            full_native_action_budget=metadata['full_native_action_budget'], original_role='train',
            source_hints_are_training_only_not_unassisted_capability=True,
            full_source_data_admitted=self.full_source_data_admitted)

    def _validate(self, record, position):
        source, level = position // 4, LEVELS[position % 4]
        if (record['position'] != position or record['source_position'] != source or record['level'] != level
                or record['producer']['source_position'] != source or record['producer']['level'] != level
                or record['reader']['source_position'] != source or record['reader']['level'] != level):
            raise ValueError('Every exact original source/view position must retain its complete native proof')
        proof, fresh = record['producer']['native_profile_control_proof'], record['reader']
        metadata = proof['native_view']
        if (fresh['actual_closed_producer_proof_sha256'] != value_digest(proof)
                or fresh['independent_native_descriptor'] != metadata or proof['source_position'] != source
                or proof['nominal_missing_non_EOS_tokens'] != level or proof['scope'] != 'complete-source-catalogue'
                or proof['source_information'] != self.oracle.source_information(source)
                or proof['oracle_binding'] != self.oracle.binding() or proof['complete_legacy_catalogue_sha256'] != self.catalogue.digest
                or fresh['all_original_unconditional_CE_targets_retained'] is not True
                or record['producer']['all_original_unconditional_CE_targets_retained'] is not True):
            raise ValueError('Original producer/fresh reader/oracle/source/whole CEview identity must match')
        eligible = fresh['native_profile_eligible_in_both_readers']
        if type(eligible) is not bool or eligible and (proof['complete'] is not True or fresh['all_original_actual_control_semantics_agree'] is not True
                or fresh['actual_fresh_native_gold']['reward'] != 1 or fresh['actual_fresh_native_prefix']['reward'] is None
                or not 0 <= fresh['actual_fresh_native_prefix']['reward'] < 1):
            raise ValueError('Only complete actually nontrivial controls in both native readers grant execution eligibility')
        if metadata != self.catalogue.rows[(source, level)]['exact_native_descriptor']:
            raise ValueError('The whole original complete source mask, prefix and native horizon must remain exact')
        return self._info(position, source, level, metadata, eligible)

    def information(self, position):
        if type(position) is not int or not 0 <= position < len(self): raise ValueError('Only an original train presentation is allowed')
        return copy.deepcopy(self._information[position])

    def row(self, position):
        info = self.information(position)
        if self.purpose == 'CPU-mechanics': return dict(information=info, CPU_control_proof=copy.deepcopy(self._CPU_proof))
        _, record = self.records.read(position)
        if self._validate(record, position) != info: raise ValueError('An original admitted source/view record changed')
        return record

    def batch(self, position, device): return self.corpus.batch(self.information(position)['original_source_position'], device)

    @contextmanager
    def episode(self, position):
        self.guard.check()
        info = self.information(position)
        if self.purpose == 'CPU-mechanics':
            yield QualifiedNativeTokenSuffixEpisode(self.oracle, self._CPU_path)
        elif not info['native_execution_eligible']:
            self.row(position)
            yield None  # Caller still trains every complete original CEtarget.
        else:
            proof = self.row(position)['producer']['native_profile_control_proof']
            with tempfile.TemporaryDirectory(prefix='own-admitted-native-source-') as directory:
                path = Path(directory) / 'closed-proof.json'
                path.write_text(json.dumps(proof, ensure_ascii=True, allow_nan=False) + '\n')
                yield QualifiedNativeTokenSuffixEpisode(self.oracle, path, catalogue=self.catalogue)
        self.guard.check()

    def complete_schedule(self, *, epochs=1, width=4, seed=989711):
        if any(type(v) is not int or v < 1 for v in [epochs, width, seed]) or width > 4:
            raise ValueError('Explicit complete epochs and one-to-four actual source replicas required')
        if self.purpose != 'CPU-mechanics': return schedule_source_views(epochs=epochs, width=width, seed=seed)
        entries = []
        for epoch in range(epochs):
            order = list(self.slots)
            random.Random(seed + epoch).shuffle(order)
            entries.extend(order[i:i+width] for i in range(0, len(order), width))
        return entries

    def budget(self, entries):
        if (not isinstance(entries, list) or not entries
                or any(not isinstance(e, list) or not 1 <= len(e) <= 4 or len(e) != len(set(e)) for e in entries)):
            raise ValueError('Nonempty original source groups without duplicate presentation slots are required')
        rows = [self.information(p) for e in entries for p in e]
        return dict(updates=len(entries), source_presentations=len(rows),
            distinct_original_human_sources=len({i['original_source_position'] for i in rows}),
            complete_original_context_tokens=sum(i['original_context_tokens'] for i in rows),
            original_response_EOS_targets=sum(i['original_response_EOS_targets'] for i in rows),
            full_native_generation_horizon_sum=sum(i['full_native_action_budget'] for i in rows),
            execution_eligible_presentations=sum(i['native_execution_eligible'] for i in rows),
            CE_only_presentations=sum(not i['native_execution_eligible'] for i in rows),
            actual_single_source_three_empty_rank_steps=sum(len(e) == 1 for e in entries))

    def identity(self, entries):
        self.budget(entries)
        return value_digest(dict(format=FORMAT, purpose=self.purpose, manifest_sha256=self.manifest_sha256,
            admission_sha256=self.admission_sha256, full_source_data_admitted=self.full_source_data_admitted,
            complete_source_view_schedule=entries))
