"""Exact native-data protocol backed by the separately recovered source chain.

This loader never upgrades the original failed admission. Both new native
control passes, full indexed readback, a closed data-only receipt and all
original whole-source targets are required before the object is returned.
"""
import json
from pathlib import Path

from .active_execution_input_bindings_v1 import merge_bindings
from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .exact_waiting_process_retirement_v1 import terminal
from .expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from .indexed_gzip_records_v1 import Records
from .native_code_execution_oracle_v2 import NativeCodeExecutionOracle
from .native_suffix_train_data_v1 import NativeSuffixTrainData, FORMAT, SCOPE, LEVELS
from .recovered_legacy_suffix_catalogue_v1 import load_recovered_legacy_catalogue

LOADER = 'own-recovered-complete-native-suffix-data-loader-v3'


def check_admission(admitted, manifest, receipt, owner, *, admission_sha256, manifest_sha256):
    if (receipt.get('complete') is not True or receipt.get('data_only_training_admitted') is not True
            or receipt.get('admission_sha256') != admission_sha256 or admitted.get('owner') != owner
            or admitted.get('source_loader') != LOADER or admitted.get('training_admitted') is not True
            or admitted.get('admitted_role') != 'train' or admitted.get('scope') != SCOPE
            or admitted.get('runtime_or_GPU_admission_not_granted') is not True
            or admitted.get('no_neural_model_GPUcontext_or_lease_started') is not True
            or admitted.get('model_updates') != 0 or admitted.get('manifest_sha256') != manifest_sha256
            or admitted.get('original_sources') != 3721 or admitted.get('all_native_views') != 14884
            or admitted.get('each_level_original_context_tokens') != 2287522
            or admitted.get('each_level_original_response_EOS_targets') != 584218
            or admitted.get('actual_complete_indexed_records_freshly_read_back') != 14884
            or admitted.get('full_source_views_per_epoch') != 14884
            or admitted.get('actual_global_updates_per_epoch') != 3724
            or admitted.get('actual_single_source_three_empty_rank_training_steps_per_epoch') != 4
            or type(admitted.get('native_execution_eligible_views')) is not int
            or type(admitted.get('CE_only_views')) is not int
            or not 0 <= admitted['native_execution_eligible_views'] <= 14884
            or admitted['native_execution_eligible_views'] + admitted['CE_only_views'] != 14884
            or manifest.get('format') != FORMAT or manifest.get('records') != 14884
            or manifest.get('original_sources') != 3721 or manifest.get('original_levels') != list(LEVELS)
            or manifest.get('separate_recovered_source_transport_v3') is not True
            or manifest.get('all_original_failed_and_CE_only_views_retained') is not True
            or manifest.get('source_training_admitted') is not False):
        raise ValueError('Only the exact complete recovered original-source data-only receipt may be loaded')


def load_recovered_native_suffix_data(root, *, purpose='optimization'):
    root = Path(root).resolve()
    folder = root / 'data/native-human-code-suffix-v3'
    completed = root / 'reports/native-human-code-suffix-data-admission-CPU-v3'
    required = [folder / 'admission.json', folder / 'manifest.json', completed / 'complete.json', completed / 'plan.json']
    if (purpose not in ['optimization', 'capacity'] or not all(p.is_file() and not p.is_symlink() for p in required)
            or (completed / 'failure.json').exists()):
        raise ValueError('Separate complete recovered native-data admission is still required')
    def read(path): return json.loads(path.read_text())
    admitted, manifest, receipt, plan = [read(p) for p in required]
    owner = plan['owner']
    check_admission(admitted, manifest, receipt, owner,
        admission_sha256=file_hash(required[0]), manifest_sha256=file_hash(required[1]))
    if not terminal(owner):
        raise ValueError('The exact complete recovered native-data publisher must be terminal')
    original = root / 'reports/native-human-code-suffix-data-admission-CPU-v1'
    failure = read(original / 'failure.json')
    if (failure['owner'] != dict(pid=167672, ticks='56126055') or not terminal(failure['owner'])
            or (original / 'complete.json').exists() or plan.get('original_failed_admission_preserved') != failure):
        raise ValueError('The exact original failed admission must remain failed and preserved')
    previous = root / 'reports/native-human-code-suffix-data-admission-CPU-v2'
    previous_failure = read(previous / 'failure.json')
    if (previous_failure['owner'] != dict(pid=188617, ticks='57073359') or not terminal(previous_failure['owner'])
            or (previous / 'complete.json').exists() or plan.get('previous_failed_admission_preserved') != previous_failure):
        raise ValueError('The exact previous failed native-data admission must remain failed and preserved')
    files = dict(admitted['source_files_sha256'])
    for p in [Path(__file__).resolve(), original / 'failure.json', previous / 'failure.json']:
        if files.get(str(p)) != file_hash(p):
            raise ValueError('The recovered loader and original failure must be explicitly source-bound')
    merge_bindings(files, {str(p): file_hash(p) for p in required})
    for name, digest in [('native-views.records.gz', manifest['pack_sha256']), ('native-views.index', manifest['index_sha256'])]:
        p = folder / name
        if files.get(str(p)) != digest or file_hash(p) != digest:
            raise ValueError('Every complete indexed native record and original offset must retain its exact bytes')
    # Retain the exact immutable V1 data ABI used by the verified distributed
    # stage. Its constructor is not patched, and its failed paths stay failed.
    result = NativeSuffixTrainData.__new__(NativeSuffixTrainData)
    result.root, result.folder, result.source_files_sha256 = root, folder, files
    result.guard = FileBindings(files)
    result.corpus = ExpandedExecutionCorpus(root)
    result.oracle = NativeCodeExecutionOracle(result.corpus)
    result.catalogue = load_recovered_legacy_catalogue(result.oracle)
    result.records = Records(folder / 'native-views.records.gz', folder / 'native-views.index', count=14884)
    try:
        result.purpose, result.full_source_data_admitted = purpose, True
        result.admission_sha256, result.manifest_sha256 = file_hash(required[0]), file_hash(required[1])
        result.slots, result._information = list(range(14884)), []
        with result.oracle.checked_source_batch():
            for position in result.slots:
                result._information.append(result._validate(result.records.read(position)[1], position))
        if (sum(i['native_execution_eligible'] for i in result._information) != admitted['native_execution_eligible_views']
                or sum(not i['native_execution_eligible'] for i in result._information) != admitted['CE_only_views']
                or any(sum(i['original_context_tokens'] for i in result._information if i['level'] == level) != 2287522
                    or sum(i['original_response_EOS_targets'] for i in result._information if i['level'] == level) != 584218
                    for level in LEVELS)):
            raise ValueError('All complete original contexts, answer/EOS masks and actual execution dispositions must remain')
        result.guard.check(full=True)
    except BaseException:
        result.records.close()
        raise
    return result
