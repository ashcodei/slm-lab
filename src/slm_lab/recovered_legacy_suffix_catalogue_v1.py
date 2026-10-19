"""Load the separate, complete recovered legacy controls with explicit provenance.

The immutable V1 catalogue protocol is retained for native episodes. This
factory initializes that exact protocol from the verified recovery, without
changing the failed reader's files or claiming native-profile admission.
"""
import json
from pathlib import Path

from .bounded_gzip_observation_stream_v1 import lines
from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .exact_waiting_process_retirement_v1 import terminal
from .native_code_execution_oracle_v2 import NativeCodeExecutionOracle
from .parallel_function_repair_stage_v1 import value_digest
from .qualified_native_token_suffix_episode_v1 import CompleteLegacySuffixCatalogue


def check_reports(source, recovered, recovery_plan, original_failure):
    if (source.get('complete') is not True or recovered.get('complete') is not True
            or recovered.get('integrity_readback_passed') is not True
            or recovered.get('owner') != recovery_plan.get('owner')
            or recovered.get('source_training_admitted') is not False
            or recovered.get('production_RL_admitted') is not False
            or source.get('source_training_admitted') is not False
            or recovered.get('original_disk_failed_reader') != dict(pid=158779, ticks='55770044')
            or recovered.get('original_failure_preserved') != original_failure
            or original_failure.get('owner') != dict(pid=158779, ticks='55770044')
            or 'Original disk reserve remains unavailable after the bounded wait' not in original_failure.get('error', '')
            or recovered.get('retained_controls_not_relabelled_as_new_executions') is not True
            or recovered.get('every_original_unconditional_CE_context_and_target_retained') is not True
            or recovered.get('no_source_or_level_dropped_for_failure_triviality_abstention_or_disagreement') is not True
            or recovered.get('neural_forwards') != 0 or recovered.get('optimizer_updates') != 0
            or recovered.get('GPUcontexts') != 0):
        raise ValueError('Complete unadmitted original/recovered scopes and actual failed-reader provenance are required')
    a, b = source['actual_counts'], recovered['actual_counts']
    old, new = recovered['actual_retained_counts'], recovered['actual_new_counts']
    if (a['complete_original_sources'] != 3721 or a['full_reference_runs'] != 3721 or a['prefix_control_runs'] != 14884
            or b['complete_original_sources'] != 3721 or b['fresh_full_reference_runs'] != 3721
            or b['fresh_prefix_control_runs'] != 14884 or b['all_native_views_rebuilt'] != 14884
            or not 0 < old['complete_original_sources'] < 3721
            or not 0 < new['complete_original_sources'] < 3721
            or any(old.get(k, 0) + new.get(k, 0) != v for k, v in b.items())
            or any(c['fresh_full_reference_runs'] != c['complete_original_sources']
                or c['fresh_prefix_control_runs'] != 4 * c['complete_original_sources']
                or c['all_native_views_rebuilt'] != 4 * c['complete_original_sources'] for c in [old, new])
            or recovered['all_four_level_metadata_totals'] != {str(level): dict(views=3721,
                original_context_tokens=2287522, original_response_EOS_targets=584218) for level in [4, 16, 64, 0]}):
        raise ValueError('All 3721 actual human sources, 14884 views, 18605 controls and original complete CE targets must remain')


def load_recovered_legacy_catalogue(oracle):
    if type(oracle) is not NativeCodeExecutionOracle or oracle.corpus.role != 'train':
        raise ValueError('The exact original train-source native oracle is required')
    root = oracle.corpus.root
    producer = root / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
    reader = root / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v4'
    old = root / 'reports/full-public-token-suffix-readback-gzip-recovery-CPU-v2'
    def read(path): return json.loads(path.read_text())
    a, b, plan, failure = (read(p) for p in [producer / 'complete.json', reader / 'complete.json',
        reader / 'plan.json', old / 'failure.json'])
    check_reports(a, b, plan, failure)
    if (not terminal(a['owner']) or not terminal(b['owner']) or (reader / 'failure.json').exists()
            or (producer / 'failure.json').exists() or (old / 'complete.json').exists()):
        raise ValueError('Only terminal complete separate recovery may be loaded; original failed scope stays failed')
    # Check every original runtime, source, test, control and recovery binding.
    FileBindings(b['source_files_sha256'])
    catalogue_path = Path(b['actual_view_disposition_file'])
    source_controls, reader_controls = Path(a['complete_control_file']), Path(b['actual_fresh_execution_control_file'])
    expected = {catalogue_path: b['actual_all3721source14884view_dispositions_sha256'],
        source_controls: a['actual_all18605_runtime_controls_sha256'],
        reader_controls: b['actual_all18605fresh_execution_controls_sha256']}
    if any(file_hash(p) != h for p, h in expected.items()):
        raise ValueError('Every complete original and recovered observation must retain its exact bytes')
    # Initialize the unchanged exact V1 protocol from these newly verified
    # inputs. No live class, constructor, oracle or global is monkeypatched.
    result = CompleteLegacySuffixCatalogue.__new__(CompleteLegacySuffixCatalogue)
    result.oracle = oracle
    paths = [producer / 'complete.json', reader / 'complete.json', reader / 'plan.json', old / 'failure.json',
        catalogue_path, source_controls, reader_controls, Path(__file__).resolve()]
    result.files = {str(p): file_hash(p) for p in paths}
    result.guard = FileBindings(result.files)
    result.rows = {}
    for number, (_, row) in enumerate(lines(catalogue_path)):
        source, level = number // 4, [4, 16, 64, 0][number % 4]
        if (source >= 3721 or row['source_position'] != source or row['nominal_missing_non_EOS_tokens'] != level
                or row['corpus_id'] != oracle.corpus.row(source)['id']
                or row['source_training_admitted'] is not False or row['production_RL_admitted'] is not False
                or row['all_original_unconditional_CE_targets_retained'] is not True
                or type(row['bounded_nontrivial_control_evidence']) is not bool):
            raise ValueError('Every recovered source/view must retain its original order, identity, controls and full CE anchor')
        result.rows[(source, level)] = row
    if len(result.rows) != 14884:
        raise ValueError('Every original human source and all four native views are required')
    result.digest = value_digest(result.files)
    return result
