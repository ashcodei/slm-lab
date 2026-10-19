"""Recover the exact full code-source scope after a terminal disk-reserve failure.

Wait for the actual old source owner. Never restart it, bind its live journal,
or change an existing neural worker. Retain every completed original byte-LF
control row, then finish all remaining sources using the unchanged old oracle.
The original1GiBraw quota and40GiBother-writer floor remain; gzip reserves one
bounded next record instead of the whole uncompressed quota on every append.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/full-public-token-suffix-qualification-CPU-v1'
OUT = ROOT / 'reports/full-public-token-suffix-qualification-gzip-recovery-CPU-v2'
PARENT = dict(pid=146084, ticks='55111340')
MAX_OUTPUT = 1024**3
RESERVE = 40 * 1024**3


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_record(corpus, position, record, metadata):
    from slm_lab.public_token_suffix_curriculum_v1 import LEVELS, view, descriptor, nontrivial
    row = corpus.row(position)
    ids, labels = corpus.sequence(position)
    if (record['source_position'] != position or record['corpus_id'] != row['id'] or record['kind'] != row['kind']
            or len(record['views']) != 4 or record['original_CE_context_tokens'] != len(ids)
            or record['original_CE_response_EOS_targets'] != len(ids) - row['prefix_positions']
            or record['source_role'] != 'train' or record['source_training_admitted'] is not False
            or record['production_RL_admitted'] is not False or record['counterfactuals_not_student_samples'] is not True
            or not record['every_original_source_and_complete_CE_anchor_retained']):
        raise ValueError('Every retained original source, complete CEtarget and role must reconstruct')
    reference = record['full_original_reference_execution']
    counts = Counter(full_reference_runs=1, full_reference_reward1=int(reference.get('reward') == 1.),
        full_reference_abstentions=int(reference.get('reward') is None), complete_original_sources=1)
    for level, control in zip(LEVELS, record['views']):
        value = view(corpus.tokenizer, row['original'], ids, labels, level)
        actual = control['counterfactual_prefix_then_EOS_execution']
        expected = dict(source_position=position, corpus_id=row['id'], kind=row['kind'], **descriptor(value))
        eligible = (nontrivial(value, reference, actual) if reference.get('reward') is not None and actual.get('reward') is not None else False)
        if (metadata[(position, level)] != expected or control['nominal_missing_non_EOS_tokens'] != level
                or control['native_view'] != descriptor(value) or control['original_prefix_text'] != value['public_reference_prefix_text']
                or control['proposed_nontrivial_bounded_execution_view'] != eligible or control['source_training_admitted'] is not False):
            raise ValueError('Every retained exact4level native view and raw execution disposition must reconstruct')
        counts.update(prefix_control_runs=1, prefix_only_reward1_rejected=int(actual.get('reward') == 1.),
            prefix_control_abstentions=int(actual.get('reward') is None), proposed_nontrivial_views=int(eligible))
    return counts


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread source-only recovery required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import atomic
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    plan = read(SOURCE / 'plan.json')
    if plan['owner'] != PARENT: raise ValueError('The exact original producer birth is required')
    tests = ROOT / 'reports/full-token-suffix-gzip-recovery-v2-tests.log'
    if '3 passed' not in tests.read_text(): raise ValueError('All retained actual prefix and source/mask/reward rejection controls must pass')
    stream_tests = ROOT / 'reports/bounded-gzip-observation-stream-v1-tests.log'
    if '5 passed' not in stream_tests.read_text(): raise ValueError('All lossless LF/quota/guard/interruption controls must pass')
    files = dict(plan['source_files_sha256'])
    extra = [Path(__file__), SOURCE / 'plan.json', tests, stream_tests,
        ROOT / 'src/slm_lab/bounded_gzip_observation_stream_v1.py', ROOT / 'tests/test_full_token_suffix_gzip_recovery_v2.py']
    files.update({str(p): sha(p) for p in extra})
    states = {p: (Path(p).stat().st_dev, Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    def intact():
        if os.sched_getscheduler(0) != os.SCHED_IDLE or any((Path(p).stat().st_dev, Path(p).stat().st_ino,
                Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != v for p, v in states.items()):
            raise ValueError('All exact original source bytes and idle source-only policy must remain unchanged')
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), original_exact_source_owner=PARENT,
        source_files_sha256=files, original_sources=3721, native_views=14884, complete_reference_prefix_controls=18605,
        original_unconditional_CE_contexts=2287522, original_unconditional_response_EOS_targets=584218,
        maximum_total_uncompressed_observation_bytes=MAX_OUTPUT, minimum_other_writer_reserve_bytes=RESERVE,
        output_encoding='lossless_gzip_byte_LF_JSON', original_execution_oracle_and_case_budgets_unchanged=True,
        wait_for_actual_original_owner_terminal_before_any_recovery=True, source_training_admitted=False,
        production_RL_admitted=False, models_constructed=0, neural_forwards=0, GPUcontexts=0, goal_achieved=False))
    try:
        expires = time.monotonic() + 7 * 86400
        while not terminal(PARENT):
            intact()
            if time.monotonic() > expires: raise ValueError('The exact predecessor waiting deadline expired')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact original source owner to become terminal',
                exact_original_owner_live=True, no_journal_consumed_or_neural_worker_changed=True))
            time.sleep(15)
        if (SOURCE / 'complete.json').exists() and not (SOURCE / 'failure.json').exists():
            atomic(OUT / 'not-needed.json', dict(complete=True, original_source_completed_without_recovery=True,
                original_complete_file=str(SOURCE / 'complete.json'), original_complete_sha256=sha(SOURCE / 'complete.json'),
                no_recovery_or_extra_execution_performed=True, goal_achieved=False))
            return
        failure = read(SOURCE / 'failure.json')
        if failure['owner'] != PARENT or 'Original disk reserve remains unavailable after the bounded wait' not in failure['error']:
            raise ValueError('Only the actual preserved terminal disk-reserve failure may use this recovery')
        import torch
        from slm_lab.bounded_gzip_observation_stream_v1 import Writer, RawBudget, lines
        from slm_lab.codenet_checkpoint_v1 import FileBindings
        from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
        from slm_lab.local_source_closure_v2 import python_sources
        from slm_lab.public_token_suffix_curriculum_v1 import LEVELS, view, descriptor, nontrivial
        torch.set_num_threads(1)
        if torch.cuda.is_initialized(): raise ValueError('Source recovery cannot acquire a GPUcontext')
        metadata_path = SOURCE / 'all14884-native-view-descriptors.jsonl'
        previous_journal = SOURCE / 'all3721-reference-and14884-prefix-controls.jsonl'
        md = read(ROOT / 'reports/full-public-token-suffix-metadata-readback-CPU-v1/complete.json')
        if not md['passed'] or md['source_files_sha256'][str(metadata_path)] != sha(metadata_path):
            raise ValueError('The unchanged complete independently rebuilt original metadata is required')
        files.update({str(p): sha(p) for p in [SOURCE / 'failure.json', previous_journal, metadata_path,
            ROOT / 'reports/full-public-token-suffix-metadata-readback-CPU-v1/complete.json',
            *python_sources(ROOT, [Path(__file__)])]})
        guard = FileBindings(files)
        guard.check(full=True)
        atomic(OUT / 'closed-original-source-bindings.json', dict(owner=owner, original_owner_terminal=True,
            original_failure_preserved=failure, source_files_sha256=files))
        corpus = ExpandedExecutionCorpus(ROOT)
        metadata_rows = [r for _, r in lines(metadata_path)]
        metadata = {(r['source_position'], r['nominal_missing_non_EOS_tokens']): r for r in metadata_rows}
        if len(metadata_rows) != 14884 or len(metadata) != 14884:
            raise ValueError('Every complete original metadata view is required')
        def checked():
            intact()
            guard.check()
            if torch.cuda.is_initialized(): raise ValueError('Source recovery cannot create a CUDAcontext')
        def resource_only():
            if os.sched_getscheduler(0) != os.SCHED_IDLE or torch.cuda.is_initialized():
                raise ValueError('Lossless source writes must remain idle and CUDA-free')
        def score(position, response):
            try: return corpus.score(position, response, stopped_on_eos=True)
            except Exception as exc: return dict(reward=None, status='preserved_execution_adapter_abstention', error=repr(exc))
        counts = Counter()
        retained = 0
        prefix_digest = hashlib.sha256()
        budget = RawBudget(MAX_OUTPUT, retained_source_bytes=metadata_path.stat().st_size)
        journal = OUT / 'all3721-reference-and14884-prefix-controls.jsonl.gz'
        with Writer(journal, root=ROOT, budget=budget, reserve_bytes=RESERVE, guard=resource_only) as output:
            for raw, row in lines(previous_journal):
                counts.update(validate_record(corpus, retained, row, metadata))
                output.append_raw(raw)
                prefix_digest.update(raw)
                retained += 1
            if prefix_digest.hexdigest() != sha(previous_journal) or not 0 < retained < 3721:
                raise ValueError('Every closed prior control byte must be retained; no torn or omitted prefix is allowed')
            checked()
            atomic(OUT / 'retained-original-prefix.json', dict(original_sources=retained,
                exact_old_journal_uncompressed_sha256=prefix_digest.hexdigest(), actual_old_counts=dict(counts),
                all_prior_observation_bytes_retained=True, original_failure_not_relabelled_complete=True))
            for position in range(retained, len(corpus)):
                checked()
                row = corpus.row(position)
                ids, labels = corpus.sequence(position)
                reference = score(position, row['original']['response'])
                controls = []
                for level in LEVELS:
                    value = view(corpus.tokenizer, row['original'], ids, labels, level)
                    actual = score(position, value['public_reference_prefix_text'])
                    eligible = (nontrivial(value, reference, actual) if reference.get('reward') is not None and actual.get('reward') is not None else False)
                    controls.append(dict(nominal_missing_non_EOS_tokens=level, native_view=descriptor(value),
                        original_prefix_text=value['public_reference_prefix_text'], counterfactual_prefix_then_EOS_execution=actual,
                        proposed_nontrivial_bounded_execution_view=eligible, source_training_admitted=False))
                record = dict(source_position=position, corpus_id=row['id'], kind=row['kind'], difficulty=row['original'].get('difficulty'),
                    full_original_reference_execution=reference, views=controls, original_CE_context_tokens=len(ids),
                    original_CE_response_EOS_targets=len(ids) - row['prefix_positions'],
                    every_original_source_and_complete_CE_anchor_retained=True, counterfactuals_not_student_samples=True,
                    source_role='train', source_training_admitted=False, production_RL_admitted=False)
                counts.update(validate_record(corpus, position, record, metadata))
                output.append(record)
                if (position + 1) % 16 == 0 or position + 1 == len(corpus):
                    atomic(OUT / 'progress.json', dict(owner=owner, phase='finish all original sources after exact failed predecessor',
                        actual_counts=dict(counts), retained_original_sources=retained, new_actual_sources=position + 1 - retained,
                        original_sources=3721, actual_compressed_output=output.receipt(), source_training_admitted=False, neural_forwards=0))
                if (position + 1) % 64 == 0: output.fsync()
        if counts['complete_original_sources'] != 3721 or counts['prefix_control_runs'] != 14884:
            raise ValueError('Every original source and all4unchanged runtime controls must finish')
        guard.check(full=True)
        atomic(OUT / 'complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), owner=owner,
            source_files_sha256=files, actual_counts=dict(counts), retained_original_sources=retained,
            new_actual_reference_executions=3721-retained, new_actual_prefix_executions=4*(3721-retained),
            actual_metadata_path=str(metadata_path), actual_all14884_native_descriptors_sha256=sha(metadata_path),
            actual_all18605_runtime_controls_sha256=sha(journal), complete_control_file=str(journal),
            lossless_observation_stream=output.receipt(), original_failed_source_owner_and_failure_preserved=True,
            all_original3721sources_and_every_unconditional_CE_target_preserved=True,
            original_full_CE_contexts=2287522, original_full_CE_response_EOS_targets=584218,
            all_failures_abstentions_and_trivial_prefixes_preserved=True,
            source_training_admitted=False, production_RL_admitted=False, student_generations=0, model_updates=0,
            GPUcontexts=0, existing_neural_workers_unchanged=True, goal_achieved=False))
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            source_training_admitted=False, optimizer_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__': main()
