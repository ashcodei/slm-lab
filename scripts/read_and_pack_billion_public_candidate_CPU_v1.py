"""Wait for the full candidate, independently re-fetch sources, then admit data.

No model, optimizer, GPU context or generation is constructed. Every observed
publisher-prefix disposition and accepted whole native sequence is rebuilt.
Nonpilot original Parquets are independently downloaded and byte verified a
second time, one at a time, rather than trusting retained producer row copies.
Admission permits future training; it does not launch or claim any training.
"""
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'reports/public-billion-causal-candidate-CPU-v2'
CANDIDATE = ROOT/'data/public-billion-causal-candidate-v2'
OUT = ROOT/'reports/public-billion-causal-candidate-v2-full-readback-v1'
PACK = ROOT/'data/public-billion-causal-windows-v1'
QUEUE = ROOT/'reports/public-billion-causal-candidate-v2-full-readback-v1-queue.json'
SOURCE_OWNER = dict(pid=3815469, ticks='43593948')
QUOTAS = {'code:Python':200000000, 'code:JavaScript':75000000, 'code:C++':75000000,
    'code:C':50000000, 'code:Go':25000000, 'code:Rust':25000000, 'code:Java':25000000,
    'code:TypeScript':25000000, 'math':250000000, 'web':250000000}
RESERVE = 40*1024**3


def validate_source_receipt(receipt, number, catalogs, positions, previous_family, rebuild):
    source = receipt['source']
    family = source['family']
    order = ['code', 'math', 'web']
    if family not in order or receipt['source_number'] != number:
        raise ValueError('Original source file order and every receipt must remain exact')
    if previous_family is None:
        if family != 'code':raise ValueError('The full original source order starts with code')
    elif family != previous_family:
        if (order.index(family) != order.index(previous_family)+1
                or not rebuild.family_full(previous_family)):
            raise ValueError('No original family or unsatisfied training quota may be skipped')
    if positions[family] >= len(catalogs[family]) or source != catalogs[family][positions[family]]:
        raise ValueError('Every observed source must follow the complete immutable publisher catalog prefix')
    if (receipt['actual_verified_original_file_bytes'] != source['size']
            or receipt['actual_verified_original_file_sha256'] != source['sha256']):
        raise ValueError('Exact pinned whole original publisher bytes required')
    positions[family] += 1
    return family


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or PACK.exists() or QUEUE.exists():
        raise ValueError('Fresh GPU-hidden independent source/causal-window reader required')
    import numpy as np
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    import preflight_public_pretraining_sources_CPU_v1 as transport
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.held_text_overlap_v1 import HeldTextOverlap
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.public_billion_readback_v1 import Rebuilder
    from slm_lab.public_causal_windows_v1 import windows
    source_plan = json.loads((SOURCE/'plan.json').read_text())
    if source_plan['owner'] != SOURCE_OWNER or source_plan['fixed_train_target_quotas'] != QUOTAS:
        raise ValueError('Only the exact corrected complete billion-target source producer is eligible')
    tests = ROOT/'reports/public-causal-windows-and-billion-readback-v1-final-tests.log'
    driver_tests = ROOT/'reports/public-billion-full-reader-driver-v1-tests.log'
    if '7 passed' not in tests.read_text() or '2 passed' not in driver_tests.read_text():
        raise ValueError('Actual complete-target, role, native-corruption and full-source order controls must pass')
    seeds = [Path(__file__), ROOT/'tests/test_public_causal_windows_v1.py',
        ROOT/'tests/test_public_billion_readback_v1.py', ROOT/'tests/test_public_billion_full_reader_driver_v1.py']
    files = {str(p):file_hash(p) for p in [tests, driver_tests, SOURCE/'plan.json', *python_sources(ROOT, seeds)]}
    files.update(source_plan['source_files_sha256'])
    guard = FileBindings(files)
    guard.check(full=True)
    OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    plan = dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, exact_source_owner=SOURCE_OWNER,
        source_files_sha256=files, minimum_original_train_targets=10**9, fixed_train_target_quotas=QUOTAS,
        independently_refetch_each_removed_original_Parquet_and_verify_publisher_LFS=True,
        replay_every_original_prefix_disposition_and_accepted_record=True,
        every_original_text_and_EOS_target_retained_once_in4096_context_windows=True,
        one_token_context_overlap_and_all_BOS_masked=True,
        within_role_cross_document_causal_attention_permitted=True,
        long_document_context_resets_at_window_boundary_not_full_unbounded_context=True,
        no_GPU_neural_model_optimizer_or_live_teacher_loaded=True,
        admission_is_data_only_not_neural_training_or_quality_gain=True,
        all_ten_final_requirements_incomplete=True, goal_achieved=False)
    atomic_json(OUT/'plan.json', plan)
    atomic_json(QUEUE, plan)
    started = time.monotonic()
    try:
        while not ((SOURCE/'complete.json').exists() and terminal(SOURCE_OWNER)):
            if (SOURCE/'failure.json').exists() or terminal(SOURCE_OWNER):
                raise ValueError('The exact full corrected source producer did not complete; no partial corpus is admitted')
            guard.check()
            atomic_json(OUT/'status.json', dict(state='waiting_for_exact_complete_public_source_candidate',
                owner=owner, source_owner=SOURCE_OWNER, model_updates=0, goal_achieved=False))
            time.sleep(10)
        done = json.loads((SOURCE/'complete.json').read_text())
        manifest_path = CANDIDATE/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (done.get('complete') is not True or (SOURCE/'failure.json').exists()
                or done['actual_train_targets'] < 10**9 or done['optimization_admitted'] is not False
                or not done['all_training_quotas_satisfied'] or manifest['fixed_train_target_quotas'] != QUOTAS):
            raise ValueError('Every original candidate source quota must actually finish before full readback')
        files.update(done['source_files_sha256'])
        files.update(manifest['source_files_sha256'])
        files[str(SOURCE/'complete.json')] = file_hash(SOURCE/'complete.json')
        guard = FileBindings(files)
        guard.check(full=True)
        catalog_path = SOURCE/'source-catalogs.json'
        catalogs = json.loads(catalog_path.read_text())
        registry = json.loads((ROOT/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json').read_text())
        if len(registry) != 32953:
            raise ValueError('Every original unchanged declared held stem required')
        overlap = HeldTextOverlap(registry)
        tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer-corpus/tokenizer.json'))
        PACK.mkdir()
        temp = OUT/'transient-original-readback';temp.mkdir()
        transport.OUT = OUT/'transport';transport.OUT.mkdir();transport.RESERVE = RESERVE
        native_folder = CANDIDATE/'whole-native-documents'
        positions = dict(code=0, math=0, web=0)
        previous_family = None
        source_count = 0
        last = time.monotonic()
        with ExitStack() as stack:
            documents_stream = stack.enter_context(gzip.open(native_folder/'whole-original-documents.jsonl.gz', 'rt'))
            documents = (json.loads(line) for line in documents_stream)
            ledger_stream = stack.enter_context(gzip.open(CANDIDATE/'observed-original-prefix-dispositions.jsonl.gz', 'rt'))
            ledger = (json.loads(line) for line in ledger_stream)
            native = {r:stack.enter_context((native_folder/(r+'.tokens.u16')).open('rb')) for r in ['train', 'valid']}
            rebuild = Rebuilder(QUOTAS, tokenizer, overlap, documents, native)
            proof_stream = stack.enter_context((OUT/'independent-original-source-proofs.jsonl').open('x'))
            for line in (CANDIDATE/'original-source-file-receipts.jsonl').open():
                receipt = json.loads(line);source_count += 1
                family = validate_source_receipt(receipt, source_count, catalogs, positions, previous_family, rebuild)
                if rebuild.family_full(family):
                    raise ValueError('The producer must stop a source family immediately after its declared quotas finish')
                previous_family = family
                spec = receipt['source']
                first = catalogs[family][0]
                work = None
                if spec['sha256'] == first['sha256'] and spec['path'] == first['path']:
                    path = ROOT/'data/public-pretraining-source-preflight-v1'/family/'original.parquet'
                else:
                    if shutil.disk_usage(ROOT).free < RESERVE+spec['size']:
                        raise ValueError('Complete independent pinned source must fit above the persistent reserve')
                    work = temp/str(source_count);work.mkdir();transport.DATA = work
                    path, _ = transport.fetch(spec)
                if path.stat().st_size != spec['size'] or file_hash(path) != spec['sha256']:
                    raise ValueError('Independently observed original public bytes differ from the immutable publisher digest')
                parquet = pq.ParquetFile(path)
                prefix = receipt['original_prefix_rows_observed']
                if (type(prefix) is not int or not 0 < prefix <= parquet.metadata.num_rows
                        or parquet.metadata.num_rows != receipt['original_file_total_rows']):
                    raise ValueError('Every original source file population and complete observed prefix required')
                observed = 0;dispositions = Counter()
                for batch in parquet.iter_batches(batch_size=128, use_threads=False):
                    for row in batch.to_pylist():
                        entry = next(ledger, None)
                        if entry is None:
                            raise ValueError('Every independently fetched original prefix row needs its original disposition')
                        dispositions[rebuild.observe(row, spec, observed, source_count, entry)] += 1
                        observed += 1
                        if time.monotonic()-last >= 10:
                            guard.check()
                            atomic_json(OUT/'progress.json', dict(at=datetime.now(timezone.utc).isoformat(),
                                actual_source_files_observed=source_count, actual_original_rows_rebuilt=rebuild.observed,
                                independently_rebuilt_candidate=rebuild.summary(),
                                seconds=time.monotonic()-started, optimization_admitted=False, model_updates=0, goal_achieved=False))
                            last = time.monotonic()
                        if observed == prefix:break
                    if observed == prefix:break
                if (observed != prefix or dict(dispositions) != receipt['source_prefix_dispositions']
                        or (prefix < parquet.metadata.num_rows and not rebuild.family_full(family))):
                    raise ValueError('Every original source prefix, exclusion count and quota stop must independently rebuild')
                proof_stream.write(json.dumps(dict(source_number=source_count, source=spec,
                    actual_independently_observed_original_file_bytes=path.stat().st_size,
                    actual_independently_observed_original_file_sha256=file_hash(path),
                    actual_original_prefix_rows_rebuilt=observed, independently_rebuilt_dispositions=dict(dispositions)))+'\n')
                proof_stream.flush();os.fsync(proof_stream.fileno())
                if work is not None:
                    if work.parent != temp or not work.name.isdecimal():
                        raise ValueError('Never remove another producer or prior source file')
                    shutil.rmtree(work)
            if next(ledger, None) is not None:
                raise ValueError('No unobserved source disposition may remain')
            summary = rebuild.finish()
        if source_count != done['actual_source_files'] or rebuild.observed != done['original_rows_observed']:
            raise ValueError('Every original source file and observed row count must reconcile')
        if dict(rebuild.dispositions) != done['actual_prefix_dispositions']:
            raise ValueError('All actual original accepted/excluded prefix totals must reconcile')
        for name, value in summary.items():
            if done[name] != value or manifest[name] != value:
                raise ValueError('Every original train/valid target, document and context denominator must rebuild: '+name)
        database = sqlite3.connect('file:'+str(native_folder/'exact-original-body-shas.sqlite3')+'?mode=ro', uri=True)
        try:
            count = 0
            for (sha,) in database.execute('SELECT sha FROM documents'):
                if sha.hex() not in rebuild.seen:
                    raise ValueError('Original exact-body deduplication database differs')
                count += 1
            if count != len(rebuild.seen):raise ValueError('Exact original body database population differs')
        finally:
            database.close()
        roles = {}
        for role in ['train', 'valid']:
            starts = np.asarray(rebuild.starts[role], dtype='<u8')
            index = windows(rebuild.offsets[role], starts, 4096)
            np.save(PACK/(role+'.document-starts.npy'), starts, allow_pickle=False)
            np.save(PACK/(role+'.windows.npy'), index, allow_pickle=False)
            roles[role] = dict(documents=len(starts), windows=len(index), native_context_tokens=rebuild.offsets[role],
                target_tokens=int(index['targets'].astype(np.uint64).sum()))
            if roles[role]['target_tokens'] != summary['actual_'+role+'_targets']:
                raise ValueError('The bounded causal windows must retain every original text/EOS target once')
        guard.check(full=True)
        own_files = [p for p in PACK.iterdir() if p.is_file()]
        pack_manifest_path = PACK/'manifest.json'
        pack_manifest = dict(format='own-public-causal-windows-v1', context_limit=4096, roles=roles,
            native_files={r:str(native_folder/(r+'.tokens.u16')) for r in ['train','valid']},
            source_files_sha256={**files, **{str(p):file_hash(p) for p in own_files}},
            within_role_cross_document_attention=True, all_BOS_masked=True, one_token_window_context_overlap=True,
            all_original_text_and_EOS_targets_once=True, original_documents_not_discarded_or_tail_clipped=True,
            long_document_attention_context_bounded_to4096=True, semantic_and_all_history_independence_not_certified=True,
            source_quality_is_not_human_preference_or_execution_correctness=True,
            original_underlying_page_rights_not_relicensed=True, model_updates=0, goal_achieved=False)
        atomic_json(pack_manifest_path, pack_manifest)
        proof_path = OUT/'readback-complete.json'
        if any(name in sys.modules for name in ['slm_lab.public_billion_candidate_v1',
                'slm_lab.public_pretraining_intake_v1', 'slm_lab.public_pretraining_intake_v2']):
            raise ValueError('Independent source reconstruction must not execute the producer intake or writer modules')
        atomic_json(proof_path, dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
            all_original_source_rows_and_native_targets_rebuilt=True, every_declared_quota_complete=True,
            independently_refetched_original_publisher_LFS_bytes=True, intake_and_writer_modules_not_imported=True,
            **summary, actual_original_source_files=source_count, actual_original_rows_rebuilt=rebuild.observed,
            all32953original_declared_held_stems_and_lexical_copy_candidates_screened=True,
            source_files_sha256={**files, str(pack_manifest_path):file_hash(pack_manifest_path),
                str(OUT/'independent-original-source-proofs.jsonl'):file_hash(OUT/'independent-original-source-proofs.jsonl')},
            source_data_only_not_neural_training_or_quality_gain=True, model_updates=0, goal_achieved=False))
        admission_path = PACK/'admission.json'
        atomic_json(admission_path, dict(admitted_to_training=True, full_source_reader_complete=True,
            manifest_sha256=file_hash(pack_manifest_path), independent_readback=str(proof_path),
            independent_readback_sha256=file_hash(proof_path), only_train_role_eligible_for_optimization=True,
            actual_train_targets=summary['actual_train_targets'], native_context_limit=4096,
            neural_training_updates=0, human_preferences=False, execution_reward=False, goal_achieved=False))
        atomic_json(OUT/'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
            **summary, data_admitted_to_training=True, actual_train_targets_admitted=summary['actual_train_targets'],
            source_files_sha256={str(proof_path):file_hash(proof_path), str(admission_path):file_hash(admission_path),
                str(pack_manifest_path):file_hash(pack_manifest_path), str(OUT/'plan.json'):file_hash(OUT/'plan.json')},
            actual_neural_training_updates=0, useful_model_gain_not_measured=True,
            seconds=time.monotonic()-started, all_ten_final_requirements_incomplete=True, goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'), actual_train_targets=summary['actual_train_targets'],
            data_admitted=True, neural_updates=0)), flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
            all_original_source_and_partial_reader_artifacts_preserved=True, neural_jobs_not_signalled=True,
            goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
