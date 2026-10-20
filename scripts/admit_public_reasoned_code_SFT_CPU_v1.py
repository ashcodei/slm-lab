"""Read every packed native token/mask before weak-SFT-only admission.

Waits for the exact whole-source packer. Reconstructs arrays from original
text independently of its formatter. No program execution or model training.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = Path('/tmp/slm-public-reasoned-code-pack-v1')
SOURCE_READER = Path('/tmp/slm-public-reasoned-code-output-readback-v1')
SCREEN = Path('/tmp/slm-public-reasoned-code-screen-v2')
JOIN = Path('/tmp/slm-public-reasoned-code-question-join-v2')
CANDIDATE = Path('/tmp/slm-public-reasoned-code-candidate-v1')
OUT = ROOT / 'reports/public-reasoned-code-pack-admission-v1'
DATA = ROOT / 'data/public-reasoned-code-SFT-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def live(owner):
    path = Path('/proc') / str(owner['pid']) / 'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == str(owner['birth']) and fields[0] != 'Z'


def main():
    if (OUT.exists() or (DATA / 'admission.json').exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.nice(0) != 19 or os.environ.get('TOKENIZERS_PARALLELISM') != 'false'
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 complete native-token admission readback required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    tests = ROOT / 'reports/public-reasoned-code-native-readback-v1-tests.log'
    assert '3 passed' in tests.read_text()
    predecessor = json.loads((PRODUCER / 'owner-plan.json').read_text())
    direct = [Path(__file__).resolve(), PRODUCER / 'owner-plan.json', tests,
        ROOT / 'src/slm_lab/public_reasoned_code_native_readback_v1.py',
        ROOT / 'tests/test_public_reasoned_code_native_readback_v1.py',
        ROOT / 'src/slm_lab/codenet_token_pack_v1.py']
    files = {str(path): sha(path) for path in direct}
    OUT.mkdir()

    def emit(name, value):
        with (OUT / name).open('x') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    emit('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_predecessor=predecessor['owner'], complete_original_shard_positions_required=20000,
        original_context_response_EOS_and_masks_rebuilt_without_producer_formatter=True,
        every_verified_fitting_view_and_every_original_exclusion_required=True,
        admission_scope_if_all_checks_pass='public_synthetic_weak_SFT_only',
        reserved_roles_optimization_forbidden=True, RL_reward_or_human_preference_admission_forbidden=True,
        CPU_threads=1, nice=19, scheduler='SCHED_IDLE',
        data_admitted_to_training=False, new_model_updates=0, original_workers_unchanged=True))
    try:
        while live(predecessor['owner']):
            print(json.dumps(dict(phase='waiting_exact_complete_native_pack', predecessor=predecessor['owner'])), flush=True)
            time.sleep(30)
        if (PRODUCER / 'failure.json').exists() or not (PRODUCER / 'complete.json').exists():
            raise ValueError('Exact packer lacks complete successful evidence; never restart it')
        source = json.loads((PRODUCER / 'complete.json').read_text())
        assert source['complete'] and source['original_shard_rows'] == 20000
        assert source['data_admitted_to_training'] is False
        assert sha(DATA / 'manifest.json') == source['dataset_manifest_sha256']
        decisions_path = PRODUCER / 'every-original-shard-pack-disposition.jsonl'
        assert sha(decisions_path) == source['decisions_sha256']
        proof = json.loads((SOURCE_READER / 'complete.json').read_text())
        assert proof['complete'] and proof['all6116_fixed_candidate_positions_reconciled']
        assert proof['actual_values_recompared_without_importing_producer_comparators']
        assert proof['original_stage_roles_and_views_preserved'] and proof['data_admitted_to_training'] is False
        positive_path = SOURCE_READER / 'verified-finite-case-positives.json'
        assert sha(positive_path) == proof['positive_index_sha256']
        files.update(source['source_files_sha256'])
        for path in [PRODUCER / 'complete.json', DATA / 'manifest.json', decisions_path,
                SOURCE_READER / 'complete.json', positive_path, SCREEN / 'complete.json',
                SCREEN / 'every-original-row-screen.jsonl', JOIN / 'complete.json', JOIN / 'every-candidate-disposition.jsonl']:
            files[str(path)] = sha(path)
        assert all(sha(path) == digest for path, digest in files.items())
        if shutil.disk_usage(ROOT).free < 48 * 1024**3:
            raise ValueError('Preserve48GiB other-writer persistent floor during data-only admission')
        stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel')) < 52 * 1024**3:
            raise ValueError('4GiB own plus48GiB other-job nonfile memory headroom required')
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
        from slm_lab.codenet_token_pack_v1 import CompleteCodePack, FILES
        from slm_lab.public_reasoned_code_native_readback_v1 import verify
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        manifest = json.loads((DATA / 'manifest.json').read_text())
        assert manifest['dataset_admitted_to_training'] is False
        assert manifest['provenance']['weak_SFT_only'] is True
        assert manifest['provenance']['use_as_RL_reward_oracle_forbidden'] is True
        for name in FILES:
            assert sha(DATA / name) == manifest['files_sha256'][name]
            files[str(DATA / name)] = manifest['files_sha256'][name]
        assert sha(DATA / 'tokenizer.json') == sha(ROOT / 'data/tokenizer-corpus/tokenizer.json')
        assert sha(DATA / 'model-config.json') == sha(ROOT / 'data/public-reasoning-curriculum-v1/model-config.json')
        assert sha(DATA / 'publisher-README.md') == sha(CANDIDATE / 'README.publisher.md')
        tokenizer = Tokenizer.from_file(str(DATA / 'tokenizer.json'))
        assert tokenizer.get_vocab_size() == 32768
        settings = json.loads((DATA / 'tokenizer.json').read_text())
        assert settings.get('truncation') is None and settings.get('padding') is None
        for word, token in [('<pad>', 0), ('<bos>', 1), ('<eos>', 2), ('<unk>', 3)]:
            assert tokenizer.token_to_id(word) == token
        positives = {row['original_shard_position']: row for row in json.loads(positive_path.read_text())}
        assert len(positives) == source['verified_original_positives']
        screened = [json.loads(line) for line in (SCREEN / 'every-original-row-screen.jsonl').open()]
        assert len(screened) == 20000
        verified = {row['original_shard_position']: row for row in map(json.loads,
            (SOURCE_READER / 'every-fixed-candidate-readback.jsonl').open())}
        packs = {role: CompleteCodePack(DATA, role, purpose='audit') for role in ('train', 'validation')}
        assert sum(len(pack) for pack in packs.values()) == source['complete_native_views']
        cursors = Counter()
        counts = Counter()
        groups = defaultdict(set)
        position = 0
        packed = 0
        journal_path = OUT / 'every-original-native-token-readback.jsonl'
        with (JOIN / 'every-candidate-disposition.jsonl').open() as join_stream, decisions_path.open() as decisions, \
                gzip.open(DATA / 'original-rows.jsonl.gz', 'rt') as rows, journal_path.open('x') as journal:
            for batch in pq.ParquetFile(CANDIDATE / 'original-python-00000.parquet').iter_batches(
                    columns=['id', 'question_id', 'dataset', 'split', 'index', 'solution', 'r1_generation'],
                    batch_size=16, use_threads=False):
                for original in batch.to_pylist():
                    joined, decision = json.loads(next(join_stream)), json.loads(next(decisions))
                    source_row = screened[position]
                    assert joined['position'] == position and joined['id'] == original['id'] == source_row['id']
                    actual_views = []
                    if position in positives:
                        positive = positives[position]
                        assert positive['fixed_pre_execution_role'] == source_row['fixed_pre_execution_role']
                        assert positive['source_role'] == source_row['source_role']
                        for view in positive['fitting_views']:
                            row = json.loads(next(rows))
                            fixed = positive['fixed_pre_execution_role']
                            role = 'train' if positive['source_role'] == 'train' else 'validation'
                            if (row['id'] != 'public-reasoned-code:' + str(position) + ':' + view + ':' + original['id']
                                    or row['view'] != view or row['original_shard_position'] != position
                                    or row['original_question_id'] != original['question_id']
                                    or row['upstream_dataset'] != original['dataset'] or row['upstream_index'] != int(original['index'])
                                    or row['verified_original_result_sha256'] != positive['verified_original_result_sha256']
                                    or row['original_tested_code_utf8_sha256'] != hashlib.sha256(original['solution'].encode()).hexdigest()
                                    or row['token_pack_index'] != packed or row.get('image')):
                                raise ValueError('Every complete original view, verified program and native pack position must match')
                            assert int(packs[role].positions[cursors[role]]) == packed
                            ids, labels = packs[role].sequence(cursors[role])
                            result = verify(original, joined, fixed, row, ids, labels, tokenizer)
                            counts[role + '_examples'] += 1
                            counts[role + '_context_tokens'] += result['complete_context_tokens']
                            counts[role + '_target_tokens'] += result['response_EOS_targets']
                            counts[role + '_' + view + '_examples'] += 1
                            groups[role].add(row['group_id'])
                            cursors[role] += 1
                            actual_views.append(dict(view=view, pack_position=packed))
                            packed += 1
                        expected_disposition = 'verified_weak_SFT_views_packed'
                    elif source_row['fresh_execution_pending']:
                        expected_disposition = 'preserved_execution_' + verified[position]['original_status']
                    else:
                        expected_disposition = 'preserved_screen_' + source_row['status']
                    expected_decision = dict(original_shard_position=position, source_id=original['id'],
                        original_source_role=source_row['source_role'], original_fixed_role=source_row.get('fixed_pre_execution_role'),
                        disposition=expected_disposition, packed_views=actual_views, data_admitted_to_training=False)
                    if decision != expected_decision:
                        raise ValueError('Every original pack inclusion/exclusion and fixed source role must reconstruct')
                    journal.write(json.dumps(dict(original_shard_position=position, source_id=original['id'],
                        original_decision_rebuilt=True, complete_native_views_read_back=actual_views,
                        independent_native_arrays_and_EOS_masks_exact=True, data_admitted_to_training=False), separators=(',', ':')) + '\n')
                    position += 1
                    if position % 500 == 0:
                        journal.flush()
                        print(json.dumps(dict(phase='complete_original_native_readback', rows=position, total=20000,
                            complete_native_views=packed)), flush=True)
            assert join_stream.readline() == '' and decisions.readline() == '' and rows.readline() == ''
        assert position == 20000 and packed == source['complete_native_views']
        assert not groups['train'].intersection(groups['validation'])
        for role in ('train', 'validation'):
            assert cursors[role] == len(packs[role])
            assert {key: counts[role + '_' + key] for key in ('examples', 'context_tokens', 'target_tokens')} == manifest['counts'][role]
        assert all(sha(path) == digest for path, digest in files.items())
        admission = dict(format='slm-public-reasoned-code-weak-SFT-admission-v1', owner=owner,
            at=datetime.now(timezone.utc).isoformat(), admitted_to_training=True,
            dataset_manifest_sha256=sha(DATA / 'manifest.json'), source_files_sha256=files,
            admitted_use='weak_supervised_coding_only', synthetic_teacher_data=True,
            response_origin='Public precomputed DeepSeek-R1', external_teacher_weights_loaded=False, live_teacher_API_used=False,
            human_preferences=False, own_model_on_policy=False, use_as_RL_reward_oracle_forbidden=True,
            independent_semantic_correctness_certified=False, reasoning_text_not_certified_by_code_tests=True,
            complete_original_source_decisions=20000, complete_source_execution_candidates_read_back=6116,
            complete_packed_rows_read_back=packed, positive_train_rows=counts['train_examples'],
            separately_reserved_rows=counts['validation_examples'], actual_native_counts=dict(counts),
            original_fixed_roles_and_every_whole_native_input_and_EOS_mask_rebuilt=True,
            declared_duplicate_component_cross_role_intersections=0,
            historical_or_semantic_independence_not_certified=True, new_neural_updates=0, goal_achieved=False)
        pending = DATA / 'admission.pending.json'
        with pending.open('x') as stream:
            stream.write(json.dumps(admission, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        emit('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, data_only_weak_SFT_admitted=True, admission_sha256=sha(pending),
            source_dataset_manifest_sha256=sha(DATA / 'manifest.json'), original_source_positions=20000,
            complete_native_rows=packed, actual_native_counts=dict(counts),
            full_native_text_arrays_and_roles_rebuilt_independently=True,
            readback_journal_sha256=sha(journal_path), original_fixed_role_cross_intersections=0,
            reserved_optimization_forbidden=True, use_as_RL_reward_oracle_forbidden=True,
            no_program_or_neural_execution=True, new_model_updates=0, model_quality_gain_established=False,
            original_workers_unchanged=True, goal_achieved=False))
        if (DATA / 'admission.json').exists():
            raise ValueError('Never replace existing dataset admission')
        os.rename(pending, DATA / 'admission.json')
        print(json.dumps(dict(complete=True, weak_SFT_admitted=True, native_rows=packed, counts=dict(counts))), flush=True)
    except BaseException as error:
        emit('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, model_quality_gain_established=False, original_workers_unchanged=True))
        raise


if __name__ == '__main__':
    main()
