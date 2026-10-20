"""Pack whole independently checked code/reasoning targets with sealed roles.

Waits for the exact complete source/output reader. Produces an unadmitted
candidate pack; an independent native-token readback is still mandatory.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
READER = Path('/tmp/slm-public-reasoned-code-output-readback-v1')
SCREEN = Path('/tmp/slm-public-reasoned-code-screen-v2')
JOIN = Path('/tmp/slm-public-reasoned-code-question-join-v2')
CACHE = Path('/tmp/slm-public-reasoned-code-train-questions-v1')
CANDIDATE = Path('/tmp/slm-public-reasoned-code-candidate-v1')
METADATA = Path('/tmp/slm-public-reasoned-code-question-metadata-v1')
OUT = Path('/tmp/slm-public-reasoned-code-pack-v1')
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
    if (OUT.exists() or DATA.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or os.environ.get('TOKENIZERS_PARALLELISM') != 'false'
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread complete source packing required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    tests = ROOT / 'reports/public-reasoned-code-pack-rows-v1-tests.log'
    assert '3 passed' in tests.read_text()
    predecessor = json.loads((READER / 'owner-plan.json').read_text())
    tokenizer_path = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    config_path = ROOT / 'data/public-reasoning-curriculum-v1/model-config.json'
    cards = [METADATA / (slug + '-README.md') for slug in ('deepmind--code_contests', 'codeparrot--apps', 'BAAI--TACO')]
    cards.append(ROOT / 'data/code-contests-original-train-cache-v2/publisher-README.md')
    direct = [Path(__file__).resolve(), READER / 'owner-plan.json', tests,
        ROOT / 'tests/test_public_reasoned_code_pack_rows_v1.py',
        ROOT / 'src/slm_lab/public_reasoned_code_pack_rows_v1.py',
        ROOT / 'src/slm_lab/public_reasoned_code_metadata_v1.py',
        ROOT / 'src/slm_lab/public_synthetic_code_screen_policy_v1.py',
        ROOT / 'src/slm_lab/held_text_overlap_v1.py', ROOT / 'src/slm_lab/codenet_token_pack_v1.py',
        tokenizer_path, config_path, CANDIDATE / 'README.publisher.md', *cards]
    files = {str(path): sha(path) for path in direct}
    OUT.mkdir()

    def emit(name, value):
        with (OUT / name).open('x') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    emit('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_predecessor=predecessor['owner'], all_original_shard_positions_required=20000,
        every_verified_fitting_code_and_reasoning_view_retained=True,
        original_train_reserved_component_roles_never_reassigned=True,
        whole_source_question_starter_response_and_EOS_preserved=True,
        persistent_pack_maximum_bytes=1024**3, persistent_free_floor_bytes=48 * 1024**3,
        maximum_all_source_decision_bytes=64 * 1024**2,
        CPU_threads=1, nice=19, scheduler='SCHED_IDLE',
        supervision='public synthetic weak SFT only; no human-preference or RL reward admission',
        source_output_readback_and_independent_native_token_readback_required=True,
        data_admitted_to_training=False, model_updates=0, original_workers_unchanged=True))
    writer = None
    try:
        while live(predecessor['owner']):
            print(json.dumps(dict(phase='waiting_exact_complete_original_output_reader', predecessor=predecessor['owner'])), flush=True)
            time.sleep(30)
        if (READER / 'failure.json').exists() or not (READER / 'complete.json').exists():
            raise ValueError('Exact complete output reader failed or lacks final evidence; never restart it')
        proof = json.loads((READER / 'complete.json').read_text())
        assert proof['complete'] and proof['all6116_fixed_candidate_positions_reconciled']
        assert proof['declared_duplicate_component_cross_role_intersections'] == 0
        assert proof['data_admitted_to_training'] is False and proof['RL_reward_admitted'] is False
        positives_path = READER / 'verified-finite-case-positives.json'
        observations_path = READER / 'every-fixed-candidate-readback.jsonl'
        assert sha(positives_path) == proof['positive_index_sha256'] and sha(observations_path) == proof['ledger_sha256']
        files.update(proof['source_files_sha256'])
        for path in [READER / 'complete.json', positives_path, observations_path,
                SCREEN / 'complete.json', SCREEN / 'every-original-row-screen.jsonl',
                JOIN / 'complete.json', JOIN / 'every-candidate-disposition.jsonl', CACHE / 'complete.json']:
            files[str(path)] = sha(path)
        assert all(sha(path) == digest for path, digest in files.items())
        if shutil.disk_usage(ROOT).free < 49 * 1024**3:
            raise ValueError('1GiB pack allowance above48GiB other-writer persistent reserve required')
        stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel')) < 52 * 1024**3:
            raise ValueError('4GiB own plus48GiB other-job nonfile memory reserve required')
        import pyarrow as pa
        import pyarrow.parquet as pq
        from tokenizers import Tokenizer
        from slm_lab.codenet_token_pack_v1 import CodePackWriter
        from slm_lab.public_reasoned_code_pack_rows_v1 import build, REVISION, PUBLISHER_FILE
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        positives = {r['original_shard_position']: r for r in json.loads(positives_path.read_text())}
        verified = {r['original_shard_position']: r for r in map(json.loads, observations_path.open())}
        assert len(verified) == 6116 and set(positives) == {p for p, r in verified.items() if r['finite_case_positive_verified']}
        screened = [json.loads(line) for line in (SCREEN / 'every-original-row-screen.jsonl').open()]
        assert len(screened) == 20000
        cache = json.loads((CACHE / 'complete.json').read_text())
        assert sha(cache['questions_file']) == cache['questions_file_sha256']
        needed = {(screened[p]['dataset'], screened[p]['source_index']) for p in positives}
        questions = {}
        for line in Path(cache['questions_file']).open():
            row = json.loads(line)
            if (row['dataset'], row['index']) in needed:
                questions[(row['dataset'], row['index'])] = row
        assert len(questions) == len(needed)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        assert tokenizer.get_vocab_size() == 32768
        raw_tokenizer = json.loads(tokenizer_path.read_text())
        assert raw_tokenizer.get('truncation') is None and raw_tokenizer.get('padding') is None
        config = json.loads(config_path.read_text())
        assert config['max_text_length'] == 4096 and config['vocab_size'] == 32768 and config['num_layers'] == 35
        DATA.mkdir()
        shutil.copyfile(tokenizer_path, DATA / 'tokenizer.json')
        shutil.copyfile(config_path, DATA / 'model-config.json')
        shutil.copyfile(CANDIDATE / 'README.publisher.md', DATA / 'publisher-README.md')
        license_text = ('NVIDIA Corporation, OpenCodeReasoning-2 (2025)\nRevision: ' + REVISION + '\n' +
            'Released dataset license: CC-BY-4.0, retaining upstream terms and attribution.\n' +
            'Source: https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2/tree/' + REVISION + '\n' +
            'Changes: select independently checked finite-test-agreeing examples; preserve complete original code or R1 response; add own native BOS/EOS and fixed stage roles.\n' +
            'Public precomputed DeepSeek-R1 synthetic text, not outside pretrained weights or a live teacher API.\n' +
            'Weak SFT only; not human feedback, semantic correctness proof, or an RL reward oracle.\n')
        for card in cards:
            license_text += '\nRetained upstream publisher card: ' + card.name + '\n' + card.read_text() + '\n'
        (DATA / 'DATASET-LICENSE.txt').write_text(license_text)
        writer = CodePackWriter(DATA, maximum_bytes=1024**3, minimum_free_gib=48)
        counts = Counter()
        expected_views = sum(len(row['fitting_views']) for row in positives.values())
        emit('execution-plan.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            verified_original_positives=len(positives), expected_complete_native_views=expected_views,
            source_revision=REVISION, source_file=PUBLISHER_FILE,
            tokenizer_sha256=sha(tokenizer_path), model_config_sha256=sha(config_path),
            no_program_or_neural_execution=True, data_admitted_to_training=False))
        group_roles = {}
        position = 0
        decision_path = OUT / 'every-original-shard-pack-disposition.jsonl'
        with (JOIN / 'every-candidate-disposition.jsonl').open() as joined_rows, decision_path.open('x') as decisions:
            for batch in pq.ParquetFile(CANDIDATE / 'original-python-00000.parquet').iter_batches(
                    columns=['id', 'question_id', 'dataset', 'split', 'index', 'solution', 'r1_generation', 'pass_rate', 'license'],
                    batch_size=16, use_threads=False):
                for original in batch.to_pylist():
                    joined = json.loads(next(joined_rows))
                    source = screened[position]
                    assert joined['position'] == position and joined['id'] == source['id'] == original['id']
                    assert source['original_shard_position'] == position
                    packed = []
                    if position in positives:
                        positive = positives[position]
                        question = questions[(original['dataset'], int(original['index']))]
                        for view in positive['fitting_views']:
                            row, ids, labels = build(original, question, joined, source, positive, verified[position], view, tokenizer)
                            packed.append(dict(view=view, pack_position=len(writer.entries)))
                            writer.add(row, ids, labels)
                            counts[row['split'] + '_examples'] += 1
                            counts[row['split'] + '_' + view + '_examples'] += 1
                            counts[row['split'] + '_response_EOS_targets'] += sum(value != -100 for value in labels)
                            group = row['group_id']
                            if group in group_roles and group_roles[group] != row['split']:
                                raise ValueError('A fixed duplicate component cannot cross native pack roles')
                            group_roles[group] = row['split']
                        disposition = 'verified_weak_SFT_views_packed'
                    elif source['fresh_execution_pending']:
                        disposition = 'preserved_execution_' + verified[position]['original_status']
                    else:
                        disposition = 'preserved_screen_' + source['status']
                    decision = dict(original_shard_position=position, source_id=original['id'],
                        original_source_role=source['source_role'], original_fixed_role=source.get('fixed_pre_execution_role'),
                        disposition=disposition, packed_views=packed, data_admitted_to_training=False)
                    raw = json.dumps(decision, ensure_ascii=True, separators=(',', ':')) + '\n'
                    if decisions.tell() + len(raw.encode()) > 64 * 1024**2:
                        raise ValueError('Preserve every original decision within its bounded allowance')
                    decisions.write(raw)
                    counts[disposition] += 1
                    position += 1
                    if position % 500 == 0:
                        print(json.dumps(dict(phase='whole_source_native_packing', rows=position, total=20000,
                            native_views=len(writer.entries))), flush=True)
            assert joined_rows.readline() == ''
        assert position == 20000 and len(writer.entries) == expected_views
        manifest = writer.finish(dict(source_files_sha256=files, dataset='nvidia/OpenCodeReasoning-2', revision=REVISION,
            source_file=PUBLISHER_FILE, synthetic_teacher_data=True, response_origin='Public precomputed DeepSeek-R1',
            weak_SFT_only=True, human_preferences=False, own_model_on_policy=False, use_as_RL_reward_oracle_forbidden=True,
            finite_case_agreement_not_semantic_or_efficiency_proof=True,
            reasoning_text_not_independently_certified_by_code_execution=True,
            original_fixed_train_reserved_roles_preserved=True, full_original_decisions_sha256=sha(decision_path),
            original_code_or_entire_R1_target_never_truncated=True, historical_or_semantic_independence_not_certified=True,
            independent_complete_native_token_readback_required=True))
        assert all(sha(path) == digest for path, digest in files.items())
        emit('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, original_shard_rows=20000, verified_original_positives=len(positives),
            complete_native_views=expected_views, counts=dict(counts), actual_pack_counts=manifest['counts'],
            dataset_manifest_sha256=sha(DATA / 'manifest.json'), decisions_sha256=sha(decision_path),
            all_original_positions_and_whole_positive_views_preserved=True,
            original_fixed_roles_preserved=True, declared_duplicate_component_cross_role_intersections=0,
            data_admitted_to_training=False, use_as_RL_reward_oracle_forbidden=True,
            independent_complete_native_token_readback_still_required=True,
            new_program_executions=0, model_updates=0, model_quality_gain_established=False,
            original_workers_unchanged=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, native_views=expected_views, counts=dict(counts))), flush=True)
    except BaseException as error:
        if writer is not None and not writer.tokens.closed:
            writer.close()
        emit('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, data_admitted_to_training=False, original_workers_unchanged=True))
        raise


if __name__ == '__main__':
    main()
