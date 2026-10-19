"""Reencode every later instruction and enumerate expanded prospective replay."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_later_instruction_replay_v2 import FAMILIES, KINDS, LaterInstructionReplay
from slm_lab.codenet_replay_schedule_v1 import coverage_replay_schedule
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exclusive_inference_resources_v1 import identity


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/codenet-later-instruction-replay-v2'
PREP = ROOT / 'reports/codenet-later-instruction-replay-v2'
REPORTS = ROOT / 'reports/codenet-later-instruction-replay-v2-independent-audit'


def read(path):
    return json.loads(path.read_text())


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message,
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or REPORTS.exists():
        raise ValueError('New CPU-only independent audit required; preserve earlier evidence')
    if shutil.disk_usage(ROOT).free < 16 * 1024**3 + 32 * 1024**2:
        raise ValueError('Retain16GiB plus32MiB independent-report commitment')
    REPORTS.mkdir()
    launch_path = ROOT / 'reports/codenet-later-instruction-replay-v2-launch.json'
    launch = read(launch_path)
    while True:
        actor = identity(launch['pid'])
        if actor is None or actor['ticks'] != str(launch['start_ticks']) or actor['state'] in {'Z', 'X'}:
            break
        assert actor['argv'] == launch['argv']
        assert file_hash(ROOT / 'scripts/prepare_codenet_later_instruction_replay_v2.py') == launch['source_sha256']
        status('waiting for exact live original instruction preparation actor', preceding_pid=launch['pid'], preceding_start_ticks=launch['start_ticks'])
        time.sleep(5)
    prepared = read(PREP / 'results.json')
    manifest = read(DATA / 'manifest.json')
    assert file_hash(DATA / 'manifest.json') == prepared['manifest_sha256']
    bound = {**prepared['source_files_sha256'], str(DATA / 'manifest.json'): prepared['manifest_sha256'],
        str(Path(__file__)): file_hash(Path(__file__)), str(launch_path): file_hash(launch_path)}
    bound.update({str(DATA / name): digest for name, digest in manifest['artifact_files_sha256'].items()})
    guard = FileBindings(bound)
    reader = LaterInstructionReplay(DATA)
    token = Tokenizer.from_file(str(ROOT / 'data/codenet-durable-pack-v2/tokenizer.json'))
    assert file_hash(ROOT / 'data/codenet-durable-pack-v2/tokenizer.json') == manifest['tokenizer_sha256']
    counts, contexts, targets = Counter(), Counter(), Counter()
    cursor = 0
    groups = set()
    with (DATA / 'records.jsonl').open() as inventory:
        for source in manifest['sources']:
            with Path(source['path']).open('rb') as stream:
                while raw := stream.readline():
                    offset = stream.tell() - len(raw)
                    row = json.loads(raw)
                    assert row['split'] == 'train' and not row.get('image') and row['external_model_teacher_used'] is False
                    kind = source['kind']
                    name = 'publisher_exercises:' + row['language'] if kind == 'exercises' else (
                        FAMILIES[6] if kind == 'javascript' else 'complete_' + row['language'] + '_source_headers')
                    prompt = row['prompt'] if row['prompt'].rstrip().endswith('Assistant:') else 'User: ' + row['prompt'] + '\nAssistant:'
                    prompt_ids, response_ids = token.encode(prompt).ids, token.encode(row['response']).ids
                    ids = [token.token_to_id('<bos>')] + prompt_ids + response_ids + [token.token_to_id('<eos>')]
                    count = len(response_ids) + 1
                    assert len(ids) <= 4096
                    digest = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).digest()
                    item = reader.index[cursor]
                    assert int(item['offset']) == offset and int(item['bytes']) == len(raw)
                    assert int(item['family']) == FAMILIES.index(name)
                    assert item['row_sha256'].tobytes() == hashlib.sha256(raw).digest()
                    assert item['input_sha256'].tobytes() == digest
                    assert int(item['context_tokens']) == len(ids) and int(item['target_tokens']) == count
                    record = json.loads(next(inventory))
                    assert record == dict(id=row['id'], family=name, source_kind=kind, source_offset=offset,
                        source_bytes=len(raw), source_row_sha256=hashlib.sha256(raw).hexdigest(),
                        full_input_ids_sha256=digest.hex(), context_tokens=len(ids), target_tokens=count)
                    for repetition in range(4 if kind == 'exercises' else 1):
                        actual = reader.row(name, repetition * manifest['unique_family_counts'][name] + counts[name])
                        expected = dict(row)
                        if kind == 'exercises':
                            expected.update(id=row['id'] + '-replay-' + str(repetition), original_exercise_record_id=row['id'],
                                exercise_replay_index=repetition, curriculum_source='original_exercism_tasks')
                        assert {k: v for k, v in actual.items() if not k.startswith('_replay_')} == expected
                        assert actual['_replay_input_sha256'] == digest.hex()
                    counts[name] += 1
                    contexts[name] += len(ids)
                    targets[name] += count
                    groups.add(row['group_id'])
                    cursor += 1
                    if cursor % 1000 == 0:
                        status('independently reconstructing every complete later-source input and explicit exposure', unique_source_rows=cursor)
        assert inventory.read() == ''
    assert cursor == len(reader.index) == manifest['unique_source_rows'] == 48584
    assert dict(counts) == manifest['unique_family_counts']
    assert dict(contexts) == manifest['unique_source_context_tokens'] and dict(targets) == manifest['unique_source_response_targets']
    for path in [ROOT / 'data/exercism-code-v1/valid.jsonl', ROOT / 'data/code-search-expanded-v1/sft.valid.jsonl']:
        bound[str(path)] = file_hash(path)
        assert not groups & {json.loads(line)['group_id'] for line in path.open()}
    names, target_lengths = [], []
    for folder in ['codenet-replay-v1', 'codenet-fact-replay-v1', 'codenet-dialogue-replay-v1', 'codenet-image-replay-v1']:
        path = ROOT / 'data' / folder
        m = read(path / 'manifest.json')
        index = np.load(path / 'index.npy', mmap_mode='r', allow_pickle=False)
        bound[str(path / 'manifest.json')] = file_hash(path / 'manifest.json')
        bound[str(path / 'index.npy')] = m['artifact_files_sha256']['index.npy']
        assert file_hash(path / 'index.npy') == m['artifact_files_sha256']['index.npy']
        for i, name in enumerate(m['family_names']):
            values = index['target_tokens'][index['family'] == i]
            assert len(values) == m['family_counts'][name]
            names.append(name)
            target_lengths.append(values.astype(np.uint32))
    for i, name in enumerate(FAMILIES):
        values = reader.index['target_tokens'][reader.positions[name]].astype(np.uint32)
        names.append(name)
        target_lengths.append(np.tile(values, 4 if i < 6 else 1))
    populations = tuple((name, len(values)) for name, values in zip(names, target_lengths))
    assert len(populations) == 28 and sum(size for _, size in populations) == 398505
    insufficient = []
    for accumulation in [17, 18]:
        try:
            coverage_replay_schedule(populations, 5279, 4 * accumulation)
        except ValueError:
            insufficient.append(accumulation)
        else:
            raise ValueError('A smaller replay layout must not claim complete expanded coverage')
    schedule = coverage_replay_schedule(populations, 5279, 76)
    offsets = np.cumsum([0] + [size for _, size in populations], dtype=np.uint64)
    exposures = np.zeros(int(offsets[-1]), dtype='<u4')
    draws = np.empty(5279 * 76, dtype='<u4')
    orders = {name: EpochOrder(size, 71 + 1000 + i) for i, (name, size) in enumerate(populations)}
    cursor = 0
    for step in range(5279):
        name, occurrence = schedule.at(step)
        family = names.index(name)
        groups_by_rank = [orders[name].rank_groups(occurrence, rank, 4, 1, 19) for rank in range(4)]
        for micro in range(19):
            for rank in range(4):
                position = int(offsets[family]) + groups_by_rank[rank][micro][0]
                exposures[position] += 1
                draws[cursor] = position
                cursor += 1
    assert cursor == len(draws) == 401204 and np.all(exposures > 0)
    by_family = {}
    for i, (name, size) in enumerate(populations):
        used = exposures[int(offsets[i]):int(offsets[i + 1])]
        by_family[name] = dict(source_examples=size, reached=int(np.count_nonzero(used)),
            exposures=int(used.sum()), response_target_exposures=int((used.astype(np.uint64) * target_lengths[i]).sum()))
    np.save(REPORTS / 'prospective-draw-positions-u32.npy', draws, allow_pickle=False)
    np.save(REPORTS / 'prospective-exposure-counts-u32.npy', exposures, allow_pickle=False)
    guard.check(full=True)
    FileBindings(bound).check(full=True)
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=bound,
        source_manifest_sha256=prepared['manifest_sha256'], original_unique_source_rows_reconstructed=48584,
        explicit_original_replay_records_reconstructed=50372, unique_family_counts=dict(counts),
        original_source_context_tokens=dict(contexts), original_source_response_targets=dict(targets),
        original_row_metadata_and_four_exercise_annotations_preserved=True, protected_held_group_overlap=0,
        every_original_input_ID_and_response_target_count_rederived=True,
        expanded_prepared_replay_families=28, expanded_prepared_replay_examples=398505,
        expanded_unique_original_record_inputs=396717, prospective_optimizer_steps=5279,
        rejected_insufficient_replay_accumulations=insufficient, minimum_passing_replay_accumulation=19,
        prospective_actual_example_exposures=len(draws), prospective_replay_response_target_exposures=sum(v['response_target_exposures'] for v in by_family.values()),
        every_prepared_replay_example_reached=True, by_family=by_family,
        artifact_files_sha256={n: file_hash(REPORTS / n) for n in ['prospective-draw-positions-u32.npy', 'prospective-exposure-counts-u32.npy']},
        no_GPU_context_created=True, new_neural_training_started=False, final_base_match=False,
        systems_packed_pretraining_and_finalized_paired_photos_still_pending=True,
        adopted28_family_runner=False, production_stage_ready=False, goal_achieved=False,
        scope='Full independent original later-instruction index/mask/count/metadata readback and prospective28-family '
            'draw enumeration. Repeated exercises and header variants are not new independent programs. '
            'No model update, final-base/all-later coverage, source semantic correctness or capability gain claimed.')
    atomic_json(REPORTS / 'results.json', result)
    status('every original later instruction and expanded prospective replay draw independently reconstructed', original_source_rows=48584, explicit_replay_records=50372, expanded_families=28, new_neural_training_started=False)
    print(json.dumps({k: result[k] for k in ['original_unique_source_rows_reconstructed', 'explicit_original_replay_records_reconstructed',
        'expanded_prepared_replay_families', 'expanded_prepared_replay_examples', 'minimum_passing_replay_accumulation', 'prospective_replay_response_target_exposures']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if REPORTS.exists():
            status('independent instruction audit stopped; inspect preserved evidence', error=str(error))
        raise
