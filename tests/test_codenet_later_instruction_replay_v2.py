import hashlib
import json
import os

import numpy as np
import pytest

from slm_lab.codenet_later_instruction_replay_v2 import (FAMILIES, KINDS, LaterInstructionReplay,
    exercise_exposure, source_family)
from slm_lab.codenet_replay_v1 import INDEX_DTYPE, FullReplayEncoder
from slm_lab.corpus import file_hash

from test_codenet_runner_v1 import Tokenizer, small_config


def artifact(tmp_path):
    folder = tmp_path / 'replay'
    folder.mkdir()
    encoder = FullReplayEncoder(Tokenizer(), small_config(), tmp_path, 16)
    rows = []
    for i, name in enumerate(FAMILIES):
        if i < 6:
            language = name.split(':')[1]
            row = dict(id='exercise-' + language, group_id='exercism:' + language, language=language,
                source='exercism/' + language.lower(), publisher_reference=True,
                task='public_code_exercise', response='a' * (35 if i == 0 else 2))
        else:
            language = 'JavaScript' if i == 6 else 'Go' if i == 7 else 'Ruby'
            code = 'function f(){}' if i == 6 else 'func f(){}' if i == 7 else 'def f; end'
            row = dict(id='header-' + language, group_id='github:fixture:' + language, language=language,
                source='CodeSearchNet', original_partition='train', parent_instruction_id='original-' + language,
                complete_source_header='header', task='complete_documented_code_parameters' if i == 6 else 'complete_documented_code_header',
                response='```' + language.lower() + '\n' + code + '\n```', source_code_sha256=hashlib.sha256(code.encode()).hexdigest())
        rows.append(dict(row, split='train', prompt='p', external_model_teacher_used=False))
    sources, entries = [], []
    for kind, subset in zip(KINDS, [rows[:6], rows[6:7], rows[7:]]):
        path = tmp_path / (kind + '.jsonl')
        with path.open('wb') as stream:
            for row in subset:
                raw = (json.dumps(row) + '\n').encode()
                ids, labels = encoder.sequence(row, 'response')
                name = source_family(row, kind)
                entries.append((stream.tell(), len(raw), FAMILIES.index(name), len(ids), sum(y != -100 for y in labels),
                    np.void(hashlib.sha256(raw).digest()), np.void(hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).digest())))
                stream.write(raw)
        manifest = tmp_path / (kind + '-manifest.json')
        manifest.write_text(json.dumps(dict(train_examples=len(subset), files_sha256={'train.jsonl': file_hash(path)}, tokenizer_sha256='b' * 64)))
        sources.append(dict(kind=kind, path=str(path), sha256=file_hash(path), manifest=str(manifest),
            manifest_sha256=file_hash(manifest), rows=len(subset)))
    np.save(folder / 'index.npy', np.asarray(entries, dtype=INDEX_DTYPE), allow_pickle=False)
    (folder / 'records.jsonl').write_text('fixture-record-inventory\n')
    (folder / 'source-admission.json').write_text(json.dumps(dict(passed=True, held_roles_excluded=True,
        source_manifests_sha256={s['kind']: s['manifest_sha256'] for s in sources}, receipts={kind: {'passed': True} for kind in KINDS})))
    manifest = dict(format='slm-complete-later-instruction-replay-v2', family_names=list(FAMILIES), sources=sources,
        tokenizer_sha256='b' * 64, unique_source_rows=9, complete_fit_rows=27,
        unique_family_counts={name: 1 for name in FAMILIES},
        family_counts={name: 4 if i < 6 else 1 for i, name in enumerate(FAMILIES)},
        artifact_files_sha256={name: file_hash(folder / name) for name in ['index.npy', 'records.jsonl', 'source-admission.json']})
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    return folder, rows, encoder


def test_every_original_source_and_four_explicit_exercise_exposures_preserve_complete_input(tmp_path):
    folder, originals, encoder = artifact(tmp_path)
    reader = LaterInstructionReplay(folder)
    assert sum(reader.size(name) for name in reader.families) == 27
    for i, name in enumerate(FAMILIES):
        for position in range(reader.size(name)):
            row = reader.row(name, position)
            expected = exercise_exposure(originals[i], position) if i < 6 else originals[i]
            assert {k: v for k, v in row.items() if not k.startswith('_replay_')} == expected
            assert encoder.sequence(row, 'response') == encoder.sequence(originals[i], 'response')
    ids, labels = encoder.sequence(reader.row(FAMILIES[0], 3), 'response')
    assert sum(y != -100 for y in labels) == 36 > small_config().max_text_length // 2
    assert encoder.truncated == 0


def test_changed_original_bytes_with_restored_size_and_mtime_are_rejected(tmp_path):
    folder, _, _ = artifact(tmp_path)
    reader = LaterInstructionReplay(folder)
    from pathlib import Path
    source = Path(reader.sources['exercises']['path'])
    stat = source.stat()
    raw = source.read_bytes().replace(b'"prompt": "p"', b'"prompt": "q"', 1)
    source.write_bytes(raw)
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match='Exact original'):
        reader.row(FAMILIES[0], 0)


def test_duplicate_or_noncontiguous_original_index_rejects_even_with_new_artifact_hash(tmp_path):
    folder, _, _ = artifact(tmp_path)
    index = np.load(folder / 'index.npy', allow_pickle=False)
    index[1]['offset'] = 0
    np.save(folder / 'index.npy', index, allow_pickle=False)
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['artifact_files_sha256']['index.npy'] = file_hash(folder / 'index.npy')
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Every original'):
        LaterInstructionReplay(folder)


@pytest.mark.parametrize('fault', ['held', 'teacher', 'partition', 'body'])
def test_wrong_source_role_or_implementation_cannot_be_returned(tmp_path, fault):
    _, rows, _ = artifact(tmp_path)
    row = dict(rows[6])
    if fault == 'held':
        row['split'] = 'valid'
    elif fault == 'teacher':
        row['external_model_teacher_used'] = True
    elif fault == 'partition':
        row['original_partition'] = 'valid'
    else:
        row['response'] = row['response'].replace('f()', 'g()')
    with pytest.raises(ValueError):
        source_family(row, 'javascript')


@pytest.mark.parametrize('position', [-1, 4, True])
def test_exercise_repetition_budget_is_explicit(position):
    with pytest.raises(ValueError):
        exercise_exposure({'id': 'fixture'}, position)
