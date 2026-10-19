from collections import Counter
import json

import numpy as np
import pytest

from slm_lab.corpus import file_hash
from slm_lab.public_causal_windows_v1 import CausalPublicPack, windows


def test_context_boundary_EOS_long_documents_and_internal_BOS_preserve_each_target_once():
    # Different document lengths put EOS, BOS and ordinary tokens on boundaries.
    for context in [3, 7, 4096]:
        docs = [[1, *([4+p]*n), 2] for p, n in enumerate([1, 4, 4093, 4094, 8193, 2])]
        native = np.asarray([t for d in docs for t in d], dtype='<u2')
        starts = np.cumsum([0]+[len(d) for d in docs[:-1]]).tolist()
        index = windows(len(native), starts, context)
        consumed = Counter()
        for row in index:
            offset, length = int(row['offset']), int(row['length'])
            positions = [p for p in range(offset+1, offset+length) if native[p] != 1]
            assert len(positions) == int(row['targets'])
            consumed.update(positions)
        assert consumed == Counter({p:1 for p, t in enumerate(native) if p and t != 1})
        assert sum(int(row['targets']) for row in index) == sum(len(d)-1 for d in docs)
        assert all(consumed[p] == 1 for p, t in enumerate(native) if t == 2)


def fixture(tmp_path):
    files = []
    roles = {}
    paths = {}
    for role, docs in [('train', [[1, 4, 5, 2], [1, 6, 7, 8, 2]]), ('valid', [[1, 40, 41, 2]])]:
        native = np.asarray([n for doc in docs for n in doc], dtype='<u2')
        path = tmp_path/(role+'.tokens.u16')
        native.tofile(path)
        starts = np.cumsum([0]+[len(d) for d in docs[:-1]]).astype('<u8')
        index = windows(len(native), starts, 4)
        np.save(tmp_path/(role+'.windows.npy'), index, allow_pickle=False)
        np.save(tmp_path/(role+'.document-starts.npy'), starts, allow_pickle=False)
        files.extend([path, tmp_path/(role+'.windows.npy'), tmp_path/(role+'.document-starts.npy')])
        paths[role] = str(path)
        roles[role] = dict(target_tokens=sum(len(d)-1 for d in docs))
    manifest = dict(format='own-public-causal-windows-v1', context_limit=4, native_files=paths, roles=roles,
                    source_files_sha256={str(p):file_hash(p) for p in files})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))


def test_actual_loader_padding_targets_and_sealed_role_are_exact_without_granting_admission(tmp_path):
    fixture(tmp_path)
    pack = CausalPublicPack(tmp_path, 'train')
    batch = pack.batch(list(range(len(pack))))
    assert int((batch['labels'][:, 1:] != -100).sum()) == 7
    assert not bool((batch['input_ids'] == 40).any())
    assert bool((batch['labels'][batch['input_ids'] == 1] == -100).all())
    assert bool((batch['labels'][~batch['attention_mask']] == -100).all())
    with pytest.raises(ValueError, match='Held public'):
        CausalPublicPack(tmp_path, 'valid', 'optimization')
    with pytest.raises(FileNotFoundError):
        CausalPublicPack(tmp_path, 'train', 'optimization')


def test_native_corruption_and_incomplete_or_replaced_source_admission_are_rejected(tmp_path):
    fixture(tmp_path)
    admission = dict(admitted_to_training=True, full_source_reader_complete=False,
        manifest_sha256=file_hash(tmp_path/'manifest.json'))
    (tmp_path/'admission.json').write_text(json.dumps(admission))
    with pytest.raises(ValueError, match='independent source admission'):
        CausalPublicPack(tmp_path, 'train', 'optimization')
    path = tmp_path/'train.tokens.u16'
    raw = bytearray(path.read_bytes());raw[2] ^= 1;path.write_bytes(raw)
    with pytest.raises(ValueError, match='bytes changed'):
        CausalPublicPack(tmp_path, 'train')
