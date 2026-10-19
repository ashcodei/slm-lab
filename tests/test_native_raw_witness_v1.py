import copy
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from slm_lab.native_raw_witness_v1 import reconstruct
from slm_lab.native_record_stream_v1 import NativeRecordStream


def test_whole_unicode_record_is_reconstructed_across_shards_before_decoding(tmp_path):
    tokenizer = Tokenizer.from_file(str(Path(__file__).resolve().parents[1] / 'data/tokenizer-corpus/tokenizer.json'))
    text = 'Straße cafe\u0301 and Δ. A whole source record crosses the file boundary.'
    ids = tokenizer.encode(text).ids; split = len(ids) // 2
    arrays = [[1, *ids[:split]], [*ids[split:], 2, 1, 7, 8, 2]]; shards = []
    stream = NativeRecordStream(); records = []
    for index, values in enumerate(arrays):
        name = f'{index}.u16'; np.asarray(values, dtype='<u2').tofile(tmp_path / name)
        shards.append(dict(path=name, tokens=len(values))); records.extend(stream.feed(np.asarray(values), dict(shard=name)))
    records.extend(stream.finish()); original, witness = records[0]
    actual, digest = reconstruct(tmp_path, shards, witness)
    assert actual == original == ids and tokenizer.decode(actual) == tokenizer.decode(ids)
    assert len(digest) == 64
    bad = copy.deepcopy(witness); bad['start'] = dict(shard='1.u16', token_offset=0); bad['token_count'] = len(ids) - split
    with pytest.raises(ValueError, match='preceding shard'): reconstruct(tmp_path, shards, bad)
    bad = copy.deepcopy(witness); bad['token_count'] -= 1
    with pytest.raises(ValueError, match='Every declared'): reconstruct(tmp_path, shards, bad)
    with pytest.raises(ValueError, match='resource limit'): reconstruct(tmp_path, shards, witness, maximum_tokens=len(ids) - 1)


def test_unterminated_final_record_is_explicit_and_internal_boundaries_reject(tmp_path):
    values = np.asarray([1, 11, 12, 13], dtype='<u2'); values.tofile(tmp_path / 'only.u16')
    shards = [dict(path='only.u16', tokens=4)]; stream = NativeRecordStream()
    rows = list(stream.feed(values, dict(shard='only.u16'))) + list(stream.finish())
    actual, _ = reconstruct(tmp_path, shards, rows[0][1]); assert actual == [11, 12, 13]
    bad = copy.deepcopy(rows[0][1]); bad['start']['token_offset'] = 0; bad['token_count'] = 4
    with pytest.raises(ValueError, match='record-boundary'): reconstruct(tmp_path, shards, bad)
