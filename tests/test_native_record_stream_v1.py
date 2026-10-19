import numpy as np
import pytest
from tokenizers import Tokenizer
from pathlib import Path

from slm_lab.held_question_scanner_v1 import QuestionScanner
from slm_lab.native_record_stream_v1 import NativeRecordStream


def test_unicode_question_crossing_token_shard_boundary_is_reconstructed_before_decoding():
    tokenizer = Tokenizer.from_file(str(Path(__file__).resolve().parents[1]/'data/tokenizer-corpus/tokenizer.json'))
    text = 'Prefix. Quelle est la masse de Ångström et Straße? Suffix.'
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    # Every split includes a split inside at least one multibyte character or BPE
    # token sequence; complete decoding must give the same original text.
    for split in range(1, len(ids)):
        stream = NativeRecordStream(); scanner = QuestionScanner([
            dict(audit_id='question', question='masse de A\u030angström et STRASSE?'),
            dict(audit_id='cross', question='Suffix. Prefix.')])
        rows = list(stream.feed(np.array([1]+ids[:split], dtype='<u2'), dict(shard=0)))
        rows += list(stream.feed(np.array(ids[split:]+[2,1]+ids+[2], dtype='<u2'), dict(shard=1)))
        rows += list(stream.finish())
        assert len(rows) == 2 and all(tokenizer.decode(row[0], skip_special_tokens=False) == text for row in rows)
        for values, source in rows: scanner.native(tokenizer, values, source)
        assert set(scanner.matches) == {'question'}
        assert stream.input_tokens == stream.output_tokens + stream.boundary_tokens
        assert stream.records == 2 and rows[0][1]['start']['shard'] == 0


def test_oversized_unfinished_record_fails_closed_and_final_tail_is_not_lost():
    stream = NativeRecordStream(maximum_record_tokens=3)
    assert not list(stream.feed(np.array([7,8],dtype='<u2'), dict(shard=0)))
    with pytest.raises(ValueError, match='resource bound'):
        list(stream.feed(np.array([9,10],dtype='<u2'), dict(shard=1)))
    stream = NativeRecordStream()
    assert not list(stream.feed(np.array([1,7,8],dtype='<u2'), dict(shard=0)))
    assert list(stream.finish())[0][0] == [7,8]
    assert stream.input_tokens == 3 and stream.output_tokens == 2 and stream.boundary_tokens == 1
