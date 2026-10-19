import pytest
import json
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from slm_lab.corpus import file_hash
from slm_lab.exposure_audit import StreamMatcher, scan_corpus


def tokenizer():
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {text: index for index, text in enumerate(['<pad>', '<bos>', '<eos>', '<unk>', *alphabet])}
    tok = Tokenizer(models.BPE(vocab, [], unk_token='<unk>'))
    tok.add_special_tokens(['<pad>', '<bos>', '<eos>', '<unk>'])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
    return tok


def test_unicode_casefold_and_long_whitespace_match_across_every_byte_boundary():
    tok = tokenizer(); matcher = StreamMatcher(tok, {'question-a': 'STRASSE CAFÉ'})
    text = 'Prefix Straße' + ' ' * 1000 + 'café suffix'
    assert tok.decode(tok.encode(text).ids) == text
    for index, token in enumerate(tok.encode(text).ids): matcher.feed([token], 'shard-' + str(index % 3))
    matches = matcher.finish()
    assert list(matches) == ['question-a'] and matches['question-a']['document_index'] == 0


def test_question_words_in_unrelated_documents_are_never_joined_into_a_match():
    tok = tokenizer(); matcher = StreamMatcher(tok, {'question-a': 'alpha beta'})
    matcher.feed(tok.encode('alpha').ids + [tok.token_to_id('<eos>')] + tok.encode('beta').ids)
    assert matcher.finish() == {}


def test_shared_question_wording_preserves_all_distinct_benchmark_identities():
    tok = tokenizer(); matcher = StreamMatcher(tok, {'knowledge:a': 'Question text?', 'math:b': 'question TEXT?'})
    matcher.feed(tok.encode('Intro. Question text? End.').ids)
    assert set(matcher.finish()) == {'knowledge:a', 'math:b'}


def test_empty_patterns_and_ids_outside_own_vocabulary_are_rejected():
    tok = tokenizer()
    with pytest.raises(ValueError): StreamMatcher(tok, {'a': '  '})
    matcher = StreamMatcher(tok, {'a': 'question'})
    with pytest.raises(ValueError): matcher.feed([99999])


def packed_source(root):
    tok = tokenizer(); path = root / 'tokenizer.json'; tok.save(str(path))
    ids = tok.encode('prefix Straße café suffix').ids + [tok.token_to_id('<eos>')]
    # Split the UTF-8 representation of é itself across different token files.
    split = len(tok.encode('prefix Straße caf').ids) + 1
    shards = []
    for index, values in enumerate([ids[:split], ids[split:]]):
        shard = root / ('train-' + str(index) + '.bin'); np.asarray(values, dtype='<u2').tofile(shard)
        shards.append({'path': shard.name, 'tokens': len(values), 'sha256': file_hash(shard)})
    (root / 'corpus.json').write_text(json.dumps({'tokenizer_sha256': file_hash(path), 'dtype': '<u2',
             'splits': {'train': {'tokens': len(ids), 'shards': shards}}}))
    return path


def test_actual_packed_files_preserve_unicode_question_matches_across_shards(tmp_path):
    path = packed_source(tmp_path)
    result = scan_corpus(tmp_path, path, {'question-a': 'STRASSE CAFÉ'}, lambda value: None)
    assert set(result['matches']) == {'question-a'} and result['actual_training_shards_rehashed'] == 2


def test_changed_packed_bytes_cannot_receive_a_completed_exposure_audit(tmp_path):
    path = packed_source(tmp_path)
    (tmp_path / 'train-0.bin').write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='training shard changed'):
        scan_corpus(tmp_path, path, {'question-a': 'STRASSE CAFÉ'}, lambda value: None)
