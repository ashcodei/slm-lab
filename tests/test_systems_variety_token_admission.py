import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from slm_lab.corpus import stable_id
from slm_lab.systems_variety_training_admission import verify_stream


def inputs():
    tokenizer = Tokenizer(models.WordLevel({'<unk>': 0, '<eos>': 1, 'original': 2, 'code': 3, 'second': 4}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    groups = {split: next('file' + str(i) for i in range(1000) if ('valid' if int(stable_id('file' + str(i))[:8], 16) % 10 == 0 else 'train') == split)
              for split in ('train', 'valid')}
    rows = [{'group_id': groups['train'], 'text': 'original code'}, {'group_id': groups['valid'], 'text': 'second code'}]
    arrays = {'train': [np.array([2, 3], dtype=np.uint16), np.array([1], dtype=np.uint16)],
              'valid': [np.array([4, 3, 1], dtype=np.uint16)]}
    return rows, tokenizer, arrays


def test_complete_document_and_eos_are_verified_across_shard_boundaries():
    rows, tokenizer, arrays = inputs(); result = verify_stream(rows, tokenizer, arrays)
    assert result['actual_complete_tokens'] == {'train': 3, 'valid': 3} and result['source_file_group_overlap'] == 0


@pytest.mark.parametrize('change', ['token', 'eos', 'missing', 'extra', 'partition', 'original_text'])
def test_rehashed_streams_still_require_the_original_complete_token_sequences(change):
    rows, tokenizer, arrays = inputs()
    if change == 'token': arrays['train'][0][0] = 4
    elif change == 'eos': arrays['train'][1][0] = 0
    elif change == 'missing': arrays['train'].pop()
    elif change == 'extra': arrays['train'].append(np.array([2], dtype=np.uint16))
    elif change == 'partition': arrays['train'], arrays['valid'] = arrays['valid'], arrays['train']
    else: rows[0]['text'] = 'second code'
    with pytest.raises(ValueError): verify_stream(rows, tokenizer, arrays)
