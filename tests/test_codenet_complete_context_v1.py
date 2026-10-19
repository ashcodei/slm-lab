from types import SimpleNamespace

import pytest

from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder, CompleteContextDoesNotFit, problem_program_row
from slm_lab.training import Encoder


class Tokenizer:
    def token_to_id(self, text):
        return {'<bos>': 1, '<eos>': 2, '<pad>': 0}[text]

    def get_vocab_size(self):
        return 1000

    def encode(self, text):
        return SimpleNamespace(ids=list(map(ord, text)))


def encoder(tmp_path, cls=CompleteCodeEncoder):
    config = SimpleNamespace(max_text_length=64, image_size=224, vocab_size=1000,
                             image_token_count=lambda _: 0)
    return cls(Tokenizer(), config, tmp_path)


def test_program_longer_than_half_window_retains_every_target(tmp_path):
    row = {'prompt': 'p', 'response': 'r' * 40}
    complete = encoder(tmp_path)
    ids, labels = complete.sequence(row, 'response')
    assert ids == [1] + list(map(ord, 'User: p\nAssistant:')) + [ord('r')] * 40 + [2]
    assert [x for x in labels if x != -100] == [ord('r')] * 40 + [2]
    assert complete.truncated == 0
    legacy = encoder(tmp_path, Encoder)
    legacy.sequence(row, 'response')
    assert legacy.truncated == 1


def test_whole_window_boundary_and_overflow_never_clip_prompt_or_program(tmp_path):
    e = encoder(tmp_path)
    prefix = len(e.tokens(e.prompt({'prompt': 'p'})))
    size = 64 - prefix - 2
    ids, labels = e.sequence({'prompt': 'p', 'response': 'x' * size}, 'response')
    assert len(ids) == len(labels) == 64
    with pytest.raises(CompleteContextDoesNotFit) as caught:
        e.sequence({'prompt': 'p', 'response': 'x' * (size + 1)}, 'response')
    assert (caught.value.prompt_tokens, caught.value.response_tokens, caught.value.limit) == (prefix, size + 1, 64)
    assert e.truncated == 0


def test_inherited_batch_masks_only_padding_and_prompt(tmp_path):
    e = encoder(tmp_path)
    batch = e.batch([{'prompt': 'p', 'response': 'x' * 40}, {'prompt': 'p', 'response': 'y'}], 'response', 'cpu')
    assert (batch['labels'][0] != -100).sum().item() == 41
    assert (batch['labels'][1] != -100).sum().item() == 2
    assert batch['labels'][1, -1].item() == -100
    assert batch['attention_mask'][1].sum().item() < batch['attention_mask'][0].sum().item()


def test_program_framing_preserves_original_code_and_runtime():
    p = {'problem_id': 'p00001', 'text': 'Read two integers and print their sum.',
         'requires_media_review': False, 'sample_parse_issues': []}
    m = {'problem_id': 'p00001', 'submission_id': 's000000001', 'language': 'Python',
         'original_language': 'Python (2.7.6)', 'status': 'Accepted'}
    code = '# original source\na,b=map(int,raw_input().split())\nprint a+b\n'
    row = problem_program_row(p, m, code)
    assert row['response'] == code and p['text'] in row['prompt']
    assert row['group_id'] is None and row['split'] is None
    assert 'Python (2.7.6)' in row['prompt']
    p['requires_media_review'] = True
    with pytest.raises(ValueError, match='media review'):
        problem_program_row(p, m, code)
