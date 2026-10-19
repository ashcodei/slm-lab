import copy
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from slm_lab.goruby_code_headers import HeaderParser
from slm_lab.goruby_header_admission import reconstruct, sha, unchanged
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder

ROOT = Path(__file__).resolve().parents[1]


def inputs(language, code):
    old = code.split('{', 1)[0][:500]
    row = {'id': 'original', 'group_id': 'github:owner/repo:file', 'language': language,
        'source': 'CodeSearchNet', 'license': 'MIT', 'source_url': 'https://example.test/pinned',
        'source_code_sha256': sha(code), 'prompt': 'Implement the documented method.\nSignature/context: ' + old + '\n\nFollow the documented behavior.',
        'response': '```' + language.lower() + '\n' + code + '\n```'}
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    encoder = Encoder(tokenizer, ModelConfig(vocab_size=4, max_text_length=256), '.', 448)
    return row, encoder


@pytest.mark.parametrize('language,code,expected', [
    ('Go', 'func inspect(value interface{}) interface{} { return value }', 'func inspect(value interface{}) interface{} '),
    ('Go', 'func create() struct { Name string } { return struct { Name string }{} }', 'func create() struct { Name string } '),
    ('Ruby', 'def send(value = {})\n value\nend', 'def send(value = {})'),
    ('Ruby', 'def self.send(value)\n value\nend', 'def self.send(value)'),
    ('Ruby', 'def 名前(value = {})\n value\nend', 'def 名前(value = {})'),
])
def test_real_grammar_restores_interface_without_changing_original_code_or_metadata(language, code, expected):
    original, encoder = inputs(language, code); parsed = HeaderParser(ROOT).measure(language, code)
    variant, length = reconstruct(original, parsed, encoder)
    assert variant['complete_source_header'] == expected and variant['response'] == original['response']
    assert variant['license'] == original['license'] and variant['group_id'] == original['group_id']
    assert variant['source_url'] == original['source_url'] and variant['split'] == 'train' and length <= 256


def test_method_body_copied_into_ruby_signature_is_removed_from_prompt_context():
    code = 'def send(value)\n copied_body_statement(value)\nend'
    original, encoder = inputs('Ruby', code); parsed = HeaderParser(ROOT).measure('Ruby', code)
    variant, _ = reconstruct(original, parsed, encoder)
    assert parsed['method_body_text_in_context'] and 'copied_body_statement' not in variant['prompt']
    assert 'copied_body_statement' in variant['response']


def test_altered_parent_implementation_or_prepared_license_is_rejected():
    code = 'func inspect(value interface{}) interface{} { return value }'
    original, encoder = inputs('Go', code); parsed = HeaderParser(ROOT).measure('Go', code)
    expected, _ = reconstruct(original, parsed, encoder)
    altered = copy.deepcopy(expected); altered['license'] = 'different'
    with pytest.raises(ValueError): unchanged(altered, expected)
    original['source_code_sha256'] = 'another implementation'
    with pytest.raises(ValueError): reconstruct(original, parsed, encoder)


def test_long_code_target_is_not_silently_clipped_when_interface_is_rebuilt():
    code = 'func inspect(value interface{}) interface{} { ' + 'value; ' * 300 + 'return value }'
    original, encoder = inputs('Go', code); parsed = HeaderParser(ROOT).measure('Go', code)
    with pytest.raises(ValueError, match='clipped'): reconstruct(original, parsed, encoder)
