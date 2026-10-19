import copy

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from slm_lab.complete_js_parameter_admission import reconstruct, sha, unchanged
from slm_lab.model import ModelConfig
from slm_lab.training import Encoder


def original_and_parser():
    code = 'function configure(options = {}) { return options; }'
    row = {'id': 'original-task', 'group_id': 'github:owner/repo:file.js', 'source': 'CodeSearchNet',
           'language': 'JavaScript', 'source_url': 'https://example.test/pinned/file.js', 'license': 'MIT',
           'source_code_sha256': sha(code), 'prompt': 'Implement configuration.\nSignature/context: function configure(options = \n\nKeep the documented default.',
           'response': '```javascript\n' + code + '\n```'}
    parsed = {'parsed': True, 'parameter_text_cut': True, 'original_full_header': 'function configure(options = {}) ',
              'node_type': 'FunctionDeclaration', 'context': 'original-source'}
    tokenizer = Tokenizer(models.WordLevel({'<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    encoder = Encoder(tokenizer, ModelConfig(vocab_size=4, max_text_length=128), '.', 448)
    return row, parsed, encoder


def test_reconstruction_restores_default_object_parameter_without_rewriting_code_or_source():
    original, parsed, encoder = original_and_parser()
    result, length = reconstruct(original, parsed, encoder)
    assert 'Signature/context: function configure(options = {}) \n\n' in result['prompt']
    assert result['response'] == original['response'] and result['source_url'] == original['source_url']
    assert result['group_id'] == original['group_id'] and result['license'] == original['license']
    assert result['id'] != original['id'] and result['split'] == 'train' and length <= 128


@pytest.mark.parametrize('field', ['response', 'prompt', 'license', 'source_url', 'group_id', 'split', 'actual_target_tokens'])
def test_altered_prepared_answer_source_split_or_targets_are_not_admitted(field):
    original, parsed, encoder = original_and_parser(); expected, _ = reconstruct(original, parsed, encoder)
    altered = copy.deepcopy(expected); altered[field] = 'changed'
    with pytest.raises(ValueError): unchanged(altered, expected)


def test_incomplete_parser_result_or_wrong_source_code_identity_cannot_make_a_variant():
    original, parsed, encoder = original_and_parser()
    for mutation in ('unparsed', 'unchanged_header', 'wrong_code', 'wrong_header'):
        source = copy.deepcopy(original); header = copy.deepcopy(parsed)
        if mutation == 'unparsed': header['parsed'] = False
        if mutation == 'unchanged_header': header['parameter_text_cut'] = False
        if mutation == 'wrong_code': source['source_code_sha256'] = 'different'
        if mutation == 'wrong_header': header['original_full_header'] = 'another implementation'
        with pytest.raises(ValueError): reconstruct(source, header, encoder)


def test_long_original_code_is_rejected_instead_of_inserting_a_partial_target():
    original, parsed, encoder = original_and_parser()
    code = 'function configure(options = {}) { ' + 'value; ' * 150 + '}'
    original.update(response='```javascript\n' + code + '\n```', source_code_sha256=sha(code))
    with pytest.raises(ValueError, match='clipped'): reconstruct(original, parsed, encoder)
