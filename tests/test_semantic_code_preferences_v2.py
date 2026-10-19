import hashlib
import json
from types import SimpleNamespace

import pytest

from slm_lab.code_quality import execute, instruction_code
from slm_lab.corpus import file_hash
from slm_lab.evaluation_provenance import capture
from slm_lab.semantic_code_preferences_v2 import prepare


class Encoder:
    config = SimpleNamespace(max_text_length=2048)
    bos = 1
    eos = 2
    truncated = 0

    def prompt(self, row): return row['prompt']
    def tokens(self, text): return list(text.encode())
    def sequence(self, row, field):
        prompt, response = self.tokens(row['prompt']), self.tokens(row[field])
        return [self.bos] + prompt + response + [self.eos], [-100] * (len(prompt) + 1) + response + [self.eos]


def fixture(tmp_path, rejected='def identity(x):\n    return 0\n'):
    data = tmp_path / 'verified_python.jsonl'
    row = {'id': 'original-one', 'function': 'identity', 'variant_index': 0,
           'original_program_sha256': 'algorithm-one', 'group_id': 'publisher:file',
           'source': 'public test fixture', 'license': 'CC0',
           'prompt': 'Implement identity(x), returning x.',
           'response': 'def identity(x):\n    return x\n',
           'verification_cases': [{'args_literal': '(1,)', 'expected_literal': '1'},
                                  {'args_literal': '(2,)', 'expected_literal': '2'}]}
    data.write_text(json.dumps(row) + '\n')
    exposures = tmp_path / 'exposures.jsonl'
    exposures.write_text(json.dumps({'pool': 'verified_python', 'id': row['id']}) + '\n')
    (tmp_path / 'manifest.json').write_text(json.dumps({'files_sha256':
        {path.name: file_hash(path) for path in [data, exposures]}}))
    checkpoint = tmp_path / 'own.pt'; checkpoint.write_bytes(b'own model identity fixture')
    diagnostic = tmp_path / 'after.json'
    report = {'data_sha256': file_hash(data), 'actual_exposures_sha256': file_hash(exposures),
        'checkpoint_sha256': file_hash(checkpoint), 'exact_language_and_critic_preserved': True,
        'examples': 1, 'evaluation_provenance': capture(),
        'results': [{'id': row['id'], 'original_program_sha256': row['original_program_sha256'],
            'function': row['function'], 'prompt_sha256': hashlib.sha256(row['prompt'].encode()).hexdigest(),
            'published_cases': 2, 'target_tokens': len(row['response'].encode()) + 1,
            'raw_response': rejected, 'candidate': execute({'language': 'Python',
                'code': instruction_code(rejected), 'function': row['function'],
                'cases': row['verification_cases'], 'stress': []})}]}
    diagnostic.write_text(json.dumps(report))
    return [data, exposures, diagnostic, checkpoint, tmp_path / 'pairs', Encoder()], row, report


def test_runnable_wrong_result_retains_exact_public_positive_and_own_negative(tmp_path):
    args, original, report = fixture(tmp_path)
    manifest = prepare(*args)
    pair = json.loads((args[4] / 'train.jsonl').read_text())
    assert manifest['train_examples'] == 1 and manifest['data_ready']
    assert pair['chosen'] == original['response'] and pair['rejected'] == report['results'][0]['raw_response']
    assert pair['group_id'] == original['group_id'] and pair['feedback_source'] == 'synthetic'
    assert pair['verification']['preferred']['functional_passed']
    assert pair['verification']['rejected']['error'] == 'Hidden case mismatch'
    assert not manifest['human_feedback'] and not manifest['external_teacher_used']


@pytest.mark.parametrize('response,reason', [
    ('def identity(x) return x', 'invalid_syntax'),
    ('def different(x):\n    return 0\n', 'missing_callable_or_restricted_module'),
    ('def identity(x):\n    return 1 / 0\n', 'runtime_error_or_timeout_instead_of_wrong_result'),
    ('def identity(x):\n    return x\n', 'own_generation_already_passes'),
])
def test_other_failure_families_are_not_silently_labeled_wrong_result_pairs(tmp_path, response, reason):
    args, _, _ = fixture(tmp_path, response)
    manifest = prepare(*args)
    assert manifest['train_examples'] == 0 and manifest['excluded'] == {reason: 1}
    assert not manifest['data_ready'] and (args[4] / 'train.jsonl').read_text() == ''


def test_forged_failure_and_changed_prompt_are_rejected(tmp_path):
    args, _, report = fixture(tmp_path)
    report['results'][0]['candidate']['passed_cases'] = 2
    args[2].write_text(json.dumps(report))
    with pytest.raises(ValueError, match='not reproducible'): prepare(*args)
    report['results'][0]['candidate']['passed_cases'] = 0
    report['results'][0]['prompt_sha256'] = 'forged prompt'
    args[2].write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Changed or repeated'): prepare(*args)
    assert not args[4].exists()


def test_unexposed_generation_is_rejected_even_with_consistent_source_checksums(tmp_path):
    args, _, report = fixture(tmp_path)
    args[1].write_text(json.dumps({'pool': 'verified_python', 'id': 'different'}) + '\n')
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    manifest['files_sha256'][args[1].name] = file_hash(args[1])
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    report['actual_exposures_sha256'] = file_hash(args[1]); args[2].write_text(json.dumps(report))
    with pytest.raises(ValueError, match='actually exposed'): prepare(*args)


def test_duplicate_algorithms_and_existing_output_are_rejected(tmp_path):
    args, _, report = fixture(tmp_path)
    report['results'] *= 2; report['examples'] = 2; args[2].write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Changed or repeated'): prepare(*args)
    args[4].mkdir()
    with pytest.raises(ValueError, match='Preserve existing'): prepare(*args)
