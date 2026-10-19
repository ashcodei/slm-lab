import json

from slm_lab.code_execution_preferences import prepare
from slm_lab.corpus import file_hash


def test_execution_preferences_preserve_raw_code_and_do_not_claim_human_feedback(tmp_path):
    data = tmp_path / 'source'; data.mkdir()
    row = {'id': 'one', 'function': 'identity', 'variant_index': 0, 'original_program_sha256': 'algorithm',
           'group_id': 'source:file', 'source': 'public test fixture', 'license': 'CC0',
           'prompt': 'Implement identity(x), returning x.', 'response': '```python\ndef identity(x):\n    return x\n```',
           'verification_cases': [{'args_literal': '(1,)', 'expected_literal': '1'},
                                  {'args_literal': '(2,)', 'expected_literal': '2'}]}
    (data / 'train.jsonl').write_text(json.dumps(row) + '\n')
    (data / 'manifest.json').write_text(json.dumps({'train_sha256': file_hash(data / 'train.jsonl')}))
    checkpoint = tmp_path / 'own.pt'; checkpoint.write_bytes(b'own checkpoint identity fixture')
    bad = '```python\ndef identity(x):\n    return 0\n```'
    diagnostic = tmp_path / 'diagnostic.json'
    diagnostic.write_text(json.dumps({'data_sha256': file_hash(data / 'train.jsonl'),
        'checkpoint_sha256': file_hash(checkpoint), 'exact_language_and_critic_preserved': True,
        'results': [{'id': 'one', 'original_program_sha256': 'algorithm', 'function': 'identity',
                     'raw_response': bad, 'candidate': {'functional_passed': False}}]}))
    manifest = prepare(data / 'train.jsonl', diagnostic, checkpoint, tmp_path / 'pairs')
    pair = json.loads((tmp_path / 'pairs/train.jsonl').read_text())
    assert manifest['train_examples'] == 1 and manifest['human_feedback'] is False
    assert pair['feedback_source'] == 'synthetic' and pair['chosen'] == row['response'] and pair['rejected'] == bad
    assert pair['verification']['preferred']['functional_passed'] is True
    assert pair['verification']['rejected']['functional_passed'] is False
