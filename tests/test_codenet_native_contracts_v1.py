from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from slm_lab.codenet_complete_context_v1 import CompleteCodeEncoder, CompleteContextDoesNotFit
from slm_lab.codenet_native_contracts_v1 import CONTRACT, native_inventory, with_native_contract


def fixture():
    code = """import sys
from numba.pycc import CC
if sys.argv[-1] == 'ONLINE_JUDGE':
    cc = CC('my_module')
    cc.compile()
from my_module import main
print(main(input()))
"""
    row = dict(id='original', language='Python', original_judge_status='Accepted',
        original_archive_member='Project_CodeNet/data/p00001/Python/s000000001.py',
        original_source_body_sha256=hashlib.sha256(code.encode()).hexdigest(),
        source_problem_id='p00001', group_id='p00001', split='train', response=code,
        prompt='Write a complete Python program.\nTarget language/runtime: Python (3.8.2).\n'
        'Use standard input and output as the problem specifies. Return only the complete program.\n\n'
        'Original complete statement\nincluding its sample input/output.')
    q = {k: row[k] for k in ['original_archive_member', 'original_source_body_sha256', 'source_problem_id', 'group_id', 'split']}
    q.update(source_declared_preparation_flag='ONLINE_JUDGE', numba_pycc_generated_modules=['my_module'],
             generated_modules_imported_at_runtime=['my_module'])
    return row, q


class CharacterTokenizer:
    def token_to_id(self, text):
        return {'<pad>': 0, '<bos>': 1, '<eos>': 2}[text]

    def get_vocab_size(self):
        return 32768

    def encode(self, text):
        return SimpleNamespace(ids=list(map(ord, text)))


def test_contract_preserves_source_statement_roles_and_full_answer_tokens(tmp_path):
    row, q = fixture(); before = deepcopy(row)
    revised = with_native_contract(row, q, 'a' * 64)
    assert row == before and revised['response'] == row['response']
    assert revised['prompt'].split('\n\n', 1)[1] == row['prompt'].split('\n\n', 1)[1]
    assert CONTRACT in revised['prompt'] and 'may continue into input' in CONTRACT
    assert revised['split'] == row['split'] and revised['group_id'] == row['group_id']
    config = SimpleNamespace(vocab_size=32768, image_size=224, max_text_length=4096,
                             image_token_count=lambda _: 0)
    enc = CompleteCodeEncoder(CharacterTokenizer(), config, tmp_path)
    old_ids, old_labels = enc.sequence(row, 'response'); ids, labels = enc.sequence(revised, 'response')
    assert [x for x in labels if x != -100] == [x for x in old_labels if x != -100]
    assert len(ids) > len(old_ids) and labels[-1] == 2 and not enc.truncated
    config.max_text_length = len(old_ids)
    with pytest.raises(CompleteContextDoesNotFit):
        enc.sequence(revised, 'response')


@pytest.mark.parametrize('field,value', [('split','test'), ('group_id','another'),
    ('original_archive_member','another.py'), ('original_source_body_sha256','0'*64)])
def test_changed_source_identity_or_held_role_is_rejected(field, value):
    row, q = fixture(); row[field] = value
    with pytest.raises(ValueError, match='identity and sealed role'):
        with_native_contract(row, q, 'a' * 64)


def test_source_digest_and_actual_module_evidence_are_both_required():
    row, q = fixture(); row['response'] += '# changed bytes\n'
    with pytest.raises(ValueError, match='no longer supports'):
        with_native_contract(row, q, 'a' * 64)
    row, q = fixture(); row['response'] = row['response'].replace('from my_module import main','from other_module import main')
    row['original_source_body_sha256'] = q['original_source_body_sha256'] = hashlib.sha256(row['response'].encode()).hexdigest()
    with pytest.raises(ValueError, match='no longer supports'):
        with_native_contract(row, q, 'a' * 64)


def test_optional_compilation_inventory_does_not_rewrite_the_source():
    row, _ = fixture()
    row['response'] = "from numba.pycc import CC\nUSE_NUMBA = False\ndef optional():\n    cc=CC('my_module')\n    cc.compile()\nprint(input())\n"
    before = deepcopy(row); facts = native_inventory(row)
    assert facts['declared_CC_modules'] == ['my_module'] and not facts['generated_module_imports']
    assert row == before and 'execution_contract' not in row
    row['language'] = 'C++'; assert native_inventory(row) is None


def test_a_contract_cannot_be_applied_twice():
    row, q = fixture(); revised = with_native_contract(row, q, 'a' * 64)
    with pytest.raises(ValueError, match='twice'):
        with_native_contract(revised, q, 'a' * 64)
