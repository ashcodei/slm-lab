import pytest

from slm_lab.verified_code_diagnostics import select


class Tokenizer:
    def encode(self, text):
        return type('Encoding', (), {'ids': list(text)})()


def test_seen_code_selection_does_not_confuse_name_variants_or_oversized_programs():
    def row(identifier, algorithm, variant=0, response='code'):
        return {'id': identifier, 'original_program_sha256': algorithm, 'variant_index': variant,
                'response': response, 'verification_cases': [{'args_literal': '()', 'expected_literal': '0'}]}
    rows = [row('original', 'a'), row('renamed', 'a', 1), row('second', 'b'), row('long', 'c', response='x' * 300)]
    chosen = select(rows, Tokenizer(), count=2)
    assert {r['original_program_sha256'] for r in chosen} == {'a', 'b'}
    assert all(r['variant_index'] == 0 for r in chosen)
    assert select(list(reversed(rows)), Tokenizer(), count=2) == chosen
    with pytest.raises(ValueError, match='Not enough'):
        select(rows, Tokenizer(), count=3)
