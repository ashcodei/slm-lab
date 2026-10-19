from types import SimpleNamespace

import pytest

from slm_lab.instruction_generalization_trial import LANGUAGES, complete_encoding, schedule


class Encoder:
    config = SimpleNamespace(max_text_length=32)
    bos = 1; eos = 2
    def tokens(self, text): return list(text.encode())
    def prompt(self, row): return row['prompt']
    def sequence(self, row, field):
        prompt = self.tokens(row['prompt']); response = self.tokens(row[field])
        return [self.bos] + prompt + response + [self.eos], [-100] * (len(prompt) + 1) + response + [self.eos]


def test_original_complete_prompt_response_and_loss_mask_are_required():
    row = {'prompt': 'task', 'response': 'answer'}; encoder = Encoder()
    assert complete_encoding(row, encoder) == {'context_tokens':12, 'target_tokens':7}
    with pytest.raises(ValueError): complete_encoding({'prompt':'a'*25, 'response':'answer'}, encoder)
    with pytest.raises(ValueError): complete_encoding({'prompt':'a', 'response':'b'*16}, encoder)


def test_mutated_actual_response_mask_is_rejected():
    class Broken(Encoder):
        def sequence(self, row, field):
            ids, labels = super().sequence(row, field); labels[1] = 9; return ids, labels
    with pytest.raises(ValueError): complete_encoding({'prompt':'task','response':'answer'}, Broken())


def test_each_three_updates_covers_every_original_module_language_equally():
    pools = {name: [object()] * 10 for name in [*LANGUAGES,'verified_python','human_instruction']}
    pools['verified_python'] = [{'group_id':str(i//2)} for i in range(10)]
    result = schedule(pools, 1500, 11)
    from collections import Counter
    counts = Counter(pick['pool'] for row in result for pick in row)
    assert all(counts[name] == 500 for name in LANGUAGES)
    assert counts['verified_python'] == counts['human_instruction'] == 1500
    assert result == schedule(pools, 1500, 11)


def test_every_source_pool_is_required_and_no_empty_language_is_silently_removed():
    pools = {name: [object()] for name in [*LANGUAGES,'verified_python','human_instruction']}
    pools['verified_python'] = [{'group_id':'original'}]
    pools['Rust'] = []
    with pytest.raises(ValueError): schedule(pools, 12, 11)
    del pools['Rust']
    with pytest.raises(ValueError): schedule(pools, 12, 11)


def test_original_algorithms_are_seen_before_repeated_name_variants():
    pools = {name: [object()] * 8 for name in [*LANGUAGES,'human_instruction']}
    pools['verified_python'] = [{'group_id':str(i//4)} for i in range(8)]
    result = schedule(pools, 8, 17)
    names = [pools['verified_python'][row[2]['index']]['group_id'] for row in result]
    assert len(set(names[:2])) == 2 and names[0] == names[2] == names[4] == names[6]
