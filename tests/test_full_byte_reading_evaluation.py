import pytest
import torch

from slm_lab import full_byte_reading_evaluation as evaluation
from slm_lab.ocr_evaluation import transcription_scores


def row(identifier, group, text, family='real_word'):
    return {'id': identifier, 'group_id': group, 'response': text, 'byte_family': family, 'stored_image_sha256': identifier}


def test_controls_require_different_original_image_groups_and_transcripts():
    rows = [row('a', 'one', 'same'), row('b', 'one', 'other'), row('c', 'two', 'third')]
    choices = evaluation.counterparts(rows); indexed = {r['id']: r for r in rows}
    for case in rows:
        other = indexed[choices[case['id']]]
        assert other['group_id'] != case['group_id'] and other['response'] != case['response']
    with pytest.raises(ValueError): evaluation.counterparts([row('a', 'one', 'same'), row('b', 'two', 'same')])
    with pytest.raises(ValueError): evaluation.counterparts([row('a', 'one', 'first'), row('a', 'two', 'second')])


def test_cer_counts_characters_instead_of_averaging_word_error_ratios():
    rows = []
    for expected, generated in [('a', ''), ('abcdefghij', 'abcdefghij')]:
        result = {'group_id': 'same-photo'}
        for name in ['proper', 'blank', 'wrong']:
            result[name + '_scores'] = transcription_scores(expected, generated)
            result[name + '_valid_utf8'] = True
        rows.append(result)
    result = evaluation.summarize(rows)
    assert result['examples'] == 2 and result['original_groups'] == 1
    assert result['proper_character_error'] == pytest.approx(1 / 11)


def test_all_cases_are_inferred_and_actual_counterpart_outputs_reused(tmp_path, monkeypatch):
    rows = [row('a', 'photo-one', 'A'), row('b', 'photo-two', 'B'), row('c', 'photo-three', 'C')]
    monkeypatch.setattr(evaluation, 'pixels', lambda r: torch.full((3, 32, 512), ord(r['response']) / 100.))
    sizes = []
    class Reader:
        def eval(self): return self
        def __call__(self, pixels):
            sizes.append(len(pixels)); logits = torch.zeros(len(pixels), 128, 257); logits[:, :, 0] = 1
            for index, value in enumerate(pixels):
                if value.max() > 0: logits[index, 0, round(float(value[0, 0, 0]) * 100) + 1] = 10
            return logits
    report = evaluation.evaluate(Reader(), rows, tmp_path / 'reading.json', 'cpu', batch_size=2)
    assert sizes == [2, 1, 1] and report['overall']['examples'] == 3
    assert report['overall']['proper_exact'] == 3 and report['overall']['wrong_exact'] == 0
    indexed = {r['id']: r for r in report['results']}
    for result in report['results']:
        assert result['wrong_generated'] == indexed[result['wrong_image_id']]['proper_generated']
        assert result['wrong_pixels_sha256'] == indexed[result['wrong_image_id']]['input_pixels_sha256']
