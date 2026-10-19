import itertools
import json
from pathlib import Path

import pytest

from slm_lab.diverse_images import jsonl
from slm_lab.rendered_text import HELD_OUT_FONTS, prepare, render, text_group

FONTS = Path('/usr/share/fonts/truetype/liberation')
NOTICE = Path('/usr/share/doc/fonts-liberation/copyright')
pytestmark = pytest.mark.skipif(not NOTICE.exists(), reason='Liberation fonts unavailable')


def test_renderer_keeps_visible_ink_and_is_reproducible():
    font = FONTS / 'LiberationSans-Regular.ttf'
    first, _ = render('Agj012!?', font, 5)
    again, _ = render('Agj012!?', font, 5)
    assert first.tobytes() == again.tobytes()
    assert first.width > 100 and first.height > 30
    low, high = first.convert('L').getextrema()
    assert high - low >= 150
    with pytest.raises(ValueError, match='supported ASCII'):
        render('unsupported\nline', font, 5)


def test_lettering_splits_hold_out_text_and_fonts_and_protect_real_labels(tmp_path):
    books = tmp_path / 'books.jsonl'
    words = ['word' + ''.join(chars) for chars in itertools.islice(itertools.product('abcdefghijklm', repeat=2), 120)]
    source = {'source': 'Project Gutenberg', 'license': 'Public-domain-US', 'id': 'book-chunk',
              'group_id': 'gutenberg:example', 'title': 'Fixture', 'source_url': 'https://example.org/book',
              'text': ' '.join(words)}
    books.write_text(json.dumps(source) + '\n')
    questions = tmp_path / 'questions.jsonl'
    questions.write_text(json.dumps({'id': 'reserved', 'question': 'A reserved benchmark question'}) + '\n')
    protected = tmp_path / 'real-valid.jsonl'
    protected.write_text(json.dumps({'task': 'word_ocr', 'response': 'A'}) + '\n')
    output = tmp_path / 'output'
    args = (output, books, questions, protected, FONTS, NOTICE, 150, 40, 31)
    manifest = prepare(*args)
    train = list(jsonl(output / 'train.jsonl')); valid = list(jsonl(output / 'valid.jsonl'))
    assert len(train) == 150 and len(valid) == 40
    assert {text_group(row['response']) for row in train}.isdisjoint(text_group(row['response']) for row in valid)
    assert all(text_group(row['response']) != text_group('A') for row in train)
    assert not {row['font'] for row in train} & HELD_OUT_FONTS
    assert {row['font'] for row in valid} <= HELD_OUT_FONTS
    for size in [224, 448]:
        assert {row[f'model_pixel_sha256_{size}'] for row in train}.isdisjoint(
            row[f'model_pixel_sha256_{size}'] for row in valid)
    assert any(row['public_text_origin'] for row in train)
    assert prepare(*args) == manifest
    # Partial and completed collections must reject a changed generation budget.
    (output / 'manifest.json').unlink()
    with pytest.raises(ValueError, match='different configuration'):
        prepare(*args[:6], 151, 40, 31)
    resumed = prepare(*args)
    assert resumed['files_sha256'] == manifest['files_sha256']
