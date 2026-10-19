import copy
from pathlib import Path

import pytest
from PIL import Image

from slm_lab.ai2d_native_eval_inputs_v1 import native_pixels, native_prompt, opaque_labels, source_gold


def test_original_choices_and_gold_are_separate_and_not_pruned():
    choices = ['same', '{}', 'same', 'other']
    prompt = native_prompt('Original question?', choices)
    assert 'A. same\nB. {}\nC. same\nD. other' in prompt
    assert source_gold(dict(correctAnswer='2', answerTexts=choices)) == 2
    changed = dict(correctAnswer=0, answerTexts=choices)
    assert native_prompt('Original question?', changed['answerTexts']) == prompt
    with pytest.raises(ValueError): source_gold(dict(correctAnswer=True, answerTexts=choices))


def test_ABC_pixels_depend_only_on_literal_source_geometry_and_replacement_labels():
    font = Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf')
    original = Image.new('RGB', (128, 96), 'white')
    annotations = dict(text=dict(T0=dict(rectangle=[[10, 10], [70, 50]], replacementText='A', value='ANSWER MUST NOT BE USED'),
        T1=dict(rectangle=[[80, 60], [110, 60]], replacementText='B', value='ignored')), relationships=dict(secret='ignored'))
    image, trace = opaque_labels(original, annotations, font)
    changed = copy.deepcopy(annotations); changed['text']['T0']['value'] = 'different gold text'; changed['relationships'] = {}
    other, _ = opaque_labels(original, changed, font)
    assert image.tobytes() == other.tobytes()
    assert image.getpixel((10, 10)) == (25, 249, 255) and original.getpixel((10, 10)) == (255, 255, 255)
    assert trace[1]['source_zero_height_box_skipped']
    padded, raw_sha, tensor_sha = native_pixels(image)
    assert padded.size == (448, 448) and len(raw_sha) == len(tensor_sha) == 64 and raw_sha != tensor_sha
