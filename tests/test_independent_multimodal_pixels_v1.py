import hashlib

import numpy as np
from PIL import Image
import pytest

from slm_lab.independent_multimodal_pixels_v1 import observe, merge, differences
from slm_lab.multimodal_pixel_protocols_v1 import observe_pixels
from slm_lab.shared_word_data_v1 import image_arrays, array_digest


@pytest.mark.parametrize('mode,shape,extension', [('RGB', (23, 71), 'png'), ('RGB', (71, 23), 'jpg'),
    ('L', (35, 35), 'png'), ('RGBA', (19, 103), 'png')])
def test_actual_independent_decodes_match_training_and_producer_in_both_geometries(tmp_path, mode, shape, extension):
    h, w = shape
    channels = 1 if mode == 'L' else len(mode)
    values = np.arange(h * w * channels, dtype=np.uint8).reshape((h, w) if channels == 1 else (h, w, channels))
    path = tmp_path / ('actual.' + extension)
    Image.fromarray(values, mode=mode).save(path)
    before = path.read_bytes()
    fresh = observe(path, word=True)
    assert fresh == observe_pixels(path, word=True)
    with Image.open(path) as image:
        word, native = image_arrays(image)
    assert fresh['word_float_CHW64x448_sha256'] == array_digest(word)
    assert fresh['native_float_CHW448_sha256'] == array_digest(native)
    assert fresh['stored_image_sha256'] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before


def test_solid_raw_source_stays_solid_even_when_native_padding_has_contrast(tmp_path):
    path = tmp_path / 'white.png'
    Image.new('RGB', (113, 17), (255, 255, 255)).save(path)
    fresh = observe(path, word=True)
    assert fresh['source_is_spatially_constant_RGB']
    assert fresh['source_channel_extrema'] == [[255, 255]] * 3
    assert fresh == observe_pixels(path, word=True)


def test_wrong_protocol_conflicts_and_corrupt_physical_sources_never_pass(tmp_path):
    path = tmp_path / 'actual.png'
    Image.new('RGB', (19, 23), (1, 2, 3)).save(path)
    fresh = observe(path, word=False)
    assert differences({'native_RGB448_sha256': fresh['native_float_CHW448_sha256']}, fresh) == ['native_RGB448_sha256']
    expected = {}
    merge(expected, {'stored_image_sha256': fresh['stored_image_sha256']})
    with pytest.raises(ValueError, match='conflicting'):
        merge(expected, {'stored_image_sha256': 'b' * 64})
    for invalid in [{'unknown': 'a' * 64}, {'source_dimensions': [1, 0]}, {'source_RGB_sha256': 'bad'}]:
        with pytest.raises(ValueError):
            merge({}, invalid)
    path.write_bytes(b'corrupt image')
    with pytest.raises(Exception):
        observe(path, word=True)
