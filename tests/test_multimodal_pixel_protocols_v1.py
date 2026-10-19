import hashlib

import numpy as np
from PIL import Image
import pytest

from slm_lab.multimodal_pixel_protocols_v1 import observe_pixels,merge_expectations,mismatches
from slm_lab.shared_word_data_v1 import image_arrays,array_digest


def sample(tmp_path):
    data=np.arange(23*71*3,dtype=np.uint8).reshape(23,71,3)
    path=tmp_path/'source.png';Image.fromarray(data).save(path)
    return path


def test_actual_file_RGB_and_both_canonical_float_protocols_match_without_source_writes(tmp_path):
    path=sample(tmp_path);before=path.read_bytes();actual=observe_pixels(path,word=True)
    with Image.open(path) as image:word,native=image_arrays(image)
    assert actual['stored_image_sha256']==hashlib.sha256(before).hexdigest()
    assert actual['word_float_CHW64x448_sha256']==array_digest(word)
    assert actual['native_float_CHW448_sha256']==array_digest(native)
    assert actual['source_dimensions']==[71,23] and actual['native_shape']==[3,448,448] and actual['word_shape']==[3,64,448]
    assert path.read_bytes()==before and not actual['source_is_spatially_constant_RGB']


def test_hash_protocol_substitution_is_rejected(tmp_path):
    actual=observe_pixels(sample(tmp_path),word=True)
    assert not mismatches({'native_RGB448_sha256':actual['native_RGB448_sha256']},actual)
    assert mismatches({'native_RGB448_sha256':actual['native_float_CHW448_sha256']},actual)==['native_RGB448_sha256']
    assert actual['native_RGB448_sha256']!=actual['native_float_CHW448_sha256']


def test_conflicting_metadata_is_not_silently_coalesced():
    values={};merge_expectations(values,{'stored_image_sha256':'a'*64});merge_expectations(values,{'stored_image_sha256':'a'*64})
    with pytest.raises(ValueError,match='Conflicting'):merge_expectations(values,{'stored_image_sha256':'b'*64})
    for bad in [{'unknown':'a'*64},{'native_RGB448_sha256':'bad'},{'source_dimensions':[0,1]}]:
        with pytest.raises(ValueError):merge_expectations({},bad)


def test_solid_source_is_detected_even_when_padding_adds_contrast(tmp_path):
    path=tmp_path/'solid.png';Image.new('RGB',(100,20),(255,255,255)).save(path)
    actual=observe_pixels(path,word=True)
    assert actual['source_is_spatially_constant_RGB']
    assert actual['source_channel_extrema']==[[255,255]]*3


def test_physical_tampering_changes_checksum_and_corrupt_images_do_not_pass(tmp_path):
    path=sample(tmp_path);first=observe_pixels(path)
    Image.new('RGB',(71,23),(1,2,3)).save(path);second=observe_pixels(path)
    assert mismatches({'stored_image_sha256':first['stored_image_sha256']},second)==['stored_image_sha256']
    path.write_bytes(b'not an image')
    with pytest.raises(Exception):observe_pixels(path)
