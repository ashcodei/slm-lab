import hashlib
from PIL import Image
import pytest

from slm_lab import real_word_refinement as refinement
from slm_lab.evaluation_provenance import pixels_sha256
from slm_lab.more_real_words import crop_tensor


def test_chunk_sampler_covers_every_old_and_new_word_once_and_repeats_seed():
    rows = [{'group_id': str(index // 17)} for index in range(103)]
    sampler = refinement.GroupChunks(rows); first = list(sampler)
    assert sorted(first) == list(range(103)) and len(set(first)) == 103
    assert first == list(refinement.GroupChunks(rows))
    sampler.epoch = 1; assert list(sampler) != first
    with pytest.raises(ValueError): refinement.GroupChunks(rows, chunk_size=0)


def test_frame_cache_preserves_hash_bound_pixels_and_stays_within_memory_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(refinement, 'ROOT', tmp_path); path = tmp_path / 'photo.png'
    Image.new('RGB', (100, 100), 'red').save(path); raw = path.read_bytes(); image = Image.open(path).convert('RGB')
    row = {'image': str(path), 'source_image_sha256': hashlib.sha256(raw).hexdigest(), 'native_dimensions': [100, 100],
           'native_crop_bounds': [10, 10, 90, 90], 'reader_input_pixels_sha256': pixels_sha256(crop_tensor(image, [10, 10, 90, 90]))}
    cache = refinement.NativeFrames(maximum_bytes=30000)
    value = cache.pixels(row); assert pixels_sha256(value) == row['reader_input_pixels_sha256'] and cache.bytes == 30000
    row['reader_input_pixels_sha256'] = 'wrong'
    with pytest.raises(ValueError): cache.pixels(row)
    # A fresh cache cannot admit changed disk bytes. An already admitted frame
    # remains a verified snapshot; final source rehashing is required by training.
    Image.new('RGB', (100, 100), 'blue').save(path)
    with pytest.raises(ValueError): refinement.NativeFrames().image(row)


def test_oversized_native_frame_is_returned_without_unbounded_caching(tmp_path, monkeypatch):
    monkeypatch.setattr(refinement, 'ROOT', tmp_path); path = tmp_path / 'photo.png'; Image.new('RGB', (100, 100)).save(path)
    row = {'image': str(path), 'source_image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'native_dimensions': [100, 100]}
    cache = refinement.NativeFrames(maximum_bytes=29999); assert cache.image(row).size == (100, 100)
    assert not cache.frames and cache.bytes == 0
