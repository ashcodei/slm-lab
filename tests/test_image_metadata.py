import io
import pytest
from PIL import Image
from slm_lab.image_metadata import strip_png_metadata
from slm_lab.diverse_images import pixel_hash


def test_large_png_profile_is_removed_without_changing_pixels():
    image=Image.new('RGB',(32,32),'red');buffer=io.BytesIO()
    image.save(buffer,format='PNG',icc_profile=b'profile'*(200_000))
    raw=buffer.getvalue()
    with pytest.raises(ValueError,match='Decompressed data too large'):Image.open(io.BytesIO(raw))
    clean,removed=strip_png_metadata(raw);assert removed==['iCCP']
    with Image.open(io.BytesIO(clean)) as decoded:assert pixel_hash(decoded)==pixel_hash(image)
    assert strip_png_metadata(clean)==(clean,[])


def test_corrupted_pixel_chunk_is_not_silently_preserved():
    buffer=io.BytesIO();Image.new('RGB',(16,16),'blue').save(buffer,format='PNG')
    raw=bytearray(buffer.getvalue());raw[20]^=1
    with pytest.raises(ValueError,match='CRC mismatch'):strip_png_metadata(raw)
