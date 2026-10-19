"""Independent physical input readback of the actual RGB and FP32 protocols.

Do not call the producer's observation or word-array implementation. Decode
the original bytes and compute both actual training geometries explicitly.
These observations certify input bytes and geometry, never label semantics.
"""
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HASH_FIELDS = frozenset({'stored_image_sha256', 'source_RGB_sha256', 'native_RGB448_sha256',
    'native_float_CHW448_sha256', 'word_float_CHW64x448_sha256'})


def file_state(path):
    stat = Path(path).stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def merge(target, additions):
    for name, value in additions.items():
        if value is None:
            continue
        if name in HASH_FIELDS:
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in '0123456789abcdef' for c in value)):
                raise ValueError('Every named original pixel protocol requires its complete SHA256')
        elif name == 'source_dimensions':
            if type(value) is not list or len(value) != 2 or any(type(x) is not int or x < 1 for x in value):
                raise ValueError('Exact positive original physical dimensions required')
        else:
            raise ValueError('Unknown physical pixel protocol')
        if name in target and target[name] != value:
            raise ValueError('One physical source has conflicting original protocol expectations: ' + name)
        target[name] = value


def normalized_CHW(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Actual decoded three-channel byte pixels required')
    scaled = np.divide(rgb, np.float32(127.5), dtype=np.float32)
    scaled = np.subtract(scaled, np.float32(1), dtype=np.float32)
    values = np.ascontiguousarray(np.moveaxis(scaled, 2, 0))
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Complete finite actual FP32 model input required')
    return values


def observe(path, *, word):
    path = Path(path)
    before = file_state(path)
    if before['bytes'] > 64 * 1024**2:
        raise ValueError('Complete original image must fit the declared physical byte bound')
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as opened:
        # The actual native training protocol performs RGB conversion without
        # EXIF rotation or alpha compositing; use exactly that public input.
        image = opened.convert('RGB')
        source = np.asarray(image, dtype=np.uint8)
        padded = ImageOps.pad(image, (448, 448), method=Image.Resampling.BICUBIC,
            color=(128, 128, 128), centering=(.5, .5))
        native_RGB = np.asarray(padded, dtype=np.uint8)
        native = normalized_CHW(native_RGB)
        roundtrip = np.rint((np.moveaxis(native, 0, 2) + 1) * 127.5).clip(0, 255).astype(np.uint8)
        if native.shape != (3, 448, 448) or not np.array_equal(roundtrip, native_RGB):
            raise ValueError('Actual native geometry and byte-to-float-to-byte roundtrip must be exact')
        low = source.reshape(-1, 3).min(axis=0)
        high = source.reshape(-1, 3).max(axis=0)
        result = dict(path=str(path), stored_image_sha256=hashlib.sha256(raw).hexdigest(),
            source_dimensions=[image.width, image.height], source_RGB_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
            native_RGB448_sha256=hashlib.sha256(native_RGB.tobytes()).hexdigest(),
            native_float_CHW448_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
            native_shape=list(native.shape), native_dtype=str(native.dtype),
            source_channel_extrema=[[int(lo), int(hi)] for lo, hi in zip(low, high)],
            source_is_spatially_constant_RGB=bool(np.array_equal(low, high)),
            RGB_and_float_hash_protocols_distinct=True)
        if word:
            resized = image.resize((448, 64), resample=Image.Resampling.BICUBIC)
            crop = normalized_CHW(np.asarray(resized, dtype=np.uint8))
            if crop.shape != (3, 64, 448):
                raise ValueError('Whole word input must retain its actual64by448geometry')
            result.update(word_float_CHW64x448_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                word_shape=list(crop.shape), word_dtype=str(crop.dtype))
    after = file_state(path)
    if after != before or len(raw) != before['bytes']:
        raise ValueError('Every physical image byte and file identity must remain unchanged during readback')
    result['source_file_state'] = after
    return result


def differences(expected, actual):
    checked = {}
    merge(checked, expected)
    return [name for name, value in checked.items() if actual.get(name) != value]
