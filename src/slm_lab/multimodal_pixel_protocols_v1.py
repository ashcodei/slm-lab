"""Exact CPU source/padded-RGB/normalized-word protocol observations.

These hashes certify reproducible input bytes and transforms, never whether a
caption or transcription is semantically correct. No model or GPU is loaded.
"""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps

from .shared_word_data_v1 import image_arrays,array_digest

HASH_FIELDS={'stored_image_sha256','source_RGB_sha256','native_RGB448_sha256',
    'native_float_CHW448_sha256','word_float_CHW64x448_sha256'}


def merge_expectations(target,values):
    for field,value in values.items():
        if value is None:continue
        if field in HASH_FIELDS:
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Every expected hash must retain an explicit original protocol and exact SHA256')
        elif field=='source_dimensions':
            if not isinstance(value,list) or len(value)!=2 or any(type(v) is not int or v<1 for v in value):
                raise ValueError('Original positive source dimensions required')
        else:raise ValueError('Unknown pixel hash protocol')
        if field in target and target[field]!=value:raise ValueError('Conflicting original expectations for one physical path/protocol: '+field)
        target[field]=value


def file_state(path):
    s=Path(path).stat()
    return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)


def observe_pixels(path,*,word=False):
    path=Path(path);before=file_state(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):digest.update(block)
    with Image.open(path) as opened:image=opened.convert('RGB')
    source_rgb=image.tobytes();extrema=image.getextrema()
    padded=ImageOps.pad(image,(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128))
    rgb=np.asarray(padded,dtype=np.uint8)
    native=np.ascontiguousarray((rgb.astype(np.float32)/127.5-1).transpose(2,0,1))
    roundtrip=np.rint((native.transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)
    if native.shape!=(3,448,448) or native.dtype!=np.float32 or not np.isfinite(native).all() or not np.array_equal(rgb,roundtrip):
        raise ValueError('Actual normalized native input and paddedRGB roundtrip must remain exact')
    result=dict(path=str(path),stored_image_sha256=digest.hexdigest(),source_dimensions=list(image.size),
        source_RGB_sha256=hashlib.sha256(source_rgb).hexdigest(),native_RGB448_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
        native_float_CHW448_sha256=array_digest(native),native_shape=list(native.shape),native_dtype=str(native.dtype),
        source_is_spatially_constant_RGB=all(lo==hi for lo,hi in extrema),source_channel_extrema=[list(pair) for pair in extrema],
        RGB_and_float_hash_protocols_distinct=True)
    if word:
        actual_word,actual_native=image_arrays(image)
        if (not np.array_equal(native,actual_native) or actual_word.shape!=(3,64,448)
                or actual_word.dtype!=np.float32 or not np.isfinite(actual_word).all()):
            raise ValueError('Canonical word/native input arrays and every original geometry must match')
        result.update(word_float_CHW64x448_sha256=array_digest(actual_word),word_shape=list(actual_word.shape),word_dtype=str(actual_word.dtype))
    after=file_state(path)
    if before!=after:raise ValueError('The physical source changed while its pixels were being checked')
    result['source_file_state']=after
    return result


def mismatches(expected,actual):
    if any(k not in HASH_FIELDS|{'source_dimensions'} for k in expected):raise ValueError('Unknown original pixel protocol')
    return [field for field,value in expected.items() if actual.get(field)!=value]
