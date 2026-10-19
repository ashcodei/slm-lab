import ctypes
import hashlib

import pytest

from slm_lab.verified_simple_riegeli_v1 import BLOCK, BLOCK_HEADER, SIGNATURE, VerifiedSimpleRiegeli, Zstd, chunk_end, varint


def encoded(value):
    result = bytearray()
    while value >= 128: result.append((value & 127) | 128); value >>= 7
    result.append(value); return bytes(result)


def fixture(tmp_path, records, compressed=False, kind=0x72):
    # Structural fixtures intentionally have zero internal HighwayHash values.
    # The reader's explicit integrity contract is whole-file expected SHA256.
    sizes = b''.join(encoded(len(r)) for r in records); values = b''.join(records)
    def zstd(raw):
        codec = Zstd(); lib = codec.lib
        lib.ZSTD_compressBound.argtypes = [ctypes.c_size_t]; lib.ZSTD_compressBound.restype = ctypes.c_size_t
        lib.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        lib.ZSTD_compress.restype = ctypes.c_size_t
        source = ctypes.create_string_buffer(raw); cap = lib.ZSTD_compressBound(len(raw)); target = ctypes.create_string_buffer(cap)
        count = lib.ZSTD_compress(target, cap, source, len(raw), 1)
        assert not lib.ZSTD_isError(count)
        return encoded(len(raw)) + target.raw[:count]
    a, b = (zstd(sizes), zstd(values)) if compressed else (sizes, values)
    data = bytes([0x7a if compressed else 0]) + encoded(len(a)) + a + b
    begin = 64; end = chunk_end(begin, len(data), len(records)); result = bytearray(end); result[:64] = SIGNATURE
    logical = b'\0' * 8 + len(data).to_bytes(8, 'little') + b'\0' * 8 + bytes([kind]) + len(records).to_bytes(7, 'little') + len(values).to_bytes(8, 'little') + data
    at = begin
    for byte in logical:
        if at % BLOCK == 0:
            result[at:at + BLOCK_HEADER] = b'\0' * 8 + (at - begin).to_bytes(8, 'little') + (end - at).to_bytes(8, 'little'); at += BLOCK_HEADER
        result[at] = byte; at += 1
    path = tmp_path / ('compressed' if compressed else 'plain'); path.write_bytes(result)
    return path, hashlib.sha256(result).hexdigest()


def test_whole_records_cross_physical_block_headers_without_losing_bytes(tmp_path):
    original = [b'a' * 70000, 'Straße α and code'.encode(), b'']
    path, sha = fixture(tmp_path, original); reader = VerifiedSimpleRiegeli(path, sha)
    actual = list(reader)
    assert [r for r, _ in actual] == original and reader.records == 3
    assert reader.decoded_bytes == sum(map(len, original)) and reader.types == {'s': 1, 'r': 1}
    path.write_bytes(path.read_bytes()[:-1] + b'X')
    with pytest.raises(ValueError, match='SHA256'): VerifiedSimpleRiegeli(path, sha)


def test_native_zstd_buffers_and_resource_or_unknown_type_fail_closed(tmp_path):
    original = [b'code' * 100, b'another original record']
    path, sha = fixture(tmp_path, original, compressed=True)
    assert [r for r, _ in VerifiedSimpleRiegeli(path, sha)] == original
    with pytest.raises(ValueError, match='resource bounds'): list(VerifiedSimpleRiegeli(path, sha, maximum_decoded_chunk=16))
    path, sha = fixture(tmp_path, original, kind=0x74)
    with pytest.raises(ValueError, match='Unsupported original data'): list(VerifiedSimpleRiegeli(path, sha))
    with pytest.raises(ValueError, match='Incomplete'): varint(b'\x80')
    with pytest.raises(ValueError, match='overflow'): varint(b'\xff' * 10)
