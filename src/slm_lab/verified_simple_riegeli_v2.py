"""Bounded sequential reader for simple Riegeli chunks in verified source files.

Implements Google's pinned public format specification:64KiB block headers,
chunk boundaries, varint size vectors and uncompressed/Zstd simple records.
Metadata carry no records and may be skipped. Unsupported data encodings fail.
Internal HighwayHash is NOT computed: callers must supply the already verified
publisher whole-file SHA256; that digest is checked before and after reading.
This is not a general corruption-recovery or arbitrary-untrusted-file reader.
"""
import ctypes
import hashlib
from pathlib import Path

BLOCK = 65536
BLOCK_HEADER = 24
USABLE = BLOCK - BLOCK_HEADER
SIGNATURE = bytes.fromhex('''
83 af 70 d1 0d 88 4a 3f 00 00 00 00 00 00 00 00
40 00 00 00 00 00 00 00 91 ba c2 3c 92 87 e1 a9
00 00 00 00 00 00 00 00 e1 9f 13 c0 e9 b1 c3 72
73 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
''')
assert len(SIGNATURE) == 64


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(2 * 1024**2), b''): result.update(block)
    return result.hexdigest()


def varint(data, offset=0):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data): raise ValueError('Incomplete original uint64 varint')
        byte = data[offset]; offset += 1
        if shift == 63 and byte > 1: raise ValueError('Original uint64 varint overflow')
        value |= (byte & 127) << shift
        if not byte & 128: return value, offset
    raise ValueError('Original uint64 varint exceeds its whole encoding')


def chunk_end(begin, data_size, records):
    size = 40 + data_size
    with_overhead = begin + size + ((size + (begin + USABLE - 1) % BLOCK) // USABLE) * BLOCK_HEADER
    position = begin + records
    remaining = BLOCK - 1 - (position + BLOCK - 1) % BLOCK
    rounded = position + max(0, remaining - (USABLE - 1))
    return max(with_overhead, rounded)


class Zstd:
    def __init__(self):
        self.path = Path('/lib/x86_64-linux-gnu/libzstd.so.1').resolve(strict=True)
        self.library_sha256 = digest(self.path); self.lib = ctypes.CDLL(str(self.path))
        self.lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ZSTD_decompress.restype = ctypes.c_size_t
        self.lib.ZSTD_isError.argtypes = [ctypes.c_size_t]; self.lib.ZSTD_isError.restype = ctypes.c_uint
        self.lib.ZSTD_findFrameCompressedSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.ZSTD_findFrameCompressedSize.restype = ctypes.c_size_t
        self.lib.ZSTD_versionNumber.restype = ctypes.c_uint

    def decompress(self, encoded, size):
        source = ctypes.create_string_buffer(encoded)
        frame = self.lib.ZSTD_findFrameCompressedSize(source, len(encoded))
        if self.lib.ZSTD_isError(frame) or frame != len(encoded): raise ValueError('Exactly one complete original Zstd frame required')
        target = ctypes.create_string_buffer(max(1, size))
        actual = self.lib.ZSTD_decompress(target, size, source, len(encoded))
        if self.lib.ZSTD_isError(actual) or actual != size: raise ValueError('Every original decompressed Zstd byte must match its bounded declared size')
        return target.raw[:size]


def unpack(data, compression, maximum, zstd, *, expected=None):
    if compression == 0:
        result = data
        if len(result) > maximum: raise ValueError('Original uncompressed buffer exceeds its declared bound')
    elif compression == 0x7a:
        size, offset = varint(data)
        if size > maximum: raise ValueError('Original decompressed buffer exceeds its declared bound')
        if expected is not None and size != expected: raise ValueError('Original value buffer decoded length differs from its chunk header')
        result = zstd.decompress(data[offset:], size)
    else: raise ValueError('Unsupported original simple-chunk compression; no records skipped')
    if expected is not None and len(result) != expected: raise ValueError('Every original decoded value byte must be present')
    return result


class VerifiedSimpleRiegeli:
    def __init__(self, path, expected_sha256, *, maximum_decoded_chunk=512 * 1024**2, maximum_encoded_chunk=128 * 1024**2):
        self.path = Path(path); self.expected_sha256 = expected_sha256
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_sha256)
                or any(type(v) is not int or v < 1 for v in [maximum_decoded_chunk, maximum_encoded_chunk])):
            raise ValueError('Original publisher whole-file SHA256 and explicit positive bounds required')
        if digest(self.path) != expected_sha256: raise ValueError('Exact original publisher file SHA256 required before reading')
        self.maximum_decoded = maximum_decoded_chunk; self.maximum_encoded = maximum_encoded_chunk; self.zstd = Zstd()
        self.records = 0; self.chunks = 0; self.decoded_bytes = 0; self.types = {}

    def __iter__(self):
        if self.records or self.chunks: raise ValueError('A fresh original source reader is required for each full pass')
        size = self.path.stat().st_size
        with self.path.open('rb') as source:
            if source.read(64) != SIGNATURE: raise ValueError('Exact original64-byte Riegeli file signature required')
            source.seek(0)
            while source.tell() < size:
                begin = source.tell(); headers = []
                if 0 < begin % BLOCK <= BLOCK_HEADER: raise ValueError('Chunk begins inside an original block header')
                def read_logical(count):
                    result = bytearray()
                    while count:
                        position = source.tell()
                        if position % BLOCK == 0:
                            raw = source.read(BLOCK_HEADER)
                            if len(raw) != BLOCK_HEADER: raise ValueError('Incomplete original block header')
                            headers.append((position, int.from_bytes(raw[8:16], 'little'), int.from_bytes(raw[16:24], 'little')))
                        available = BLOCK - source.tell() % BLOCK
                        amount = min(count, available); value = source.read(amount)
                        if len(value) != amount: raise ValueError('Incomplete original logical chunk')
                        result.extend(value); count -= amount
                    return bytes(result)
                header = read_logical(40); encoded = int.from_bytes(header[8:16], 'little')
                kind = header[24]; records = int.from_bytes(header[25:32], 'little'); decoded = int.from_bytes(header[32:40], 'little')
                if encoded > self.maximum_encoded or decoded > self.maximum_decoded or records > 1_000_000:
                    raise ValueError('Original chunk exceeds declared full-record resource bounds')
                end = chunk_end(begin, encoded, records)
                if end > size or end < source.tell(): raise ValueError('Original complete chunk extent must fit its verified file')
                data = read_logical(encoded)
                # Any rare padding remains part of this same chunk, including
                # intervening block headers. It contains no problem records.
                while source.tell() < end:
                    position = source.tell()
                    if position % BLOCK == 0:
                        raw = source.read(BLOCK_HEADER)
                        if len(raw) != BLOCK_HEADER: raise ValueError('Incomplete original padding block header')
                        headers.append((position, int.from_bytes(raw[8:16], 'little'), int.from_bytes(raw[16:24], 'little')))
                    else: source.seek(min(end, position + BLOCK - position % BLOCK))
                if source.tell() != end or any(previous != position - begin or following != end - position for position, previous, following in headers):
                    raise ValueError('Every original block-to-chunk distance must reconstruct exactly')
                self.chunks += 1; self.types[chr(kind)] = self.types.get(chr(kind), 0) + 1
                if kind in [0x73, 0x70]:
                    if records or decoded or (kind == 0x73 and encoded): raise ValueError('Signature/padding may contain no original records')
                    continue
                if kind == 0x6d:
                    if records: raise ValueError('Metadata may contain no original problem records')
                    continue
                if kind != 0x72: raise ValueError('Unsupported original data chunk type; no problem records skipped')
                if not data: raise ValueError('Complete original simple-chunk buffers required')
                compression = data[0]; sizes_length, offset = varint(data, 1)
                if offset + sizes_length > len(data): raise ValueError('Complete original compressed size vector required')
                lengths = unpack(data[offset:offset + sizes_length], compression, 10 * records, self.zstd)
                values = unpack(data[offset + sizes_length:], compression, self.maximum_decoded, self.zstd, expected=decoded)
                cursor = position = 0
                for index in range(records):
                    length, cursor = varint(lengths, cursor)
                    if position + length > len(values): raise ValueError('Every original record must fit its complete value buffer')
                    record = values[position:position + length]; position += length; self.records += 1; self.decoded_bytes += length
                    yield record, dict(chunk_begin=begin, index_in_chunk=index, chunk_records=records,
                        original_record_bytes=length, original_record_sha256=hashlib.sha256(record).hexdigest())
                if cursor != len(lengths) or position != decoded: raise ValueError('Every original size/value byte and declared record must be consumed exactly')
        if digest(self.path) != self.expected_sha256: raise ValueError('Original publisher file changed during its complete read')
