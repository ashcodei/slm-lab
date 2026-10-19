"""Bounded, identity-checked HTTP range access for original large ZIP files."""
from collections import OrderedDict
import io
import re
import urllib.request

from .public_sources import AGENT


def checked_range(response, start, end, size, etag):
    if response.status != 206: raise ValueError('Server ignored range request; refuse full archive response')
    if response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}': raise ValueError('Source returned another byte range')
    if response.headers.get('ETag') != etag: raise ValueError('Original archive identity changed')
    raw = response.read(end - start + 2)
    if len(raw) != end - start + 1: raise ValueError('Incomplete/oversized original archive range')
    return raw


class RemoteZipFile(io.RawIOBase):
    def __init__(self, url, block_size=1 << 20, max_range=16 << 20):
        super().__init__(); self.url = url; self.block_size = block_size; self.max_range = max_range
        request = urllib.request.Request(url, method='HEAD', headers={'User-Agent': AGENT})
        with urllib.request.urlopen(request, timeout=30) as response:
            self.size = int(response.headers['Content-Length']); self.etag = response.headers.get('ETag')
            self.identity = {'url': url, 'bytes': self.size, 'etag': self.etag,
                             'version_id': response.headers.get('x-amz-version-id'),
                             'last_modified': response.headers.get('Last-Modified')}
        if not self.etag or self.size <= 0: raise ValueError('Original range source lacks a stable size/identity')
        self.position = 0; self.cache = OrderedDict(); self.fetched_bytes = 0; self.requests = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        if whence not in {0, 1, 2}: raise ValueError('Invalid seek origin')
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if not 0 <= position <= self.size: raise ValueError('Range seek outside original archive')
        self.position = position; return position

    def fetch(self, start, end):
        if end - start + 1 > self.max_range: raise ValueError('Original range exceeds bounded request size')
        request = urllib.request.Request(self.url, headers={'User-Agent': AGENT,
                                        'Range': f'bytes={start}-{end}', 'If-Match': self.etag})
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = checked_range(response, start, end, self.size, self.etag)
        self.fetched_bytes += len(raw); self.requests += 1; return raw

    def read(self, count=-1):
        if count < 0: count = self.size - self.position
        count = min(count, self.size - self.position)
        if count == 0: return b''
        if count > self.max_range: raise ValueError('Refuse unbounded archive read')
        if count > self.block_size:
            raw = self.fetch(self.position, self.position + count - 1); self.position += count; return raw
        result = bytearray()
        while len(result) < count:
            start = self.position // self.block_size * self.block_size
            if start not in self.cache:
                self.cache[start] = self.fetch(start, min(start + self.block_size, self.size) - 1)
                while len(self.cache) > 4: self.cache.popitem(last=False)
            self.cache.move_to_end(start); block = self.cache[start]; offset = self.position - start
            part = block[offset:offset + count - len(result)]
            result.extend(part); self.position += len(part)
        return bytes(result)
