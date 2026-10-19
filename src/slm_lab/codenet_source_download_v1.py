"""Bounded conditional download; a partial archive is never an admitted corpus."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import http.client
import os
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request


def download(url, path, expected_bytes, etag, progress, *,
             minimum_free_bytes=12 * 1024 ** 3, attempts=3, pause=time.sleep):
    path = Path(path)
    if expected_bytes <= 0 or not etag or etag.startswith('W/') or attempts < 1:
        raise ValueError('Require bounded bytes, a strong object identity and attempts')
    if shutil.disk_usage(path.parent).free < expected_bytes + minimum_free_bytes:
        raise ValueError('Preserve temporary checkpoint/archive capacity')
    digest = hashlib.sha256()
    offset = checkpoint = 0
    with path.open('xb') as stream:
        for attempt in range(attempts):
            # A failed local write may have stored an uncommitted suffix.
            stream.seek(offset)
            stream.truncate(offset)
            headers = {'User-Agent': 'SLMLabResearch/0.1', 'If-Match': etag,
                       'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            try:
                request = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.headers.get('ETag') != etag:
                        raise ValueError('Publisher object identity changed')
                    if response.status != (206 if offset else 200):
                        raise ValueError('Publisher did not honor complete/ranged transfer')
                    if int(response.headers.get('Content-Length', '-1')) != expected_bytes - offset:
                        raise ValueError('Publisher object byte length differs')
                    if offset and response.headers.get('Content-Range') != f'bytes {offset}-{expected_bytes - 1}/{expected_bytes}':
                        raise ValueError('Resumed byte interval differs')
                    while offset < expected_bytes:
                        block = response.read(min(8 * 1024 ** 2, expected_bytes - offset))
                        if not block:
                            raise OSError('Publisher transfer ended before declared bytes')
                        if shutil.disk_usage(path.parent).free < minimum_free_bytes + len(block):
                            raise ValueError('Temporary reserve would be consumed')
                        if stream.write(block) != len(block):
                            raise ValueError('Incomplete local source write')
                        digest.update(block)
                        offset += len(block)
                        if offset - checkpoint >= 32 * 1024 ** 2 or offset == expected_bytes:
                            stream.flush()
                            progress(offset, digest.hexdigest(), None)
                            checkpoint = offset
                stream.flush()
                os.fsync(stream.fileno())
                return {'bytes': offset, 'sha256': digest.hexdigest(), 'etag': etag}
            except (OSError, http.client.HTTPException) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}:
                    raise
                if attempt + 1 == attempts:
                    raise
                delay = 2 ** (attempt + 1)
                if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                    value = error.headers.get('Retry-After')
                    try:
                        delay = max(60, int(value))
                    except (TypeError, ValueError):
                        try:
                            date = parsedate_to_datetime(value)
                            if date.tzinfo is None:
                                date = date.replace(tzinfo=timezone.utc)
                            delay = max(60, (date - datetime.now(timezone.utc)).total_seconds())
                        except (TypeError, ValueError, OverflowError):
                            delay = 300
                stream.flush()
                progress(offset, digest.hexdigest(), {'retry_after_seconds': delay,
                                                    'error': str(error)[:240]})
                while delay > 0:
                    interval = min(15, delay)
                    pause(interval)
                    delay -= interval
    raise RuntimeError('No complete publisher transfer')
