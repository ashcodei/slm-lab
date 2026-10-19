from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from slm_lab.codenet_source_download_v1 import download


@contextmanager
def publisher(mode):
    data = bytes(range(256)) * 256
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append(dict(self.headers))
            start = 20000 if self.headers.get('Range') else 0
            self.send_response(206 if start else 200)
            self.send_header('ETag', '"other-object"' if mode == 'changed' else '"fixed-object"')
            self.send_header('Content-Length', str(len(data) - start + (mode == 'oversize')))
            if start:
                left = 0 if mode == 'wrong_range' else start
                self.send_header('Content-Range', f'bytes {left}-{len(data)-1}/{len(data)}')
            self.end_headers()
            block = data[start:]
            if not start and mode in {'interrupted', 'wrong_range'}:
                block = block[:20000]
                self.close_connection = True
            try:
                self.wfile.write(block)
                self.wfile.flush()
            except OSError:
                pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/archive', data, calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_interrupted_transfer_resumes_exact_bytes_under_same_object(tmp_path):
    path = tmp_path / 'source.partial'
    events = []
    with publisher('interrupted') as (url, data, calls):
        result = download(url, path, len(data), '"fixed-object"',
                          lambda *event: events.append(event), minimum_free_bytes=0, pause=lambda _: None)
    assert path.read_bytes() == data
    assert result['sha256'] == hashlib.sha256(data).hexdigest()
    assert [call.get('Range') for call in calls] == [None, 'bytes=20000-']
    assert all(call['If-Match'] == '"fixed-object"' for call in calls)
    assert events[0][0] == 20000 and events[0][2]['error']
    assert events[-1][0] == len(data) and events[-1][2] is None


@pytest.mark.parametrize('mode,message,retained', [
    ('changed', 'object identity changed', 0),
    ('oversize', 'byte length differs', 0),
    ('wrong_range', 'byte interval differs', 20000),
])
def test_changed_object_or_interval_cannot_be_admitted(tmp_path, mode, message, retained):
    path = tmp_path / 'source.partial'
    with publisher(mode) as (url, data, calls):
        with pytest.raises(ValueError, match=message):
            download(url, path, len(data), '"fixed-object"', lambda *_: None,
                     minimum_free_bytes=0, pause=lambda _: None)
    assert path.read_bytes() == data[:retained]
    assert len(calls) == (2 if mode == 'wrong_range' else 1)
