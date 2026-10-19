import gzip
import hashlib
import io
import json
import random
import tarfile

import pytest

from slm_lab.codenet_candidate_bodies_v1 import copy_bodies

BODY = 'Project_CodeNet/data/p00001/Python/s000000001.py'
SECOND = 'Project_CodeNet/data/p00002/C++/s000000002.cpp'
AUX = 'Project_CodeNet/derived/duplicates/identical_problem_clusters'


def source(tmp_path, records=None):
    path = tmp_path / 'source.tar.gz'
    records = records if records is not None else [(BODY, b'print(1)\n'), (SECOND, b'int main() {}\n'), (AUX, b'p00001 p00002\n')]
    with tarfile.open(path, 'w:gz') as archive:
        for name, raw in records:
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return path


def test_source_store_offsets_and_hashes_preserve_original_bytes(tmp_path):
    path = source(tmp_path)
    output = tmp_path / 'selected'
    result = copy_bodies(path, output, {SECOND: 999, BODY: 9}, {AUX: 14}, lambda _: None, minimum_free_bytes=0)
    index = [json.loads(x) for x in (output / 'source-index.jsonl').read_text().splitlines()]
    raw = (output / 'source-bodies.bin').read_bytes()
    assert raw == b'print(1)\nint main() {}\n'
    assert result['publisher_declared_size_mismatches'] == 1
    assert result['source_body_store_sha256'] == hashlib.sha256(raw).hexdigest()
    for entry in index:
        value = raw[entry['body_offset']:entry['body_offset']+entry['body_bytes']]
        assert hashlib.sha256(value).hexdigest() == entry['body_sha256']
    assert (output / result['auxiliary_members'][AUX]['cache_relative_path']).read_bytes() == b'p00001 p00002\n'
    assert result['whole_gzip_stream_drained_crc_checked'] and not result['source_code_executed']


def test_missing_source_member_cannot_report_complete_pool(tmp_path):
    with pytest.raises(ValueError, match='members missing'):
        copy_bodies(source(tmp_path, [(BODY, b'x')]), tmp_path / 'out', {BODY: 1, SECOND: 1}, {}, lambda _: None, minimum_free_bytes=0)


def test_duplicate_selected_member_cannot_override_source_identity(tmp_path):
    with pytest.raises(ValueError, match='Duplicate/non-regular'):
        copy_bodies(source(tmp_path, [(BODY, b'x'), (BODY, b'y')]), tmp_path / 'out', {BODY: 1}, {}, lambda _: None, minimum_free_bytes=0)


def test_auxiliary_size_must_match_original_inventory(tmp_path):
    with pytest.raises(ValueError, match='auxiliary member identity or bytes'):
        copy_bodies(source(tmp_path), tmp_path / 'out', {BODY: 9}, {AUX: 1}, lambda _: None, minimum_free_bytes=0)


def test_retained_byte_budget_stops_source_copy(tmp_path):
    with pytest.raises(ValueError, match='retained-byte bound'):
        copy_bodies(source(tmp_path), tmp_path / 'out', {BODY: 9}, {}, lambda _: None,
                    minimum_free_bytes=0, maximum_retained_bytes=5)


def test_gzip_corruption_after_selected_members_still_prevents_completion(tmp_path):
    tail = 'Project_CodeNet/data/p00003/Python/s000000003.py'
    path = source(tmp_path, [(BODY, b'print(1)\n'), (tail, random.Random(3).randbytes(1024 ** 2))])
    raw = bytearray(path.read_bytes())
    raw[-8] ^= 1
    path.write_bytes(raw)
    with pytest.raises(gzip.BadGzipFile):
        copy_bodies(path, tmp_path / 'out', {BODY: 9}, {}, lambda _: None, minimum_free_bytes=0)


def test_unselected_headers_do_not_accumulate_in_memory(tmp_path, monkeypatch):
    rows = [(f'Project_CodeNet/data/p00002/Python/s{n:09d}.py', b'x') for n in range(2, 1002)]
    path = source(tmp_path, [(BODY, b'print(1)\n'), *rows])
    original = tarfile.TarFile.next
    retained = []
    def next_header(self):
        retained.append(len(self.members))
        return original(self)
    monkeypatch.setattr(tarfile.TarFile, 'next', next_header)
    result = copy_bodies(path, tmp_path / 'out', {BODY: 9}, {}, lambda _: None, minimum_free_bytes=0)
    assert result['original_code_headers'] == 1001 and result['source_bodies'] == 1
    assert max(retained) <= 1
