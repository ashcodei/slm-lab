import gzip
import io
import tarfile

import pytest

from slm_lab.codenet_archive_inspection_v1 import inspect_archive


def source(tmp_path, extra=None):
    path = tmp_path / 'source.tar.gz'
    files = {
        'Project_CodeNet/README.md': b'Original publisher documentation.',
        'Project_CodeNet/data/p00001/Python/s000000001.py': b'print(input())\n',
        'Project_CodeNet/data/p00001/C++/s000000002.cpp': b'int main() { return 0; }\n',
        'Project_CodeNet/problem_descriptions/p00001.html': b'<p>Original problem.</p>',
        'Project_CodeNet/metadata/p00001.csv': b'submission_id,problem_id,language,status\ns000000001,p00001,Python,Accepted\ns000000002,p00001,C++,Wrong Answer\n',
        'Project_CodeNet/metadata/problem_list.csv': b'id,name\np00001,Test\n',
        'Project_CodeNet/derived/identical_problems.txt': b'p00001 p00002\n',
    }
    if extra:
        files.update(extra)
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in files.items():
            item = tarfile.TarInfo(name)
            if data is None:
                item.type = tarfile.SYMTYPE
                item.linkname = '/etc/passwd'
                archive.addfile(item)
            else:
                item.size = len(data)
                archive.addfile(item, io.BytesIO(data))
    return path


def test_original_metadata_retained_and_submissions_remain_unextracted(tmp_path):
    archive = source(tmp_path)
    output = tmp_path / 'inspection'
    result = inspect_archive(archive, output, lambda _: None, minimum_free_bytes=0)
    assert result['metadata_submission_rows'] == 2
    assert result['metadata_status_counts'] == {'Accepted': 1, 'Wrong Answer': 1}
    assert result['metadata_accepted_by_language'] == {'Python': 1}
    assert result['original_code_files_by_language'] == {'Python': 1, 'C++': 1}
    assert result['description_files'] == 1 and result['code_header_and_metadata_row_counts_match']
    assert result['whole_gzip_stream_drained_crc_checked']
    assert not (output / 'data').exists()
    assert (output / 'derived/identical_problems.txt').read_text() == 'p00001 p00002\n'


@pytest.mark.parametrize('name,data', [
    ('Project_CodeNet/../../escaped', b'bad'),
    ('Project_CodeNet/metadata/link', None),
])
def test_traversal_and_link_members_stop_inspection(tmp_path, name, data):
    archive = source(tmp_path, {name: data})
    with pytest.raises(ValueError, match='root/path|links/devices'):
        inspect_archive(archive, tmp_path / 'inspection', lambda _: None, minimum_free_bytes=0)
    assert not (tmp_path / 'escaped').exists()


def test_corrupt_gzip_trailer_cannot_complete_inspection(tmp_path):
    path = source(tmp_path)
    raw = bytearray(path.read_bytes())
    raw[-8] ^= 1
    path.write_bytes(raw)
    with pytest.raises(gzip.BadGzipFile):
        inspect_archive(path, tmp_path / 'inspection', lambda _: None, minimum_free_bytes=0)


def test_decompression_budget_stops_even_a_small_compressed_archive(tmp_path):
    archive = source(tmp_path)
    with pytest.raises(ValueError, match='decompression exceeds'):
        inspect_archive(archive, tmp_path / 'inspection', lambda _: None,
                        minimum_free_bytes=0, max_decompressed_bytes=500)


def test_streaming_does_not_retain_all_submission_headers(tmp_path, monkeypatch):
    archive = source(tmp_path, {
        f'Project_CodeNet/data/p00002/Python/s{index:09d}.py': b'print(0)\n'
        for index in range(3, 1003)})
    original = tarfile.TarFile.next
    retained_headers = []
    def next_header(self):
        retained_headers.append(len(self.members))
        return original(self)
    monkeypatch.setattr(tarfile.TarFile, 'next', next_header)
    result = inspect_archive(archive, tmp_path / 'inspection', lambda _: None, minimum_free_bytes=0)
    assert result['original_code_files_by_language']['Python'] == 1001
    assert max(retained_headers) <= 1
