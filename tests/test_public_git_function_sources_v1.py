import ast
import hashlib
import io
import tarfile

import pytest

from slm_lab.public_git_function_sources_v1 import archive_files,blob_sha,repository_function,source_entry
from test_public_function_contracts_v1 import PROGRAM
from test_public_pretraining_intake_v1 import fixture


def test_original_archive_every_file_and_git_blob_bytes_are_verified_without_extracting_paths(tmp_path):
    raw=PROGRAM.encode();path=tmp_path/'source.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        info=tarfile.TarInfo('Python-commit/algorithms/example.py');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    tree=dict(truncated=False,tree=[dict(path='algorithms/example.py',type='blob',mode='100644',size=len(raw),sha=blob_sha(raw))])
    assert list(archive_files(path,tree,'Python-commit'))[0][:2] == ('algorithms/example.py',raw)
    tree['tree'][0]['sha']='a'*40
    with pytest.raises(ValueError,match='Git blob'):list(archive_files(path,tree,'Python-commit'))


def test_archive_parent_escape_and_truncated_publisher_tree_are_rejected(tmp_path):
    path=tmp_path/'source.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        info=tarfile.TarInfo('Python-commit/../example.py');info.size=1;tar.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='exact pinned repository root'):
        list(archive_files(path,dict(truncated=False,tree=[]),'Python-commit'))
    with pytest.raises(ValueError,match='complete pinned Git tree'):
        list(archive_files(path,dict(truncated=True,tree=[]),'Python-commit'))


def test_repository_source_contract_uses_exact_original_bytes_without_inventing_publisher_partitions():
    tokenizer,overlap,_,_=fixture();raw=PROGRAM.encode()
    git=dict(path='math/example.py',type='blob',mode='100644',size=len(raw),sha=blob_sha(raw))
    original,reason=source_entry(git['path'],raw,git,repository='author/project',revision='c'*40,
        repository_license_sha256='d'*64,tokenizer=tokenizer,overlap=overlap)
    assert reason is None and original['publisher_partition'] is None
    assert original['original_UTF8_sha256']==hashlib.sha256(raw).hexdigest()
    candidate,reason=repository_function(original,ast.parse(PROGRAM).body[0],tokenizer,overlap)
    assert reason is None and candidate['original_publisher_partition'] is None
    assert candidate['response']==ast.get_source_segment(PROGRAM,ast.parse(PROGRAM).body[0])
    assert 'original_publisher_role_is_train_derived_source_role_stays_sealed' not in candidate


def test_embedded_docstring_markdown_is_explicitly_qualified_instead_of_rewriting_or_crashing():
    tokenizer,overlap,_,_=fixture()
    text=PROGRAM.replace('Move the final item','```python\n    ```\n    Move the final item');raw=text.encode()
    git=dict(path='example.py',type='blob',mode='100644',size=len(raw),sha=blob_sha(raw))
    original,reason=source_entry(git['path'],raw,git,repository='author/project',revision='c'*40,
        repository_license_sha256='d'*64,tokenizer=tokenizer,overlap=overlap)
    assert reason is None
    value,reason=repository_function(original,ast.parse(text).body[0],tokenizer,overlap)
    assert value is None and reason=='embedded_markdown_fence_not_supported_by_declared_legacy_runtime_contract'
