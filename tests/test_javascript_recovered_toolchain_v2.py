import copy
import json
from pathlib import Path

import pytest

from slm_lab.corpus import file_hash
from slm_lab.javascript_recovered_toolchain_v2 import BOOKKEEPING, validate_toolchain


def fixture(root, recovered=True):
    prefix = root / 'runtime'
    parser = prefix / 'node_modules/@babel/parser'
    parser.mkdir(parents=True)
    (parser / 'package.json').write_text(json.dumps(dict(version='7.29.9')))
    (parser / 'index.js').write_text('original-parser-fixture')
    book = prefix / BOOKKEEPING
    book.write_bytes(b'old')
    node = root / 'node'
    node.write_bytes(b'original-node-fixture')
    old = dict(toolchain_root=str(prefix), node_path=str(node), node_sha256=file_hash(node), node_version='fixture',
        dependency_files_sha256={str(p.relative_to(prefix)): file_hash(p) for p in [book, parser / 'package.json', parser / 'index.js']})
    previous = root / 'previous.json'
    previous.write_text(json.dumps(old))
    digest = file_hash(previous)
    current = copy.deepcopy(old)
    if recovered:
        book.write_bytes(b'new')
        current['dependency_files_sha256'][BOOKKEEPING] = file_hash(book)
        current['recovery'] = dict(previous_manifest='previous.json', previous_manifest_sha256=digest,
            changed_files={BOOKKEEPING: dict(previous=old['dependency_files_sha256'][BOOKKEEPING], recovered=file_hash(book))})
    tools = root / '.tools'
    tools.mkdir()
    path = tools / 'javascript-toolchain-v1.json'
    path.write_text(json.dumps(current))
    return digest, path, current


@pytest.mark.parametrize('recovered', [False, True])
def test_original_or_bookkeeping_only_recovery_verifies_actual_all_dependency_bytes(tmp_path, recovered):
    expected, path, current = fixture(tmp_path, recovered)
    proof = validate_toolchain(tmp_path, expected, '7.29.9')
    assert proof['changed_inventory_entries'] == ([BOOKKEEPING] if recovered else [])
    assert proof['verified_dependency_files'] == 3
    assert proof['actual_toolchain_manifest_sha256'] == file_hash(path)


@pytest.mark.parametrize('fault', ['declared_parser', 'actual_parser', 'node', 'old_snapshot',
    'inventory', 'metadata', 'bookkeeping_receipt', 'parser_version'])
def test_runtime_parser_or_unsupported_manifest_change_rejects(tmp_path, fault):
    expected, path, current = fixture(tmp_path)
    prefix = Path(current['toolchain_root'])
    if fault == 'declared_parser':
        current['dependency_files_sha256']['node_modules/@babel/parser/index.js'] = 'd' * 64
    elif fault == 'actual_parser':
        (prefix / 'node_modules/@babel/parser/index.js').write_text('altered-parser-fixture')
    elif fault == 'node':
        Path(current['node_path']).write_bytes(b'altered-node-fixture')
    elif fault == 'old_snapshot':
        with (tmp_path / 'previous.json').open('a') as stream:
            stream.write(' ')
    elif fault == 'inventory':
        current['dependency_files_sha256']['extra.js'] = 'd' * 64
    elif fault == 'metadata':
        current['node_version'] = 'different'
    elif fault == 'bookkeeping_receipt':
        current['recovery']['changed_files'] = {}
    path.write_text(json.dumps(current))
    with pytest.raises(ValueError):
        validate_toolchain(tmp_path, expected, 'different' if fault == 'parser_version' else '7.29.9')
