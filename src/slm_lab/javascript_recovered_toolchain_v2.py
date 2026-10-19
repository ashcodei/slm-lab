"""Prove recovered Node/parser byte identity, allowing npm bookkeeping only."""
import json
from pathlib import Path

from .corpus import file_hash


BOOKKEEPING = 'node_modules/.package-lock.json'


def validate_toolchain(root, expected_preparation_sha256, expected_parser_version):
    root = Path(root).resolve()
    path = root / '.tools/javascript-toolchain-v1.json'
    current_sha = file_hash(path)
    current = json.loads(path.read_text())
    changed = []
    if current_sha != expected_preparation_sha256:
        recovery = current.get('recovery')
        if not isinstance(recovery, dict):
            raise ValueError('Different toolchain requires an exact recorded recovery chain')
        previous_path = (root / recovery['previous_manifest']).resolve()
        if not previous_path.is_relative_to(root) or file_hash(previous_path) != expected_preparation_sha256:
            raise ValueError('Exact original preparation toolchain snapshot required')
        old = json.loads(previous_path.read_text())
        if recovery.get('previous_manifest_sha256') != expected_preparation_sha256:
            raise ValueError('Recovery must reference the exact original preparation manifest')
        old_files = old['dependency_files_sha256']
        new_files = current['dependency_files_sha256']
        if set(old_files) != set(new_files):
            raise ValueError('Recovery cannot add or remove any original dependency inventory entry')
        changed = [name for name in old_files if old_files[name] != new_files[name]]
        if changed != [BOOKKEEPING]:
            raise ValueError('Only npm package-lock bookkeeping may differ; parser/runtime changes reject')
        if ({k: v for k, v in old.items() if k != 'dependency_files_sha256'}
                != {k: v for k, v in current.items() if k not in {'dependency_files_sha256', 'recovery'}}):
            raise ValueError('Every other original toolchain field must remain unchanged')
        expected_delta = {BOOKKEEPING: dict(previous=old_files[BOOKKEEPING], recovered=new_files[BOOKKEEPING])}
        if recovery.get('changed_files') != expected_delta:
            raise ValueError('Exact recorded bookkeeping delta required')
    if file_hash(current['node_path']) != current['node_sha256']:
        raise ValueError('Actual pinned Node executable bytes changed')
    prefix = Path(current['toolchain_root'])
    for relative, expected in current['dependency_files_sha256'].items():
        member = Path(relative)
        if member.is_absolute() or '..' in member.parts or file_hash(prefix / member) != expected:
            raise ValueError('Actual pinned parser/dependency bytes changed')
    parser = prefix / 'node_modules/@babel/parser/package.json'
    version = json.loads(parser.read_text())['version']
    if version != expected_parser_version:
        raise ValueError('Exact original Babel parser version required')
    return dict(preparation_toolchain_manifest_sha256=expected_preparation_sha256,
        actual_toolchain_manifest_sha256=current_sha, parser_version=version,
        actual_node_sha256=current['node_sha256'], verified_dependency_files=len(current['dependency_files_sha256']),
        changed_inventory_entries=changed, runtime_and_parser_bytes_unchanged=True,
        scope='Only a proved npm bookkeeping regeneration is accepted; all actual recorded dependency and Node bytes are verified. '
            'Every original source/header/mask still requires complete reconstruction by the admission caller.')
