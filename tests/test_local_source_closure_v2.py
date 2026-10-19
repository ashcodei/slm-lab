import json

import pytest

from slm_lab.corpus import file_hash
from slm_lab.local_source_closure_v2 import python_sources, evidence_bindings


def test_lazy_relative_absolute_package_and_script_imports_are_all_included(tmp_path):
    package=tmp_path/'src/pkg'; package.mkdir(parents=True); scripts=tmp_path/'scripts'; scripts.mkdir()
    (package/'__init__.py').write_text('from . import left\n')
    (package/'left.py').write_text('def later():\n from .right import useful\n return useful\n')
    (package/'right.py').write_text('from pkg import tail\nuseful=1\n')
    (package/'tail.py').write_text('import json\n')
    driver=scripts/'owner.py'; driver.write_text('from pkg import left\nimport another\n')
    (scripts/'another.py').write_text('import pkg.right\n')
    found=python_sources(tmp_path,[driver])
    assert set(found)=={p.resolve() for p in [driver,scripts/'another.py',*package.glob('*.py')]}


def test_nested_evidence_maps_preserve_every_original_SHA_and_reject_conflicts(tmp_path):
    source=tmp_path/'source.py'; source.write_text('x=1\n')
    child=tmp_path/'child.json'; child.write_text(json.dumps(dict(source_files_sha256={str(source):file_hash(source)})))
    parent=tmp_path/'parent.json'; parent.write_text(json.dumps(dict(source_files_sha256={str(child):file_hash(child)})))
    result=evidence_bindings({str(parent):file_hash(parent)})
    assert result[str(source)]==file_hash(source) and result[str(child)]==file_hash(child)
    with pytest.raises(ValueError,match='Conflicting'):
        evidence_bindings({str(parent):file_hash(parent),str(source):'a'*64})


def test_valid_large_integer_case_values_do_not_require_global_parser_changes(tmp_path):
    import sys
    source=tmp_path/'source.py'; source.write_text('x=1\n')
    child=tmp_path/'big.json'
    child.write_text('{"raw_case_value":'+('9'*9131)+',"source_files_sha256":'+json.dumps({str(source):file_hash(source)})+'}')
    limit=sys.get_int_max_str_digits()
    result=evidence_bindings({str(child):file_hash(child)})
    assert result[str(source)]==file_hash(source) and sys.get_int_max_str_digits()==limit
