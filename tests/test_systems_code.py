import hashlib

import pytest

from slm_lab.systems_code import reviewed_identity


def test_unrecognized_license_requires_the_exact_reviewed_bytes_and_family():
    raw = b'Publisher license fixture'; repo = 'example/library'
    reviews = {repo: {'expected_family': 'BSD-3-Clause', 'license_sha256': hashlib.sha256(raw).hexdigest()}}
    assert 'recorded review' in reviewed_identity(repo, raw, 'NOASSERTION', 'BSD-3-Clause', reviews)
    with pytest.raises(ValueError): reviewed_identity(repo, raw + b'changed terms', 'NOASSERTION', 'BSD-3-Clause', reviews)
    with pytest.raises(ValueError): reviewed_identity(repo, raw, 'NOASSERTION', 'MIT', reviews)
    with pytest.raises(ValueError): reviewed_identity(repo, raw, 'GPL-3.0', 'BSD-3-Clause', reviews)


def test_unrecognized_unreviewed_source_cannot_be_admitted():
    with pytest.raises(ValueError): reviewed_identity('unknown/library', b'terms', 'NOASSERTION', 'MIT', {})
