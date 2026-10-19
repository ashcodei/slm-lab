import pytest

from slm_lab.systems_variety import effective_license, license_matches


def test_original_file_license_overrides_the_repository_family_without_relicensing():
    text = '// SPDX-License-Identifier: BSD-3-Clause OR GPL-2.0-only\nint example(void) { return 1; }'
    values, source = effective_license(text, 'Apache-2.0')
    assert values == ['BSD-3-Clause OR GPL-2.0-only'] and source == 'original file SPDX declarations'


def test_the_kernel_syscall_exception_is_retained_in_its_original_expression():
    text = '/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */'
    values, _ = effective_license(text, 'GPL-2.0')
    assert values == ['GPL-2.0-only WITH Linux-syscall-note']


@pytest.mark.parametrize('value', ['LicenseRef-Proprietary', 'BSD-3-Clause OR unknown', '', 'GPL-2.0 NOT MIT'])
def test_unrecognized_file_license_declarations_do_not_enter_source_collection(value):
    with pytest.raises(ValueError): effective_license('// SPDX-License-Identifier: ' + value + '\n', 'MIT')


def test_absent_file_spdx_is_described_as_a_repository_declaration():
    values, scope = effective_license('int hello(void) { return 1; }', 'MIT')
    assert values == ['MIT'] and 'repository license declaration' in scope


def test_license_family_detection_never_turns_arbitrary_readme_text_into_permission():
    assert not license_matches(b'This project is probably open source.', 'MIT')
    assert license_matches(b'Permission is hereby granted, free of charge, to any person obtaining a copy...', 'MIT')
    assert not license_matches(b'GNU LESSER GENERAL PUBLIC LICENSE\nVersion 2.1, February 1999', 'GPL-2.0')


def test_original_lwip_third_clause_is_recognized_without_accepting_advertising_licenses():
    text = b'Redistribution and use in source and binary forms.\n3. The name of the author may not be used to endorse or promote products.'
    assert license_matches(text, 'BSD-3-Clause')
    assert not license_matches(text + b' All advertising materials must display an acknowledgement.', 'BSD-3-Clause')


def test_tcpdump_multiline_plural_author_clause_and_current_affero_family_are_distinct():
    text = b'Redistribution and use in source and binary forms.\n3. The names of the authors\nmay not be used to endorse products.'
    assert license_matches(text, 'BSD-3-Clause')
    affero = b'GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3, 19 November 2007'
    assert license_matches(affero, 'AGPL-3.0')
    assert not license_matches(affero, 'Apache-2.0') and not license_matches(affero, 'GPL-2.0')
