import pytest

from slm_lab.exercise_rust_runner import extract_files, inventory


def test_rust_source_files_and_original_package_target_are_delivered_without_extra_prose():
    text = 'File: src/lib.rs\n```rust\npub fn result() -> u32 { 42 }\n```\n\nFile: Cargo.toml\n```toml\n[package]\nname="original"\n```'
    files = extract_files(text, ['src/lib.rs', 'Cargo.toml'])
    assert set(files) == {'src/lib.rs', 'Cargo.toml'}


@pytest.mark.parametrize('change', ['escape', 'tests', 'build_script', 'repeat', 'prose', 'missing'])
def test_unrequested_paths_test_replacement_and_incomplete_delivery_are_rejected(change):
    text = 'File: src/lib.rs\n```rust\npub fn result() -> u32 { 42 }\n```'
    if change == 'escape': text = text.replace('src/lib.rs', '../lib.rs')
    elif change == 'tests': text = text.replace('src/lib.rs', 'tests/original.rs')
    elif change == 'build_script': text = text.replace('src/lib.rs', 'build.rs')
    elif change == 'repeat': text += '\n' + text
    elif change == 'prose': text += '\ntrust me: all tests pass'
    else: text = text.replace('src/lib.rs', 'src/other.rs')
    with pytest.raises(ValueError): extract_files(text, ['src/lib.rs'])


def output():
    return ('running 2 tests\ntest first ... ok\ntest invalid - should panic ... ok\n'
            'test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\n').encode()


def test_complete_native_inventory_includes_expected_panics():
    measured = inventory(output())
    assert measured['cases'] == ['first', 'invalid'] and measured['all_passed'] and not measured['skipped']


@pytest.mark.parametrize('change', ['early_exit', 'missing_case', 'duplicate_case', 'counts', 'filtered', 'missing_summary'])
def test_successful_exit_or_claimed_summary_cannot_replace_complete_registered_case_evidence(change):
    raw = output()
    if change == 'early_exit': raw = b'all tests passed'
    elif change == 'missing_case': raw = raw.replace(b'test first ... ok\n', b'')
    elif change == 'duplicate_case': raw = raw.replace(b'test invalid - should panic ... ok', b'test first ... ok')
    elif change == 'counts': raw = raw.replace(b'2 passed', b'10 passed')
    elif change == 'filtered': raw = raw.replace(b'0 filtered out', b'1 filtered out')
    else: raw = raw.split(b'test result:')[0]
    with pytest.raises(ValueError): inventory(raw)


def test_ignored_registered_cases_cannot_earn_a_full_suite_pass():
    raw = output().replace(b'test first ... ok', b'test first ... ignored').replace(b'2 passed', b'1 passed').replace(b'0 ignored', b'1 ignored')
    measured = inventory(raw)
    assert not measured['all_passed'] and measured['skipped'] == ['first']
