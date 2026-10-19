import json

import pytest

from slm_lab.exercise_cpp_runner import extract_files, parse_junit, validate_sources


def test_multifile_extraction_preserves_code_and_requires_all_requested_files():
    response = 'File: grains.cpp\n```cpp\n#include "grains.h"\nint square(int n) { return n; }\n```\n\nFile: grains.h\n```cpp\n#pragma once\nint square(int);\n```'
    files = extract_files(response, ['grains.cpp', 'grains.h'])
    assert files['grains.cpp'].endswith('return n; }\n')
    validate_sources(files)
    with pytest.raises(ValueError, match='filenames'): extract_files(response, ['grains.cpp'])


@pytest.mark.parametrize('name', ['../test.cpp', 'test/catch.hpp', '/tmp/main.cpp'])
def test_solution_cannot_overwrite_harness_or_escape_workspace(name):
    with pytest.raises(ValueError, match='filename'): extract_files(f'File: {name}\n```cpp\nint x;\n```')
    with pytest.raises(ValueError, match='filename'): validate_sources({name: 'int x;'})


@pytest.mark.parametrize('code', ['#define EXERCISM_RUN_ALL_TESTS 0', '#undef REQUIRE',
                                  '#include "test/catch.hpp"', '#define REQUIRE(x) 0',
                                  '#define CATCH_CONFIG_DISABLE', 'int main() { return 0; }'])
def test_candidate_cannot_disable_tests_through_preprocessor_or_replace_main(code):
    with pytest.raises(ValueError): validate_sources({'answer.cpp': code})


def test_valid_include_guard_and_unknown_local_include():
    validate_sources({'answer.h': '#ifndef ANSWER_H\n#define ANSWER_H\n#include <vector>\n#endif'})
    with pytest.raises(ValueError): validate_sources({'answer.h': '#include "secret.h"'})


def test_junit_requires_nonempty_inventory_and_checks_nested_failures():
    valid = b'<testsuites><testsuite tests="2" failures="0" errors="0"><testcase classname="global" name="one"/><testcase classname="global" name="two"/></testsuite></testsuites>'
    report = parse_junit(valid)
    assert report['assertions'] == 2 and len(report['cases']) == 2 and report['all_passed']
    assert json.loads(json.dumps(report)) == report
    broken = valid.replace(b'name="two"/>', b'name="two"><failure/></testcase>')
    assert not parse_junit(broken)['all_passed']
    with pytest.raises(ValueError): parse_junit(b'<testsuites><testsuite tests="0" failures="0" errors="0"/></testsuites>')
