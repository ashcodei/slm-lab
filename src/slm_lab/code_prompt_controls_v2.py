"""Remove appended examples and Python doctests from known-source prompts."""
import doctest
import re

from .code_prompt_controls_v1 import without_examples
from .code_quality import instruction_code
from .verified_code import renamed
import hashlib


def _example_lines(text):
    removed = set()
    for example in doctest.DocTestParser().get_examples(text):
        length = len(example.source.splitlines()) + len(example.want.splitlines())
        removed.update(range(example.lineno, example.lineno + length))
    return removed


def strip_doctests(text):
    lines = text.splitlines(keepends=True); removed = set(); index = 0
    # Commented doctests are not recognized by Python's normal parser.
    while index < len(lines):
        if not re.match(r'^\s*#', lines[index]): index += 1; continue
        end = index
        while end < len(lines) and re.match(r'^\s*#', lines[end]): end += 1
        block = ''.join(re.sub(r'^\s*# ?', '', line) for line in lines[index:end])
        if '>>>' in block:
            removed.update(index + line for line in _example_lines(block))
        index = end
    partial = ''.join(line for i, line in enumerate(lines) if i not in removed)
    removed = _example_lines(partial)
    result = ''.join(line for i, line in enumerate(partial.splitlines(keepends=True)) if i not in removed)
    if re.search(r'^\s*(?:#\s*)?>>>', result, re.MULTILINE):
        raise ValueError('Unremoved Python example prompt requires inspection')
    return result


def without_doctests(row):
    result = without_examples(row)
    result['prompt'] = strip_doctests(result['prompt'])
    result['prompt_condition'] = 'original_signatures_and_prose_without_appended_or_doctest_examples'
    return result


def new_function_without_doctests(row):
    result = without_doctests(row)
    old = row['function']; new = 'task_' + hashlib.sha256(row['id'].encode()).hexdigest()[:10]
    pattern = r'(^[ \t]*def )' + re.escape(old) + r'\('
    prompt, count = re.subn(pattern, lambda match: match[1] + new + '(', result['prompt'], flags=re.MULTILINE)
    if not count: raise ValueError('Original requested function declaration is absent')
    result.update(prompt=prompt, function=new, original_function=old,
                  response='```python\n' + renamed(instruction_code(row['response']), old, new) + '\n```',
                  prompt_condition='unseen_function_identifier_without_appended_or_doctest_examples')
    return result
