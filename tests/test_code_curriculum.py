from slm_lab.code_curriculum import extract


def row(text):
    return {'text':'File: example.py\nLanguage: .py\n\n'+text,'language_extension':'.py',
        'source':'Example','source_url':'https://example.test/example.py','revision':'revision',
        'license':'MIT','group_id':'example.py'}


def test_curriculum_retains_required_imports_and_literal_constants():
    pairs=extract(row('import math\nLIMIT = 2\ndef f(x):\n    """Return the square root of x plus the documented limit."""\n    return math.sqrt(x)+LIMIT\n'))
    assert len(pairs)==1
    assert 'import math' in pairs[0]['response'] and 'LIMIT = 2' in pairs[0]['response']
    assert 'math.sqrt(x)+LIMIT' not in pairs[0]['prompt']


def test_curriculum_excludes_unknown_module_state_and_unavailable_helpers():
    text='def f(x):\n    """Return an external model prediction for the input feature."""\n    return MODEL.predict(x)\n'
    assert not extract(row(text))
    text='def helper(x):\n    return x+1\ndef f(x):\n    """Return the value produced by an unavailable helper implementation."""\n    return helper(x)\n'
    assert not extract(row(text))
