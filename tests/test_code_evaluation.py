from slm_lab.code_evaluation import check


def test_function_completion_runs_without_external_access():
    result=check('def add_one(n):\n    return n+1\n','add_one',[(0,1),(-2,-1)])
    assert result['passed'] and result['passed_cases']==2


def test_executor_rejects_imports_and_attribute_access():
    result=check('def f(n):\n    import os\n    return n\n','f',[(1,1)])
    assert not result['passed'] and 'restricted Python subset' in result['error']
    result=check('def f(n):\n    return n.__class__\n','f',[(1,1)])
    assert not result['passed']
