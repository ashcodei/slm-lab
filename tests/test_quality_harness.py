from slm_lab.code_quality import execute,efficiency
from slm_lab.code_tasks import TASKS,request
from slm_lab.knowledge_evaluation import interval,letter


def test_correctness_is_distinct_from_syntax():
    task=TASKS[0]
    correct=execute(request(task,'Python',task[4]));assert correct['functional_passed']
    wrong=execute(request(task,'Python','def sum_even(xs):\n    return 0'))
    assert wrong['syntax_valid'] and not wrong['functional_passed']
    broken=execute(request(task,'Python','def sum_even(xs)\n    return 0'))
    assert not broken['syntax_valid']


def test_quadratic_algorithm_passes_small_cases_but_fails_efficiency():
    task=next(t for t in TASKS if t[0]=='stable_unique')
    code='def stable_unique(xs):\n    out=[]\n    for x in xs:\n        if x not in out:out.append(x)\n    return out'
    candidate=execute(request(task,'Python',code));reference=execute(request(task,'Python',task[4]))
    assert candidate['syntax_valid'] and candidate['functional_passed']
    assert not efficiency(candidate,reference,task[3])[0]


def test_javascript_reference_executes_without_host_access():
    task=TASKS[0];result=execute(request(task,'JavaScript',task[5]))
    assert result['syntax_valid'] and result['functional_passed'] and result['efficiency_completed']
    attack=execute(request(task,'JavaScript','function sum_even(xs) {return process.env.SECRET;}'))
    assert attack['syntax_valid'] and not attack['functional_passed']


def test_knowledge_parser_and_confidence_interval_are_bounded():
    assert letter('A',10)==0 and letter('The answer is J.',10)==9
    assert letter('print(A)',10) is None and letter('J',4) is None
    assert interval(0,40)[0]==0 and interval(40,40)[1]==1
def test_instruction_code_extracts_formatting_without_repairing_programs():
    from slm_lab.code_quality import instruction_code
    assert instruction_code('Here it is:\n```python\ndef broken(:\n```\nExplanation')=='def broken(:'
    assert instruction_code('def f():\n    return 1')=='def f():\n    return 1'
