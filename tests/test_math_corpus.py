from fractions import Fraction
import pytest

from slm_lab.math_corpus import arithmetic, assemble, checked_calculations, final_answer, numeric


def test_decimal_answers_are_exact_and_do_not_guess_from_prose():
    assert final_answer('Work: 99\n#### -1,250.50') == Fraction(-2501, 2)
    assert final_answer('#### 1\nMore work\n#### 2.00') == 2
    for text in ['The answer is 42.', '#### 1/2', '#### 4 + 2', '#### 10kg', '#### 1,23', '#### nan']:
        assert final_answer(text) is None
    assert numeric('0.10') == Fraction(1, 10)
    assert numeric('.50') == Fraction(1, 2)
    assert numeric('2.') == 2


def test_source_arithmetic_is_rational_and_rounding_is_bounded():
    assert arithmetic('(0.1 + 0.2) * 10') == 3
    assert checked_calculations('<<18*.5=9.00>> <<2*20*.01=.4>>\n#### 9') == 2
    assert checked_calculations('<<1/3=0.33>> <<5-2=3>>\n#### 3') == 2
    for answer in ['<<1/3=0.32>>\n#### 1', '<<5/2=2>>\n#### 2']:
        with pytest.raises(ValueError): checked_calculations(answer)


@pytest.mark.parametrize('expression', ["__import__('os').system('echo bad')", 'True+1', '2**1000000', '1/0', '[1][0]', '9**9**9'])
def test_source_annotations_cannot_execute_code_or_unbounded_arithmetic(expression):
    with pytest.raises(ValueError): arithmetic(expression)


def test_protected_questions_bad_calculations_and_duplicate_rows_are_excluded():
    good = {'question': 'How many apples in two pairs?', 'answer': 'Two pairs have <<2*2=4>>4 apples.\n#### 4'}
    test = [{'question': 'How many pears?', 'answer': '#### 2'}]
    train = [good, good, {'question': ' HOW many pears! ', 'answer': '<<1+1=2>>2\n#### 2'},
             {'question': 'Reserved?', 'answer': '<<1+1=2>>2\n#### 2'},
             {'question': 'Bad source?', 'answer': '<<2+2=5>>5\n#### 5'}]
    splits, excluded = assemble(train, test, ['Reserved?'])
    retained = splits['train']+splits['valid']
    assert len(retained) == 1 and len(excluded) == 4 and len(splits['test']) == 1
    assert retained[0]['published_calculations_checked'] == 1
    assert '<<' not in retained[0]['response']
    assert '#### 4' in retained[0]['response']
    assert splits['test'][0]['answer_numerator'] == 2
