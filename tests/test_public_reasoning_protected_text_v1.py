"""Declared complete protected stems and bidirectional literal-copy controls."""
from collections import Counter
from pathlib import Path

from slm_lab.public_reasoning_protected_text_v1 import overlaps,population


def candidate(identity,question,prompt=None):
    return dict(id=identity,prompt=prompt or question,independent_answer='42',original_entry=dict(question=question))


def test_complete_actual_original_population_and_only_input_fields():
    rows=population(Path(__file__).resolve().parents[1]);counts=Counter(r['family'] for r in rows)
    assert len(rows)==len({r['audit_id'] for r in rows})==31349
    assert counts['full_QA_and_MBPP']==17561 and counts['APPS_original_test']==5000
    assert counts['whole_human_preferences']==4642 and counts['AI2D_original_test']==3088
    assert all(set(row)=={'audit_id','family','question','source_path','source_position'} for row in rows)


def test_both_literal_directions_and_unicode_word_boundaries():
    held=[dict(audit_id='held-small',family='toy',question='What is 17 plus 25?'),
          dict(audit_id='held-long',family='toy',question='Extra context. How many bits are set in 17? Additional context.')]
    rows=[candidate('forward','Additional text. What is 17 plus 25?'),
          candidate('reverse','How many bits are set in 17?'),candidate('clean','A separate algorithm problem.')]
    result,settings=overlaps(rows,held)
    assert any(r['kind']=='protected_stem_in_complete_candidate' and r['protected_id']=='held-small' for r in result['forward'])
    assert any(r['kind']=='candidate_question_in_protected_input' and r['protected_id']=='held-long' for r in result['reverse'])
    assert result['clean']==[] and settings['both_literal_directions']


def test_decorated_copies_and_protected_gold_independence():
    stem=' '.join('unique'+str(i) for i in range(90))
    held=[dict(audit_id='held',family='toy',question=stem,gold='never consulted')]
    decorated=' '.join('**'+word+'**' for word in stem.split())
    rows=[candidate('decorated',decorated)]
    first,_=overlaps(rows,held);second,_=overlaps(rows,[{**held[0],'gold':'changed'}])
    assert first==second
    assert any(r['kind']=='rare_word_shingle_copy' for r in first['decorated'])
    assert not any(r['kind']=='protected_stem_in_complete_candidate' for r in first['decorated'])
