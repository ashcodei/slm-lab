import copy

import pytest

from slm_lab.codeio_whole_text_screen_v2 import inspect,blocked_components,whole_text
from slm_lab.held_text_overlap_v1 import HeldTextOverlap


def row():
    return dict(problem_description='Return a string.',io_requirements='x: str; return str',
        refcode='def main_solution(x):\n    return x\n',funcname='main_solution',
        ios=[dict(input={'x':'ordinary'},output={'nested':['ordinary']})])


def test_late_code_comment_and_nested_cached_value_are_screened_without_cropping():
    stem=' '.join('protected'+str(i) for i in range(90));index=HeldTextOverlap([dict(audit_id='held',question=stem)])
    source=row();assert inspect(source,index)['literal_held_stem_ids']==[]
    code=copy.deepcopy(source);code['refcode']+='\n# '+'preamble '*8000+stem
    result=inspect(code,index);assert result['literal_held_stem_ids']==['held'] and not result['source_text_cropped']
    cached=copy.deepcopy(source);cached['ios'][0]['output']['nested']=['padding '*8000,stem]
    result=inspect(cached,index);assert result['literal_held_stem_ids']==['held']
    assert stem in whole_text(cached)


def test_component_veto_and_uninspectable_original_values_cannot_silently_pass():
    clear=dict(literal_held_stem_ids=[],lexical_held_stem_ids=[])
    records=[dict(fixed_role={'component_sha256':c},prior_component_quarantined=False,
        whole_source_screen_qualification=False,screen=copy.deepcopy(clear)) for c in ['a','a','b','c']]
    records[1]['screen']['literal_held_stem_ids']=['held'];records[2]['whole_source_screen_qualification']=True
    assert blocked_components(records)=={'a','b'}
    for value in [float('nan'),float('inf'),'contains\0sentinel']:
        source=row();source['ios'][0]['output']=value
        with pytest.raises(ValueError):whole_text(source)
