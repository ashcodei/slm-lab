import bz2
import pickle
import pytest
from slm_lab.passive_pickle import load_primitives
from slm_lab.wikipedia_corpus import articles,clean_article
from slm_lab.code_search_corpus import applicable_license,python_body


def test_primitive_license_metadata_loads_but_callable_pickles_do_not_execute(tmp_path):
    value={'org/repo':[('LICENSE','MIT text')], 'flag':True,'nested':{'name':None}}
    assert load_primitives(pickle.dumps(value,protocol=3))==value
    assert load_primitives(pickle.dumps(value,protocol=5))==value
    class Executable:
        def __reduce__(self):return (eval,('1+1',))
    with pytest.raises(ValueError,match='opcode'):load_primitives(pickle.dumps(Executable()))


def test_wikipedia_uses_page_ids_and_skips_redirects_and_nonarticle_namespaces(tmp_path):
    xml=b'''<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
    <page><title>Chemistry</title><ns>0</ns><id>11</id><revision><id>900</id><timestamp>date</timestamp><text>Article body</text></revision></page>
    <page><title>Template</title><ns>10</ns><id>12</id><revision><id>901</id><text>Skip</text></revision></page>
    <page><title>Redirect</title><ns>0</ns><id>13</id><redirect title="Other"/><revision><id>902</id><text>Skip</text></revision></page>
    <page><title>Biology</title><ns>0</ns><id>14</id><revision><id>903</id><text>Second body</text></revision></page>
    </mediawiki>'''
    path=tmp_path/'source.bz2';path.write_bytes(bz2.compress(xml));rows=list(articles(path))
    assert [(r['page_id'],r['revision']) for r in rows]==[('11','900'),('14','903')]


def test_article_cleaner_preserves_math_and_removes_reference_text():
    markup=('A chemical explanation with substantive prose. '*20)+'{{math|x^2}}<ref>HIDDEN_REF</ref>\n== References ==\nBIBLIOGRAPHY_ONLY\n[[Category:Chemistry]]'
    text,domain,categories=clean_article(markup,'Chemistry')
    assert 'x^2' in text and 'HIDDEN_REF' not in text and 'BIBLIOGRAPHY_ONLY' not in text
    assert domain=='chemistry' and categories==['Chemistry']


def test_closest_license_scope_and_conflicting_notices_are_handled():
    mit='Permission is hereby granted, free of charge. The above copyright notice must remain.'
    apache='Apache License Version 2.0, January 2004'
    notices=[('org/repo/LICENSE',mit),('org/repo/sub/LICENSE',apache)]
    assert applicable_license('org/repo','main.py',notices)['license']=='MIT'
    assert applicable_license('org/repo','sub/module.py',notices)['license']=='Apache-2.0'
    assert applicable_license('org/repo','sub/module.py',notices+[('org/repo/sub/COPYING',mit)]) is None
    assert applicable_license('other/repo','main.py',notices) is None


def test_python_documented_implementation_keeps_logic_without_duplicate_comment():
    source='def add(a, b):\n    """Add two values."""\n    return a + b\n'
    body=python_body(source)
    assert 'return a + b' in body and 'Add two values' not in body
    assert python_body('def broken(:') is None


def test_legacy_surrogate_license_notice_is_preserved_as_metadata():
    import json
    text='Permission is hereby granted, free of charge. The above copyright notice. Legacy byte: \udca9'
    record=applicable_license('org/repo','main.py',[('org/repo/LICENSE',text)])
    assert record['license']=='MIT' and len(record['license_text_sha256'])==64
    serialized=json.dumps({'text':text},ensure_ascii=True).encode('utf-8')
    assert json.loads(serialized)['text']==text
