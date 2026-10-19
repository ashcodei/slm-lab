import json

import numpy as np

from slm_lab.corpus import ShardWriter
from slm_lab.open_education_pack_v1 import BookComponents,book_roles,WholeTokenWindows


def test_duplicate_connected_books_share_roles_with_domain_holdout():
    graph=BookComponents(['a','b','c','d','e']);graph.union('a','b')
    groups=graph.groups();roles=book_roles(groups,dict(a='math',b='math',c='math',d='physics',e='physics'),73)
    assert graph.find('a')==graph.find('b')
    assert set(roles[g] for g,books in groups.items() if books[0] in ('a','c'))=={'train','valid'}
    assert set(roles[g] for g,books in groups.items() if books[0] in ('d','e'))=={'train','valid'}


def test_every_real_next_token_including_short_tail_is_supervised_once(tmp_path):
    writer=ShardWriter(tmp_path,'train',np.dtype('<u2'),5)
    values=[1,5,6,2,1,7,8,9,10,11,12,2];writer.append(values);writer.flush()
    (tmp_path/'corpus.json').write_text(json.dumps(dict(tokenizer_sha256='a'*64,splits=dict(train=dict(shards=writer.files)))))
    corpus=WholeTokenWindows(tmp_path,'a'*64,4,'train')
    actual=[];positions=[]
    for i in range(len(corpus)):
        row=corpus[i]
        actual.extend(v for v in row['labels'][1:] if v!=-100)
        positions.extend(range(row['original_first_target_position'],row['original_last_target_position_exclusive']))
    assert actual==values[1:]
    assert positions==list(range(1,len(values)))
    assert corpus[len(corpus)-1]['attention_mask']==[True,True,True,False]
