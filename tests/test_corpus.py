import hashlib
import json
import numpy as np
from slm_lab.corpus import PackedCorpus


def test_packed_windows_span_shard_boundary_without_missing_targets(tmp_path):
    a=np.arange(0,5,dtype='<u2');b=np.arange(5,11,dtype='<u2')
    a.tofile(tmp_path/'a.bin');b.tofile(tmp_path/'b.bin')
    manifest={'tokenizer_sha256':'test','dtype':'<u2','splits':{'train':{'shards':[
        {'path':'a.bin','tokens':5},{'path':'b.bin','tokens':6}]}}}
    (tmp_path/'corpus.json').write_text(json.dumps(manifest))
    corpus=PackedCorpus(tmp_path,'test',length=4)
    assert len(corpus)==3
    assert corpus[0]['tokens']==[0,1,2,3]
    assert corpus[1]['tokens']==[3,4,5,6]
    assert corpus[2]['tokens']==[6,7,8,9]


def test_source_chunks_stay_in_one_split_and_domain_counts_match(tmp_path):
    from argparse import Namespace
    from tokenizers import Tokenizer, models, pre_tokenizers
    from slm_lab.corpus import pack
    words=[f'document{i}' for i in range(40)]
    tokenizer=Tokenizer(models.WordLevel({'<unk>':0,'<eos>':1,**{w:i+2 for i,w in enumerate(words)}},unk_token='<unk>'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    token_path=tmp_path/'tokenizer.json';tokenizer.save(str(token_path))
    rows=[{'text':word+' '+word,'group_id':f'book-{i//2}','domain':'mathematics'} for i,word in enumerate(words)]
    source=tmp_path/'source.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    root=tmp_path/'packed'
    pack(Namespace(jsonl=str(source),dataset=None,tokenizer=str(token_path),output=str(root),max_tokens=1000,shard_tokens=13))
    manifest=json.loads((root/'corpus.json').read_text())
    tokens={s:set(np.concatenate([np.fromfile(root/f['path'],dtype='<u2') for f in v['shards']]).tolist())
            for s,v in manifest['splits'].items()}
    for i in range(0,40,2):
        assert (i+2 in tokens['train']) == (i+3 in tokens['train'])
        assert (i+2 in tokens['valid']) == (i+3 in tokens['valid'])
        assert (i+2 in tokens['train']) != (i+2 in tokens['valid'])
    assert sum(manifest['domain_tokens']['mathematics'].values()) == manifest['total_tokens']


def test_mixture_preserves_tokens_and_rejects_incompatible_tokenizer(tmp_path):
    from slm_lab.mix_corpora import mix
    import pytest
    components=[]
    for name,values in [('code',[2,3,4,5]),('reference',[6,7,8,9])]:
        root=tmp_path/name;root.mkdir()
        for split in ['train','valid']:np.asarray(values,dtype='<u2').tofile(root/(split+'.bin'))
        manifest={'tokenizer_sha256':'same','dtype':'<u2','source':{'name':name},
            'splits':{s:{'tokens':4,'shards':[{'path':s+'.bin','tokens':4}]} for s in ['train','valid']}}
        (root/'corpus.json').write_text(json.dumps(manifest));components.append((name,root,None))
    mixed=tmp_path/'mixed';mix(components,mixed)
    corpus=PackedCorpus(mixed,'same',4)
    assert corpus[0]['tokens']==[2,3,4,5]
    assert corpus[1]['tokens']==[5,6,7,8]
    root=components[1][1];manifest=json.loads((root/'corpus.json').read_text())
    manifest['tokenizer_sha256']='different';(root/'corpus.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='same tokenizer'):mix(components,tmp_path/'invalid')
