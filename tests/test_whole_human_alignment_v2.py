import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from slm_lab.model import ModelConfig
from slm_lab.whole_human_alignment_v2 import WholeResponseEncoder, PackedWriter, split_human_ranked_pair


def encoder():
    token=Tokenizer(models.WordLevel({'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3,'Assistant:':4,'User:':5,'x':6,'y':7},unk_token='<unk>'))
    token.pre_tokenizer=pre_tokenizers.WhitespaceSplit()
    config=ModelConfig(vocab_size=8,hidden_size=16,intermediate_size=32,num_layers=1,num_heads=4,num_kv_heads=2,
        max_text_length=64,vision_hidden_size=8,vision_intermediate_size=16,vision_layers=1,vision_heads=2,
        image_size=16,patch_size=4,pool_size=2)
    return WholeResponseEncoder(token,config)


def test_response_over_old_half_window_remains_whole_and_over_context_rejects():
    native=encoder()
    ids,start=native.sequence(dict(prompt='Assistant:',response='x '*40),'response')
    assert ids==[1,4]+[6]*40+[2] and start==2
    with pytest.raises(ValueError,match='full context'):
        native.sequence(dict(prompt='Assistant:',response='x '*63),'response')


def test_original_pair_suffix_whitespace_and_prefix_are_preserved_exactly():
    raw=dict(chosen='\n\nHuman: x\n\nAssistant:  y\n',rejected='\n\nHuman: x\n\nAssistant: x  ')
    pair=split_human_ranked_pair(raw)
    assert pair['prompt']+pair['chosen']==raw['chosen']
    assert pair['prompt']+pair['rejected']==raw['rejected']
    assert pair['chosen']=='  y\n' and pair['rejected']==' x  '
    with pytest.raises(ValueError,match='complete final source prompt'):
        split_human_ranked_pair({**raw,'rejected':raw['rejected'].replace('Human: x','Human: y')})


def test_native_binary_offsets_cover_both_whole_preference_responses(tmp_path):
    native=encoder();pair=dict(id='human-ranking-fixture',prompt='Assistant:',chosen='x '*40,rejected='y '*12)
    sequences={field:native.sequence(pair,field) for field in ('chosen','rejected')}
    writer=PackedWriter(tmp_path,'preferences.train');writer.append(pair,sequences);summary=writer.close()
    row=json.loads((tmp_path/'preferences.train.jsonl').read_text());tokens=np.fromfile(tmp_path/'preferences.train.tokens.u16',dtype='<u2')
    assert summary['examples']==1 and summary['response_EOS_targets']==54
    assert summary['maximum_whole_response_EOS']==41
    for field,(ids,start) in sequences.items():
        index=row['native'][field]
        assert index['target_start']==start
        assert tokens[index['offset']:index['offset']+index['length']].tolist()==ids
    assert sum(v['length'] for v in row['native'].values())==len(tokens)
