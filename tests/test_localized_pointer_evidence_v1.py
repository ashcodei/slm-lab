import copy
import json

import pytest
from PIL import Image, ImageOps
from tokenizers import Tokenizer

from slm_lab.localized_pointer_evidence_v1 import (
    dense_target, expected_rows, image_layout, original_queries, require_row)


def source():
    return dict(image_id='original-photo', annotator_id=17, dataset_id='publisher-label-is-not-role',
        caption='red flowers a green tree', timed_caption=[
            dict(utterance='red flowers', start_time=.2, end_time=.8),
            dict(utterance='a green tree', start_time=1., end_time=1.6)], traces=[[
                dict(x=.1, y=.2, t=t) for t in [.2, .3, .6, .8]]+[
                dict(x=.8, y=.6, t=t) for t in [1., 1.1, 1.4, 1.6]]])


def test_raw_transcript_samples_tokens_roles_and_geometry_reconstruct():
    record=source();raw=json.dumps(record).encode()+b'\n'
    tokenizer=Tokenizer.from_file('checkpoints/dialogue-retention-v1/systems-variety-code/tokenizer.json')
    photo=dict(image_id=record['image_id'], split='validation', group_id='original-source-group', stored_dimensions=[400,200])
    rows,reviews=expected_rows(record,raw,'validation-original.jsonl',211,photo,tokenizer)
    assert reviews==0 and len(rows)==2 and all(r['optimization_forbidden'] for r in rows)
    assert rows[0]['source_point_refs']==[[0,0],[0,1],[0,2],[0,3]]
    assert rows[0]['attention_grid_histogram']==[[4*14+1,4]]
    assert rows[1]['attention_grid_histogram']==[[7*14+11,4]]
    assert rows[0]['query_input_ids']==[1,*tokenizer.encode('red flowers').ids,2]
    assert rows[0]['source_record']['offset']==211 and rows[0]['source_record']['bytes']==len(raw)
    assert rows[0]['source_dataset_id']!='validation'
    require_row(rows[0], rows[0])
    for key,value in [('query','blue flowers'),('split','train'),('optimization_forbidden',False),
            ('source_point_refs',[[0,4],[0,5],[0,6],[0,7]]),('attention_grid_histogram',[[1,4]]),
            ('query_input_ids',[1,5,2])]:
        bad=copy.deepcopy(rows[0]);bad[key]=value
        with pytest.raises(ValueError,match='original source reconstruction'):require_row(bad,rows[0])
    counts,valid=dense_target(rows[0],[400,200])
    assert sum(counts)==4 and counts[57]==4 and not valid[0]
    for dims in [(400,200),(201,400),(448,448),(896,333),(448,97)]:
        bounds,mask=image_layout(dims)
        actual=ImageOps.pad(Image.new('L',dims,255),(448,448),method=Image.Resampling.NEAREST,color=0)
        assert bounds==(actual.getbbox()[0],actual.getbbox()[1],actual.getbbox()[2]-actual.getbbox()[0],actual.getbbox()[3]-actual.getbbox()[1])
        assert mask==[actual.crop((x*32,y*32,(x+1)*32,(y+1)*32)).getbbox() is not None for y in range(14) for x in range(14)]


def test_changed_original_sources_and_padding_targets_cannot_supply_labels():
    record=source();record['caption']='rewritten transcript'
    with pytest.raises(ValueError,match='complete transcript'):original_queries(record)
    record=source();record['traces'][0][1]['t']=.1
    with pytest.raises(ValueError,match='times must be ordered'):original_queries(record)
    record=source();record['traces'][0][0]['x']=float('nan')
    with pytest.raises(ValueError,match='sample must be finite'):original_queries(record)
    record=source()
    for point in record['traces'][0][:4]:point['x']=1.2
    assert [r['query'] for r in original_queries(record)]==['a green tree']
    for hist,count in [([[0,4]],4),([[57,2],[57,2]],4),([[57,-4]],4),([[57,4]],5)]:
        row=dict(image_size=448,grid_side=14,attention_grid_histogram=hist,source_point_count=count)
        with pytest.raises(ValueError):dense_target(row,[400,200])
