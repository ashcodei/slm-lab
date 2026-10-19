import copy
import math

import pytest

from slm_lab.point_grounding_protocol_v1 import control_pairs, fit_position_prior, prior_logits, score, summarize


def row(identity, photo, cell, query='red flower', role='validation'):
    return dict(id=identity,image_id=photo,query=query,split=role,optimization_forbidden=role!='train',
        attention_grid_histogram=[[cell,4]],source_point_count=4)


def test_original_train_only_prior_and_matched_geometry_aware_scores():
    prior=fit_position_prior([row('t1','train',0,role='train'),row('t2','train',3,role='train')],cells=4)
    assert prior['counts']==[2.,1.,1.,2.] and prior['fitted_original_training_queries']==2
    with pytest.raises(ValueError,match='original training queries'):
        fit_position_prior([row('held','test',0)],cells=4)
    result=score([math.log(3),0.,0.,math.log(2)],[3.,0.,0.,1.],[True]*4,grid=2)
    assert result['predicted_cell']==0 and result['top1_observed_attention_cell']
    assert result['original_sample_mass_at_top1']==.75
    assert result['soft_target_cross_entropy']==pytest.approx(math.log(7)-.75*math.log(3)-.25*math.log(2))
    assert result['expected_normalized_grid_center_distance']==pytest.approx(.25*math.sqrt(.5))
    masked=score([100.,0.,0.,1.],[0.,0.,0.,4.],[False,True,True,True],grid=2)
    assert masked['predicted_cell']==3 and masked['original_sample_mass_at_top1']==1
    assert len(prior_logits('training_position',[False,True,True,True],prior))==4
    with pytest.raises(ValueError):score([0.]*4,[4.,0.,0.,0.],[False,True,True,True],grid=2)
    with pytest.raises(ValueError):prior_logits('center',[False,True],center_cell=0)


def test_deterministic_same_photo_query_and_distinct_photo_controls_are_recounted():
    rows=[row('a','first',0),row('b','first',195,'blue sky'),row('c','second',0)]
    photos={'first':dict(source_photo_aliases=['12']),'second':dict(source_photo_aliases=['34'])}
    pairs=control_pairs(rows,photos)
    assert pairs==control_pairs(reversed(rows),photos)
    by_id={p['id']:p for p in pairs}
    assert by_id['a']['wrong_query_id']=='b' and by_id['b']['wrong_query_id']=='a' and by_id['c']['wrong_query_id'] is None
    assert all(p['wrong_image_id']!=p['image_id'] for p in pairs)
    observations=[]
    for r in rows:
        counts=[0.]*196;counts[r['attention_grid_histogram'][0][0]]=4
        for condition in ['proper','blank_image','wrong_image','uniform','center','training_position']:
            logits=[0.]*196
            if condition=='proper':logits[r['attention_grid_histogram'][0][0]]=5
            observations.append(dict(id=r['id'],image_id=r['image_id'],condition=condition,scores=score(logits,counts,[True]*196)))
        if by_id[r['id']]['wrong_query_id']:
            observations.append(dict(id=r['id'],image_id=r['image_id'],condition='wrong_query',scores=score([0.]*196,counts,[True]*196)))
    report=summarize(observations)
    assert report['conditions']['proper']['query_mean']['top1_observed_attention_cell']==1
    assert report['conditions']['wrong_query']['queries']==2
    assert report['conditions']['blank_image']['query_mean']['top1_observed_attention_cell']==pytest.approx(2/3)
    with pytest.raises(ValueError,match='same original queries'):summarize(observations[1:])
    altered=copy.deepcopy(observations);altered[0]['image_id']='different target'
    with pytest.raises(ValueError,match='original target photo'):summarize(altered)
