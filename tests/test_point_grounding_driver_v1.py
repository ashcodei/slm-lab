import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from slm_lab.localized_pointer_evidence_v1 import dense_target,image_layout
from slm_lab.point_grounding_protocol_v1 import control_pairs,fit_position_prior,prior_logits,score
from slm_lab.point_grounding_readback_v1 import recount_attention,bootstrap_photo_gain


def test_active_math_owner_or_original_work_rejects_before_gpu_queries_signals_and_output(monkeypatch,tmp_path):
    path=Path('scripts/train_point_grounding_joint_v1.py').resolve()
    spec=importlib.util.spec_from_file_location('checked_point_driver',path);driver=importlib.util.module_from_spec(spec)
    sys.path.insert(0,str(path.parent))
    try:
        spec.loader.exec_module(driver)
    finally:
        sys.path.pop(0)
    reports=tmp_path/'not-created';monkeypatch.setattr(driver,'REPORTS',reports);monkeypatch.setattr(driver,'OUTPUT',tmp_path/'no-weights')
    monkeypatch.setattr(driver.resources,'same',lambda actor:dict(pid=actor['pid'],ticks=actor['ticks'],state='R'))
    actions=[]
    monkeypatch.setattr(driver.subprocess,'run',lambda *a,**k:actions.append('GPU query'))
    monkeypatch.setattr(driver.os,'kill',lambda *a,**k:actions.append('signal'))
    with pytest.raises(ValueError,match='math owner must actually end'):
        driver.parent({},dict(pid=123,ticks='456'))
    with pytest.raises(ValueError,match='Never preempt original work'):
        driver.idle_original()
    assert not actions and not reports.exists() and not driver.OUTPUT.exists()
    reports.mkdir();child=dict(pid=999999,ticks='one-exact-birth');(reports/'children.json').write_text(__import__('json').dumps([child]))
    states={child['pid']:dict(**child,state='R'),driver.WAITER['pid']:dict(**driver.WAITER,state='T')}
    monkeypatch.setattr(driver.resources,'same',lambda actor:states.get(actor['pid']))
    def stop_registered(actor):
        actions.append('stop exact child');states.pop(actor['pid'])
    def resume_original(pid,signum):
        assert pid==driver.WAITER['pid'] and signum==driver.signal.SIGCONT and child['pid'] not in states
        actions.append('resume original');states[pid]['state']='S'
    monkeypatch.setattr(driver.resources,'stop_child',stop_registered);monkeypatch.setattr(driver.os,'kill',resume_original)
    driver.restore_owned(dict(waiter=driver.WAITER),'unit owned restoration')
    driver.restore_owned(dict(waiter=driver.WAITER),'second cleanup must have no extra effect')
    assert actions==['stop exact child','resume original']


def test_raw_attention_controls_scores_priors_and_paired_photo_gains_reconstruct():
    photos={key:dict(image_id=key,source_photo_aliases=[key],stored_dimensions=[448,448],model_pixel_sha256_448=key*64) for key in ['a','b']}
    rows=[dict(id=key,image_id=key,split='validation',optimization_forbidden=True,query='flower '+key,
        query_input_ids=[1,4+i,2],attention_grid_histogram=[[9+i,4]],source_point_count=4,image_size=448,grid_side=14)
        for i,key in enumerate(photos)]
    pairs=control_pairs(rows,photos);pairing={p['id']:p for p in pairs}
    prior=fit_position_prior([dict(rows[0],id='train-source',split='train',optimization_forbidden=False)])
    observations=[]
    import hashlib
    for row in rows:
        target,valid=dense_target(row,[448,448]);cell=row['attention_grid_histogram'][0][0]
        for condition in ['proper','blank_image','wrong_image','uniform','center','training_position']:
            logits=[0.]*196
            if condition in ['uniform','center','training_position']:
                logits=prior_logits(condition,valid,prior,center_cell=105)
            elif condition=='proper':logits[cell]=5.
            image=None if condition=='blank_image' else pairing[row['id']]['wrong_image_id'] if condition=='wrong_image' else row['image_id']
            sha=hashlib.sha256(bytes([128,128,128])*(448*448)).hexdigest() if image is None else photos[image]['model_pixel_sha256_448']
            observations.append(dict(id=row['id'],image_id=row['image_id'],condition=condition,
                actual_query_id=row['id'],actual_query_input_ids=row['query_input_ids'],actual_image_id=image,
                actual_448px_RGB_sha256=sha,original_attention_grid_histogram=row['attention_grid_histogram'],
                original_target_valid_cells=valid,logits=logits,scores=score(logits,target,valid)))
    summary,conditions=recount_attention(observations,rows,photos,pairs,prior)
    assert summary['conditions']['proper']['queries']==2
    gain=bootstrap_photo_gain(conditions['proper'],conditions['wrong_image'],replicates=50)
    assert gain['equal_photo_mean_CE_gain']>0 and gain['percentile95_interval'][0]>0
    for index,field,value in [(0,'actual_image_id','b'),(0,'actual_query_input_ids',[1,9,2]),(2,'original_attention_grid_histogram',[[1,4]])]:
        changed=copy.deepcopy(observations);changed[index][field]=value
        with pytest.raises(ValueError,match='must reconstruct'):recount_attention(changed,rows,photos,pairs,prior)
    with pytest.raises(ValueError,match='Missing actual original'):recount_attention(observations[:-1],rows,photos,pairs,prior)
