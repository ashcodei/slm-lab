import copy
import json

import pytest

from slm_lab.codeio_case_segments_v1 import Segments
from slm_lab.codeio_published_cases_v1 import contract
from slm_lab.codeio_whole_cases_v2 import execute

BINDINGS = {'controlled_fixture_source': 'a'*64}


def actual_record(position):
    cases=contract(dict(refcode='def main_solution(*,x):\n    return {"value": [x+1, True]}\n',
        funcname='main_solution',ios=[dict(input={'x':2},output={'value':[3,True]}),dict(input={'x':-1},output={'value':[0,True]})]))
    observations=[execute(cases,seed=seed,seconds=10) for seed in [0,1]]
    noop=execute(cases,seed=0,seconds=10,program='def main_solution(**kwargs):\n    return None\n')
    assert all(o['all_original_cached_values_match'] for o in observations) and not noop['all_original_cached_values_match']
    return dict(original_position=position,original_row_sha256='b'*64,
        fixed_role=dict(source_role='train' if position%2==0 else 'reserved'),
        source_optimization_forbidden=position%2==1,whole_component_quarantined=False,
        all_original_cases_and_seeds_match=True,status='all_original_cached_values_match_two_local_seed_contexts',
        total_original_cases=cases['total_cases'],original_program_sha256=cases['original_program_sha256'],
        original_cases_sha256=cases['original_cases_sha256'],reference_observations=observations,no_OP_control=noop,
        source_only_no_training_admission=True,independent_semantic_correctness_certified=False,RL_reward_admitted=False)


def store(path,**kwargs):
    return Segments(path,bindings=BINDINGS,total=3,width=2,minimum_free_bytes=0,
        maximum_total_bytes=1024**2,maximum_segment_bytes=512*1024,**kwargs)


def test_real_confined_case_observations_recover_closed_orphan_and_actual_tail(tmp_path,monkeypatch):
    first=actual_record(0);second=copy.deepcopy(first);second.update(original_position=1,
        fixed_role={'source_role':'reserved'},source_optimization_forbidden=True)
    original=store(tmp_path)
    def fail():raise OSError('controlled interruption after durable segment publication')
    monkeypatch.setattr(original,'_write_cursor',fail)
    with pytest.raises(OSError):original.append([first,second])
    assert len(list(tmp_path.glob('segment-*.json.gz')))==1
    pending=tmp_path/'interrupted-original-source.pending';pending.write_bytes(b'partial original attempt retained')
    recovered=store(tmp_path)
    assert recovered.next_position==2 and list(recovered.rows())==[first,second]
    tail=copy.deepcopy(first);tail['original_position']=2
    recovered.append([tail])
    again=store(tmp_path)
    assert again.next_position==3 and list(again.rows())==[first,second,tail]
    assert json.loads((tmp_path/'cursor.json').read_text())['complete']
    assert pending.read_bytes()==b'partial original attempt retained'
    with pytest.raises(ValueError):again.append([tail])


def test_partial_positive_wrong_role_source_binding_or_corrupt_closed_frame_rejects(tmp_path):
    row=actual_record(0);other=copy.deepcopy(row);other.update(original_position=1,
        fixed_role={'source_role':'reserved'},source_optimization_forbidden=True)
    writer=store(tmp_path)
    for change in ['partial','role','position','seed']:
        bad=copy.deepcopy([row,other])
        if change=='partial':bad[0]['reference_observations'][0]['records'].pop()
        elif change=='role':bad[1]['source_optimization_forbidden']=False
        elif change=='position':bad[1]['original_position']=2
        else:bad[0]['reference_observations'][1]['seed']=0
        with pytest.raises(ValueError):writer.append(bad)
        assert writer.next_position==0 and not list(tmp_path.glob('segment-*.json.gz'))
    writer.append([row,other])
    with pytest.raises(ValueError):Segments(tmp_path,bindings={'changed':'c'*64},total=3,width=2,
        minimum_free_bytes=0,maximum_total_bytes=1024**2,maximum_segment_bytes=512*1024)
    cursor=tmp_path/'cursor.json';saved=cursor.read_bytes();value=json.loads(saved);value['next_original_position']=1
    cursor.write_text(json.dumps(value))
    with pytest.raises(ValueError):store(tmp_path)
    cursor.write_bytes(saved)
    path=next(tmp_path.glob('segment-*.json.gz'));raw=bytearray(path.read_bytes());raw[len(raw)//2]^=1;path.write_bytes(raw)
    with pytest.raises(Exception):store(tmp_path)
