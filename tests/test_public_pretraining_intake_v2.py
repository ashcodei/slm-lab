from copy import deepcopy
import json
from pathlib import Path

import pyarrow.parquet as pq

from slm_lab.held_text_overlap_v1 import HeldTextOverlap
from slm_lab.public_pretraining_intake_v1 import candidate as first
from slm_lab.public_pretraining_intake_v2 import candidate,language
from slm_lab.public_billion_candidate_v1 import key
from test_public_pretraining_intake_v1 import fixture


def test_actual_publisher_GO_alias_reaches_Go_quota_without_changing_original_fields_or_bytes():
    t,q,s,r=fixture();r=dict(r,language='GO',path='main.go',code='package main\nfunc seven(n int) int { return n * 7 }\n');original=deepcopy(r)
    item,reason=candidate(r,s,t,q)
    assert item is not None and r==original and item['original_publisher_row']==original
    assert item['language']=='Go' and item['original_publisher_language']=='GO'
    assert key(s['family'],language(r['language']))=='code:Go'
    assert t.decode(item['whole_native_ids'][1:-1],skip_special_tokens=False)==r['code']
    assert item['whole_native_labels']==[-100,*item['whole_native_ids'][1:]]


def test_every_non_GO_native_sequence_role_and_code_license_decision_remains_unchanged():
    t,q,s,r=fixture();a,why=first(r,s,t,q);b,other=candidate(r,s,t,q)
    assert why==other
    assert all(b[k]==v for k,v in a.items())
    assert candidate(dict(r,language='GO',license='gpl-3.0'),s,t,q)[0] is None


def test_three_real_original_publisher_GO_rows_are_accepted_with_lossless_native_masks():
    root=Path(__file__).resolve().parents[1];t,_,_,_=fixture()
    q=HeldTextOverlap(json.loads((root/'reports/CodeContests-native-held-text-v2/unchanged-held-text-registry.json').read_text()))
    plan=json.loads((root/'reports/public-pretraining-source-preflight-CPU-v1/plan.json').read_text());s=next(x for x in plan['pinned_sources'] if x['family']=='code')
    path=root/'data/public-pretraining-source-preflight-v1/code/original.parquet';count=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128,use_threads=False):
        for row in batch.to_pylist():
            if row.get('language')!='GO':continue
            item,why=candidate(row,s,t,q)
            if item is None:continue
            assert item['original_publisher_row']['language']=='GO' and item['language']=='Go'
            assert t.decode(item['whole_native_ids'][1:-1],skip_special_tokens=False)==row['code']
            assert item['complete_causal_targets']==len(item['whole_native_ids'])-1
            count+=1
            if count==3:break
        if count==3:break
    assert count==3
