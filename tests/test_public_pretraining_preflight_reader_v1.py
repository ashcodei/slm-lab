from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

from slm_lab.public_pretraining_intake_v1 import candidate
from test_public_pretraining_intake_v1 import fixture


def reader():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('public_preflight_reader_unit',root/'scripts/read_public_pretraining_preflight_CPU_v1.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_whole_raw_source_token_mask_and_EOS_corruption_reject():
    m=reader();t,q,s,r=fixture();entry,_=candidate(r,s,t,q);proof=m.verify_document(t,q,s,r,entry)
    assert proof['targets']==len(entry['whole_native_ids'])-1
    for change in [lambda x:x['whole_native_ids'].pop(),lambda x:x['whole_native_labels'].__setitem__(2,-100),
                   lambda x:x.update(original_public_text=r['code'][:25]),lambda x:x.update(optimization_admitted=True)]:
        bad=deepcopy(entry);change(bad)
        with pytest.raises(ValueError):m.verify_document(t,q,s,r,bad)


def test_per_file_license_cannot_be_replaced_by_catalog_or_changed_repository_attribution():
    m=reader();t,q,s,r=fixture();entry,_=candidate(r,s,t,q);bad=deepcopy(entry)
    bad['rights']['publisher_per_file_license']='apache-2.0'
    with pytest.raises(ValueError,match='per-file license'):m.verify_document(t,q,s,r,bad)
    bad=deepcopy(entry);bad['original_public_URL']='https://github.com/another/project'
    with pytest.raises(ValueError,match='attribution'):m.verify_document(t,q,s,r,bad)
