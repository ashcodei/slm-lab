import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import shutil
import sys

import pytest


def driver():
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'scripts'))
    spec=importlib.util.spec_from_file_location('code_first_visual_reader_unit',root/'scripts/assess_AI2D_fixed_exposure_strata_v4.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_actual_predeclared_all3088_source_groups_are_reused():
    module=driver();folder=module.PREPARED
    preparation=json.loads((folder/'preparation-complete.json').read_text())
    rows=list(map(json.loads,(folder/'all3088-fixed-source-exposure-flags.jsonl').open()))
    population=SimpleNamespace(identity_sha256=preparation['original_population_sha256'],cases=[{'id':r['id']} for r in rows])
    actual,binding=module.prepared(population,folder)
    assert actual==rows and len(rows)==3088 and len(module.ENDPOINTS)==8
    assert binding['source_group_sizes']['generic_question_in_inspected_text']==16
    assert binding['source_group_sizes']['no_generic_question_match_in_inspected_text']==3072
    assert module.VISUAL.name=='full-AI2D-pipeline-v4'


def test_changed_actual_flags_or_source_population_are_rejected(tmp_path):
    module=driver();original=module.PREPARED
    for name in ['preparation-complete.json','all3088-fixed-source-exposure-flags.jsonl','fixed-original-cohort-identities.json']:
        shutil.copyfile(original/name,tmp_path/name)
    preparation=json.loads((tmp_path/'preparation-complete.json').read_text())
    rows=list(map(json.loads,(tmp_path/'all3088-fixed-source-exposure-flags.jsonl').open()))
    population=SimpleNamespace(identity_sha256='changed',cases=[{'id':r['id']} for r in rows])
    with pytest.raises(ValueError):module.prepared(population,tmp_path)
    population.identity_sha256=preparation['original_population_sha256']
    rows[0]['memberships']=[]
    (tmp_path/'all3088-fixed-source-exposure-flags.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError):module.prepared(population,tmp_path)
