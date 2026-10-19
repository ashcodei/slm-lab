"""Operational-only routing preserves all ten original scientific controllers."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from slm_lab.corpus import file_hash

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_function_repair_GPU_lineage_v1 import transform


def test_all_ten_exact_declared_routing_and_only_human_gate_and_bindings_change():
    mapping=json.loads((ROOT/'reports/function-repair-GPU-lineage-v1-file-map.json').read_text())
    assert len(mapping['stages'])==10 and len({r['queue'] for r in mapping['stages']})==10
    for i,row in enumerate(mapping['stages']):
        old=(ROOT/'scripts'/row['previous_script']).read_text();new=(ROOT/'scripts'/row['recovery_script']).read_text()
        assert new==transform(old,mapping['permitted_literal_replacements'],human=i==0)
        before={n.name:n for n in ast.parse(old).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        after={n.name:n for n in ast.parse(new).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        assert before.keys()==after.keys()
        for name in before:
            if i==0 and name=='previous_ready':
                assert ast.unparse(after[name].body[0])=='if not study_released(ROOT):\n    return False'
                after[name].body=after[name].body[1:]
            if i==0 and name=='parent':
                additions=next(n for n in ast.walk(after[name]) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='additions' for t in n.targets))
                assert len(additions.value.elts)>3
                additions.value.elts=additions.value.elts[:1]+additions.value.elts[4:]
            assert ast.dump(before[name])==ast.dump(after[name]),(row['stage'],name)
        assert row['queue'] in new
    for path,digest in mapping['source_files_sha256'].items():assert file_hash(path)==digest


def test_human_gate_checks_study_restoration_before_original_public_readiness():
    mapping=json.loads((ROOT/'reports/function-repair-GPU-lineage-v1-file-map.json').read_text())
    source=ROOT/'scripts'/mapping['stages'][0]['recovery_script']
    fn=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='previous_ready')
    calls=[]
    namespace={'ROOT':ROOT,'study_released':lambda root:calls.append('study') or False,
        'owner':SimpleNamespace(resources=SimpleNamespace(same=lambda owner:(_ for _ in ()).throw(AssertionError('Original gate ran too early'))))}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),namespace)
    assert namespace['previous_ready']({'pid':1,'ticks':'1'}) is False and calls==['study']
    namespace['study_released']=lambda root:calls.append('released') or True
    namespace['owner'].resources.same=lambda owner:calls.append('original') or True
    assert namespace['previous_ready']({'pid':1,'ticks':'1'}) is False
    assert calls==['study','released','original']


def test_factory_endpoint_and_full_population_routes_are_preserved():
    from slm_lab.public_pretraining_endpoints_v1 import ENDPOINTS
    assert len(ENDPOINTS)==9
    mapping=json.loads((ROOT/'reports/function-repair-GPU-lineage-v1-file-map.json').read_text())
    for row in mapping['stages']:
        old=ast.parse((ROOT/'scripts'/row['previous_script']).read_text())
        new=ast.parse((ROOT/'scripts'/row['recovery_script']).read_text())
        before=[ast.dump(n) for n in ast.walk(old) if isinstance(n,ast.ImportFrom) and n.module and
            any(x in n.module for x in ['policy_v','endpoints_v','full_held','full_original','whole_human'])]
        after=[ast.dump(n) for n in ast.walk(new) if isinstance(n,ast.ImportFrom) and n.module and
            any(x in n.module for x in ['policy_v','endpoints_v','full_held','full_original','whole_human'])]
        assert before==after
        before_numbers=sorted(repr(n.value) for n in ast.walk(old) if isinstance(n,ast.Constant) and type(n.value) in {int,float})
        after_numbers=sorted(repr(n.value) for n in ast.walk(new) if isinstance(n,ast.Constant) and type(n.value) in {int,float})
        assert before_numbers==after_numbers


def test_handoff_scope_changes_only_operational_births_and_deadlines():
    from handoff_function_repair_GPU_lineage_v1 import scope
    old=dict(at=1,expires_at_unix=2,owner={'pid':42},exact_predecessor={'pid':41},source_files_sha256={'old':'a'*64},
        full_QA_questions=16961,full_MBPP_problems=600,AI2D_image_conditions=3,arms=['chosen_CE','human_DPO'])
    new=dict(old,at=3,expires_at_unix=4,owner={'pid':142},exact_predecessor={'pid':141},source_files_sha256={'new':'b'*64})
    assert scope(old)==scope(new)
    assert scope(old)!=scope(dict(new,full_MBPP_problems=500))


def test_handoff_refuses_any_learned_output_and_only_accepts_original_empty_strata_plan(tmp_path,monkeypatch):
    import handoff_function_repair_GPU_lineage_v1 as operator
    monkeypatch.setattr(operator,'ROOT',tmp_path);monkeypatch.setattr(operator,'observe',lambda identity,command:identity)
    scripts=tmp_path/'scripts';scripts.mkdir();birth=dict(pid=42,ticks='42')
    (scripts/'original.py').write_text("REPORTS = ROOT/'reports/learned'\nOUTPUT=ROOT/'checkpoints/learned'\n")
    row=dict(stage='learned',previous_script='original.py',previous_queue='reports/old-queue.json')
    assert operator.unstarted(row,birth)==birth
    (tmp_path/'checkpoints/learned').mkdir(parents=True)
    with pytest.raises(ValueError):operator.unstarted(row,birth)
    (scripts/'strata.py').write_text("OUT = ROOT / 'reports/AI2D-fixed-exposure-strata-recovery-v1'\n")
    row=dict(stage='AI2D-fixed-exposure-strata-v5',previous_script='strata.py',previous_queue='reports/strata-queue.json')
    folder=tmp_path/'reports/AI2D-fixed-exposure-strata-recovery-v1';folder.mkdir(parents=True)
    content=json.dumps(dict(owner=birth,actual_neural_assessments=0,model_updates=0))
    (folder/'plan.json').write_text(content);(tmp_path/row['previous_queue']).write_text(content)
    assert operator.unstarted(row,birth)==birth
    (folder/'unexpected.json').write_text('{}')
    with pytest.raises(ValueError):operator.unstarted(row,birth)
