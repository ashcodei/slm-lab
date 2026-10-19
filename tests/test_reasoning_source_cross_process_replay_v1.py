"""Independent interpreters must replay the actual unchanged public sources."""
import json
import os
from pathlib import Path
import subprocess
import sys


def test_all_twelve_actual_generators_replay_with_explicit_python_hash_seed():
    root=Path(__file__).resolve().parents[1]
    code="""
import json
from slm_lab.reasoning_source_loader_v1 import load
from slm_lab.reasoning_source_oracles_v1 import CONFIGS,expected
rows=[]
with load('data/reasoning-gym-pinned-source-v1') as factory:
    for task,options in CONFIGS.items():
        dataset=factory.create_dataset(task,seed=981113,size=16,**options)
        for index in range(16):
            entry=dataset[index]
            assert expected(entry)==entry['answer']
            rows.append(entry)
print(json.dumps(rows,sort_keys=True))
"""
    env={**os.environ,'PYTHONHASHSEED':'0','CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':str(root/'src')}
    first=subprocess.check_output([sys.executable,'-c',code],env=env,cwd=root,timeout=60)
    second=subprocess.check_output([sys.executable,'-c',code],env=env,cwd=root,timeout=60)
    changed=subprocess.check_output([sys.executable,'-c',code],env={**env,'PYTHONHASHSEED':'1'},cwd=root,timeout=60)
    assert first==second
    first_rows=json.loads(first);changed_rows=json.loads(changed)
    assert len(first_rows)==len(changed_rows)==192
    differences=[(a['metadata']['source_dataset'],a['metadata']['source_index'])
                 for a,b in zip(first_rows,changed_rows) if a!=b]
    assert differences and {name for name,_ in differences}=={'ransom_note','isomorphic_strings'}
