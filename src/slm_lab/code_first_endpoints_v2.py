"""Fixed code-first lineage and native loaders, declared before final scores."""
import json
from pathlib import Path

from .corpus import file_hash

ENDPOINTS=('phase_initial','phase_trained','code_first_trained','human_chosen_CE',
    'human_DPO','code_anchor_CE','code_anchor_execution','standalone_final')
COMPARISONS=(('phase_initial','phase_trained'),('phase_trained','code_first_trained'),
    ('code_first_trained','human_chosen_CE'),('code_first_trained','human_DPO'),
    ('human_chosen_CE','human_DPO'),('human_DPO','code_anchor_CE'),
    ('human_DPO','code_anchor_execution'),('code_anchor_CE','code_anchor_execution'),
    ('code_anchor_execution','standalone_final'))


def descriptors(root):
    root=Path(root).resolve();result={}
    for name in ENDPOINTS:
        if name.startswith('phase_'):
            arm=name.removeprefix('phase_');stage='phase-joint-v3';kind='foundation';filename=arm+'/model.pt'
        elif name=='code_first_trained':
            arm='trained';stage='code-first-continuation-v2';kind='foundation';filename='trained/model.pt'
        elif name.startswith('human_'):
            arm='human_DPO' if name=='human_DPO' else 'chosen_CE'
            stage='phase-human-preference-v6';kind='human';filename=arm+'-actor.pt'
        elif name.startswith('code_anchor_'):
            arm=name.removeprefix('code_');stage='broad-anchored-code-v5';kind='code';filename=arm+'-actor.pt'
        else:
            arm='final';stage='code-first-final-standalone-v2';kind='standalone';filename='model.pt'
        plan=root/'reports'/stage/'plan.json'
        if kind=='standalone':plan=root/'reports/code-first-final-standalone-CPU-v2/complete.json'
        result[name]=dict(name=name,kind=kind,arm=arm,checkpoint=str(root/'checkpoints'/stage/filename),plan=str(plan))
    return result


def expected_values(spec):
    return 984399690 if spec['kind'] in {'foundation','standalone'} else 999706442


def load_native(spec,device='cpu'):
    from .dense_foundation_inference_v1 import DenseNativeView
    if spec['kind']=='standalone':
        from .own_dense_standalone_v1 import load
        if file_hash(spec['plan'])!=spec['plan_sha256']:raise ValueError('Exact immutable standalone completion receipt required')
        receipt=json.loads(Path(spec['plan']).read_text())
        if (receipt.get('complete') is not True or receipt.get('completed_updates')!=3721
                or receipt.get('checkpoint_sha256')!=spec['checkpoint_sha256']
                or receipt.get('all_actual_exported_parameters_and_complete_native_probes_cold_load_bit_exact') is not True):
            raise ValueError('Complete fixed source actor and actual standalone cold-load proof required')
        model,metadata=load(spec['checkpoint'],expected_checkpoint_sha256=spec['checkpoint_sha256'],expected_sources=receipt['standalone_sources'])
        native=DenseNativeView(model.to(device)).eval()
        if file_hash(spec['plan'])!=spec['plan_sha256']:raise ValueError('Standalone receipt changed during native loading')
    else:
        if spec['kind']=='foundation':from .fast_joint_inference_v2 import load
        elif spec['kind']=='human':from .phase_human_policy_v2 import load
        elif spec['kind']=='code':from .broad_anchored_policy_v2 import load
        else:raise ValueError('Exact predeclared own native endpoint required')
        model,metadata=load(spec['checkpoint'],spec['plan'],spec['arm'],expected_checkpoint_sha256=spec['checkpoint_sha256'],
            expected_plan_sha256=spec['plan_sha256'],device=device)
        native=DenseNativeView(model).eval() if spec['kind']=='foundation' else model.policy.eval()
    if sum(p.numel() for p in native.parameters())!=expected_values(spec):
        raise ValueError('Every actual own student value must match its fixed under1B endpoint')
    return native,metadata
