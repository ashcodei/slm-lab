"""Independent CPU readback of the retained complete execution-training fixture."""
import json
import math
from pathlib import Path

import torch

from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.recurrent_ablation import state_digest


def main():
    ready=Path('reports/execution-training-v5-readiness.json');r=json.loads(ready.read_text())
    for p,d in r['files_sha256'].items():
        if file_hash(p)!=d:raise ValueError('Readiness bound bytes differ: '+p)
    folder=Path(r['CPU_integration']['retained_artifacts'])
    state=torch.load(folder/'lifecycle/resume.pt',weights_only=True,mmap=True)
    exported=torch.load(folder/'lifecycle/inference.pt',weights_only=True,mmap=True)
    assert state['steps']==state['planned_steps']==60 and len(state['metrics'])==60
    assert len(state['worker_rng_states'])==4
    assert fingerprint(state['residuals'])==fingerprint(exported['residuals'])
    assert state['residual_state_sha256']==state_digest(state['residuals'])
    assert all(bool(torch.isfinite(t).all()) for t in state['residuals'].values())
    for moment in state['optimizer']['state'].values():
        assert int(moment['step'])==60
        for key in ['exp_avg','exp_avg_sq']:assert bool(torch.isfinite(moment[key]).all())
    for i,metric in enumerate(state['metrics']):
        assert metric['step']==i+1
        dpo=metric['execution_DPO'];rows=dpo['records']
        assert dpo['global_pairs']==8 and len(rows)==8
        assert {(x['rank'],x['local_exposure']) for x in rows}=={(rank,j) for rank in range(4) for j in range(2)}
        expected=dpo['alpha']*sum(x['unscaled_DPO'] for x in rows)/8
        assert math.isclose(dpo['weighted_DPO'],expected,rel_tol=1e-12,abs_tol=1e-12)
        for row in rows:
            x=dpo['beta']*row['reference_anchored_margin']
            sigmoid=math.exp(-x)/(1+math.exp(-x)) if x>=0 else 1/(1+math.exp(x))
            assert math.isclose(row['gradient_coefficient'],dpo['alpha']*dpo['beta']*sigmoid*4/8,rel_tol=1e-12,abs_tol=1e-12)
        ce=metric['objectives']
        assert math.isclose(metric['CE_summed_objective'],sum(v['coefficient']*v['token_mean_nll'] for v in ce.values()),rel_tol=1e-12)
        assert metric['total_target_tokens']==sum(v['global_target_tokens'] for v in ce.values())+sum(x['chosen_targets']+x['rejected_targets'] for x in rows)
    progress=state['sampler_state']
    assert progress['execution_source_cursors']['positive']['examples']==240
    assert progress['execution_source_cursors']['pair']['epoch_and_cursor']==[68,4]
    base=torch.load(folder/'base.pt',weights_only=True)
    assert state_digest(base['model'])==state['frozen_backbone_state_sha256']
    report=dict(readiness_sha256=file_hash(ready),review_source_sha256=file_hash(__file__),all_bound_files_match=True,
        complete_CPU_state_has60_updates_all4_RNG_and_finite_Adam_step60=True,
        inference_residuals_equal_complete_state=True,original_frozen_CPU_backbone_matches=True,
        all60_global_pair_records_CE_normalizers_DPO_aggregates_and_coefficients_recomputed=True,
        pair_and_positive_full_cursors_retained=True,focused_checks=10,
        actual_full857M_execution_capacity_run=False,new_full857M_auxiliary_updates=0,
        candidate_promoted=False,high_end_capability_established=False,goal_achieved=False,
        limitations=['Own8688-parameter CPU integration and actual admitted-data interfaces; not full857M GPU training or improved capability.',
            'Kernel recovery survives external-cache deletion. Production preserves its immutable bound capacity/reference evidence.'])
    atomic_json(Path('reports/execution-training-v5-final-review.json'),report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
