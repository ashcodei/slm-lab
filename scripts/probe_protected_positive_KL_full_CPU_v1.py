"""Exercise nonzero own-reference KL on the actual full-size trained control.

Gradient-only check: no optimizer step, no benchmark training, no quality claim.
"""
from datetime import datetime,timezone
import os
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/protected-positive-KL-full-CPU-v1.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Hide every GPU')
    if OUTPUT.exists():raise ValueError('Preserve prior full-size positive-KL evidence')
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_inference_v2 import fresh,load
    from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups
    from slm_lab.own_code_training_stage_v1 import PreparedCodeStream
    from slm_lab.protected_joint_stage_v1 import backward_protected
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);started=time.monotonic()
    folder=ROOT/'reports/protected-joint-full-CPU-v2';plan_path=folder/'plan.json';plan=json.loads(plan_path.read_text())
    model,_=load(folder/'model.pt',plan_path,'trained',expected_checkpoint_sha256=file_hash(folder/'model.pt'),expected_plan_sha256=file_hash(plan_path),device='cpu')
    reference,_=fresh(plan,'cpu');reference.eval().requires_grad_(False);optimizer=make_optimizer(model,plan['training_options'])
    stream=PreparedCodeStream('generation',ROOT/'data/APPS-verified-generation-v1',folder/'tokenizer.json',model.config)
    position=plan['actual_short_source_controls'][0][1];batch=stream.encoder.batch([dict(packed_code_index=position)],'response','cpu')
    before=tensor_digest(model.named_parameters());frozen=tensor_digest(reference.named_parameters())
    targets=int((batch['labels'][:,1:]!=-100).sum());contexts=int(batch['attention_mask'][:,1:].sum())
    model.train();model.network.gradient_checkpointing=True
    metric=backward_protected(model,batch,kind='text',global_targets=targets,global_context_targets=contexts,reference=reference)
    groups=parameter_groups(model)
    if (metric['full_context_reference_KL_sum']<=0 or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for _,p in groups['language'])
            or any(p.grad is not None for name in ['vision','word'] for _,p in groups[name])
            or any(p.grad is not None for p in reference.parameters()) or optimizer.state
            or before!=tensor_digest(model.named_parameters()) or frozen!=tensor_digest(reference.named_parameters())):
        raise ValueError('Actual positive full-size reference KL needs finite language-only gradients and unchanged policy/reference/no Adam state')
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),actual_policy_parameters=model.counts()['total_unique'],
        training_only_reference_parameters=reference.counts()['total_unique'],complete_original_train_position=position,
        complete_original_contexts=contexts+1,original_response_EOS_targets=targets,actual_gradient_only_metric=metric,
        actual_nonzero_KL_and_all317_language_gradients_finite=True,policy_reference_and_inactive_groups_unchanged=True,
        actual_optimizer_steps=0,production_updates=0,source_plan_sha256=file_hash(plan_path),source_sha256=file_hash(__file__),
        full_size_GPU_capacity_or_held_quality_not_claimed=True,seconds=time.monotonic()-started,goal_achieved=False))
    print(json.dumps(dict(KL_sum=metric['full_context_reference_KL_sum'],seconds=time.monotonic()-started)),flush=True)


if __name__=='__main__':main()
