"""Observe repeated actual source/zero-actor forwards without relaxing a gate."""
from datetime import datetime,timezone
import json,os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/zero-actor-recovery-diagnostic-CPU-v1'

def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Fresh CPU diagnostic required')
    import torch
    from slm_lab.own_dense_standalone_v1 import load
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source=ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/plan.json';plan=json.loads(source.read_text())
    checkpoint=ROOT/'checkpoints/code-first-continuation-v3/standalone/model.pt'
    base,_=load(checkpoint,expected_checkpoint_sha256=file_hash(checkpoint),expected_sources=plan['expected_standalone_sources'])
    model=OwnDenseActor(base).eval();fixed=tensor_digest(model.parameters() if False else model.named_parameters())
    rows=json.loads((ROOT/'reports/completed-code-public-function-rollouts-CPU-v1/all14-original-source-train-records.json').read_text())
    pair=json.loads((ROOT/'data/own-function-trajectory-preferences-v1/all-original-function-trajectory-pairs.jsonl').read_text().splitlines()[0])
    ids=torch.tensor([pair['original_native_prompt_ids']]);OUT.mkdir();outputs=[];results=[]
    with torch.no_grad():
        for label,target in [('base1',base),('base2',base),('actor1',model),('actor2',model),('base3',base),('actor3',model)]:
            logits=target(input_ids=ids)['logits'].clone();outputs.append((label,logits))
            result=dict(label=label,max_error_vs_first=float((logits-outputs[0][1]).abs().max()),exact_vs_first=torch.equal(logits,outputs[0][1]),last_argmax=int(logits[0,-1].argmax()))
            results.append(result);atomic_json(OUT/'progress.json',dict(actual_forwards=results,neural_updates=0,goal_achieved=False))
    if tensor_digest(model.named_parameters())!=fixed:raise ValueError('No diagnostic may alter any model value')
    atomic_json(OUT/'complete.json',dict(complete=True,at=datetime.now(timezone.utc).isoformat(),actual_forwards=results,
        source_checkpoint_sha256=file_hash(checkpoint),all_values_unchanged=True,torch_version=str(torch.__version__),
        CPU_capability=torch.backends.cpu.get_cpu_capability(),float32_matmul_precision=torch.get_float32_matmul_precision(),
        neural_updates=0,goal_achieved=False))
    print(json.dumps(results))

if __name__=='__main__':main()
