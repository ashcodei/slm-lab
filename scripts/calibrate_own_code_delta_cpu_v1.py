"""Whole original train-context neural checks; not a code-correctness benchmark."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
    raise ValueError('This calibration must have no visible GPUs')

from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from slm_lab.apps_human_solution_view_v1 import AppsHumanSolutions
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_code_delta_inference_v1 import load
from slm_lab.training import load_checkpoint

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/own-code-delta-v1'
OUTPUT=ROOT/'checkpoints/own-code-delta-v1/model.pt'
BASE=ROOT/'checkpoints/extended-curriculum-v1/joint/model.pt'
TARGET=ROOT/'checkpoints/apps-full-language-v1/model.pt'
SOURCE=ROOT/'data/public-apps-human-solution-view-v1'


def main():
    if (REPORTS/'CPU-calibration-plan.json').exists() or (REPORTS/'CPU-calibration-complete.json').exists():
        raise ValueError('Preserve original calibration and partial observations; no implicit restart')
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    complete=json.loads((REPORTS/'complete.json').read_text());plan=json.loads((REPORTS/'plan.json').read_text())
    if '1 passed' not in (ROOT/'reports/own-code-delta-inference-v1-tests.log').read_text():
        raise ValueError('Actual reference-bound own inference roundtrip check required')
    source=AppsHumanSolutions(SOURCE)
    index=source.pack.index[source.pack.positions[source.positions]]
    eligible=[i for i,length in enumerate(index['length']) if 3<=int(length)<=256]
    selected=sorted(eligible,key=lambda i:hashlib.sha256(('own-code-delta-calibration-v1:'+source.records[i]['id']).encode()).hexdigest())[:4]
    if len(selected)!=4:
        raise ValueError('Four complete original train contexts within the fixed256token calibration budget required')
    paths=[Path(__file__).resolve(),ROOT/'src/slm_lab/own_code_delta_inference_v1.py',
        ROOT/'src/slm_lab/own_code_delta_v1.py',ROOT/'tests/test_own_code_delta_inference_v1.py',
        ROOT/'reports/own-code-delta-inference-v1-tests.log',REPORTS/'plan.json',REPORTS/'complete.json',
        OUTPUT,BASE,TARGET,SOURCE/'admission.json']
    guard=FileBindings({str(p):file_hash(p) for p in paths})
    atomic_json(REPORTS/'CPU-calibration-plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256={str(p):file_hash(p) for p in paths},source_positions=selected,
        original_source_ids=[source.records[i]['id'] for i in selected],
        source_selection='First4SHA-ordered complete admitted train contexts with full length<=256, fixed before neural outputs.',
        device='CPU',precision='FP32',threads=2,held_data_used=False,neural_updates=0,
        functional_or_algorithmic_correctness_not_measured=True,goal_achieved=False))
    routed,metadata=load(OUTPUT,BASE,BASE.parent/'tokenizer.json',REPORTS/'plan.json',
        expected_checkpoint_sha256=complete['actual_factors_checkpoint_sha256'],expected_base_sha256=plan['general_base_sha256'],
        expected_target_sha256=plan['own_code_target_sha256'],expected_plan_sha256=complete['source_plan_sha256'])
    target,target_metadata=load_checkpoint(TARGET,'cpu',training_state=False);target.eval()
    records=[];started=time.monotonic()
    with torch.no_grad(),(REPORTS/'CPU-calibration-observations.jsonl').open('x') as sink:
        for position in selected:
            example=source.example(position);ids,labels=example['positive']
            if len(ids)>256 or ids[-1]!=2 or labels[-1]!=2:
                raise ValueError('Every actual original prompt/program/EOS must remain whole')
            inputs=torch.tensor([ids]);gold=torch.tensor([labels]);mask=gold[:,1:]!=-100
            general=routed(inputs,labels=gold,route='general')
            reference=target(inputs,labels=gold)
            compressed=routed(inputs,labels=gold,route='code')
            again=routed(inputs,labels=gold,route='general')
            if not torch.equal(general['logits'],again['logits']):
                raise ValueError('General route logits must remain exactly unchanged around code inference')
            reference_logp=reference['logits'][:,:-1].float().log_softmax(-1)
            compressed_logp=compressed['logits'][:,:-1].float().log_softmax(-1)
            kl=(reference_logp.exp()*(reference_logp-compressed_logp)).sum(-1)
            a=reference_logp.argmax(-1);b=compressed_logp.argmax(-1)
            record=dict(id=example['id'],source_position=position,whole_context_tokens=len(ids),response_EOS_targets=int(mask.sum()),
                original_input_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
                frozen_general_response_NLL=float(general['loss']),own_full_code_response_NLL=float(reference['loss']),
                compressed_code_response_NLL=float(compressed['loss']),
                code_source_to_compressed_target_mean_KL=float(kl[mask].mean()),
                code_source_to_compressed_target_argmax_agreements=int(((a==b)&mask).sum()),
                frozen_general_logits_unchanged=True)
            if not all(torch.isfinite(out['loss']) for out in [general,reference,compressed]):
                raise ValueError('All actual own-model source losses must be finite')
            records.append(record);sink.write(json.dumps(record,separators=(',',':'))+'\n');sink.flush()
            print('whole_train_context_calibrated',example['id'],record['compressed_code_response_NLL'],flush=True)
    guard.check(full=True);source.guard.check(full=True)
    if torch.cuda.is_initialized():
        raise ValueError('CPU source calibration must not initialize CUDA')
    total=sum(r['response_EOS_targets'] for r in records)
    scores={k:sum(r[k]*r['response_EOS_targets'] for r in records)/total for k in [
        'frozen_general_response_NLL','own_full_code_response_NLL','compressed_code_response_NLL','code_source_to_compressed_target_mean_KL']}
    atomic_json(REPORTS/'CPU-calibration-complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        original_whole_training_contexts=len(records),actual_response_EOS_targets=total,weighted_scores=scores,
        code_source_argmax_agreements=sum(r['code_source_to_compressed_target_argmax_agreements'] for r in records),
        exact_frozen_general_logits_preserved=True,actual_model_counts=metadata['model_counts'],
        actual_factors_checkpoint_sha256=file_hash(OUTPUT),observations_sha256=file_hash(REPORTS/'CPU-calibration-observations.jsonl'),
        elapsed_seconds=time.monotonic()-started,CPU_FP32_not_original_CUDA_BF16_protocol=True,
        source_calibration_not_held_code_correctness_or_retention_benchmark=True,
        held_data_used=False,neural_training_updates=0,CUDA_initialized=False,candidate_promoted=False,goal_achieved=False))


if __name__=='__main__':main()
