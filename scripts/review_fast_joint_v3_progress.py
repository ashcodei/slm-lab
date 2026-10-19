"""Source/schedule/runtime progress reader; never a capability-gain claim."""
from collections import Counter
from datetime import datetime,timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/fast-joint-v3'
OUTPUT=ROOT/'reports/fast-joint-v3-progress-readback-v1.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Progress reader must hide all GPUs')
    if OUTPUT.exists():raise ValueError('Preserve previous progress evidence')
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.fast_joint_data_v1 import FastJointSources
    from slm_lab.model import ModelConfig
    import assess_joint_java_comparison_v1 as resources
    torch.set_num_threads(2)
    plan=json.loads((REPORTS/'plan.json').read_text());progress=json.loads((REPORTS/'progress.json').read_text())
    raw=(REPORTS/'training-metrics.jsonl').read_bytes();sealed=raw[:raw.rfind(b'\n')+1]
    records=[json.loads(line) for line in sealed.splitlines()]
    if len(records)<25 or len(records)<progress['actual_step']:raise ValueError('At least 25 actual production records required')
    initializer=torch.load(plan['initial_dense_settings']['initializer'],map_location='cpu',weights_only=True,mmap=True)
    data=FastJointSources(ROOT,ModelConfig(**initializer['config']));del initializer
    if data.budget()!=plan['complete_source_budget']:raise ValueError('Complete original source epoch differs')
    # Reconstruct the curriculum separately, without importing its scheduler.
    batches=[]
    for ordinal,name in enumerate(sorted(data.lengths)):
        positions=list(range(len(data.lengths[name])));rng=random.Random(plan['seed']+ordinal*1009);rng.shuffle(positions)
        width=plan['global_family_batch_widths'][name];groups=[]
        for offset in range(0,len(positions),width*32):
            selected=positions[offset:offset+width*32];selected.sort(key=lambda p:(data.lengths[name][p],p))
            for start in range(0,len(selected),width):groups.append(selected[start:start+width])
        rng.shuffle(groups)
        batches.extend((Fraction(2*i+1,2*len(groups)),ordinal,name,g) for i,g in enumerate(groups))
    batches.sort();actual=Counter();contexts=Counter();targets=Counter();seen={name:set() for name in data.lengths}
    if len(batches)!=plan['planned_steps']:raise ValueError('Full unrepeated curriculum length differs')
    for i,r in enumerate(records):
        _,_,family,positions=batches[i]
        if r['step']!=i+1 or r['family']!=family or r['global_original_positions']!=positions:
            raise ValueError('Actual update lost complete original source order')
        if seen[family].intersection(positions):raise ValueError('A source-family row was repeated in this epoch')
        seen[family].update(positions)
        context=sum(data.lengths[family][p] for p in positions);target=sum(data.targets[family][p] for p in positions)
        if r['global_examples']!=len(positions) or r['complete_context_tokens']!=context or r['response_EOS_targets']!=target:
            raise ValueError('Actual complete source/target totals disagree')
        if not math.isclose(r['CE_sum']/target,r['token_mean_CE'],rel_tol=1e-12):raise ValueError('Actual loss arithmetic disagrees')
        if not math.isfinite(r['gradient_norm_before_clip']) or r['active_parameter_tensors']<=0:raise ValueError('Actual finite learning gradients required')
        actual[family]+=len(positions);contexts[family]+=context;targets[family]+=target
    children=json.loads((REPORTS/'children.json').read_text());live=[dict(a,current=resources.same(a)) for a in children if resources.same(a)]
    ranks=[a for a in live if a['subject'].startswith('train-rank-')]
    if len(ranks)!=4 or not resources.same(plan['owner']):raise ValueError('Exact live four-GPU trainer required')
    previous=ROOT/'reports/fast-joint-v2';oldplan=json.loads((previous/'plan.json').read_text())
    if resources.same(oldplan['owner']) or any(resources.same(a) for a in json.loads((previous/'children.json').read_text())):
        raise ValueError('Prior stalled trainer still live')
    if not (previous/'restored.json').exists():raise ValueError('Prior original work restoration required')
    samples=[]
    for _ in range(5):
        value=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
        rows=[[float(x.strip()) for x in line.split(',')] for line in value.splitlines()]
        samples.append(rows);time.sleep(1)
    import hashlib
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),production_update_records_read=len(records),
        planned_updates=len(batches),sealed_training_journal_prefix_sha256=hashlib.sha256(sealed).hexdigest(),
        sealed_training_journal_prefix_bytes=len(sealed),original_source_order_reconstructed_separately=True,
        all_observed_complete_source_targets_and_loss_arithmetic_agree=True,no_observed_source_family_repeats=True,
        examples_by_family=dict(actual),complete_context_tokens_by_family=dict(contexts),target_tokens_by_family=dict(targets),
        all_full_epoch_source_family_records=sum(v['examples'] for v in data.budget().values()),
        all_full_epoch_target_tokens=sum(v['response_EOS_targets'] for v in data.budget().values()),
        exact_live_owner=resources.same(plan['owner']),exact_live_registered_ranks=ranks,previous_owner_and_workers_terminal_restored=True,
        GPU_samples_columns=['physical_GPU','GPU_busy_percent','memory_used_MiB','power_watts'],GPU_samples=samples,
        training_GPU_busy_percent_mean=[sum(sample[i][1] for sample in samples)/len(samples) for i in range(4)],
        GPU_busy_percent_is_not_useful_FLOPs_utilization=True,held_capability_gain_from_this_run_not_yet_measured=True,
        observed_update_records_are_not_an_independent_weight_or_Adam_replay=True,
        source_plan_sha256=file_hash(REPORTS/'plan.json'),reader_source_sha256=file_hash(__file__),
        candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(observed_updates=len(records),examples=sum(actual.values()),complete_targets=sum(targets.values()),families=len(actual))),flush=True)


if __name__=='__main__':main()
