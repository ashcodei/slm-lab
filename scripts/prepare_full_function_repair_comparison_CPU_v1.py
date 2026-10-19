"""Build every actual batch of a full target-matched original/repair comparison.

All826original train functions remain. All791eligible original positives get
one additional presentation per epoch in each arm. Only that presentation's
prompt changes in the repair arm. All15source-pack validation functions stay
out of optimization, without claiming all-history independence.
"""
from datetime import datetime,timezone
import json,os
from pathlib import Path

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.function_repair_curriculum_v1 import FunctionRepairCurriculum,ARMS,schedule
from slm_lab.local_source_closure_v2 import python_sources

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/full-function-repair-comparison-sources-CPU-v1'

def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full source preparation required')
    import torch
    torch.set_num_threads(1)
    tests=ROOT/'reports/function-repair-curriculum-v1-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Actual paired-target, held-role, complete-epoch and mask controls required')
    data=FunctionRepairCurriculum(ROOT,'source_preparation');settings=dict(epochs=3,width=16,seed=979113)
    entries=schedule(data.lengths,**settings);budgets={arm:data.budget(arm,entries) for arm in ARMS}
    if (len(entries)!=306 or any(b['complete_presentations']!=4851 for b in budgets.values())
            or budgets[ARMS[0]]['original_response_EOS_targets']!=budgets[ARMS[1]]['original_response_EOS_targets']):
        raise ValueError('All1617complete original/second-view presentations in every3epochs and identical positive targets required')
    files=dict(data.source_files_sha256)
    for path in [tests,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_function_repair_curriculum_v1.py'])]:files[str(path)]=file_hash(path)
    guard=FileBindings(files);OUT.mkdir()
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19]),
        source_files_sha256=files,actual_schedule_settings=settings,actual_schedule_identity_sha256=data.identity(entries),
        actual_complete_slot_batches=entries,planned_updates_per_arm=len(entries),actual_matched_source_budgets=budgets,
        all826original_functions_retained_each_epoch=True,all791eligible_additional_positive_presentations_target_matched=True,
        original15validation_functions_never_optimized=True,full_size_model_or_optimizer_not_constructed=True,neural_updates=0,goal_achieved=False))
    observed={arm:dict(batches=0,padded_context_tokens=0,original_target_tokens=0,minimum_native_length=4096,maximum_native_length=0) for arm in ARMS}
    for i,positions in enumerate(entries):
        targets=[]
        for arm in ARMS:
            batch=data.batch(arm,positions)
            actual=[batch['labels'][row][batch['labels'][row]!=-100].tolist() for row in range(len(positions))]
            targets.append(actual);row=observed[arm]
            if (batch['input_ids'].shape[0]!=len(positions) or batch['input_ids'].shape[1]>4096
                    or any(t[-1]!=2 for t in actual) or bool((batch['labels'][~batch['attention_mask']]!=-100).any())):
                raise ValueError('Every original whole response/EOS and actual padding mask must remain exact')
            row['batches']+=1;row['padded_context_tokens']+=batch['input_ids'].numel();row['original_target_tokens']+=sum(map(len,actual))
            row['minimum_native_length']=min(row['minimum_native_length'],int(batch['attention_mask'].sum(1).min()))
            row['maximum_native_length']=max(row['maximum_native_length'],int(batch['attention_mask'].sum(1).max()))
        if targets[0]!=targets[1]:raise ValueError('Every source target in every actual compared batch must be identical')
        if i%25==0:atomic_json(OUT/'progress.json',dict(actual_complete_matched_batches=i+1,planned=len(entries),neural_updates=0,goal_achieved=False))
    if any(row['original_target_tokens']!=budgets[arm]['original_response_EOS_targets'] for arm,row in observed.items()):raise ValueError('Every whole-epoch target denominator must reconcile')
    if torch.cuda.is_initialized():raise ValueError('Source preparation must never initialize a GPU')
    guard.check(full=True)
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,planned_updates_per_arm=len(entries),
        actual_schedule_settings=settings,actual_schedule_identity_sha256=data.identity(entries),actual_matched_source_budgets=budgets,
        all306actual_matched_batches_built_on_CPU=True,every_batch_has_identical_original_response_EOS_targets=True,
        actual_complete_batch_proofs=observed,three_short_final_source_batches_preserved=True,
        full_size_model_or_optimizer_not_constructed=True,actual_CPU_training_profile_still_requires_capacity_and_recovery_on_these_sources=True,
        all_original_gpu_training_and_full_evaluation_jobs_unchanged=True,source_files_sha256={**files,str(OUT/'plan.json'):file_hash(OUT/'plan.json')},
        new_neural_updates=0,goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT/'complete.json'),planned_updates_each_arm=len(entries),actual_budgets=budgets)),flush=True)

if __name__=='__main__':main()
