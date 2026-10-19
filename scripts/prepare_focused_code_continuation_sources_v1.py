"""Exercise the complete new code epoch and every broad replay family on CPU.

This explicitly omits the still-unadmitted execution candidate and sets a
preparation-only purpose that the optimizer rejects. It is not a GPU gate,
production plan, actual training epoch, or model-quality assessment.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import time
import traceback

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.focused_code_continuation_data_v1 import CODE,REPLAY,FocusedCodeContinuationData,FocusedContinuationSchedule
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/focused-code-continuation-sources-v1-preparation'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full-source preparation required')
    torch.set_num_threads(1)
    tests=ROOT/'reports/focused-code-continuation-data-v1-final-tests.log'
    if '4 passed' not in tests.read_text():raise ValueError('Actual native padding, tail-source, admission and length-bucket tests required')
    shape_source=ROOT/'checkpoints/full-actor-fold-dense-update-CPU-v1/initial-folded-student.pt'
    raw=torch.load(shape_source,map_location='cpu',weights_only=True,mmap=True)
    config=ModelConfig(**raw['complete_dense']['config']);del raw
    # No network is instantiated and none of these mapped weight values is
    # used. Only own architecture metadata supplies token/image shape limits.
    data=FocusedCodeContinuationData(ROOT,config,purpose='source_preparation')
    schedule=FocusedContinuationSchedule(data)
    files=dict(data.source_files_sha256)
    files.update({str(p):file_hash(p) for p in [shape_source,tests,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_focused_code_continuation_data_v1.py'])]})
    guard=FileBindings(files);OUT.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        purpose='source_preparation',original_complete_code_presentations=58488,
        all_ten_replay_families=REPLAY,execution_supervision_not_admitted_or_included=True,
        own_checkpoint_used_only_for_architecture_metadata_no_neural_model_constructed=True,
        source_is_not_declared_production_initializer=True,models_constructed=0,model_updates=0,goal_achieved=False))
    observations=[];examples=contexts=targets=0;physical_padded=0
    try:
        for step,(name,positions) in enumerate(schedule.entries):
            if name!=CODE:continue
            batch=data.batch(name,positions,'cpu')
            actual_contexts=int(batch['attention_mask'].sum());actual_targets=int((batch['labels'][:,1:]!=-100).sum())
            contexts+=actual_contexts;targets+=actual_targets;examples+=len(positions);physical_padded+=batch['input_ids'].numel()
            if examples%4096<len(positions):
                atomic_json(OUT/'progress.json',dict(owner=owner,actual_complete_code_examples_rebuilt=examples,
                    actual_native_context_positions=contexts,actual_response_EOS_targets=targets,
                    model_updates=0,models_constructed=0))
        if (examples,contexts,targets)!=(58488,88610948,54063335):raise ValueError('Every admitted original code input and native target must reconstruct in real batches')
        order=list(range(data.populations[CODE]));random.Random(972317).shuffle(order)
        naive=sum(max(data.lengths[CODE][p] for p in order[i:i+16])*len(order[i:i+16]) for i in range(0,len(order),16))
        for name in REPLAY:
            position=max(range(data.populations[name]),key=lambda p:data.lengths[name][p])
            batch=data.batch(name,[position],'cpu')
            observation=dict(family=name,original_position=position,kind=data.kinds[name],
                complete_context_positions=int(batch['attention_mask'].sum()),response_EOS_targets=int((batch['labels'][:,1:]!=-100).sum()),
                actual_tensor_shapes={k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)})
            if data.kinds[name] in ('image','word') and tuple(batch['pixel_values'].shape)!=(1,3,448,448):
                raise ValueError('Every real image/word family must reconstruct the entire native448input')
            if data.kinds[name]=='word' and tuple(batch['word_pixels'].shape)!=(1,3,64,448):
                raise ValueError('Whole native byte-CTC word view must remain separate from native448caption input')
            observations.append(observation)
        if physical_padded>=naive:raise ValueError('On the actual full new code population, length buckets must reduce padding without dropping examples')
        guard.check(full=True)
        atomic_json(OUT/'actual-replay-family-extremes.json',observations)
        atomic_json(OUT/'preparation-only-schedule.json',dict(seed=schedule.seed,width=schedule.width,
            identity_sha256=schedule.identity_sha256,entries=schedule.entries,
            selected_native_budgets=schedule.selected_budget(data),purpose='source_preparation',models_constructed=0))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            every58488original_code_presentation_rebuilt_in_complete_real_native_batches=True,
            actual_code_examples=examples,actual_complete_code_context_positions=contexts,actual_code_response_EOS_targets=targets,
            full_batch_padded_code_token_slots=physical_padded,same_population_naive_random_padded_token_slots=naive,
            actual_global_batch_padding_slot_reduction_fraction=(naive-physical_padded)/naive,
            padding_slot_reduction_not_measured_GPU_speedup=True,
            all_ten_actual_original_replay_family_extremes_checked=True,
            actual_preparation_only_updates=len(schedule),selected_preparation_budgets=schedule.selected_budget(data),
            execution_source_missing_so_optimization_forbidden=True,
            no_current_training_or_held_evaluation_source_modified=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',OUT/'actual-replay-family-extremes.json',OUT/'preparation-only-schedule.json']},
            models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_code_examples=examples,
            actual_batch_padding_slot_reduction=(naive-physical_padded)/naive,
            original_replay_families=len(observations),training_started=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),models_constructed=0,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
