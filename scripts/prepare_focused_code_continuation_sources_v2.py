"""Check the new retention curriculum without repeating unchanged code work.

The complete code order and data bindings must equal the independently stored
v1 preparation. Actual new raw-code/education batches and all original replay
extremes are reconstructed. This is preparation, not training or quality gain.
"""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import traceback

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.focused_code_continuation_data_v2 import CODE,REPLAY,RAW_REPLAY,FocusedCodeContinuationData,FocusedContinuationSchedule
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/focused-code-continuation-sources-v2-preparation'
PREVIOUS=ROOT/'reports/focused-code-continuation-sources-v1-preparation'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only preparation required')
    torch.set_num_threads(1)
    tests=ROOT/'reports/focused-code-continuation-data-v2-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('All12-family schedule, source-tail and auxiliary-admission tests required')
    prior=json.loads((PREVIOUS/'complete.json').read_text())
    prior_plan=json.loads((PREVIOUS/'plan.json').read_text())
    if not prior['every58488original_code_presentation_rebuilt_in_complete_real_native_batches']:
        raise ValueError('The actual full unchanged code population must already be rebuilt')
    FileBindings(prior_plan['source_files_sha256']).check(full=True)
    old_schedule=json.loads((PREVIOUS/'preparation-only-schedule.json').read_text())
    shape_source=ROOT/'checkpoints/full-actor-fold-dense-update-CPU-v1/initial-folded-student.pt'
    raw=torch.load(shape_source,map_location='cpu',weights_only=True,mmap=True)
    config=ModelConfig(**raw['complete_dense']['config']);del raw
    data=FocusedCodeContinuationData(ROOT,config,purpose='source_preparation')
    schedule=FocusedContinuationSchedule(data)
    old_code=[(n,p) for n,p in old_schedule['entries'] if n==CODE]
    new_code=[(n,p) for n,p in schedule.entries if n==CODE]
    if old_code!=new_code:raise ValueError('Reused complete CODE batch proof must match every actual original position and batch')
    files=dict(data.source_files_sha256)
    files.update({str(p):file_hash(p) for p in [tests,shape_source,PREVIOUS/'complete.json',PREVIOUS/'plan.json',
        PREVIOUS/'preparation-only-schedule.json',*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_focused_code_continuation_data_v2.py'])]})
    guard=FileBindings(files);OUT.mkdir()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        purpose='source_preparation',all_twelve_replay_families=REPLAY,added_raw_retention_families=RAW_REPLAY,
        motivation='Fixed packed-code NLL rose from step12288 to14336 after instruction phase began; prevention hypothesis only',
        unchanged_full_CODE_real_batch_proof_reused_only_after_all_bindings_and_every_batch_position_equal=True,
        every_replay_family_extreme_actually_rebuilt=True,execution_supervision_not_admitted_or_included=True,
        models_constructed=0,model_updates=0,goal_achieved=False))
    try:
        observations=[]
        for name in REPLAY:
            position=max(range(data.populations[name]),key=lambda p:data.lengths[name][p])
            batch=data.batch(name,[position],'cpu')
            if data.kinds[name] in ('image','word') and tuple(batch['pixel_values'].shape)!=(1,3,448,448):
                raise ValueError('Every entire actual448image input required')
            if data.kinds[name]=='word' and tuple(batch['word_pixels'].shape)!=(1,3,64,448):
                raise ValueError('Entire separate native byte-CTC word view required')
            observations.append(dict(family=name,original_position=position,kind=data.kinds[name],
                actual_complete_context_positions=int(batch['attention_mask'].sum()),
                actual_response_EOS_targets=int((batch['labels'][:,1:]!=-100).sum()),
                actual_tensor_shapes={k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)}))
        selected=schedule.selected_budget(data)
        if not set(REPLAY)<=selected.keys() or any(selected[n]['response_EOS_targets']<=0 for n in RAW_REPLAY):
            raise ValueError('Every original replay family plus both raw retention objectives must occupy the real schedule')
        guard.check(full=True)
        atomic_json(OUT/'actual-all12-replay-family-extremes.json',observations)
        atomic_json(OUT/'preparation-only-schedule.json',dict(seed=schedule.seed,width=schedule.width,
            identity_sha256=schedule.identity_sha256,entries=schedule.entries,selected_native_budgets=selected,
            purpose='source_preparation',models_constructed=0))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            all12actual_replay_family_extremes_rebuilt=True,
            all58488_CODE_example_batches_identical_to_preceding_full_actual_rebuild=True,
            reused_actual_code_examples=prior['actual_code_examples'],
            reused_actual_code_context_positions=prior['actual_complete_code_context_positions'],
            reused_actual_code_response_EOS_targets=prior['actual_code_response_EOS_targets'],
            actual_preparation_only_updates=len(schedule),actual_selected_source_budgets=selected,
            added_raw_retention_families=RAW_REPLAY,
            actual_GPU_speedup_or_retention_improvement_not_established=True,
            execution_source_missing_so_optimization_forbidden=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',OUT/'actual-all12-replay-family-extremes.json',OUT/'preparation-only-schedule.json']},
            no_current_training_or_held_evaluation_source_modified=True,models_constructed=0,model_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_replay_families=len(observations),
            added_raw_replay={n:selected[n] for n in RAW_REPLAY},training_started=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),models_constructed=0,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
