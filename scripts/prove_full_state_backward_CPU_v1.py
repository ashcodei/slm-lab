"""Two real full-size state updates on an explicitly partial own CPU copy.

No GPU ownership, original-model mutation or production-source substitution.
Full Adam recovery is checked in memory; only one final native model is saved
to preserve the persistent disk budget of the existing GPU pipeline.
"""
import copy
from datetime import datetime,timezone
import gc
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np
import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings,atomic_checkpoint,capture_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.fast_joint_inference_v2 import load as load_main
from slm_lab.focused_code_continuation_data_v3 import STATE,FocusedCodeContinuationData
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.own_dense_standalone_v1 import export,load
from slm_lab.state_backward_rehearsal_v1 import rehearse,equal
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-state-backward-CPU-v1'
OUTPUT=ROOT/'checkpoints/full-state-backward-CPU-v1'
SOURCE=ROOT/'checkpoints/phase-joint-v3/step-28672/model.pt'
SOURCE_PLAN=ROOT/'reports/phase-joint-v3/plan.json'


class TwoSources:
    phase_for_step=['instruction_and_vision']*2
    phase_bounds=dict(foundation=[0,0],instruction_and_vision=[0,2])
    def __init__(self,positions):self.positions=positions
    def __len__(self):return 2
    def at(self,i):return STATE,[self.positions[i]]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or OUTPUT.exists():raise ValueError('Fresh CPU-hidden actual state rehearsal required')
    if shutil.disk_usage(ROOT).free<152*1024**3:raise ValueError('Preserve planned main/code-first allocations and original26GiB reserve')
    tests=ROOT/'reports/state-backward-rehearsal-v1-final-tests.log'
    if '1 passed' not in tests.read_text():raise ValueError('Actual state-gradient/all-leaf/RNG/Adam recovery and rejected-corruption controls required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    source_metadata=json.loads(SOURCE_PLAN.read_text());raw=torch.load(SOURCE,map_location='cpu',weights_only=True,mmap=True)
    if raw.get('arm')!='step-28672' or raw.get('completed_updates')!=28672 or raw.get('whole_source_epochs_completed')!=0:
        raise ValueError('Explicit fixed partial main28672 required; never label it the complete production main')
    del raw
    source_sha=file_hash(SOURCE);source_plan_sha=file_hash(SOURCE_PLAN)
    files={str(p):file_hash(p) for p in [SOURCE,SOURCE_PLAN,tests,*python_sources(ROOT,[Path(__file__),ROOT/'tests/test_state_backward_rehearsal_v1.py'])]}
    guard=FileBindings(files);OUT.mkdir();OUTPUT.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    def progress(**row):
        atomic_json(OUT/'progress.json',dict(owner=owner,elapsed_seconds=time.monotonic()-started,
            actual_CUDA_initialized=torch.cuda.is_initialized(),**row,production_updates=0,goal_achieved=False))
    try:
        progress(phase='loading exact own partial main on CPU')
        model,metadata=load_main(SOURCE,SOURCE_PLAN,'step-28672',expected_checkpoint_sha256=source_sha,expected_plan_sha256=source_plan_sha)
        if model.counts()['total_unique']!=984399690 or metadata['actual_completed_updates']!=28672 or torch.cuda.is_initialized():
            raise ValueError('Exact own984399690-value partial source with no CUDA initialization required')
        data=FocusedCodeContinuationData(ROOT,model.config,purpose='optimization');files.update(data.source_files_sha256)
        positions=[min(range(data.populations[STATE]),key=lambda p:(data.lengths[STATE][p],p)),
            max(range(data.populations[STATE]),key=lambda p:(data.lengths[STATE][p],-p))]
        if data.populations[STATE]!=1310 or [data.lengths[STATE][p] for p in positions]!=[354,4095]:
            raise ValueError('Exact admitted whole-state shortest/longest contexts required without clipping')
        schedule=TwoSources(positions);rates=dict(language=1e-5,vision=1e-4,word=3e-4)
        control=json.loads((ROOT/'reports/phase-joint-full-CPU-v2/plan.json').read_text())
        options=dict(control['training_options'],learning_rates=rates,phase_learning_rates=dict(instruction_and_vision=rates),maximum_microbatch=1)
        files[str(ROOT/'reports/phase-joint-full-CPU-v2/plan.json')]=file_hash(ROOT/'reports/phase-joint-full-CPU-v2/plan.json')
        word_position=next(i for i,r in enumerate(data.replay.words.rows) if r['byte_family']=='real_word')
        word=data.batch('words',[word_position],'cpu');word_image=Path(data.replay.words.rows[word_position]['image'])
        files[str(word_image)]=file_hash(word_image);guard=FileBindings(files)
        plan=dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
            original_main_inputs=dict(checkpoint=str(SOURCE),checkpoint_sha256=source_sha,plan_sha256=source_plan_sha,actual_updates=28672),
            source_is_explicitly_partial_main_not_complete_production_initializer=True,
            complete_state_family_available=1310,actual_selected_state_positions=positions,
            actual_selected_complete_context_positions=[data.lengths[STATE][p] for p in positions],
            actual_selected_response_EOS_targets=[data.targets[STATE][p] for p in positions],
            original_word_position=word_position,actual_counts=model.counts(),initial_dense_settings=source_metadata['initial_dense_settings'],
            training_options=options,planned_steps=2,phase_bounds=schedule.phase_bounds,
            complete_source_budget=data.budget(),CPU_threads=2,logical_updates=2,physical_optimizer_calls=3,
            maximum_new_checkpoint_bytes=5*1024**3,full_source_epoch_completed=False,
            in_memory_private_recovery_only_no_durable_Adam_snapshot=True,
            source_and_all_queued_GPU_jobs_must_stay_unchanged=True,
            fresh_four_L40_capacity_and_generated_quality_still_required=True,candidate_promoted=False,goal_achieved=False)
        atomic_json(OUT/'plan.json',plan);digest=file_hash(OUT/'plan.json')
        reference=copy.deepcopy(model).eval().requires_grad_(False)
        binding=dict(path=str(SOURCE),checkpoint_sha256=source_sha,parameter_sha256=tensor_digest(reference.named_parameters()),foundation_completed_updates=0)
        random.seed(978217);np.random.seed(978217);torch.manual_seed(978217)
        result=rehearse(model,data,schedule,plan,digest,reference=reference,reference_binding=binding,progress=progress)
        if result['actual_active_language_tensors']!=317 or result['actual_full_language_Adam_states']!=317:raise ValueError('All317actual full language leaves and Adam states required')
        progress(phase='checking original real-word CTC path after state learning',logical_updates=2)
        model.eval()
        with torch.no_grad():
            word_before=reference.word_logits(word['word_pixels']).detach().clone()
            if not torch.equal(word_before,model.word_logits(word['word_pixels'])):raise ValueError('State text updates must preserve actual original word-CTC logits')
        del reference;gc.collect()
        progress(phase='exporting one complete native CPU rehearsal model',logical_updates=2)
        shutil.copyfile(data.code.tokenizer_path,OUTPUT/'tokenizer.json')
        sources=dict(experimental_main28672_sha256=source_sha,rehearsal_plan_sha256=digest,
            admitted_state_manifest_sha256=file_hash(data.states.folder/'manifest.json'))
        provenance=dict(kind='partial_own_main_full_state_backward_CPU_rehearsal',actual_source_main_updates=28672,
            actual_logical_state_updates=2,actual_source_positions=positions,full_state_source_epoch_completed=False,
            not_a_production_initializer_or_generated_quality_result=True)
        final_path=OUTPUT/'model.pt';atomic_checkpoint(final_path,export(model,sources=sources,provenance=provenance,completed_updates=2,
            tokenizer_sha256=data.code.provenance['tokenizer_sha256']),5*1024**3,26*1024**3)
        batch=data.batch(STATE,[positions[1]],'cpu');kwargs={k:batch[k] for k in ('input_ids','attention_mask','pixel_values')}
        with torch.no_grad():before=model(**kwargs)['logits'].detach().clone()
        rng=capture_rng('cpu');cold,cold_metadata=load(final_path,expected_checkpoint_sha256=file_hash(final_path),expected_sources=sources)
        if not equal(rng,capture_rng('cpu')):raise ValueError('Cold native reconstruction must preserve the caller private CPU generators')
        with torch.no_grad():
            if not torch.equal(before,cold(**kwargs)['logits']) or not torch.equal(word_before,cold.word_logits(word['word_pixels'])):
                raise ValueError('Every whole4095state logit and actual original word-CTC output must cold-load exactly')
        if tensor_digest(model.named_parameters())!=tensor_digest(cold.named_parameters()) or cold_metadata['actual_counts']!=model.counts():
            raise ValueError('Every actual own exported parameter and count must cold-load exactly')
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,**result,
            actual_source_main_updates=28672,actual_unique_student_values=984399690,
            actual_whole_state_lengths=[354,4095],actual_whole_state_targets=[data.targets[STATE][p] for p in positions],
            every_exported_parameter_and_whole4095_logits_and_real_word_CTC_cold_load_exact=True,
            actual_word_CTC_logits_unchanged_by_state_text_updates=True,
            source_file_and_all_original_training_queues_unchanged=True,actual_CUDA_initialized=torch.cuda.is_initialized(),
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',final_path,OUTPUT/'tokenizer.json']},
            full_original_source_epoch_and_GPU_BF16_capacity_and_held_capability_not_proved=True,
            production_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_full_state_updates=2,actual_unique_values=984399690,
            production_initializer_unchanged=True,quality_gain_not_assessed=True)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(owner=owner,error=repr(error),traceback=traceback.format_exc(),
            original_training_and_queues_left_unchanged=True,production_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
