"""Rebuild every admitted output/state batch and preserve all old exposures."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import traceback

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.focused_code_continuation_data_v3 import CODE,EXECUTION,STATE,REPLAY,FocusedCodeContinuationData,FocusedContinuationSchedule
from slm_lab.focused_code_continuation_data_v2 import FocusedContinuationSchedule as PreviousSchedule
from slm_lab.focused_continuation_run_contract_v2 import capacity_batches
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/focused-code-continuation-sources-v3-preparation'
PREVIOUS=ROOT/'reports/focused-code-continuation-sources-v2-preparation'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only state-inclusive source preparation required')
    torch.set_num_threads(1);tests=ROOT/'reports/state-code-first-data-and-run-contract-v1-tests.log'
    if '3 passed' not in tests.read_text():raise ValueError('Exact old exposures/state tails and real trained recovery controls required')
    prior=json.loads((PREVIOUS/'complete.json').read_text());prior_plan=json.loads((PREVIOUS/'plan.json').read_text())
    if not prior['all58488_CODE_example_batches_identical_to_preceding_full_actual_rebuild'] or not prior['all12actual_replay_family_extremes_rebuilt']:
        raise ValueError('Every unchanged complete CODE source and broad replay proof required')
    FileBindings(prior_plan['source_files_sha256']).check(full=True)
    shape_source=ROOT/'checkpoints/full-actor-fold-dense-update-CPU-v1/initial-folded-student.pt'
    raw=torch.load(shape_source,map_location='cpu',weights_only=True,mmap=True);config=ModelConfig(**raw['complete_dense']['config']);del raw
    data=FocusedCodeContinuationData(ROOT,config,purpose='source_preparation');schedule=FocusedContinuationSchedule(data)
    previous=PreviousSchedule(data);old_entries=json.loads((PREVIOUS/'preparation-only-schedule.json').read_text())['entries']
    if ([(n,p) for n,p in schedule.entries if n!=STATE]!=previous.entries
            or [[n,p] for n,p in schedule.entries if n not in {STATE,EXECUTION}]!=old_entries):
        raise ValueError('Every original CODE, output and all12replay exposure must remain exactly unchanged')
    files=dict(data.source_files_sha256)
    extra=[tests,shape_source,PREVIOUS/'complete.json',PREVIOUS/'plan.json',PREVIOUS/'preparation-only-schedule.json',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_focused_code_continuation_data_v3.py',ROOT/'tests/test_focused_continuation_run_contract_v2.py'])]
    files.update({str(p):file_hash(p) for p in extra});guard=FileBindings(files);OUT.mkdir()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        purpose='source_preparation',all12original_replay_families=REPLAY,
        original_CODE_output_and_replay_exposures_preserved=True,added_family=STATE,
        all_admitted_output_and_state_records_rebuilt_in_actual_full_batches=True,
        model_configuration_only_from_old_prototype_no_model_weights_used_as_initializer=True,
        models_constructed=0,model_updates=0,goal_achieved=False))
    try:
        checked={}
        for name in [EXECUTION,STATE]:
            examples=contexts=targets=0
            for family,positions in schedule.entries:
                if family!=name:continue
                batch=data.batch(name,positions,'cpu');examples+=len(positions)
                contexts+=int(batch['attention_mask'].sum());targets+=int((batch['labels'][:,1:]!=-100).sum())
            expected=data.budget()[name]
            if (examples,contexts,targets)!=(expected['examples'],expected['complete_context_tokens'],expected['response_EOS_targets']):
                raise ValueError('Every complete real native source/output/state target must reach actual batches once')
            checked[name]=dict(actual_examples=examples,actual_complete_context_positions=contexts,actual_response_EOS_targets=targets)
        extremes=[]
        for name in REPLAY:
            position=max(range(data.populations[name]),key=lambda p:data.lengths[name][p]);batch=data.batch(name,[position],'cpu')
            if data.kinds[name] in {'image','word'} and tuple(batch['pixel_values'].shape)!=(1,3,448,448):raise ValueError('Whole original native448image required')
            if data.kinds[name]=='word' and tuple(batch['word_pixels'].shape)!=(1,3,64,448):raise ValueError('Whole original real-word CTC input required')
            extremes.append(dict(family=name,position=position,actual_complete_context_positions=int(batch['attention_mask'].sum()),
                actual_response_EOS_targets=int((batch['labels'][:,1:]!=-100).sum())))
        capacity=[]
        for name,positions in capacity_batches(data):
            batch=data.batch(name,positions,'cpu');capacity.append(dict(family=name,positions=positions,
                actual_tensor_shapes={k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)}))
        if len(capacity)!=8 or capacity[3]['family']!=STATE:raise ValueError('Actual state-source extent must enter the fresh capacity gate')
        guard.check(full=True)
        atomic_json(OUT/'actual-auxiliary-source-batches.json',checked);atomic_json(OUT/'actual-all12-replay-extremes.json',extremes)
        atomic_json(OUT/'actual-eight-capacity-input-batches.json',capacity)
        atomic_json(OUT/'preparation-only-schedule.json',dict(seed=schedule.seed,width=schedule.width,identity_sha256=schedule.identity_sha256,
            entries=schedule.entries,complete_native_budgets=data.budget(),selected_native_budgets=schedule.selected_budget(data),purpose='source_preparation',models_constructed=0))
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            all58488_CODE_example_batches_identical_to_preceding_full_actual_rebuild=True,
            all12actual_replay_family_extremes_rebuilt=True,
            all_existing_CODE_output_and_replay_exposures_exactly_preserved=True,
            all_admitted_output_and_state_examples_rebuilt_in_actual_complete_batches=True,
            actual_auxiliary_batches=checked,actual_planned_updates=len(schedule),prior_output_inclusive_updates=len(previous),
            added_whole_state_updates=len(schedule)-len(previous),actual_complete_source_budget=data.budget(),actual_selected_source_budget=schedule.selected_budget(data),
            all_eight_actual_capacity_input_extremes_rebuilt=True,fresh_four_L40_NN_capacity_and_recovery_still_required=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',OUT/'actual-auxiliary-source-batches.json',OUT/'actual-all12-replay-extremes.json',
                OUT/'actual-eight-capacity-input-batches.json',OUT/'preparation-only-schedule.json']},
            actual_model_gain_or_GPU_speedup_not_established=True,models_constructed=0,model_updates=0,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),planned_updates=len(schedule),added_state_updates=len(schedule)-len(previous),
            actual_auxiliary_batches=checked,model_training_started=False)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),models_constructed=0,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
