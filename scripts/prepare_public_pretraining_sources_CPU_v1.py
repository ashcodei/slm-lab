"""Prepare an actual complete public-pretraining schedule after full admission.

Every previous code/state/replay presentation is retained, with compute-bounded
regrouping. This constructs source tensors, not a model or optimizer. A fresh
full-size four-L40 capacity/recovery experiment remains necessary before SGD.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/public-pretraining-sources-CPU-v1'
QUEUE = ROOT/'reports/public-pretraining-sources-CPU-v1-queue.json'
READER = ROOT/'reports/public-billion-causal-candidate-v2-full-readback-v1'
PACK = ROOT/'data/public-billion-causal-windows-v1'
READER_OWNER = dict(pid=3815661, ticks='43628858')
RAW = 'raw:public_code_math_web'


def dependencies_ready():
    if json.loads((READER/'plan.json').read_text())['owner'] != READER_OWNER:
        raise ValueError('Only the exact complete original-source reader is eligible')
    if not terminal(READER_OWNER):return False
    if (READER/'failure.json').exists() or not (READER/'complete.json').exists():
        raise ValueError('The full original public-source reader must finish, not an admitted prefix')
    done = json.loads((READER/'complete.json').read_text())
    admission = json.loads((PACK/'admission.json').read_text())
    if (done.get('complete') is not True or done.get('data_admitted_to_training') is not True
            or done.get('actual_train_targets_admitted',0) < 10**9
            or admission.get('admitted_to_training') is not True or admission.get('full_source_reader_complete') is not True
            or admission.get('manifest_sha256') != file_hash(PACK/'manifest.json')):
        raise ValueError('Every actual original train quota and complete data admission must finish')
    return True


def validate_selected_budget(selected, previous):
    if set(selected)-{RAW} != set(previous) or selected.get(RAW,{}).get('response_EOS_targets',0) < 10**9:
        raise ValueError('The full billion-target raw source and every original retained family are required')
    for name, original in previous.items():
        for key in ['presentations','complete_context_tokens','response_EOS_targets']:
            if selected[name][key] != original[key]:
                raise ValueError('No previous code/state/photo/word/broad presentation or native target may be removed or repeated: '+name)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists() or QUEUE.exists():
        raise ValueError('Fresh GPU-hidden complete actual-source preparation required')
    from slm_lab.local_source_closure_v2 import python_sources
    tests = ROOT/'reports/public-pretraining-data-update-checkpoint-v1-tests.log'
    driver_tests = ROOT/'reports/public-pretraining-source-preparation-v1-tests.log'
    parallel = ROOT/'reports/public-pretraining-four-CPU-v1/complete.json'
    proof = json.loads(parallel.read_text())
    if ('7 passed' not in tests.read_text() or '2 passed' not in driver_tests.read_text()
            or not proof['all_four_actual_CPU_workers_passed']
            or proof['final_replica_examples'] != [1,0,0,0]
            or proof['serial_full_update_maximum_parameter_difference'] > 2e-6):
        raise ValueError('Actual source/modality/recovery, four-replica empty-tail and serial controls must pass')
    checkpoint = ROOT/'checkpoints/phase-joint-v3/trained/model.pt'
    previous_plan = ROOT/'reports/code-first-continuation-v3/plan.json'
    paths = [tests,driver_tests,parallel,checkpoint,previous_plan,READER/'plan.json',
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_public_pretraining_source_preparation_v1.py'])]
    files = {str(p):file_hash(p) for p in paths}
    guard = FileBindings(files)
    OUT.mkdir()
    me = dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    plan = dict(at=datetime.now(timezone.utc).isoformat(),owner=me,exact_full_source_reader=READER_OWNER,
        source_files_sha256=files,minimum_new_train_targets=10**9,
        every_previous_selected_code_state_and_all12_replay_family_exposures_required=True,
        compute_batching=dict(padded_context_budget=262144,maximum_rows=1024,image_size=448,seed=976319),
        no_neural_model_optimizer_GPU_or_teacher_loaded=True,
        full_size_four_L40_capacity_and_recovery_still_required=True,
        training_driver_and_explicit_GPU_ownership_still_required=True,production_updates=0,goal_achieved=False)
    atomic_json(QUEUE,plan);atomic_json(OUT/'plan.json',plan)
    try:
        while not dependencies_ready():
            guard.check()
            atomic_json(OUT/'status.json',dict(state='waiting_for_exact_full_billion_source_admission',
                owner=me,reader_owner=READER_OWNER,model_updates=0,goal_achieved=False))
            time.sleep(10)
        import numpy as np
        import torch
        from slm_lab.model import ModelConfig
        from slm_lab.public_pretraining_data_v1 import PublicPretrainingData,TokenBudgetSchedule
        torch.set_num_threads(1)
        raw = torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
        config = ModelConfig(**raw['model']['config']);del raw
        # Only read the architecture header; no neural network is instantiated.
        data = PublicPretrainingData(ROOT,config)
        settings = plan['compute_batching']
        schedule = TokenBudgetSchedule(data,image_prefix=config.image_token_count(settings['image_size']),
            context_budget=settings['padded_context_budget'],maximum_rows=settings['maximum_rows'],seed=settings['seed'])
        prior = json.loads(previous_plan.read_text())
        selected = schedule.selected_budget(data)
        validate_selected_budget(selected,prior['selected_source_budget'])
        files.update(data.source_files_sha256)
        for path in [READER/'complete.json',PACK/'manifest.json',PACK/'admission.json']:
            files[str(path)] = file_hash(path)
        guard = FileBindings(files)
        atomic_json(OUT/'complete-source-schedule.json',dict(settings=schedule.settings,
            identity_sha256=schedule.identity_sha256,entries=schedule.entries,selected_source_budget=selected,
            previous_retention_identity_sha256=data.retention_identity_sha256))
        extremes = {};capacity = {}
        for name in selected:
            candidates = [p for n,p in schedule.entries if n == name]
            prefix = schedule.settings['image_prefix'] if data.kinds[name] in {'image','word'} else 0
            def cost(p):return len(p)*(max(data.lengths[name][i] for i in p)+prefix)
            choices = [min(candidates,key=lambda p:(len(p),cost(p))),max(candidates,key=cost)]
            capacity[name] = choices[-1]
            rows = []
            for positions in choices:
                if len(positions) > settings['maximum_rows'] or cost(positions) > settings['padded_context_budget']:
                    raise ValueError('Actual whole source batches must fit their declared padded context bound')
                batch = data.batch(name,positions,'cpu')
                actual_contexts = int(batch['attention_mask'].sum())
                actual_targets = int((batch['labels'][:,1:] != -100).sum())
                if actual_contexts != sum(data.lengths[name][p] for p in positions) or actual_targets != sum(data.targets[name][p] for p in positions):
                    raise ValueError('Actual complete source tensors must preserve every original context and native target')
                rows.append(dict(positions=positions,examples=len(positions),padded_context_cost_bound=cost(positions),
                    actual_native_context_tokens=actual_contexts,actual_response_EOS_targets=actual_targets,
                    actual_collated_input_ids_little_u16_sha256=__import__('hashlib').sha256(batch['input_ids'].numpy().astype('<u2').tobytes()).hexdigest(),
                    actual_original_pixel_shape=list(batch['pixel_values'].shape) if batch.get('pixel_values') is not None else None,
                    actual_real_word_pixel_shape=list(batch['word_pixels'].shape) if batch.get('word_pixels') is not None else None))
                del batch
            extremes[name] = rows
            atomic_json(OUT/'progress.json',dict(complete_actual_family_extremes=list(extremes),planned_families=list(selected),
                model_updates=0,goal_achieved=False))
        if torch.cuda.is_initialized():raise ValueError('Source preparation cannot acquire any GPU')
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            complete_source_budget=data.budget(),selected_source_budget=selected,planned_steps=len(schedule),
            source_schedule_sha256=schedule.identity_sha256,actual_schedule_settings=schedule.settings,
            all_new_admitted_raw_train_windows_once=True,all_previous_selected_source_presentations_exactly_preserved=True,
            every_selected_family_has_actual_minimum_and_maximum_compute_batch_proofs=True,
            actual_family_extremes=extremes,actual_capacity_global_batches=capacity,
            all_tensors_built_on_CPU_without_model_or_optimizer=True,all_native_masks_and_padding_denominators_rebuilt=True,
            source_files_sha256={**files,str(OUT/'complete-source-schedule.json'):file_hash(OUT/'complete-source-schedule.json')},
            full_size_four_L40_capacity_recovery_and_cold_export_still_required=True,
            production_training_controller_still_required=True,new_neural_training_updates=0,
            useful_model_gain_not_established=True,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),planned_steps=len(schedule),
            new_raw_train_targets=selected[RAW]['response_EOS_targets'],neural_updates=0)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),
            every_original_job_and_artifact_preserved=True,goal_achieved=False))
        raise


if __name__ == '__main__':main()
