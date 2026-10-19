"""Read every experimental dense tensor against the unchanged own source.

Waits for the exact CPU rehearsal to finish. No network is instantiated and
no forward/backward or source program is rerun. The in-memory Adam/RNG and
cold-forward observations remain producer evidence, not a repeated proof.
"""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/full-state-backward-CPU-v1'
NATIVE=ROOT/'checkpoints/full-state-backward-CPU-v1/model.pt'
OUT=ROOT/'reports/full-state-backward-CPU-v1-readback'


def tensors(payload):
    return {group+'.'+name:value for group in ('network','heads') for name,value in payload[group].items() if name!='lm_head.weight'}


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only full-state tensor readback required')
    plan_path=SOURCE/'plan.json';source_plan=json.loads(plan_path.read_text());identity=source_plan['owner']
    files={str(p):file_hash(p) for p in [Path(__file__),plan_path]};guard=FileBindings(files)
    OUT.mkdir();deadline=time.monotonic()+7*86400;torch.set_num_threads(1)
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        exact_source_owner=identity,every_stored_native_parameter_count_shape_and_finite_value_checked=True,
        every_nonlanguage_tensor_compared_directly_with_original_partial_main=True,
        no_neural_model_or_fresh_forward_or_backward=True,
        durable_private_Adam_RNG_snapshot_absent_so_not_independently_read_back=True,
        original_source_and_GPU_jobs_left_unchanged=True,models_constructed=0,model_updates=0,goal_achieved=False))
    try:
        while not terminal(identity):
            guard.check()
            if (SOURCE/'failure.json').exists():raise ValueError('Exact source rehearsal failed; preserve all evidence without retry')
            if time.monotonic()>deadline:raise ValueError('Reader observation deadline; original source process left untouched')
            time.sleep(5)
        done_path=SOURCE/'complete.json'
        if (SOURCE/'failure.json').exists() or not done_path.is_file():raise ValueError('Exact terminal complete source rehearsal required')
        done=json.loads(done_path.read_text())
        if (done.get('complete') is not True or done.get('actual_logical_updates')!=2 or done.get('physical_optimizer_calls')!=3
                or done.get('actual_source_main_updates')!=28672 or done.get('actual_CUDA_initialized') is not False
                or source_plan.get('full_source_epoch_completed') is not False):raise ValueError('Explicit limited two-update partial-main CPU experiment required')
        bindings={**source_plan['source_files_sha256'],**done['source_files_sha256'],str(done_path):file_hash(done_path),**files}
        full_guard=FileBindings(bindings)
        source_path=Path(source_plan['original_main_inputs']['checkpoint'])
        original=torch.load(source_path,map_location='cpu',weights_only=True,mmap=True)
        native=torch.load(NATIVE,map_location='cpu',weights_only=True,mmap=True)
        if (original.get('format')!='own-fast-joint-deterministic-positions-native-model-v2'
                or original.get('arm')!='step-28672' or original.get('completed_updates')!=28672 or original.get('whole_source_epochs_completed')!=0
                or native.get('format')!='own-complete-dense-standalone-student-v1' or native.get('completed_updates')!=2
                or native.get('old_checkpoint_optimizer_or_training_reference_needed_for_inference') is not False
                or native.get('provenance',{}).get('not_a_production_initializer_or_generated_quality_result') is not True):
            raise ValueError('Exact partial-source and complete experimental native identities required')
        before,after=original['model'],native['complete_dense']
        if any(before[k]!=after[k] for k in ['config','settings','actual_counts']):raise ValueError('All original configuration/head dimensions and counts required')
        a,b=tensors(before),tensors(after)
        if (a.keys()!=b.keys() or sum(v.numel() for v in b.values())!=984399690
                or before['network']['embedding.weight'].data_ptr()!=before['network']['lm_head.weight'].data_ptr()
                or after['network']['embedding.weight'].data_ptr()!=after['network']['lm_head.weight'].data_ptr()):
            raise ValueError('Every984399690actual value and one stored tied vocabulary required')
        language={n for n in b if n.startswith('network.') and not n.startswith(('network.vision.','network.value_head.'))}
        if len(language)!=317:raise ValueError('Every actual full language tensor required')
        changed=[];fixed=[]
        for name,value in b.items():
            if value.shape!=a[name].shape or value.dtype!=a[name].dtype or value.dtype!=torch.float32 or not bool(torch.isfinite(value).all()):
                raise ValueError('Every actual native value must remain finite with unchanged full shape/dtype')
            differs=not torch.equal(value,a[name])
            if name in language:
                if differs:changed.append(name)
            else:
                if differs:raise ValueError('An unselected vision/word/critic/auxiliary tensor changed')
                fixed.append(name)
        if not changed or sorted(changed)!=sorted(done['actual_changed_language_tensor_names']):raise ValueError('Actual changed full-language tensors must match stored bytes')
        index=np.load(ROOT/'data/public-state-traces-admitted-v1/index.npy',allow_pickle=False)
        positions=source_plan['actual_selected_state_positions'];records=done['actual_update_records']
        if len(records)!=2 or len(positions)!=2:raise ValueError('Every original source update must retain one complete record')
        for i,(position,row) in enumerate(zip(positions,records)):
            item=index[position];length=int(item['length']);targets=length-int(item['target_start'])
            if (row['step']!=i+1 or row['family']!='new:public_source_state_prediction' or row['global_original_positions']!=[position]
                    or row['global_examples']!=1 or row['complete_context_tokens']!=length or row['response_EOS_targets']!=targets
                    or row['actual_active_groups']!=['language'] or row['active_parameter_tensors']!=317
                    or row['token_mean_CE']!=row['CE_sum']/targets
                    or row['full_context_reference_KL_mean']!=row['full_context_reference_KL_sum']/(length-1)
                    or row['resident_training_only_reference_parameters']!=984399690 or row['reference_forward_queried'] is not True
                    or not all(math.isfinite(row[k]) for k in ['CE_sum','token_mean_CE','gradient_norm_before_clip','full_context_reference_KL_sum'])):
                raise ValueError('Every real whole-source cursor, target/EOS and global normalization must reconstruct')
        expected_sources=dict(experimental_main28672_sha256=source_plan['original_main_inputs']['checkpoint_sha256'],
            rehearsal_plan_sha256=file_hash(plan_path),admitted_state_manifest_sha256=file_hash(ROOT/'data/public-state-traces-admitted-v1/manifest.json'))
        if (after['sources']!=expected_sources or native['tokenizer_sha256']!=file_hash(NATIVE.parent/'tokenizer.json')
                or native['provenance']['actual_source_main_updates']!=28672 or native['provenance']['full_state_source_epoch_completed'] is not False):
            raise ValueError('Native experiment/source/tokenizer provenance must remain explicit and exact')
        full_guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            actual_unique_native_values=984399690,actual_full_language_tensors=317,actual_changed_language_tensors=len(changed),
            actual_fixed_vision_word_critic_auxiliary_tensors=len(fixed),actual_all_unique_finite_tensor_shapes=len(b),
            every_recorded_source_context_target_EOS_denominator_and_cursor_rebuilt=True,
            all_original_nonlanguage_values_bit_exact=True,one_actual_stored_tied_vocabulary=True,
            actual_partial_source_main_updates=28672,actual_experimental_logical_state_updates=2,
            actual_source_training_NLL_before=done['longest_source_pre_update_NLL'],actual_source_training_NLL_after=done['longest_source_post_update_NLL'],
            source_training_NLL_is_not_generated_or_held_capability=True,
            producer_in_memory_recovery_and_cold_neural_probes_not_repeated_by_reader=True,
            no_durable_Adam_or_private_RNG_state_to_independently_verify=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',plan_path,done_path,source_path,NATIVE,NATIVE.parent/'tokenizer.json']},
            original_source_and_training_queues_unchanged=True,models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),actual_native_values=984399690,
            actual_changed_language_tensors=len(changed),actual_fixed_nonlanguage_tensors=len(fixed),generated_quality_not_assessed=True)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(owner=owner,error=repr(error),traceback=traceback.format_exc(),
            original_source_process_and_GPU_jobs_untouched=True,models_constructed=0,model_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
