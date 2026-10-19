"""Explain the observed same-source NLL rise on an isolated partial own copy.

Reproduce the exact two-update warm-Adam endpoint first. Then measure two
scaled displacements and one fresh-Adam control from the same pre-update
weights and gradient. No production hyperparameter or queue is changed.
"""
from datetime import datetime,timezone
import copy
import gc
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from prove_full_state_backward_CPU_v1 import TwoSources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.fast_joint_inference_v2 import load
from slm_lab.fast_joint_stage_v1 import make_optimizer,parameter_groups,learning_factor
from slm_lab.focused_code_continuation_data_v3 import STATE,FocusedCodeContinuationData
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.phase_joint_update_v1 import update,backward_phase
from slm_lab.state_backward_rehearsal_v1 import loss
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/full-state-backward-CPU-v1'
OUT=ROOT/'reports/state-update-direction-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden matched update-direction diagnostic required')
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    plan_path=SOURCE/'plan.json';done_path=SOURCE/'complete.json';source_plan=json.loads(plan_path.read_text());done=json.loads(done_path.read_text())
    if (not done['complete'] or not done['actual_in_memory_next_update_weights_Adam_and_private_CPU_RNG_exact']
            or not done['longest_source_post_update_NLL']>done['longest_source_pre_update_NLL']):
        raise ValueError('This diagnosis requires the actual reproducible same-source NLL increase')
    files={**source_plan['source_files_sha256'],**done['source_files_sha256']}
    files.update({str(p):file_hash(p) for p in [done_path,*python_sources(ROOT,[Path(__file__)])]})
    guard=FileBindings(files);OUT.mkdir();started=time.monotonic()
    owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        source_is_partial_main28672_plus_two_experimental_state_updates=True,
        same_pre_update_model_and_same_actual_clipped_gradient_for_all_controls=True,
        predeclared_warm_displacement_scales=[.25,.0625],fresh_Adam_control_learning_rate=source_plan['training_options']['learning_rates']['language'],
        warm_endpoint_must_match_every_retained_final_language_tensor_before_controls=True,
        clipped_combined_gradient_dot_actual_displacement_measured=True,
        source_loss_not_held_capability_and_not_production_batch16_warmup_profile=True,
        no_model_checkpoints_or_production_recipe_changes=True,CPU_threads=2,production_updates=0,goal_achieved=False))
    def progress(phase):atomic_json(OUT/'progress.json',dict(owner=owner,phase=phase,elapsed_seconds=time.monotonic()-started,
        actual_CUDA_initialized=torch.cuda.is_initialized(),production_updates=0,goal_achieved=False))
    try:
        progress('loading exact original CPU source')
        inputs=source_plan['original_main_inputs'];model,_=load(inputs['checkpoint'],ROOT/'reports/phase-joint-v3/plan.json','step-28672',
            expected_checkpoint_sha256=inputs['checkpoint_sha256'],expected_plan_sha256=inputs['plan_sha256'])
        reference=copy.deepcopy(model).eval().requires_grad_(False);data=FocusedCodeContinuationData(ROOT,model.config,purpose='optimization')
        schedule=TwoSources(source_plan['actual_selected_state_positions']);options=source_plan['training_options'];optimizer=make_optimizer(model,options)
        groups=parameter_groups(model);language=groups['language'];parameters=[p for rows in groups.values() for _,p in rows]
        fixed=[(n,p) for n,p in model.named_parameters() if id(p) not in {id(x) for _,x in language}];fixed_digest=tensor_digest(fixed)
        random.seed(978217);np.random.seed(978217);torch.manual_seed(978217)
        progress('reproducing exact first short-source update')
        first=update(model,optimizer,data,schedule,0,options,reference=reference)
        if first!=done['actual_update_records'][0]:raise ValueError('Exact original first update must reproduce before interpreting controls')
        # The original recovery rehearsal advances each private generator once
        # between its snapshot and the next source update.
        random.random();np.random.random();torch.rand(())
        before={n:p.detach().clone() for n,p in language};family,positions=schedule.at(1);batch=data.batch(family,positions,'cpu')
        targets=data.targets[STATE][positions[0]];contexts=data.lengths[STATE][positions[0]]-1
        factor=learning_factor(1,2,options['warmup_fraction'],options['cosine_floor'])
        for group in optimizer.param_groups:group['lr']=options['phase_learning_rates']['instruction_and_vision'][group['name']]*factor
        model.train();model.network.gradient_checkpointing=True;optimizer.zero_grad(set_to_none=True)
        progress('computing exact whole long-source combined gradient')
        actual=backward_phase(model,batch,phase='instruction_and_vision',kind='text',global_targets=targets,
            global_context_targets=contexts,reference=reference,reference_KL_weight=options['reference_KL_weight'])
        norm=float(torch.nn.utils.clip_grad_norm_(parameters,options['gradient_clip'],error_if_nonfinite=True))
        expected=done['actual_update_records'][1]
        if (actual['CE_sum']!=expected['CE_sum'] or actual['full_context_reference_KL_sum']!=expected['full_context_reference_KL_sum']
                or norm!=expected['gradient_norm_before_clip'] or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for _,p in language)):
            raise ValueError('Exact pre-update source loss, own-reference loss and full gradient norm must reproduce')
        optimizer.step()
        native=torch.load(ROOT/'checkpoints/full-state-backward-CPU-v1/model.pt',map_location='cpu',weights_only=True,mmap=True)
        final=native['complete_dense']['network']
        if any(not torch.equal(p,final[n.removeprefix('network.')]) for n,p in language):
            raise ValueError('Recomputed warm-Adam endpoint must equal every actual retained final language tensor')
        def direction():
            dot=squared=0.
            for name,p in language:
                delta=p.detach().double()-before[name].double()
                dot+=float((p.grad.detach().double()*delta).sum());squared+=float(delta.square().sum())
            return dict(clipped_combined_gradient_dot_displacement=dot,actual_parameter_displacement_L2=math.sqrt(squared))
        warm_direction=direction();observations=[dict(kind='warm_Adam_full_step',scale=1.,
            source_NLL=done['longest_source_post_update_NLL'],source_NLL_reused_only_after_every_weight_matches=True,**warm_direction)]
        progress('evaluating scaled warm-Adam displacements')
        for scale in [.25,.0625]:
            with torch.no_grad():
                for name,p in language:p.copy_(before[name]+scale*(final[name.removeprefix('network.')]-before[name]))
            measured=loss(model,batch)
            if not math.isfinite(measured):raise ValueError('Every actual displacement control needs finite complete-source loss')
            observations.append(dict(kind='scaled_actual_warm_parameter_displacement',scale=scale,source_NLL=measured,
                not_claimed_bit_equal_to_changing_Adam_learning_rate=True,**direction()))
            atomic_json(OUT/'partial-observations.json',dict(before_source_NLL=expected['token_mean_CE'],actual_observations=observations,
                all_fixed_heads_unchanged=tensor_digest(fixed)==fixed_digest,production_updates=0))
        progress('evaluating fresh-Adam control at the original learning rate')
        with torch.no_grad():
            for name,p in language:p.copy_(before[name])
        del optimizer;gc.collect()
        fresh=make_optimizer(model,options)
        for group in fresh.param_groups:group['lr']=options['phase_learning_rates']['instruction_and_vision'][group['name']]*factor
        fresh.step();fresh_direction=direction();measured=loss(model,batch)
        if not math.isfinite(measured):raise ValueError('Finite complete-source fresh-Adam loss required')
        observations.append(dict(kind='fresh_Adam_same_pre_update_weights_and_gradient',scale=1.,source_NLL=measured,
            both_first_and_second_moment_history_reset=True,**fresh_direction))
        if tensor_digest(fixed)!=fixed_digest or torch.cuda.is_initialized():raise ValueError('All nonlanguage values and CPU-only scope must remain exact')
        guard.check(full=True)
        atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
            actual_unique_student_values=984399690,actual_source_state_position=positions[0],actual_whole_context_positions=contexts+1,actual_response_EOS_targets=targets,
            exact_original_short_update_and_long_gradient_and_warm_final_parameters_reproduced=True,
            before_source_NLL=expected['token_mean_CE'],actual_observations=observations,
            warm_step_is_non_descent_for_measured_current_combined_gradient=warm_direction['clipped_combined_gradient_dot_displacement']>0,
            all_fixed_vision_word_critic_auxiliary_values_unchanged=True,
            source_loss_controls_not_held_code_or_visual_quality=True,
            production_batch16_and7558update_warmup_not_ablated_by_this_two_example_test=True,
            original_training_queues_hyperparameters_and_checkpoints_unchanged=True,
            source_files_sha256={str(p):file_hash(p) for p in [OUT/'plan.json',SOURCE/'complete.json',ROOT/'checkpoints/full-state-backward-CPU-v1/model.pt']},
            production_updates=0,candidate_promoted=False,goal_achieved=False))
        print(json.dumps(dict(complete=str(OUT/'complete.json'),before_source_NLL=expected['token_mean_CE'],
            actual_observations=observations,production_recipe_unchanged=True)),flush=True)
    except BaseException as error:
        atomic_json(OUT/'failure.json',dict(owner=owner,error=repr(error),traceback=traceback.format_exc(),
            original_training_and_queues_untouched=True,production_updates=0,goal_achieved=False));raise


if __name__=='__main__':main()
