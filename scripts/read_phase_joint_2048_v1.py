"""Read actual step2048 state and matched held loss without neural construction.

The large mutable resume file is observed once, not copied or promised as an
immutable retained artifact. Compact actual per-leaf hashes, clocks, RNG states
and journal are retained beside the permanent native checkpoint binding.
There is no new optimizer step, CUDA initialization or independent loss forward.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/phase-joint-step2048-readback-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh CPU-hidden historical state readback required')
    import numpy as np
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources,CompletePhaseSchedule
    from slm_lab.fast_joint_inference_v2 import FORMAT as NATIVE_FORMAT,sources
    from slm_lab.model import ModelConfig
    from slm_lab.phase_joint_checkpoint_v1 import FORMAT,validate_clocks,validate_reference
    torch.set_num_threads(1);OUT.mkdir();started=time.monotonic()
    state_path=ROOT/'checkpoints/phase-joint-v3/resume.pt'
    native_path=ROOT/'checkpoints/phase-joint-v3/step-2048/model.pt'
    plan_path=ROOT/'reports/phase-joint-v3/plan.json'
    plan=json.loads(plan_path.read_text());plan_sha=file_hash(plan_path)
    def stat(path):
        s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
    before=stat(state_path);raw_state_sha=file_hash(state_path)
    saved=torch.load(state_path,map_location='cpu',weights_only=True,mmap=True)
    native=torch.load(native_path,map_location='cpu',weights_only=True,mmap=True)
    core=saved['core'];completed=core['completed']
    if (saved['format'] != FORMAT or completed != 2048 or core['mode'] != 'train' or core['world_size'] != 4
            or len(core['rngs']) != 4 or core['run_plan_sha256'] != plan_sha
            or core['training_options'] != plan['training_options'] or saved['phase_bounds'] != plan['phase_bounds']):
        raise ValueError('Exact live step2048 phase/source/optimizer/private-worker profile required')
    validate_reference(plan,'train',2048,saved['reference_binding'],None)
    validate_clocks(core,plan,'train')
    if (native['format'] != NATIVE_FORMAT or native['completed_updates'] != 2048 or native['arm'] != 'step-2048'
            or native['run_plan_sha256'] != plan_sha or native['whole_source_epochs_completed'] != 0
            or native['model']['sources'] != sources(plan,plan_sha)):
        raise ValueError('Exact permanent partial native checkpoint required')
    state_native=core['model']
    if (state_native['arm']!='trained' or
            {k:v for k,v in native.items() if k not in ['model','arm']} !=
            {k:v for k,v in state_native.items() if k not in ['model','arm']}):
        raise ValueError('Resume and native exports must identify precisely the same committed state')
    model=native['model'];state_model=state_native['model']
    if {k:v for k,v in model.items() if k not in ['network','heads']} != {k:v for k,v in state_model.items() if k not in ['network','heads']}:
        raise ValueError('Actual native/resume architecture, source bindings and inventory must agree')
    initial_path=Path(plan['initial_dense_settings']['initializer'])
    if file_hash(initial_path) != plan['initial_dense_settings']['initializer_sha256']:
        raise ValueError('Exact own initial full-weight source required')
    initial=torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True)
    # The own initializer is an OwnDenseAssistant export, not a phase wrapper.
    if initial.get('config') != model['config'] or initial.get('actual_counts') != model['actual_counts']:
        raise ValueError('Initial and trained complete native architecture must match')
    def leaf_digest(value,*,other=None,nonnegative=False):
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or not value.is_contiguous():
            raise ValueError('Actual contiguous CPU-mapped checkpoint tensor required')
        if other is not None and (value.shape != other.shape or value.dtype != other.dtype):
            raise ValueError('Every native tensor shape and dtype must match')
        digest=hashlib.sha256();flat=value.reshape(-1)
        for start in range(0,flat.numel(),1<<20):
            chunk=flat[start:start+(1<<20)]
            if not bool(torch.isfinite(chunk).all()) or nonnegative and bool((chunk<0).any()):
                raise ValueError('Every mapped weight/moment must be finite and second moments nonnegative')
            if other is not None and not torch.equal(chunk,other.reshape(-1)[start:start+len(chunk)]):
                raise ValueError('Committed resume/native tensor differs')
            digest.update(chunk.view(torch.uint8).numpy().tobytes())
        return dict(shape=list(value.shape),dtype=str(value.dtype),values=value.numel(),sha256=digest.hexdigest())
    groups=dict(language=[],vision=[],word=[]);inventory={};changed={k:[] for k in groups};unchanged_other=[]
    unique=0;weight_hashes={}
    for section in ['network','heads']:
        if model[section].keys() != state_model[section].keys() or model[section].keys() != initial[section].keys():
            raise ValueError('Every original complete native tensor key required')
        for name,value in model[section].items():
            if section=='network' and name=='lm_head.weight':
                if (not torch.equal(value,model['network']['embedding.weight']) or
                        not torch.equal(state_model['network']['lm_head.weight'],state_model['network']['embedding.weight'])):
                    raise ValueError('One equal tied vocabulary required in both exports')
                continue
            key=section+'.'+name;unique+=value.numel()
            weight_hashes[key]=leaf_digest(value,other=state_model[section][name])
            group=('vision' if key.startswith('network.vision.') else
                'language' if key.startswith('network.') and not key.startswith('network.value_head.') else
                'word' if key.startswith('heads.word.') else None)
            equal=torch.equal(value,initial[section][name])
            if group is None:
                if not equal:raise ValueError('A never-selected head or critic changed during raw foundation')
                unchanged_other.append(key)
            else:
                groups[group].append((key,value))
                if not equal:changed[group].append(key)
    if unique != 984399690 or unique != model['actual_counts']['total_unique'] or changed['vision'] or changed['word'] or not changed['language']:
        raise ValueError('Counted own under1B student and foundation-only active-value changes required')
    adam_hashes={};selected=[]
    for group in core['optimizer']['param_groups']:
        leaves=groups[group['name']]
        if len(group['params']) != len(leaves):raise ValueError('Every actual Adam key must map to its complete parameter partition')
        if tuple(group['betas']) != tuple(plan['training_options']['betas']) or group['weight_decay'] != plan['training_options']['weight_decay']:
            raise ValueError('Exact configured Adam hyperparameters required')
        for key,(name,value) in zip(group['params'],leaves):
            selected.append(key);state=core['optimizer']['state'].get(key)
            if group['name']!='language':
                if state is not None:raise ValueError('Inactive vision/word groups must have no Adam state')
                continue
            if set(state) != {'step','exp_avg','exp_avg_sq'} or float(state['step']) != 2048:
                raise ValueError('Every active language Adam leaf must have exactly2048 updates')
            if any(state[k].shape != value.shape or state[k].dtype != value.dtype for k in ['exp_avg','exp_avg_sq']):
                raise ValueError('Complete original-shaped full-weight moments required')
            adam_hashes[name]=dict(step=2048,exp_avg=leaf_digest(state['exp_avg']),exp_avg_sq=leaf_digest(state['exp_avg_sq'],nonnegative=True))
    if len(set(selected)) != len(selected) or len(adam_hashes) != 317 or len(core['optimizer']['state']) != 317:
        raise ValueError('Every and only active unique Adam key required')
    for state in core['rngs']:
        random.Random().setstate(state['python']);n=state['numpy']
        np.random.RandomState().set_state((n['kind'],np.asarray(n['keys'],dtype=np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']))
        torch.Generator(device='cpu').set_state(state['torch'])
        cuda=state['cuda']
        if not isinstance(cuda,torch.Tensor) or cuda.dtype!=torch.uint8 or cuda.shape!=(16,):
            raise ValueError('Actual retained L40 private CUDA RNG byte profile required')
    data=ExpandedJointSources(ROOT,ModelConfig(**model['config']))
    schedule=CompletePhaseSchedule(data,plan['global_family_batch_widths'],plan['seed'])
    if schedule.phase_bounds != plan['phase_bounds'] or len(schedule) != plan['planned_steps']:
        raise ValueError('Exact original complete phase schedule required')
    journal=[]
    for line in (ROOT/'reports/phase-joint-v3/training-metrics.jsonl').read_bytes().splitlines():
        if len(journal)==2048:break
        journal.append(json.loads(line))
    if journal != core['metrics']:raise ValueError('Actual external committed journal prefix must exactly match the saved full state')
    for i,row in enumerate(journal):
        family,positions=schedule.at(i)
        if (row['step']!=i+1 or row['family']!=family or row['global_original_positions']!=positions
                or row['complete_context_tokens']!=sum(data.lengths[family][p] for p in positions)
                or row['response_EOS_targets']!=sum(data.targets[family][p] for p in positions)
                or row['active_parameter_tensors']!=317 or row['phase']!='foundation'
                or not math.isclose(row['token_mean_CE'],row['CE_sum']/row['response_EOS_targets'],rel_tol=1e-14)):
            raise ValueError('Every actual committed source cursor and native target budget must reconstruct')
    initial_loss_path=ROOT/'reports/phase-joint-v3/held-packed-initial.json'
    trained_loss_path=ROOT/'reports/phase-joint-v3/held-packed-2048.json'
    a,b=[json.loads(p.read_text()) for p in [initial_loss_path,trained_loss_path]]
    if {k:v for k,v in a.items() if k!='token_mean_NLL'} != {k:v for k,v in b.items() if k!='token_mean_NLL'}:
        raise ValueError('All held source positions, contexts, target denominator and flags must match')
    packed=data.packed;population=packed.size(packed.families[0],split='valid');positions=[i*population//128 for i in range(128)]
    if positions!=b['selected_original_positions'] or b['original_manifest_sha256']!=packed.manifest['packed_manifest_sha256']:
        raise ValueError('Exact predeclared unchanged held packed window selection required')
    held_targets=0;held=[]
    for position in positions:
        row=packed.row(packed.families[0],position,split='valid')
        if row['split']!='valid' or row['source_split']!='valid':raise ValueError('Only original valid source windows required')
        held_targets+=len(row['tokens'])-1
        held.append(dict(position=position,original_input_sha256=row['_replay_input_sha256'],targets=len(row['tokens'])-1))
    if held_targets!=524160 or held_targets!=b['complete_causal_targets']:
        raise ValueError('Every actual held causal target and original input byte hash required')
    if stat(state_path)!=before:raise ValueError('Mutable state was replaced during readback; this readback is not admitted')
    # Compact retained capture; it deliberately excludes all model/Adam values.
    capture=OUT/'captured-state-evidence.pt'
    torch.save(dict(captured_source_path=str(state_path),captured_source_stat=before,captured_source_sha256=raw_state_sha,
        completed=2048,model_leaf_hashes=weight_hashes,adam_leaf_hashes=adam_hashes,
        optimizer_param_groups=core['optimizer']['param_groups'],all_four_actual_private_RNG_states=core['rngs'],
        metrics=journal,no_model_or_Adam_values_retained=True),capture)
    atomic_json(OUT/'held-window-bindings.json',held)
    evidence=[Path(__file__),plan_path,native_path,initial_path,initial_loss_path,trained_loss_path,
        ROOT/'src/slm_lab/phase_joint_checkpoint_v1.py',ROOT/'src/slm_lab/fast_joint_checkpoint_v2.py',
        ROOT/'src/slm_lab/packed_validation_v1.py',ROOT/'src/slm_lab/fast_joint_stage_v1.py',
        ROOT/'scripts/train_phase_joint_v3.py',capture,OUT/'held-window-bindings.json']
    bindings={str(p):file_hash(p) for p in evidence}
    if bindings[str(initial_path)]!=plan['initial_dense_settings']['initializer_sha256'] or file_hash(plan_path)!=plan_sha:
        raise ValueError('Permanent observed inputs changed during readback')
    result=dict(at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        complete=True,actual_completed_updates=2048,whole_source_epoch_completed=False,
        own_exported_unique_parameters=unique,active_language_Adam_states=317,all_active_Adam_clocks=2048,
        current_source_cursor_and_complete_native_target_counts_rederived=True,
        actual_trained_targets=int(sum(r['response_EOS_targets'] for r in journal)),
        native_and_resume_all_model_tensors_bit_exact=True,all_weights_and_Adam_moments_finite=True,
        all_second_moments_nonnegative=True,changed_parameter_tensors={k:len(v) for k,v in changed.items()},
        inactive_vision_word_and_other_head_values_exactly_unchanged=True,
        four_private_Python_NumPy_TorchCPU_RNG_states_validated_without_global_mutation=True,
        four_private_CUDA_RNG_states_retained_CPU_structure_only=True,
        no_new_actual_CUDA_RNG_restore_or_next_update_replay=True,
        held_windows=128,held_original_targets=held_targets,held_positions_contexts_and_token_bytes_rederived=True,
        held_initial_NLL=a['token_mean_NLL'],held_step2048_NLL=b['token_mean_NLL'],
        relative_held_NLL_reduction=1-b['token_mean_NLL']/a['token_mean_NLL'],
        held_initial_perplexity=math.exp(a['token_mean_NLL']),held_step2048_perplexity=math.exp(b['token_mean_NLL']),
        held_NLL_values_are_producer_observations_not_an_independent_new_neural_forward=True,
        no_per_window_losses_retained_so_no_paired_uncertainty_estimate=True,
        held_next_token_prediction_gain_not_code_correctness_or_all_domain_gain=True,
        raw_mutable_resume_observation=dict(path=str(state_path),sha256_at_capture=raw_state_sha,
            stat_at_capture=before,large_original_resume_not_copied_or_retained_by_this_reader=True,
            path_may_be_atomically_replaced_at_a_later_training_checkpoint=True),
        source_files_sha256=bindings,model_generations=0,model_updates=0,
        new_code_correctness_gain_measured=False,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',result)
    print(json.dumps({k:result[k] for k in ['actual_completed_updates','actual_trained_targets','own_exported_unique_parameters',
        'active_language_Adam_states','relative_held_NLL_reduction','held_positions_contexts_and_token_bytes_rederived']}),flush=True)


if __name__=='__main__':main()
