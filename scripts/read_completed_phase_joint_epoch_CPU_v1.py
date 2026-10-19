"""Reconcile the completed full source epoch and stored final state on CPU.

Waits for all four exact training ranks to end, while the original parent may
continue native GPU assessments. Never restores the optimizer, constructs a
neural model, initializes CUDA, or replays a recorded training step.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.expanded_joint_sources_v1 import ExpandedJointSources, CompletePhaseSchedule
from slm_lab.local_source_closure_v2 import python_sources
from slm_lab.model import ModelConfig
from slm_lab.source_jsonl_records_v1 import records

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/phase-joint-v3'
MODEL = ROOT / 'checkpoints/phase-joint-v3'
OUT = ROOT / 'reports/completed-phase-joint-v3-CPU-readback-v1'


def tensors(payload):
    return {group + '.' + name: value for group in ('network', 'heads')
            for name, value in payload[group].items() if name != 'lm_head.weight'}


def factor(step, total, warmup, floor):
    begin = max(1, math.ceil(total * warmup))
    return ((step + 1) / begin if step < begin else
            floor + (1 - floor) * .5 * (1 + math.cos(math.pi * (step - begin) / max(1, total - begin - 1))))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('A fresh CPU-only completed epoch readback is required')
    torch.set_num_threads(1)
    plan_path = SOURCE / 'plan.json'; plan = json.loads(plan_path.read_text()); digest = file_hash(plan_path)
    ranks = [r for r in json.loads((SOURCE / 'children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(ranks) != 4:
        raise ValueError('All four exact original training rank identities must be retained')
    files = {str(p): file_hash(p) for p in [plan_path, *python_sources(ROOT, [Path(__file__)])]}
    guard = FileBindings(files); OUT.mkdir()
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(OUT / 'plan.json', dict(
        at=datetime.now(timezone.utc).isoformat(), owner=owner, source_files_sha256=files,
        exact_original_train_ranks=ranks, every_actual_logged_source_position_and_budget_reconciled=True,
        every_stored_native_and_optimizer_tensor_compared_or_checked=True,
        CPU_private_RNG_states_parsed_without_caller_RNG_mutation=True,
        CUDA_private_RNG_shape_only_no_CUDA_generator_or_replay=True,
        does_not_reexecute35284updates_or_verify_unarchived_per_step_input_tensors=True,
        original_main_native_assessments_and_all_queues_left_unchanged=True,
        models_constructed=0, model_updates=0, goal_achieved=False,
    ))
    try:
        while not all(terminal(r) for r in ranks):
            if (SOURCE / 'failure.json').exists():
                raise ValueError('Original main failed; retain all partial evidence')
            time.sleep(5)
        done_path = SOURCE / 'training-complete.json'; done = json.loads(done_path.read_text())
        if (done['actual_updates'] != 35284 or not done['full_selected_family_source_epoch_complete']
                or done['complete_source_budget'] != plan['complete_source_budget']):
            raise ValueError('The complete declared original source epoch must finish')
        main_guard = FileBindings(plan['source_files_sha256'])
        retained = [done_path, SOURCE / 'training-metrics.jsonl', SOURCE / 'train-reference-binding.json',
                    SOURCE / 'held-packed-35284.json', MODEL / 'initial/model.pt', MODEL / 'trained/model.pt', MODEL / 'resume.pt']
        snapshot_bindings = {str(p): file_hash(p) for p in retained}
        state = torch.load(MODEL / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
        initial = torch.load(MODEL / 'initial/model.pt', map_location='cpu', weights_only=True, mmap=True)
        native = torch.load(MODEL / 'trained/model.pt', map_location='cpu', weights_only=True, mmap=True)
        core = state['core']; payload = native['model']; options = plan['training_options']
        if (state['format'] != 'own-complete-foundation-then-instruction-reference-checkpoint-v1'
                or core['format'] != 'own-fast-joint-complete-Adam-all-worker-RNG-v2'
                or core['mode'] != 'train' or core['completed'] != 35284 or core['world_size'] != 4
                or core['run_plan_sha256'] != digest or core['training_options'] != options
                or state['phase_bounds'] != plan['phase_bounds']
                or native['format'] != 'own-fast-joint-deterministic-positions-native-model-v2'
                or native['arm'] != 'trained' or native['completed_updates'] != 35284
                or native['whole_source_epochs_completed'] != 1 or native['run_plan_sha256'] != digest):
            raise ValueError('All complete original source, native and private snapshot identities must agree')
        if any(core['model'][key] != native[key] for key in native if key != 'model'):
            raise ValueError('Private snapshot and native export metadata differ')
        if any(core['model']['model'][key] != payload[key] for key in payload if key not in ('network', 'heads')):
            raise ValueError('Every complete dense configuration, source and count must agree')
        data = ExpandedJointSources(ROOT, ModelConfig(**payload['config']))
        if data.budget() != plan['complete_source_budget']:
            raise ValueError('Every original family population and whole target budget must rebuild')
        schedule = CompletePhaseSchedule(data, plan['global_family_batch_widths'], plan['seed'])
        journal = list(records(SOURCE / 'training-metrics.jsonl'))
        if (len(schedule) != 35284 or len(journal) != len(schedule) or journal != core['metrics']
                or schedule.phase_bounds != plan['phase_bounds']):
            raise ValueError('Every actual retained source/update record must match the private cursor and original schedule')
        seen = {name: set() for name in data.lengths}; budget = {name: Counter() for name in data.lengths}
        clocks = Counter(); phase_updates = Counter()
        for step, record in enumerate(journal):
            name, positions = schedule.at(step); phase = schedule.phase_for_step[step]
            begin, end = schedule.phase_bounds[phase]; kind = data.kinds[name]
            targets = sum(data.targets[name][p] for p in positions); contexts = sum(data.lengths[name][p] for p in positions)
            active = ['language'] if kind == 'text' else ['vision'] + (['word'] if kind == 'word' else [])
            if (record['step'] != step + 1 or record['phase'] != phase
                    or record['phase_step'] != step - begin + 1 or record['phase_planned_steps'] != end - begin
                    or record['family'] != name or record['global_original_positions'] != positions
                    or record['global_examples'] != len(positions) or record['complete_context_tokens'] != contexts
                    or record['response_EOS_targets'] != targets or record['actual_active_groups'] != active
                    or record['active_parameter_tensors'] != sum({'language': 317, 'vision': 112, 'word': 16}[g] for g in active)
                    or seen[name].intersection(positions) or len(set(positions)) != len(positions)):
                raise ValueError('An actual full source position, target, active partition or phase cursor differs')
            seen[name].update(positions); clocks.update(active); phase_updates[phase] += 1
            budget[name].update(examples=len(positions), complete_context_tokens=contexts, response_EOS_targets=targets)
            reference = phase == 'instruction_and_vision'
            if (record['reference_forward_queried'] != (reference and kind == 'text')
                    or record['resident_training_only_reference_parameters'] != (984399690 if reference else 0)
                    or record['reference_KL_weight'] != (options['reference_KL_weight'] if reference else 0.)):
                raise ValueError('Every actual own-reference query and phase decision must agree')
            if any(not math.isfinite(record[k]) for k in ['CE_sum', 'CTC_word_sum', 'full_context_reference_KL_sum', 'gradient_norm_before_clip']):
                raise ValueError('Every actual loss and gradient norm must be finite')
            if record['token_mean_CE'] != record['CE_sum'] / targets:
                raise ValueError('Actual whole target normalization differs')
            if record['words'] != (len(positions) if kind == 'word' else 0):
                raise ValueError('Every actual complete word count differs')
            expected_CTC = record['CTC_word_sum'] / len(positions) if kind == 'word' else None
            expected_KL = record['full_context_reference_KL_sum'] / (contexts - len(positions)) if reference and kind == 'text' else None
            if record['mean_byte_normalized_CTC'] != expected_CTC or record['full_context_reference_KL_mean'] != expected_KL:
                raise ValueError('Actual word or complete-context own-reference normalization differs')
            multiplier = factor(step - begin, end - begin, options['warmup_fraction'], options['cosine_floor'])
            if any(record['learning_rates'][g] != options['phase_learning_rates'][phase][g] * multiplier
                   for g in ['language', 'vision', 'word']):
                raise ValueError('An actual phase-local learning rate differs')
        for name in seen:
            expected = plan['complete_source_budget'][name]
            if seen[name] != set(range(len(data.lengths[name]))) or any(budget[name][k] != expected[k] for k in budget[name]):
                raise ValueError('Every selected original family record must occur exactly once with its whole budget')
        before, after, private = tensors(initial['model']), tensors(payload), tensors(core['model']['model'])
        if (before.keys() != after.keys() or after.keys() != private.keys()
                or len(after) != 461 or sum(p.numel() for p in after.values()) != 984399690):
            raise ValueError('Every actual full-size unique tensor and value must be present')
        if payload['network']['embedding.weight'].data_ptr() != payload['network']['lm_head.weight'].data_ptr():
            raise ValueError('The stored vocabulary and language head must be one actual tied table')
        groups = {name: [] for name in ['language', 'vision', 'word']}; changed = []; fixed = []
        for name, value in after.items():
            group = ('vision' if name.startswith('network.vision.') else
                     'language' if name.startswith('network.') and not name.startswith('network.value_head.') else
                     'word' if name.startswith('heads.word.') else None)
            if (value.dtype != torch.float32 or value.shape != before[name].shape or value.dtype != before[name].dtype
                    or private[name].dtype != value.dtype
                    or not bool(torch.isfinite(value).all()) or not torch.equal(value, private[name])):
                raise ValueError('Every final full native value must be finite and exactly equal the stored private model')
            if group:
                groups[group].append(name)
                if not torch.equal(value, before[name]): changed.append(name)
            else:
                if not torch.equal(value, before[name]):
                    raise ValueError('An unused auxiliary or critic tensor changed')
                fixed.append(name)
        optimizer = core['optimizer']; keys = set(); adam_count = 0
        if len(optimizer['param_groups']) != 3 or {k: len(v) for k, v in groups.items()} != dict(language=317, vision=112, word=16):
            raise ValueError('The exact317/112/16full parameter partitions must remain')
        for actual, name in zip(optimizer['param_groups'], ['language', 'vision', 'word']):
            if (actual['name'] != name or len(actual['params']) != len(groups[name])
                    or tuple(actual['betas']) != tuple(options['betas']) or actual['weight_decay'] != options['weight_decay']
                    or actual['lr'] != journal[-1]['learning_rates'][name]):
                raise ValueError('Every original active optimizer partition and hyperparameter must agree')
            for key, parameter in zip(actual['params'], groups[name]):
                moment = optimizer['state'][key]; keys.add(key); adam_count += 1
                if (set(moment) != {'step', 'exp_avg', 'exp_avg_sq'} or moment['step'].numel() != 1
                        or float(moment['step']) != clocks[name]
                        or any(moment[k].shape != after[parameter].shape or moment[k].dtype != torch.float32
                               or not bool(torch.isfinite(moment[k]).all())
                               for k in ['exp_avg', 'exp_avg_sq'])
                        or not bool((moment['exp_avg_sq'] >= 0).all())):
                    raise ValueError('Every actual full Adam moment shape, finite value and source-active clock must agree')
        if adam_count != 445 or set(optimizer['state']) != keys:
            raise ValueError('All445and only selected parameter tensors need complete optimizer states')
        binding = state['reference_binding']
        if binding != done['own_reference_binding'] or binding != json.loads((SOURCE / 'train-reference-binding.json').read_text()):
            raise ValueError('The immutable own foundation binding must agree at every final endpoint')
        ref_path = Path(binding['path']); reference = torch.load(ref_path, map_location='cpu', weights_only=True, mmap=True)
        if file_hash(ref_path) != binding['checkpoint_sha256'] or reference['completed_updates'] != 12853:
            raise ValueError('The exact own12853foundation reference is required')
        reference_digest = hashlib.sha256()
        for name, value in tensors(reference['model']).items():
            reference_digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
            reference_digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        if reference_digest.hexdigest() != binding['parameter_sha256']:
            raise ValueError('Every actual retained own foundation parameter must match its immutable reference digest')
        rng_rows = []
        if len(core['rngs']) != 4:
            raise ValueError('All four actual private training RNG states are required')
        for rank, state_rng in enumerate(core['rngs']):
            random.Random().setstate(state_rng['python']); n = state_rng['numpy']
            np.random.RandomState().set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
            torch.Generator(device='cpu').set_state(state_rng['torch'])
            cuda = state_rng['cuda']
            if cuda.dtype != torch.uint8 or list(cuda.shape) != [16]:
                raise ValueError('Each stored private CUDA Philox state must retain its actual byte shape')
            rng_rows.append(dict(rank=rank, CPU_torch_state_bytes=state_rng['torch'].numel(), CUDA_state_bytes=cuda.numel(),
                                 CUDA_generator_not_instantiated=True))
        if torch.cuda.is_initialized():
            raise ValueError('This static reader must never initialize CUDA')
        guard.check(full=True)
        main_guard.check(full=False)
        if any(file_hash(p) != sha for p, sha in snapshot_bindings.items()):
            raise ValueError('Every now-terminal original training artifact must stay unchanged during readback')
        atomic_json(OUT / 'complete.json', dict(
            at=datetime.now(timezone.utc).isoformat(), complete=True, actual_original_updates=len(journal),
            actual_phase_updates=dict(phase_updates), actual_families=len(seen), actual_source_budget={k: dict(v) for k, v in budget.items()},
            every_actual_logged_source_position_and_complete_denominator_reconciled=True,
            all_selected_family_records_once_not_all_unique_original_examples_once=True,
            actual_declared_original_exercise_repetitions=4, actual_unique_student_values=984399690,
            actual_unique_full_tensors=461, actual_changed_selected_tensors=len(changed), actual_unused_fixed_tensors=len(fixed),
            all_native_parameters_exactly_equal_private_full_model=True, actual_full_Adam_states=adam_count,
            every_active_Adam_clock_matches_actual_family_updates=True, actual_active_group_update_counts=dict(clocks),
            actual_private_RNG_shapes_and_CPU_parse=rng_rows, every_own_reference_parameter_digest_matches=True,
            no_new_model_forward_backward_optimizer_replay_or_CUDA_generator_certificate=True,
            does_not_prove_unarchived_per_step_input_tensors_or_held_capability=True,
            source_files_sha256={**plan['source_files_sha256'], **files, **snapshot_bindings, str(ref_path): binding['checkpoint_sha256']},
            original_native_assessments_and_all_queues_unchanged=True, models_constructed=0, model_updates=0,
            candidate_promoted=False, goal_achieved=False,
        ))
        print(json.dumps(dict(complete=str(OUT / 'complete.json'), original_updates=len(journal),
            unique_student_values=984399690, changed_selected_tensors=len(changed), actual_full_Adam_states=adam_count,
            goal_achieved=False)), flush=True)
    except BaseException as error:
        atomic_json(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            original_training_and_native_assessments_untouched=True, models_constructed=0, model_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
