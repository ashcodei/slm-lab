"""Actual visual capacity replay and complete-source durable experiment loops.

These helpers do not acquire resources, admit data, choose a candidate, export
a final model or replace an owned controller. Small mechanics calls are
explicitly marked and never count as full-size training/capacity evidence.
"""
import copy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .codenet_checkpoint_v1 import capture_rng
from .joint_visual_durable_state_v2 import read_latest
from .joint_visual_objective_update_v1 import FAMILIES, named_groups
from .joint_visual_private_checkpoint_v2 import _agree, private_rng_sha256, restore_state
from .joint_visual_training_stage_v1 import JointVisualStage
from .parallel_function_repair_stage_v1 import value_digest
from .whole_human_sft_evidence_v1 import tensor_digest


def private_draws(device):
    values = [random.random(), float(np.random.rand()), float(torch.rand(()))]
    if torch.device(device).type == 'cuda':
        values.append(float(torch.rand((), device=device)))
    return values


def live(stage):
    stage.fixed.check(stage.view, stage.optimizer)
    return dict(selected_values=tensor_digest((n, p) for rows in named_groups(stage.view).values() for n, p in rows),
        optimizer=value_digest(stage.optimizer.state_dict()), completed=stage.completed,
        metrics=value_digest(stage.metrics), RNG=private_rng_sha256(capture_rng(stage.device)), failed=stage.failed)


def prove_next_update(stage, directory, *, maximum_bytes, minimum_free_bytes):
    if (type(stage) is not JointVisualStage or stage.mode not in ['mechanics', 'capacity']
            or stage.completed or len(stage._entries) < 3
            or [FAMILIES[e['family']] for e in stage._entries[:3]] != ['native', 'photo', 'word']):
        raise ValueError('A fresh actual native/photo/word capacity prefix is required')
    first = stage.next_update()
    stage.commit(directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
    shared, receipt, pointer_sha = read_latest(directory)
    expected_draws = private_draws(stage.device)
    second = stage.next_update()
    before = live(stage)
    common = shared['common']
    private = shared['private_states'][stage.rank]
    retained_RNG_sha = receipt['semantic_manifest']['all_private_rank_boundaries'][stage.rank]['private_RNG_sha256']
    controls = ['source', 'missing_active_moment', 'nonfinite_saved_selected_value']
    if stage.world > 1:
        controls.append('valid_foreign_rank_RNG')
    for control in controls:
        bad = {**common, **private}
        if stage.rank == stage.world - 1:
            if control == 'source':
                bad['sources'] = dict(common['sources'], source_admission_sha256='b' * 64)
            elif control == 'missing_active_moment':
                moments = dict(common['optimizer']['state'])
                del moments[next(iter(moments))]
                bad['optimizer'] = dict(common['optimizer'], state=moments)
            elif control == 'nonfinite_saved_selected_value':
                values = dict(common['selected_weights'])
                name = next(iter(values))
                values[name] = values[name].clone()
                values[name].reshape(-1)[0] = float('nan')
                bad['selected_weights'] = values
            else:
                peer = shared['private_states'][(stage.rank + 1) % stage.world]
                bad['private_RNG'] = peer['private_RNG']
                bad['private_RNG_sha256'] = peer['private_RNG_sha256']
        error = None
        try:
            restore_state(stage.view, stage.optimizer, stage.fixed, bad, sources=stage._sources,
                entries=stage._entries, mode=stage.private_mode, rank=stage.rank, world=stage.world,
                device=stage.device, control_group=stage.group, expected_completed=1,
                expected_private_RNG_sha256=retained_RNG_sha)
        except ValueError:
            pass
        else:
            error = 'A deliberately invalid selected/source/Adam/private-RNG state was accepted'
        if live(stage) != before:
            error = 'Rejected recovery changed actual selected values, moments, cursor, metrics or private RNG'
        _agree(error, None, stage.group, stage.world)
    stage.restore(directory, expected_pointer_sha256=pointer_sha)
    actual_draws = private_draws(stage.device)
    actual_second = stage.next_update()
    error = None
    if actual_draws != expected_draws or actual_second != second or live(stage) != before:
        error = 'Actual next-source weights/Adam/metrics/private RNG did not replay bit exactly'
    _agree(error, None, stage.group, stage.world)
    return dict(first_actual_record=first, replayed_second_actual_record=second,
        actual_logical_completed_source_entries=2, actual_physical_optimizer_calls=3,
        exact_next_source_weights_Adam_metrics_and_private_RNG=True,
        invalid_source_active_moment_and_nonfinite_saved_values_rejected_before_mutation=True,
        valid_foreign_rank_RNG_rejected_before_mutation=stage.world > 1,
        external_commit_pointer_sha256=pointer_sha,
        mechanics_only=stage.mode == 'mechanics', full_size_visual_capacity_not_inferred_from_this_prefix=True)


def train(stage, directory, journal, *, every, maximum_bytes, minimum_free_bytes,
          maximum_journal_bytes, preflight, progress, allow_mechanics=False):
    if (type(stage) is not JointVisualStage or stage.completed or stage.failed
            or stage.mode != 'train' and not (allow_mechanics is True and stage.mode == 'mechanics')
            or type(every) is not int or every < 1
            or type(maximum_journal_bytes) is not int or maximum_journal_bytes < 1
            or not callable(preflight) or not callable(progress)):
        raise ValueError('A fresh full train stage or explicit own-small mechanics loop is required')
    stream = None
    error = None
    if stage.rank == 0:
        try:
            stream = Path(journal).open('xb')
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
    try:
        _agree(error, None, stage.group, stage.world)
        baseline = stage.commit(directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
        if baseline['completed'] != 0:
            raise ValueError('The exact untrained selected initializer must be committed first')
        seconds, receipt = [], None
        for _ in stage._entries:
            error = None
            try:
                preflight()
            except Exception as exc:
                error = type(exc).__name__ + ': ' + str(exc)
            _agree(error, None, stage.group, stage.world)
            start = time.monotonic()
            row = stage.next_update()
            elapsed = time.monotonic() - start
            seconds.append(elapsed)
            error = None
            if stage.rank == 0:
                try:
                    encoded = (json.dumps(dict(step=stage.completed, actual_update_seconds=elapsed,
                        actual_record=row), allow_nan=False) + '\n').encode()
                    if len(encoded) > 2 * 1024**2 or stream.tell() + len(encoded) > maximum_journal_bytes:
                        raise ValueError('The whole source journal must stay inside its explicit reserved size')
                    stream.write(encoded)
                    stream.flush()
                except Exception as exc:
                    error = type(exc).__name__ + ': ' + str(exc)
            _agree(error, None, stage.group, stage.world)
            if stage.completed == 1 or stage.completed % every == 0 or stage.next_entry is None:
                error = None
                if stage.rank == 0:
                    try:
                        os.fsync(stream.fileno())
                    except Exception as exc:
                        error = type(exc).__name__ + ': ' + str(exc)
                _agree(error, None, stage.group, stage.world)
                receipt = stage.commit(directory, maximum_bytes=maximum_bytes, minimum_free_bytes=minimum_free_bytes)
            error = None
            if stage.rank == 0:
                try:
                    progress(copy.deepcopy(row), elapsed)
                except Exception as exc:
                    error = type(exc).__name__ + ': ' + str(exc)
            _agree(error, None, stage.group, stage.world)
        if receipt is None or receipt['completed'] != len(stage._entries) or stage.next_entry is not None:
            raise ValueError('Every complete original source must reach the final durable cursor')
        stage._boundary()
        return dict(actual_completed_source_entries=stage.completed, actual_update_seconds=seconds,
            actual_complete_source_presentations=sum(len(e['original_local_positions']) for e in stage._entries),
            actual_response_EOS_targets=sum(r['response_EOS_targets'] for r in stage.metrics),
            final_shared_receipt=receipt, actual_initial_selected_baseline_committed=True,
            production_updates=stage.completed if stage.mode == 'train' else 0,
            mechanics_only=stage.mode == 'mechanics', candidate_promoted=False, goal_achieved=False)
    except BaseException:
        stage.failed = True
        raise
    finally:
        if stream is not None:
            stream.close()
