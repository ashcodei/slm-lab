"""Independently derive a complete native from the actual joint private state.

This CPU path does not read an exported candidate's weights. A later GPU
cold check can compare this independently rebuilt native to that candidate.
"""
from .joint_selected_native_state_v1 import install
from .joint_native_fold_v1 import fold


def reconstruct(view, optimizer, directory, *, sources, entries, source_files, expected_world, expected_completed,
                expected_pointer_sha256, mode, working_budget_bytes, allow_mechanics=False):
    restored = install(view, optimizer, directory, sources=sources, entries=entries, source_files=source_files,
        expected_world=expected_world, expected_completed=expected_completed, expected_pointer_sha256=expected_pointer_sha256,
        mode=mode, allow_mechanics=allow_mechanics)
    native, folded = fold(view.actor, sources=sources, working_budget_bytes=working_budget_bytes)
    return native, dict(actual_reconstruction=restored, actual_fold=folded,
        exported_candidate_weights_not_loaded_for_source_native=True,
        training_or_GPU_or_quality_admission=False)
