"""Close the completed CPU initializer for a later owned GPU launcher.

This reader imports no neural runtime and takes no lease. Actual full-size
GPU capacity/cold, complete training and every original assessment remain
independent requirements. A live exact source owner remains a wait.
"""
import hashlib
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def completed_source(folder):
    folder = Path(folder).resolve(); plan = read(folder / 'plan.json')
    if not terminal(plan['owner']): return None
    names = ['complete.json', 'source-admission.json', 'actual-completed-human-state-readback.json']
    if (folder / 'failure.json').exists() or any(not (folder / name).is_file() or (folder / name).is_symlink() for name in names):
        raise ValueError('The exact original CPU initializer owner ended without successful complete evidence')
    done, source, state = [read(folder / name) for name in names]
    if (done.get('format') != 'own-completed-human-DPO-and-own-foundation-joint-initializer-v1'
            or done.get('complete') is not True or done.get('owner') != plan['owner']
            or done.get('actual_counts', {}).get('total_unique') != 984399690
            or done.get('actual_full_size_CPU_initializer') is not True or done.get('joint_optimizer_updates') != 0
            or done.get('GPU_contexts') != 0 or done.get('full_size_GPU_capacity_or_training_admitted') is not False
            or done.get('all_native_values_and_actual_text_photo_word_cold_outputs_exact') is not True
            or done.get('inference_needs_only_native_checkpoint_and_packaged_tokenizer') is not True
            or done.get('source_CPU_RNG_unchanged') is not True
            or done.get('full_original_baseline_and_final_assessments_still_required') is not True
            or source.get('fixed_text_arm') != 'human_DPO' or source.get('quality_selection_or_GPU_admission') is not False
            or source.get('original_full_human_and_public_source_epochs_complete') is not True
            or done.get('completed_human_state_readback') != state):
        raise ValueError('Only the actual full-size completed own fixed-source CPU initializer is eligible')
    rows = state.get('actual_rank_observations', [])
    if (state.get('all_four_final_private_states_match_exported_actor') is not True
            or state.get('actual_full_size_completed_human_state_readback') is not True
            or state.get('actual_completed_human_updates') != 5362 or state.get('original_pair_population') != 85786
            or state.get('new_Adam_calls') != 0 or state.get('GPU_training_admitted') is not False
            or [r['rank'] for r in rows] != [0, 1, 2, 3]
            or any(r['actual_completed_updates'] != 5362 or r['actual_human_pair_presentations'] != 85786
                or r['actual_Adam_states'] != 561 for r in rows)
            or len({r['actual_private_RNG_sha256'] for r in rows}) != 4
            or len({r['actual_common_actor_Adam_metrics_sha256'] for r in rows}) != 1):
        raise ValueError('All four complete original human actor/Adam/metric/private-RNG source inspections must remain')
    probes = {r['name']: r for r in done.get('actual_FP32_source_probes', [])}
    if (set(probes) != {'actual_complete_original_short_code', 'actual_complete448photo', 'actual_complete_word_native'}
            or probes['actual_complete448photo']['actual_pixel_shape'] != [1, 3, 448, 448]
            or any(not 1 <= r['complete_text_positions'] <= 4096
                or not 0 <= r['maximum_FP32_actor_to_initializer_error'] <= .002 for r in probes.values())
            or done.get('actual_word_pixel_shape', [])[-2:] != [64, 448]
            or done.get('actual_word_output_shape', [None, None])[1:] != [140, 257]
            or done.get('actual_word_dtype') != 'torch.float32'
            or done.get('actual_assembly', {}).get('every_selected_source_value_exact') is not True
            or done.get('actual_language_fold', {}).get('original_native_critic_preserved') is not True):
        raise ValueError('Every complete native value and actual original photo/word/text CPU cold probe must remain')
    checkpoint = folder / 'native/model.pt'; token = checkpoint.parent / 'tokenizer.json'
    if (Path(done.get('checkpoint', '')).resolve() != checkpoint or checkpoint.is_symlink() or token.is_symlink()
            or not checkpoint.is_file() or not token.is_file()
            or not 984399690 * 4 <= checkpoint.stat().st_size <= 5 * 1024**3):
        raise ValueError('The actual complete independently packaged native and own tokenizer must remain present')
    expected_sources = dict(completed_human_policy_sha256=source['checkpoint_sha256'], completed_human_plan_sha256=source['plan_sha256'],
        own_visual_foundation_sha256=source['source_files_sha256'][source['own_visual_foundation']],
        completed_source_admission_sha256=sha(folder / 'source-admission.json'),
        actual_four_private_state_readback_sha256=sha(folder / 'actual-completed-human-state-readback.json'))
    if done.get('sources') != expected_sources or done.get('tokenizer_sha256') != source['tokenizer_sha256']:
        raise ValueError('The exact fixed completed donors and actual independent state readback must bind the native')
    files = dict(done['source_files_sha256'])
    for path, digest in plan['source_files_sha256'].items():
        if files.get(path) != digest: raise ValueError('Every actual original initializer runtime byte must remain bound')
    for path in [folder / 'plan.json', *[folder / name for name in names]]: files[str(path)] = sha(path)
    files[str(checkpoint)] = done['checkpoint_sha256']; files[str(token)] = done['tokenizer_sha256']
    if any(sha(path) != digest for path, digest in files.items()):
        raise ValueError('A completed initializer source, private state, native or tokenizer byte changed')
    return dict(source_recipe_complete=True, original_CPU_owner=plan['owner'], source_files_sha256=files,
        initializer=dict(checkpoint=str(checkpoint), checkpoint_sha256=done['checkpoint_sha256'],
            tokenizer=str(token), tokenizer_sha256=done['tokenizer_sha256'], sources=expected_sources,
            actual_counts=done['actual_counts']), fixed_sources_not_selected_by_future_quality=True,
        original4096text_and448photo_and140frame_GPUcold_still_required=True,
        full_size_GPU_capacity_training_and_all_original_assessments_still_required=True,
        GPU_training_admitted=False, goal_achieved=False)
