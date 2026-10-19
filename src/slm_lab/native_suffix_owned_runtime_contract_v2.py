"""Preserve whole native data, prior full stages and all original assessments.

Passing this read-only contract grants no GPU lease or capacity admission.
The owned native driver must still perform actual four-L40 workload, fixed
full-horizon stress, independent fresh readback and complete training/export.
"""
import hashlib
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal

FULL = dict(updates=11172, source_presentations=44652, distinct_original_human_sources=3721,
    complete_original_context_tokens=27450264, original_response_EOS_targets=7010616,
    actual_single_source_three_empty_rank_steps=12)
PRIOR = (
    ('joint-visual-aligned-continuation-GPU-v2', 'all4733updates302627views4669243targets_complete',
     'all_full_QA_AI2D_MBPP_human_words_code_broad_candidate_suites_complete'),
    ('mixed-coding-SFT-aligned-GPU-v1', 'all84084presentations21316734targets5256updates_per_profile',
     'all_original_full_QA_AI2D_MBPP_human_words_code_broad_suites_each_candidate_complete'),
    ('batched-reasoning-aligned-GPU-v2', 'all7704presentations57240targets1926updates_per_profile',
     'all_original_full_QA_AI2D_MBPP_human_words_code_broad_suites_each_candidate_complete'),
)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def require_complete_preparation(root, prepared, *, options, prefill, maximum_state_bytes,
                                 maximum_journal_raw_bytes, maximum_assessment_bytes, reserve_bytes):
    root = Path(root).resolve()
    if (prepared.get('complete') is not True or prepared.get('full_native_budget') is None
            or any(prepared['full_native_budget'].get(k) != v for k, v in FULL.items())
            or prepared.get('native_stage_options') != options or prepared.get('native_prefill') != prefill
            or options.get('epochs') != 3 or options.get('seed') != 989711 or options.get('group_size') != 8
            or options.get('CPU_mechanics_horizon') is not None or options.get('execution_workers') != 2
            or options.get('maximum_cache_bytes', 0) < 2 * 8 * 35 * 8 * 64 * 4095 * 4
            or prefill != dict(protocol='own-native-complete-text-causal-chunked-prefill-FP32-v1', chunk_tokens=128)
            or prepared.get('original_canonical_visual_mixed_reasoning_scopes_preserved') is not True
            or prepared.get('GPU_work_started') is not False):
        raise ValueError('All original native sources, three epochs, eight whole histories and prior full scopes must remain')
    minimum = (reserve_bytes + 9 * maximum_state_bytes + 2 * prepared['maximum_native_bytes']
        + 2 * maximum_assessment_bytes + 2 * maximum_journal_raw_bytes + 64 * 1024**2)
    if prepared.get('required_free_bytes_before_launch', 0) < minimum:
        raise ValueError('Both complete source histories, atomic generations, native candidates, assessments and reserve must be funded')
    files = prepared.get('source_files_sha256', {})
    native_files = prepared.get('native_source_files_sha256', {})
    if not files or not native_files or any(files.get(p) != h for p, h in native_files.items()):
        raise ValueError('Every exact original native data byte must also belong to the closed full runtime source recipe')
    for name, scope, assessment in PRIOR:
        folder = root / 'reports' / name
        required = [folder / n for n in ['plan.json', 'complete.json', 'restored.json', 'children.json']]
        if (not all(p.is_file() and not p.is_symlink() for p in required) or (folder / 'failure.json').exists()
                or any(files.get(str(p)) != sha(p) for p in required)):
            raise ValueError('Every original full visual/mixed/reasoning stage must finish and remain source-bound')
        plan, done, restored, children = [read(p) for p in required]
        if (not terminal(plan['owner']) or any(not terminal(c) for c in children)
                or done.get('complete') is not True or done.get(scope) is not True or done.get(assessment) is not True
                or done.get('original_nine_endpoint_pipeline_preserved') is not True
                or restored.get('all_registered_point_actors_terminal') is not True):
            raise ValueError('All actual original source training and full endpoint populations must finish before native continuation')
    selected = root / 'reports/native-human-code-suffix-actual-capacity-CPU-v2/actual-complete-source-capacity-plan.json'
    if not selected.is_file() or files.get(str(selected)) != sha(selected):
        raise ValueError('The complete actual admitted native source selection is required')
    capacity = read(selected)
    if (capacity.get('geometry_only') is not False or capacity.get('full_source_data_admitted') is not True
            or capacity.get('original_human_sources') != 3721 or capacity.get('original_source_views') != 14884
            or capacity.get('epochs') != 3 or capacity.get('seed') != 989711 or not capacity.get('capacity_entries')
            or any(capacity.get('complete_training_budget', {}).get(k) != v for k, v in FULL.items())
            or capacity.get('all_12_real_single_source_three_empty_rank_training_tails_retained') is not True):
        raise ValueError('Geometry-only or partial native source selection cannot authorize GPU work')
    spec = prepared['initializer']
    fixed = root / 'checkpoints/batched-reasoning-aligned-GPU-v2/anchor_RL/native/model.pt'
    if Path(spec['checkpoint']).resolve() != fixed or files.get(str(fixed)) != spec.get('checkpoint_sha256'):
        raise ValueError('The own completed reasoning anchor_RL initializer is fixed before future candidate quality is observed')
    for p, h in files.items():
        if sha(p) != h: raise ValueError('Every tested source, full data input and completed predecessor byte must retain its exact hash')
    return True
