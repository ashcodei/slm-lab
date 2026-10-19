"""Full original joint recipe after the exact preceding experiments finish.

No neural imports, resource acquisition or partial predecessor adoption.
Readiness closes source evidence only; the owned parent repeats actual
physical funding, exclusive lease and four-L40 capacity/cold checks.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .completed_joint_initializer_admission_v1 import completed_source
from .complete_joint_storage_budget_v1 import budget
from .exact_waiting_process_retirement_v1 import terminal
from .full_experiment_launch_queue_v1 import completion
from .full_experiment_launch_queue_v2 import STAGES

FORMAT = 'own-full-original-joint-owned-source-recipe-v1'
OPTIONS = dict(maximum_examples=4, maximum_padded_tokens=16384, gradient_bucket_values=1024**2,
    gradient_clip=1., human_weight=.2, visual_weight=.3, CTC_weight=1., native_word_CE_weight=1.,
    matching_weight=.1, temperature=.07, correction_weight=.1, DPO_beta=.1,
    chosen_CE_weight=.1, project_conflicts=True, precision='bf16')
OPTIMIZER = dict(lr=3e-5, betas=[.9, .95], eps=1e-8, weight_decay=.01, foreach=False, fused=False)
ACTOR = dict(rank=16, seed=997331, parameter_limit=10**9)
SUITES = ('code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words')
FULL = dict(updates=5362, complete_native_context_tokens=91100747, native_response_EOS_targets=34500161,
    separate_CTC_UTF8_bytes=644063, code_presentations=84084, whole_human_pairs=85786, visual_views=302627,
    full_QA=16961, full_AI2D=3088, AI2D_conditions=3, full_MBPP=600, held_human_pairs=4642,
    held_words=3429, word_conditions=3, reserved_reasoning=289)


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def merge(files, incoming):
    for path, digest in incoming.items():
        if path in files and files[path] != digest: raise ValueError('Exact original and new immutable source bindings disagree')
        files[path] = digest


def prepare(root, *, queue, expected_queue_owner, initializer, measurement, runtime_sources):
    root, queue, initializer, measurement = map(lambda p: Path(p).resolve(), [root, queue, initializer, measurement])
    queued = read(queue / 'plan.json')
    if queued['owner'] != expected_queue_owner or queued['declared_stages'] != list(STAGES):
        raise ValueError('The exact original recovered four-stage observer and every full experiment must remain')
    if not terminal(expected_queue_owner): return None
    if (queue / 'failure.json').exists() or not (queue / 'complete.json').is_file():
        raise ValueError('The preceding exact full experiment queue stopped without successful completion')
    done = read(queue / 'complete.json'); rows = done.get('all_original_experiments', [])
    if (done.get('complete') is not True or done.get('all_original_training_and_evaluation_populations_retained') is not True
            or [r['stage'] for r in rows] != [s['name'] for s in STAGES]):
        raise ValueError('All complete original visual,mixed-SFT,reasoning,native-code experiments and assessments must remain')
    files = dict(runtime_sources); merge(files, queued['source_files_sha256'])
    for spec, row in zip(STAGES, rows):
        if completion(root, spec, row['owner']) != row:
            raise ValueError('Every original owned completion and terminal registered actor must independently reconcile')
        merge(files, row['original_complete_receipts_sha256'])
    own = completed_source(initializer)
    if own is None: return None
    merge(files, own['source_files_sha256'])
    size_plan, measured = read(measurement / 'plan.json'), read(measurement / 'complete.json')
    if not terminal(size_plan['owner']) or (measurement / 'failure.json').exists():
        raise ValueError('The actual full selected-state serialization measurement must finish successfully')
    funded = budget(measured); merge(files, size_plan['source_files_sha256'])
    native_plan_path = root / 'reports' / STAGES[-1]['output'] / 'plan.json'
    native = read(native_plan_path)
    for key in ['cold_inputs', 'cached_own_reference', 'pulse_sources']:
        if not native.get(key): raise ValueError('Every original full cold and assessment source must remain')
    if (sha(native['cold_inputs']['path']) != native['cold_inputs']['sha256']
            or sha(native['cached_own_reference']['report']) != native['cached_own_reference']['report_sha256']):
        raise ValueError('The actual complete original cold inputs and held own reference must retain their committed identities')
    capacity_folder = root / 'reports/complete-joint-actual-capacity-cases-CPU-v3'
    capacity_plan = read(capacity_folder / 'plan.json'); capacity_done = read(capacity_folder / 'complete.json')
    capacity_path = capacity_folder / 'actual-complete-joint-capacity-plan.json'
    capacity = read(capacity_path)
    if (not terminal(capacity_plan['owner']) or (capacity_folder / 'failure.json').exists()
            or capacity_done.get('complete') is not True or capacity_done.get('actual_complete_capacity_cases') != 47
            or capacity_done.get('actual_original_joint_steps') != FULL['updates']
            or sha(capacity_path) != capacity_done.get('capacity_plan_sha256')
            or capacity.get('complete_positive_and_negative_context_tokens') != FULL['complete_native_context_tokens']
            or capacity.get('complete_positive_and_negative_response_EOS_targets') != FULL['native_response_EOS_targets']):
        raise ValueError('All actual corrected full joint sources,47capacity cases and original source budgets must remain')
    merge(files, capacity_plan['source_files_sha256'])
    paths = [queue / 'plan.json', queue / 'complete.json', measurement / 'plan.json', measurement / 'complete.json',
        native_plan_path, capacity_folder / 'plan.json', capacity_folder / 'complete.json', capacity_path,
        Path(native['cold_inputs']['path']), Path(native['cached_own_reference']['report']),
        *[Path(p) for p in native['pulse_sources'].values()]]
    merge(files, {str(p): sha(p) for p in paths})
    for p, h in files.items():
        if sha(p) != h: raise ValueError('Every actual complete original source,proof and new runtime byte must agree')
    return dict(format=FORMAT, complete_source_recipe=True, initializer=own['initializer'], source_files_sha256=files,
        exact_original_queue_owner=expected_queue_owner, original_queue_complete_sha256=sha(queue / 'complete.json'),
        storage_budget=funded, full_original_budget=deepcopy(FULL), options=deepcopy(OPTIONS), optimizer=deepcopy(OPTIMIZER),
        actor_settings=deepcopy(ACTOR), checkpoint_every=128, private_seed=997340, suites=list(SUITES),
        expected_native_values=984399690, expected_research_values=999706442, expected_selected_values=68123713,
        expected_selected_parameter_leaves=690, capacity_plan=str(capacity_path), capacity_plan_sha256=sha(capacity_path),
        cold_inputs=deepcopy(native['cold_inputs']), cached_own_reference=deepcopy(native['cached_own_reference']),
        pulse_sources=deepcopy(native['pulse_sources']), original_nine_endpoint_pipeline_preserved=True,
        every_original_baseline_and_final_assessment_required=True, no_automatic_retry_or_promotion=True,
        maximum_examples4_requires_actual_full_GPU_capacity_and_does_not_establish_speedup=True,
        GPU_training_admitted=False, goal_achieved=False)
