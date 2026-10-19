"""Assemble a source-only native continuation from completed full experiments.

No neural module, model or GPU ownership is created here. The fixed own
reasoning RL endpoint supplies both the initializer and complete cached native
baseline; its selection never depends on future native-code quality scores.
"""
import hashlib
import json
from pathlib import Path

from .exact_waiting_process_retirement_v1 import terminal
from .native_suffix_owned_runtime_contract_v2 import FULL, PRIOR, require_complete_preparation

SUITES = ('code', 'broad', 'preferences', 'QA', 'AI2D', 'MBPP', 'words')


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def merge(files, incoming):
    for name, expected in incoming.items():
        name = str(Path(name).resolve())
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected)
                or name in files and files[name] != expected):
            raise ValueError('Every original immutable source binding must agree')
        files[name] = expected


def immutable(path):
    name = Path(path).name
    return not ('progress' in name or 'queue' in name or name in ['children.json', 'LATEST.json']
        or name.endswith(('.pending', '.lock', '.log')))


def completed_stages(root, captured):
    """Return closed stage metadata, or None while an exact owner is live/absent.

    Captures each future immutable plan as soon as it actually appears. A
    missing future plan is never labelled a live process or completed work.
    """
    root = Path(root).resolve()
    results = {}
    for name, scope, assessment in PRIOR:
        folder = root / 'reports' / name
        path = folder / 'plan.json'
        if not path.exists():
            if name in captured or (folder / 'failure.json').exists():
                raise ValueError('A captured stage plan disappeared or an unplanned stage failed')
            return None
        if path.is_symlink() or not path.is_file(): raise ValueError('An ordinary complete original stage plan is required')
        plan = read(path)
        identity = dict(owner=plan['owner'], plan_sha256=sha(path))
        if name in captured and captured[name] != identity:
            raise ValueError('An originally observed experiment owner or source recipe changed')
        captured[name] = identity
        if not terminal(plan['owner']): return None
        children = folder / 'children.json'
        if not children.is_file(): raise ValueError('The complete original stage child registry must survive')
        if any(not terminal(c) for c in read(children)): return None
        required = [folder / n for n in ['complete.json', 'restored.json']]
        if (not all(p.is_file() and not p.is_symlink() for p in required) or (folder / 'failure.json').exists()):
            raise ValueError('An exact full prior stage stopped without successful complete evidence')
        done, restored = [read(p) for p in required]
        if (done.get('complete') is not True or done.get(scope) is not True or done.get(assessment) is not True
                or done.get('original_nine_endpoint_pipeline_preserved') is not True
                or restored.get('all_registered_point_actors_terminal') is not True
                or done.get('native_values' if name.startswith('joint-visual') else 'actual_native_values') != 984399690):
            raise ValueError('Every actual full visual/mixed/reasoning scope and restored worker must remain')
        results[name] = dict(plan=plan, complete=done, folder=folder)
    return results


def full_baseline(root, checkpoint_sha256):
    """Inspect full original baseline populations, image conditions and receipts."""
    folder = Path(root).resolve() / 'reports/batched-reasoning-aligned-GPU-v2/anchor_RL/assessment'
    baseline = folder / 'standalone'
    for name, count in [('full-QA.json', 16961), ('full-AI2D.json', 3088),
            ('full-original-MBPP.json', 600), ('held-preferences.json', 4642)]:
        report = read(baseline / name)
        if len(report.get('results', [])) != count:
            raise ValueError('Every original full baseline QA/diagram/program/human population must remain')
        if name == 'full-AI2D.json' and any(set(r.get('conditions', {})) != {'proper_image', 'gray_image', 'wrong_image'} for r in report['results']):
            raise ValueError('Every baseline diagram requires all three original image conditions')
    word = read(baseline / 'all-held-words.json')
    if (len(word.get('results', [])) != 3429 or word.get('generation_budget') != 64
            or word.get('complete_held_source_words') is not True or word.get('held_source_gradients') is not False
            or [r.get('source_position') for r in word['results']] != list(range(3429))
            or any(set(r.get('conditions', {})) != {'proper', 'blank', 'wrong'} for r in word['results'])):
        raise ValueError('Every original held word, full action budget and all three image controls must remain')
    for frame in ['completion', 'instruction']:
        code = read(baseline / ('code-' + frame + '.json'))
        if code.get('task_count') != 24 or len(code.get('results', [])) != 24:
            raise ValueError('Every original native Python/JavaScript baseline case must remain')
    for name in ['knowledge', 'math', 'ocr', 'objects', 'vision']:
        if not (baseline / (name + '.json')).is_file():
            raise ValueError('Every original broad native baseline must remain')
    for suite in SUITES:
        for rank in range(4):
            proof = read(baseline / f'assessment-{suite}-{rank}-complete.json')
            if (proof.get('physical_GPU') != rank or proof.get('actual_student_values') != 984399690
                    or proof.get('checkpoint_sha256') != checkpoint_sha256 or proof.get('model_updates') != 0
                    or any(proof.get(k) is not True for k in ['all_loaded_values_unchanged',
                        'only_standalone_weights_and_tokenizer_for_neural_inference'])):
                raise ValueError('All28 actual full native baseline worker receipts must match the fixed initializer')
    reasoning = read(folder / 'all-reserved-reasoning.json')
    if (reasoning.get('complete') is not True or reasoning.get('examples') != 289
            or len(reasoning.get('results', [])) != 289
            or [r.get('position') for r in reasoning['results']] != list(range(289))
            or reasoning.get('checkpoint_sha256') != checkpoint_sha256
            or reasoning.get('optimization_forbidden') is not True
            or reasoning.get('all_original_whole_prompts_and_full64or256budgets_retained') is not True):
        raise ValueError('All289 original reserved reasoning cases and full budgets must remain')
    return baseline, {str(p.resolve()): sha(p) for p in folder.rglob('*') if p.is_file() and immutable(p)}


def assemble(root, original, capacity, stages, *, owner, source_files, driver):
    """Build a closed full recipe; storage and actual GPU admission stay separate."""
    root = Path(root).resolve()
    if set(stages) != {row[0] for row in PRIOR} or tuple(driver.SUITES) != SUITES:
        raise ValueError('All actual original stages and assessment suites are required')
    if (capacity.get('complete') is not True or capacity.get('source_selection_only') is not True
            or any(capacity.get('complete_training_budget', {}).get(k) != v for k, v in FULL.items())
            or any(capacity.get(k) != 0 for k in ['models_constructed', 'neural_forwards', 'GPU_contexts', 'GPU_leases', 'optimizer_updates'])
            or capacity.get('full_size_GPUcapacity_passed') is not False
            or capacity.get('production_training_admitted') is not False
            or not capacity.get('actual_fixed_horizon_source_positions')
            or not 0 < capacity.get('predicted_full_size_selected_logical_KV_bytes', 0) <= driver.STAGE_OPTIONS['maximum_cache_bytes']
            or capacity.get('declared_cache_bound_bytes') != driver.STAGE_OPTIONS['maximum_cache_bytes']):
        raise ValueError('Only complete actual original native-source selection, with no runtime admission, can be prepared')
    files = {}
    merge(files, source_files)
    merge(files, {p: h for p, h in original['source_files_sha256'].items() if immutable(p)})
    merge(files, capacity['source_files_sha256'])
    for name, value in stages.items():
        folder = root / 'reports' / name
        merge(files, {str(folder / n): sha(folder / n) for n in ['plan.json', 'complete.json', 'children.json', 'restored.json']})
    export_path = root / 'reports/batched-reasoning-aligned-GPU-v2/anchor_RL/native-export.json'
    spec = read(export_path)
    checkpoint = root / 'checkpoints/batched-reasoning-aligned-GPU-v2/anchor_RL/native/model.pt'
    if (spec.get('complete') is not True or spec.get('profile') != 'anchor_RL' or spec.get('completed_updates') != 1926
            or Path(spec.get('checkpoint', '')).resolve() != checkpoint or sha(checkpoint) != spec.get('checkpoint_sha256')
            or spec.get('actual_native_values') != 984399690
            or spec.get('every_native_value_and_complete_text_photo_word_cold_exact') is not True
            or spec.get('provenance', {}).get('training_profile') != 'anchor_RL'
            or spec.get('provenance', {}).get('whole_production_source_schedule_completed') is not True):
        raise ValueError('The fixed completed own reasoning anchor_RL native and full cold proof must remain')
    tokenizer = checkpoint.parent / 'tokenizer.json'
    if sha(tokenizer) != spec.get('tokenizer_sha256') or spec['tokenizer_sha256'] != original['initializer']['tokenizer_sha256']:
        raise ValueError('Every original source and fixed native initializer must retain the same own tokenizer')
    cold = original['cold_inputs']
    if sha(cold['path']) != cold['sha256']: raise ValueError('The complete original text/photo/word cold input bundle changed')
    baseline, baseline_files = full_baseline(root, spec['checkpoint_sha256'])
    merge(files, baseline_files)
    merge(files, {str(p): sha(p) for p in [export_path, checkpoint, tokenizer, Path(cold['path'])]})
    maximum_native = max(4 * 1024**3, checkpoint.stat().st_size + 64 * 1024**2)
    budget = (driver.RESERVE + 9 * driver.MAX_STATE + 2 * maximum_native
        + 2 * driver.MAX_ASSESSMENTS + 2 * driver.MAX_JOURNAL + 64 * 1024**2)
    prepared = dict(complete=True, owner=owner, source_files_sha256=files,
        exact_predecessors=original['exact_predecessors'],
        initializer={k: spec[k] for k in ['checkpoint', 'checkpoint_sha256', 'sources', 'tokenizer_sha256']},
        cold_inputs=cold, maximum_native_bytes=maximum_native,
        cached_baseline=dict(folder=str(baseline), same_full_protocol=True, exact_fixed_native_reused_not_score_selected=True),
        cached_own_reference=original['cached_own_reference'], pulse_sources=original['pulse_sources'],
        native_source_files_sha256=dict(capacity['source_files_sha256']), full_native_budget=capacity['complete_training_budget'],
        native_stage_options=dict(driver.STAGE_OPTIONS), native_prefill=dict(driver.NATIVE_PREFILL),
        required_free_bytes_before_launch=budget, original_canonical_visual_mixed_reasoning_scopes_preserved=True,
        GPU_work_started=False, models_constructed=0, neural_forwards=0, GPU_contexts=0, GPU_leases=0,
        production_updates=0, later_fresh_normal_nice0_owned_launch_required=True,
        full_size_GPUcapacity_training_export_and_quality_still_required=True, candidate_promoted=False, goal_achieved=False)
    require_complete_preparation(root, prepared, options=driver.STAGE_OPTIONS, prefill=driver.NATIVE_PREFILL,
        maximum_state_bytes=driver.MAX_STATE, maximum_journal_raw_bytes=driver.MAX_JOURNAL,
        maximum_assessment_bytes=driver.MAX_ASSESSMENTS, reserve_bytes=driver.RESERVE)
    return prepared
