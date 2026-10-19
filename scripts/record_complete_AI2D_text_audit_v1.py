"""Record complete source-text auditing without claiming model capability."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/complete-AI2D-text-audit-v1-review'


def identity(value, waiting=False):
    proc = Path('/proc') / str(value['pid']); fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
    if fields[19] != value['ticks'] or fields[0] in ['Z', 'X']: raise ValueError('Exact original live birth identity required')
    children = (proc / 'task' / str(value['pid']) / 'children').read_text().strip()
    visible = next((v.split('=', 1)[1] for v in (proc / 'environ').read_bytes().decode().split('\0') if v.startswith('CUDA_VISIBLE_DEVICES=')), None)
    if waiting and (children or visible != ''): raise ValueError('Waiting successor must remain CPU-hidden and childless')
    return dict(pid=value['pid'], ticks=fields[19], state=fields[0], children=children, CUDA_VISIBLE_DEVICES=visible)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden full-scope review required')
    prior = ROOT / 'reports/contrastive-fit-and-full-audits-v1-review/review-corrected.json'
    parent = json.loads(prior.read_text()); requirements = parent['requirements']
    text_folder = ROOT / 'reports/AI2D-full-text-exposure-audit-v1'
    reader_folder = ROOT / 'reports/AI2D-all-text-exposure-witnesses-v1-readback'
    text_path = text_folder / 'complete.json'; reader_path = reader_folder / 'complete.json'
    text = json.loads(text_path.read_text()); reader = json.loads(reader_path.read_text())
    FileBindings(reader['source_files_sha256']).check(full=True)
    signatures = reader['independent_matched_questions_by_signature']; tokens = reader['total_declared_input_token_positions_accounted_for']
    if (text['all_six_declared_scopes_completed'] is not True or text['actual_original_AI2D_questions'] != 3088
            or reader['actual_matched_source_witnesses_reconstructed'] != 23 or tokens != 3250975085
            or signatures != text['matched_questions_by_pattern'] or signatures['original_question']['any_inspected_scope'] != 16
            or any(signatures[k]['any_inspected_scope'] for k in ['question_with_ordered_options', 'whole_native_prompt'])):
        raise ValueError('Every complete original text-source scope, actual witness and signature count required')
    phase_path = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(phase_path.read_text())
    progress = json.loads((ROOT / 'reports/phase-joint-v3/progress.json').read_text())
    workers = [identity(p) for p in json.loads((ROOT / 'reports/phase-joint-v3/children.json').read_text()) if p['subject'].startswith('train-rank-')]
    if len(workers) != 4: raise ValueError('Every exact original main worker required')
    queue_names = ['phase-human-preference-v4', 'broad-anchored-code-v3', 'full-held-QA-pipeline-v1', 'full-AI2D-pipeline-v1']
    queues = {n: json.loads((ROOT / 'reports' / (n + '-queue.json')).read_text()) for n in queue_names}
    image_folder = ROOT / 'reports/AI2D-current-native-image-exposure-v1'
    image_plan_path = image_folder / 'plan.json'; image_plan = json.loads(image_plan_path.read_text())
    image_progress = json.loads((image_folder / 'progress.json').read_text())
    group_plan_path = ROOT / 'reports/AI2D-fixed-exposure-strata-v1/plan.json'; group_plan = json.loads(group_plan_path.read_text())
    code_plan_path = ROOT / 'reports/own-execution-contrast-full-CPU-v1/plan.json'; code = json.loads(code_plan_path.read_text())
    code_reader_plan_path = ROOT / 'reports/own-execution-contrast-full-CPU-v2-outcome-readback/plan.json'
    code_reader = json.loads(code_reader_plan_path.read_text())
    if (len(group_plan['fixed_cohort_names']) != 8 or group_plan['all_cohort_formulas_fixed_before_new_visual_assessment'] is not True
            or image_plan['expected_positions'] != 302627 or (image_folder / 'failure.json').exists()):
        raise ValueError('Every original image position and all pre-score source group formulas required')
    runtime = dict(at=datetime.now(timezone.utc).isoformat(), main_owner=identity(phase['owner']), actual_four_main_workers=workers,
        main_published_updates=progress['actual_updates'], main_planned_updates=35284,
        original_successor_owners={name: identity(q['owner'], True) for name, q in queues.items()},
        current_image_audit_owner=identity(image_plan['owner']), current_image_audit_progress=image_progress,
        original_code40_full_evaluation_owner=identity(code['owner']), actual_code_outcome_reader=identity(code_reader['owner'], True),
        fixed_source_group_preparation_owner=identity(group_plan['owner'], True),
        fixed_source_groups_prepared=(ROOT / 'reports/AI2D-fixed-exposure-strata-v1/preparation-complete.json').exists(),
        actual_GPU_observation=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'], text=True))
    new_code_observations = {}
    for frame in ['completion', 'instruction']:
        path = ROOT / 'reports/own-execution-contrast-full-CPU-v1/anchor_CE' / ('code-' + frame + '.json')
        if path.exists():
            value = json.loads(path.read_text()); new_code_observations[frame] = dict(
                **{k: value[k] for k in ['task_count', 'syntax_pass_rate', 'functional_pass_at_1', 'efficiency_stress_pass_rate']},
                source_sha256=file_hash(path), producer_observation_not_yet_freshly_executed_by_endpoint_reader=True)
    requirements['broad_independent_benchmarks']['evidence'] = (
        'The original all16961-question text QA job remains queued for six fixed original main/human/code endpoints:12032original '
        'MMLU-Pro questions,1319GSM8K and3610NQ-open, with original inputs/aliases, full budgets and exact source-overlap flags retained. '
        'It has not started GPU assessment. The original six-endpoint AI2D job likewise retains all3088original publisher-test questions, '
        'all original choices/gold/ambiguities and32-action budgets with proper/gray/fixed-wrong native448 images; no question/endpoint '
        'score is available yet. AI2D has814nonempty diagrams,91empty question files,76present images without a question file, and '
        'publisher ID4325 absent; the original records and complete input reconstruction are preserved. '
        f'The AI2D literal text audit is now complete across all six current/future/older inspected pools:all{tokens}declared token '
        'positions,9264signatures for3088questions. Sixteen original question cases have at least one question-only occurrence '
        '(9current,1future-human,0future-code,7Wikipedia,0code-search,6broad pool); none has an ordered-options or whole-native-prompt '
        'occurrence. The independent reader reconstructs every23reported source witness and original token budget, using actual '
        'current native batches and whole older records across manifest-ordered boundaries, without the producer token projection '
        'or substring automaton. It does not independently rescan all negative source text. Generic question occurrence is not '
        'evidence of the same diagram/choices/gold; absence of exact strings is not semantic/image/full-ancestry independence. '
        'The current native image/word audit still runs over all302627selected positions in9families; no whole-pool pixel absence '
        'or completion is claimed. Two actual held-pixel preprocessing controls, without an NN or training, independently show '
        'the benchmark and production image/word native448 processors produce exactly the same float32hash for identical inputs, '
        'and the64x448wordCTC view remains distinct. Eight source grouping rules are fixed before any visual score; their exact '
        'memberships await the complete image audit. The CPU-only owner then waits for the exact original full visual owner/workers '
        'and restoration before rederiving all18528native question-endpoint deliveries/55584condition verdicts and every source-group '
        'summary. All original questions remain the primary population; no source flag selects a winner or certifies clean data. '
        'The old175-problem supported public-code subset does not replace a full coding benchmark;600reserved original MBPP '
        'questions and the fixed64held APPS cohort remain relevant. Broad independent code/vision/knowledge quality and high-end '
        'model comparison still require actual completed assessments.')
    requirements['text_and_images']['evidence'] += (
        ' Exact preprocessing compatibility is now exercised on actual original/ABC publisher pixels through VerifiedReplayEncoder '
        'and SharedWordData.image_arrays, keeping the held role and constructing no neural model. Both original and ABC native448 '
        'hashes match the benchmark; wordCTC64x448 is distinct. These are two pixel-protocol controls, not image understanding or '
        'OCR gains. Source groups keep generic text, ordered choices, whole prompt, original diagram bytes and proper native pixels '
        'separate; no actual new visual neural score exists.')
    requirements['serious_full_model_training']['evidence'] += (
        f' The original full-weight984399690-value run remains active at{progress["actual_updates"]}/35284published updates on four '
        'original L40s. This work adds no optimizer step or promoted model; it completes source auditing, cross-protocol controls '
        'and independent witness reconstruction while the original main and40-update code assessment jobs continue. The8192held '
        'token NLL0.7746043680235744 remains a token-prediction producer observation, not code/knowledge/image accuracy.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' The source-fit comparison remains3/4CE40and3/4contrast40with every actual program freshly re-executed; no contrastive '
        'advantage. The initial model remains0/48on the independently executed complete local suite. Trained-endpoint full native '
        'code assessments are still live. Any newly available single frame recorded here is explicitly a producer observation '
        'until the separate endpoint reader freshly executes all original programs; no generalization gain is inferred from '
        'job progress or training-data fit.')
    requirements['gpu_runtime']['evidence'] += (
        ' All four original main worker identities are freshly live; original human/code/text/visual successors remain CPU-hidden '
        'and childless. The completed text auditor and witness reader ended normally. Image/source-group/code-output CPU owners '
        'remain active without preempting GPUs. No running GPU training job was restarted or original waiter released early.')
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('Every original final requirement remains explicitly incomplete')
    OUT.mkdir(); files = [Path(__file__), prior, text_path, reader_path, phase_path, image_plan_path, group_plan_path,
        code_plan_path, code_reader_plan_path, ROOT / 'tests/test_native_raw_witness_v1.py', ROOT / 'reports/native-raw-witness-v1-tests.log',
        ROOT / 'tests/test_ai2d_exposure_strata_v1.py', ROOT / 'reports/ai2d-exposure-strata-v1-tests.log',
        ROOT / 'tests/test_ai2d_training_pixel_compatibility_v1.py', ROOT / 'reports/ai2d-training-pixel-compatibility-v1-tests.log',
        *(ROOT / 'reports' / (n + '-queue.json') for n in queue_names)]
    report = dict(at=runtime['at'], requirements=requirements, actual_runtime=runtime,
        complete_literal_text_input_positions=tokens, independent_source_witnesses=23, actual_matched_original_question_cases=16,
        actual_per_signature_source_counts=signatures, new_trained_code_frame_observations=new_code_observations,
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True, no_remaining_scope_removed=True,
        source_audit_completion_not_native_capability_improvement=True,
        source_files_sha256={str(p): file_hash(p) for p in files}, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUT / 'review.json', report)
    protected = set(phase['source_files_sha256']) | set(code['source_files_sha256']) | set(image_plan['source_files_sha256'])
    protected.update(group_plan['source_files_sha256']); protected.update(code_reader['source_files_sha256'])
    protected.update(reader['source_files_sha256'])
    for q in queues.values(): protected.update(q['source_files_sha256'])
    docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', ROOT / 'reports/completion-audit.json']
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT / 'review.json'), bound_root_documents_preserved=True)), flush=True); return
    master_path = docs[-1]; master = json.loads(master_path.read_text())
    (ROOT / 'reports/completion-audit-before-complete-AI2D-text-audit-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=runtime['at'], requirements={k: r['evidence'] for k, r in requirements.items()},
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)), latest_actual_main_runtime=runtime,
        latest_complete_visual_text_source_audit=str(text_path.relative_to(ROOT)),
        latest_independent_visual_text_source_witnesses=str(reader_path.relative_to(ROOT)),
        fixed_visual_source_group_plan=str(group_plan_path.relative_to(ROOT)),
        current_goal_turn_classification='progress', all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (f'Complete visual-question text-source audit, **{runtime["at"]}**: all **{tokens:,} token positions** in six '
        'current/queued/older pools were checked against three signatures for each of3,088original AI2D questions. **16question cases '
        'match only generic question text**; none matches ordered choices or the whole native prompt. A separate reader reconstructs '
        'all **23source witnesses**, checks every original source budget and whole raw records across file boundaries, and bypasses '
        'the original projection/substring scanner. It does not repeat the whole negative scan or certify semantic/image/ancestral '
        'independence. Two actual original/ABC pixel controls confirm the benchmark and native training preprocessors hash identical '
        'inputs identically;64x448wordCTC stays distinct.\n\n'
        'The image audit remains live across302,627selected source positions. Eight source-group formulas are fixed before visual '
        'scores; a CPU-hidden owner waits for complete source audits and original full visual assessment/restoration, retaining all '
        '3,088questions as primary. **No full visual neural result or contrastive coding advantage yet.** '
        f'The original four-L40 main run is active at **{progress["actual_updates"]:,}/35,284updates**; original human/code/text/visual '
        'successors remain queued and trained native coding assessments continue. **All ten final requirements remain incomplete.** '
        '[Review](reports/complete-AI2D-text-audit-v1-review/review.json), '
        '[full text audit](reports/AI2D-full-text-exposure-audit-v1/complete.json), '
        '[independent source witnesses](reports/AI2D-all-text-exposure-witnesses-v1-readback/complete.json).\n\n')
    for path in docs[:2]:
        first, rest = path.read_text().split('\n', 1); path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), audited_token_positions=tokens,
        original_question_matches=16, independent_source_witnesses=23, main_updates=progress['actual_updates'], goal_achieved=False)), flush=True)


if __name__ == '__main__': main()
