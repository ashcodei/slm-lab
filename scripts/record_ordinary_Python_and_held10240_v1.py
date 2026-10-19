"""Record validated broader Python execution and the next fixed held loss."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/ordinary-Python-and-held10240-v1-review'


def identity(value, waiting=False):
    proc = Path('/proc') / str(value['pid']); fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
    if fields[19] != value['ticks'] or fields[0] in ['Z', 'X']: raise ValueError('Exact original live birth identity required')
    children = (proc / 'task' / str(value['pid']) / 'children').read_text().strip()
    visible = next((v.split('=', 1)[1] for v in (proc / 'environ').read_bytes().decode().split('\0') if v.startswith('CUDA_VISIBLE_DEVICES=')), None)
    if waiting and (children or visible != ''): raise ValueError('Waiting successor must remain CPU-hidden and childless')
    return dict(pid=value['pid'], ticks=fields[19], state=fields[0], children=children, CUDA_VISIBLE_DEVICES=visible)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden full-scope evidence update required')
    prior = ROOT / 'reports/complete-AI2D-text-audit-v1-review/review.json'; parent = json.loads(prior.read_text()); requirements = parent['requirements']
    folder = ROOT / 'reports/own-execution-contrast-ordinary-Python-v1'; ordinary_plan_path = folder / 'plan.json'
    ordinary_plan = json.loads(ordinary_plan_path.read_text()); controls_path = folder / 'actual-canonical-module-and-subset-controls.json'
    controls = json.loads(controls_path.read_text()); ordinary = {}; receipts = []
    if (len(controls['all12_ordinary_Python_main_guard_reference_controls']) != 12
            or not all(r['actual_original_control']['functional_passed'] for r in controls['all12_ordinary_Python_main_guard_reference_controls'])
            or controls['original_restricted_subset_main_guard_observation']['functional_passed'] is not False):
        raise ValueError('Every actual ordinary module reference and old-subset main-guard observation required')
    FileBindings(ordinary_plan['source_files_sha256']).check(full=True)
    for name in ['initial-completion', 'initial-instruction', 'anchor_CE-completion']:
        path = folder / (name + '-complete.json'); receipt = json.loads(path.read_text()); receipts.append(path)
        FileBindings(receipt['source_files_sha256']).check(full=True); ordinary[name] = receipt['actual_outcomes']
        if ordinary[name]['Python_functions'] != 12 or ordinary[name]['ordinary_Python_correct'] != 0:
            raise ValueError('This review requires every actual completed36-program ordinary-Python observation')
    ce_path = ROOT / 'reports/own-execution-contrast-CE-completion24-v1-readback/complete.json'
    ce = json.loads(ce_path.read_text()); FileBindings(ce['source_files_sha256']).check(full=True)
    if (ce['actual_trained_completion_deliveries_freshly_executed'] != 24 or ce['actual_reference_controls_freshly_executed'] != 24
            or ce['before_independently_executed_initial']['syntax_passes'] != 8
            or ce['after_independently_executed_CE40'] != dict(tasks=24, syntax_passes=11, correct=0, empirical_efficiency_stress_passes=0)):
        raise ValueError('Every actual matched initial/CE code frame and fresh reference control required')
    initial_path = ROOT / 'reports/phase-joint-v3/held-packed-initial.json'; latest_path = ROOT / 'reports/phase-joint-v3/held-packed-10240.json'
    initial, latest = [json.loads(p.read_text()) for p in [initial_path, latest_path]]
    for key in ['original_split', 'selected_original_positions', 'original_valid_population', 'evaluated_windows',
            'complete_causal_targets', 'maximum_original_context', 'original_manifest_sha256']:
        if initial[key] != latest[key]: raise ValueError('Exact same original held loss inputs/protocol required')
    nll = dict(initial=initial['token_mean_NLL'], step10240=latest['token_mean_NLL'],
        relative_reduction=(initial['token_mean_NLL'] - latest['token_mean_NLL']) / initial['token_mean_NLL'],
        step10240_perplexity=math.exp(latest['token_mean_NLL']), fixed_original_windows=128, fixed_original_targets=524160,
        original_input_protocol_and_arithmetic_verified=True, actual_neural_losses_are_producer_observations=True,
        no_fresh10240_weight_Adam_or_CUDA_replay_claim=True, not_generated_code_knowledge_or_image_accuracy=True)
    phase_path = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(phase_path.read_text())
    progress = json.loads((ROOT / 'reports/phase-joint-v3/progress.json').read_text())
    workers = [identity(p) for p in json.loads((ROOT / 'reports/phase-joint-v3/children.json').read_text()) if p['subject'].startswith('train-rank-')]
    if len(workers) != 4: raise ValueError('Every exact original main worker required')
    queue_names = ['phase-human-preference-v4', 'broad-anchored-code-v3', 'full-held-QA-pipeline-v1', 'full-AI2D-pipeline-v1']
    queues = {n: json.loads((ROOT / 'reports' / (n + '-queue.json')).read_text()) for n in queue_names}
    image_plan = json.loads((ROOT / 'reports/AI2D-current-native-image-exposure-v1/plan.json').read_text())
    image_progress = json.loads((ROOT / 'reports/AI2D-current-native-image-exposure-v1/progress.json').read_text())
    group_plan = json.loads((ROOT / 'reports/AI2D-fixed-exposure-strata-v1/plan.json').read_text())
    code = json.loads((ROOT / 'reports/own-execution-contrast-full-CPU-v1/plan.json').read_text())
    code_reader = json.loads((ROOT / 'reports/own-execution-contrast-full-CPU-v2-outcome-readback/plan.json').read_text())
    runtime = dict(at=datetime.now(timezone.utc).isoformat(), main_owner=identity(phase['owner']), actual_four_main_workers=workers,
        main_published_updates=progress['actual_updates'], main_planned_updates=35284,
        original_successor_owners={n: identity(q['owner'], True) for n, q in queues.items()},
        exact_original_code_CPU_owner=identity(code['owner']), exact_original_code_output_reader=identity(code_reader['owner'], True),
        ordinary_Python_output_assessor=identity(ordinary_plan['owner'], True),
        current_image_audit_owner=identity(image_plan['owner']), current_image_audit_progress=image_progress,
        fixed_source_group_owner=identity(group_plan['owner'], True),
        actual_GPU_observation=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'], text=True))
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' A new fresh program re-execution confirms the CE40 student completion frame has11/24syntactic passes versus8/24initially, '
        'with0/24functional and0/24empirical efficiency passes both before/after. All24original canonical reference controls '
        'freshly pass, with no timing-verdict disagreement. This is a complete local completion frame; full48-case trained '
        'endpoints are still pending. A real evaluator limitation is now documented:the old restricted Python subset rejects '
        'a correct canonical function solely for an inert ordinary main guard. A separate normal-module standard-library Python '
        'runtime now reuses existing pre-exec Landlock/seccomp confinement, preserves program bytes, suppresses ordinary main '
        'guards through an import-style module name, and keeps all expected outputs/reference programs outside the candidate '
        'process/readable filesystem. It compares every returned value strictly in the parent, without the APPS optional-list '
        'compatibility rule or injecting publisher globals. Three runtime tests prove legitimate classes/helpers/constants/'
        'stdlib imports/main guards, no scalar-for-list false acceptance, expected-value separation, denied outside files/network '
        'and bounded infinite execution. Twelve actual canonical main-guard references pass this new protocol. All36actual '
        'generated Python programs available so far—initial12completion,initial12instruction,CE40completion12—remain0correct '
        'under normal Python as well. Thus these tested outputs still fail without the old subset restrictions; no new native '
        'functional gain is demonstrated. A CPU-hidden exact-owner assessor waits to process all72Python programs across all '
        'three endpoints/two frames, with no new neural generation or training. The new protocol is a runtime-validity check, '
        'not a weight improvement, empirical efficiency result, official benchmark score or full ANY-code capability proof.')
    requirements['serious_full_model_training']['evidence'] += (
        f' The original full-weight984399690-value run remains active on four original L40s at{progress["actual_updates"]}/35284updates. '
        f'On the same128held packed windows/524160targets, producer NLL at10240 is{nll["step10240"]}, versus{nll["initial"]}initially '
        f'({100 * nll["relative_reduction"]:.6f}% lower; perplexity{nll["step10240_perplexity"]}). The reader checks every held input '
        'protocol field and report/arithmetic; it does not repeat NN loss computation or10240weight/Adam/CUDA recovery. This '
        'token-prediction improvement does not establish generated coding, independent knowledge, vision or high-end capability.')
    requirements['gpu_runtime']['evidence'] += (
        ' Every original main worker birth identity is freshly live. New ordinary-Python execution/control assessment uses CPU '
        'only and no neural model. Original human/code/text/visual successors remain CPU-hidden/childless and no live training '
        'source was edited, preempted or restarted. The separate current-image and fixed-source-group owners remain live.')
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('Every original final requirement remains explicitly incomplete')
    OUT.mkdir(); files = [Path(__file__), prior, ordinary_plan_path, controls_path, ce_path, *receipts, initial_path, latest_path, phase_path,
        ROOT / 'tests/test_ordinary_python_function_cases_v1.py', ROOT / 'reports/ordinary-python-function-cases-v1-tests.log',
        *(ROOT / 'reports' / (n + '-queue.json') for n in queue_names)]
    report = dict(at=runtime['at'], requirements=requirements, actual_runtime=runtime,
        actual_matched_reexecuted_CE_completion_frame=dict(before=ce['before_independently_executed_initial'], after=ce['after_independently_executed_CE40']),
        actual_ordinary_Python_generated_program_outcomes=ordinary, actual_reference_main_guard_controls=12,
        latest_same_protocol_held_loss=nll, previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True, no_remaining_scope_removed=True,
        no_new_functional_coding_gain_or_frontier_equivalence_established=True,
        source_files_sha256={str(p): file_hash(p) for p in files}, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUT / 'review.json', report)
    protected = set(phase['source_files_sha256']) | set(code['source_files_sha256']) | set(code_reader['source_files_sha256'])
    for p in [ordinary_plan, image_plan, group_plan]: protected.update(p['source_files_sha256'])
    for q in queues.values(): protected.update(q['source_files_sha256'])
    docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', ROOT / 'reports/completion-audit.json']
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT / 'review.json'), bound_root_documents_preserved=True)), flush=True); return
    master_path = docs[-1]; master = json.loads(master_path.read_text())
    (ROOT / 'reports/completion-audit-before-ordinary-Python-held10240-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=runtime['at'], requirements={k: r['evidence'] for k, r in requirements.items()},
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)), latest_actual_main_runtime=runtime,
        latest_matched_held10240_token_loss_observation=nll,
        latest_CE_completion_code_frame_readback=str(ce_path.relative_to(ROOT)),
        ordinary_Python_generated_program_assessment_plan=str(ordinary_plan_path.relative_to(ROOT)),
        current_goal_turn_classification='progress', all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (f'Normal Python execution and10,240-update held loss, **{runtime["at"]}**: fresh execution of every trained CE '
        'completion delivery and24reference controls confirms syntax **8/24->11/24**, functional **0/24->0/24** and empirical '
        'efficiency0/24. The earlier3/4 source-fit result is memorization/fit evidence, with no new-program correctness gain '
        'demonstrated. A separate ordinary stdlib-Python module runtime now passes12canonical main-guard controls while keeping '
        'expected answers outside the OS-confined candidate. The old subset rejects that valid guard. All **36actual generated '
        'Python programs tested so far still score0correct** under the broader runtime. It changes execution interpretation, '
        'adds no training or NN generation, and claims no efficiency result. Trained instruction/contrastive evaluations continue.\n\n'
        f'The original main run is active at **{progress["actual_updates"]:,}/35,284updates** on four L40s. Its same-input held '
        f'token NLL at10,240 is **{nll["step10240"]:.6f}**, down **{100 * nll["relative_reduction"]:.2f}%** from{nll["initial"]:.6f}; '
        'this is producer token-prediction evidence, with protocol/report arithmetic checked, not generated coding/knowledge/image '
        'accuracy. The complete visual-question text source audit/23witness readback remain valid, image/source-group audits '
        'continue, and original human/code/text/visual successors remain queued. **All ten final requirements remain incomplete.** '
        '[Review](reports/ordinary-Python-and-held10240-v1-review/review.json), '
        '[trained24-case execution readback](reports/own-execution-contrast-CE-completion24-v1-readback/complete.json), '
        '[ordinary Python assessment](reports/own-execution-contrast-ordinary-Python-v1/plan.json).\n\n')
    for path in docs[:2]:
        first, rest = path.read_text().split('\n', 1); path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual_ordinary_generated_programs=36,
        ordinary_correct=0, actual_CE_completion_correct='0/24', held10240_NLL=nll['step10240'],
        main_updates=progress['actual_updates'], goal_achieved=False)), flush=True)


if __name__ == '__main__': main()
