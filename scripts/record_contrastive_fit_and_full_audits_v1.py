"""Preserve the complete goal after actual contrastive fit and audit progress."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/contrastive-fit-and-full-audits-v1-review'


def identity(value, waiting=False):
    proc = Path('/proc') / str(value['pid']); fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
    if fields[19] != value['ticks'] or fields[0] in ['Z', 'X']: raise ValueError('Exact original live birth identity required')
    children = (proc / 'task' / str(value['pid']) / 'children').read_text().strip()
    visibility = next((v.split('=', 1)[1] for v in (proc / 'environ').read_bytes().decode().split('\0') if v.startswith('CUDA_VISIBLE_DEVICES=')), None)
    if waiting and (children or visibility != ''): raise ValueError('Queued successor must be CPU-hidden and childless')
    return dict(pid=value['pid'], ticks=fields[19], state=fields[0], children=children, CUDA_VISIBLE_DEVICES=visibility)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden complete-scope evidence update required')
    previous = ROOT / 'reports/full-AI2D-queued-v1-review/review.json'; parent = json.loads(previous.read_text())
    requirements = parent['requirements']; evidence = {}
    paths = dict(contrast_source=ROOT / 'reports/own-execution-contrast-early-contrast-source-greedy-CPU-v1-readback/complete.json',
        paired_source=ROOT / 'reports/own-execution-contrast-early-contrast-source-greedy-CPU-v1/matched-CE40-versus-contrast40-training-source-comparison.json',
        native_initial=ROOT / 'reports/own-execution-contrast-full-CPU-v2-outcome-readback/initial/complete.json')
    for name, path in paths.items():
        evidence[name] = json.loads(path.read_text())
        if 'source_files_sha256' in evidence[name]: FileBindings(evidence[name]['source_files_sha256']).check(full=True)
    paired = evidence['paired_source']; baseline = evidence['native_initial']['independent_outcomes']
    if (paired['CE40_finite_passes'] != 3 or paired['contrast40_finite_passes'] != 3
            or len(paired['matched_original_sources']) != 4
            or evidence['contrast_source']['independently_reexecuted_actual_neural_programs'] != 4
            or evidence['contrast_source']['earlier_CE40_actual_programs_also_freshly_reexecuted'] != 4
            or evidence['native_initial']['actual_raw_code_deliveries_reexecuted'] != 48
            or any(baseline['native_code'][f]['correct'] != 0 for f in ['completion', 'instruction'])):
        raise ValueError('Every actual paired source outcome and complete initial native delivery must reconcile')
    phase_path = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(phase_path.read_text())
    progress = json.loads((ROOT / 'reports/phase-joint-v3/progress.json').read_text())
    workers = [identity(p) for p in json.loads((ROOT / 'reports/phase-joint-v3/children.json').read_text()) if p['subject'].startswith('train-rank-')]
    if len(workers) != 4: raise ValueError('Every exact original main GPU worker required')
    queue_names = ['phase-human-preference-v4', 'broad-anchored-code-v3', 'full-held-QA-pipeline-v1', 'full-AI2D-pipeline-v1']
    queues = {n: json.loads((ROOT / 'reports' / (n + '-queue.json')).read_text()) for n in queue_names}
    code_queue_path = ROOT / 'reports/own-execution-contrast-full-CPU-v1-queue.json'; code = json.loads(code_queue_path.read_text())
    audit_status = {}; audit_paths = []
    for name in ['AI2D-full-text-exposure-audit-v1', 'AI2D-current-native-image-exposure-v1']:
        folder = ROOT / 'reports' / name; plan_path = folder / 'plan.json'; plan = json.loads(plan_path.read_text())
        p = json.loads((folder / 'progress.json').read_text())
        if (folder / 'failure.json').exists(): raise ValueError('Explicit audit failure must be resolved before a progress review')
        audit_status[name] = dict(owner=identity(plan['owner']), complete=(folder / 'complete.json').exists(), latest_observation=p)
        audit_paths.append(plan_path)
    source_scopes = {}
    for name in ['current_native', 'future_human', 'future_code']:
        path = ROOT / 'reports/AI2D-full-text-exposure-audit-v1' / (name + '.json'); item = json.loads(path.read_text()); audit_paths.append(path)
        source_scopes[name] = item['question_counts_by_benchmark']
        if any(k != 'AI2D:original_question' for k in source_scopes[name]): raise ValueError('This review expects only the observed generic question matches')
    text_plan = json.loads((ROOT / 'reports/AI2D-full-text-exposure-audit-v1/plan.json').read_text())
    image_plan = json.loads((ROOT / 'reports/AI2D-current-native-image-exposure-v1/plan.json').read_text())
    if (text_plan['original_AI2D_questions'] != 3088 or sum(text_plan['question_counts'].values()) != 9264
            or image_plan['expected_positions'] != 302627 or len(image_plan['expected_complete_source_positions']) != 9):
        raise ValueError('Complete original text/pixel source audit scopes required')
    reader_plan_path = ROOT / 'reports/own-execution-contrast-full-CPU-v2-outcome-readback/plan.json'
    reader = json.loads(reader_plan_path.read_text())
    runtime = dict(at=datetime.now(timezone.utc).isoformat(), main_owner=identity(phase['owner']), actual_four_main_workers=workers,
        main_published_updates=progress['actual_updates'], main_planned_updates=35284,
        original_successor_owners={name: identity(q['owner'], True) for name, q in queues.items()},
        code40_actual_full_native_assessment_owner=identity(code['owner']), code_outcome_reader=identity(reader['owner'], True),
        full_text_and_image_audits=audit_status,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'], text=True))
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' The early40-update contrastive student now completes actual unassisted512-budget greedy generation on every four original '
        'train problems. It solves3/4, exactly matching the supervised CE40 control; the same three complete human programs/EOS are '
        'generated by both models. The fourth program differs and both fail. A separate reader rederives every actual prompt/action/'
        'EOS/budget and freshly executes all four contrastive and all four retained CE deliveries. No contrastive advantage is '
        'demonstrated. Teacher-forced competition-target argmax correctness is269/283 versus276/283 for CE, yet both free-running '
        'programs fail; teacher forcing is not functional correctness. The actual initial model’s full48 native local-development '
        'outputs have now been independently freshly executed:completion8/24syntax,0/24correct;instruction14/24syntax,0/24correct; '
        'all efficiency passes0 and no timing-verdict disagreement. The producer’s trained CE/contrastive48-case assessments remain '
        'active, with an exact-birth-identity reader waiting to freshly execute every144native delivery plus12source deliveries '
        'as each immutable endpoint completes. Native code reports retain raw delivered text but not actual BPE action IDs; this '
        'reader cannot reconstruct missing IDs or rerun NN generation. These are development cases, not independent final ANY-code '
        'coverage, formal efficiency, human RLHF or frontier capability.')
    requirements['broad_independent_benchmarks']['evidence'] += (
        ' A new actual CPU-only source audit scans all3088AI2D cases using9264signatures:original question, question with every '
        'ordered original option, and whole native prompt. It covers all577662698current native context positions plus31147346 '
        'future-human and2287522future-code positions, and three older raw pools totaling2638877519additional positions. All three '
        'current/future scopes are complete:current9andfuture-human1question-only matches,future-code0; no ordered-choice or whole '
        'native-prompt match in those completed scopes. Generic question occurrence does not prove exposure to the same diagram, '
        'choices or gold. The older Wikipedia/code-search/broad pools are still being scanned, so the six-scope audit is incomplete. '
        'A separate actual native-image audit visits all302627selected image/word positions across9families, checks original file '
        'bytes and fresh actual normalized448input hashes, and keeps64x448CTC distinct. It reuses only verified same-source/protocol/'
        'expected-hash digests while still counting every original position. It is currently within the word family; no whole-pool '
        'image-overlap conclusion is drawn. Exact-pool checks do not certify semantic/perceptual/crop or all historical ancestry. '
        'No original benchmark question is removed, and the original six-endpoint full text/visual queues remain intact.')
    requirements['serious_full_model_training']['evidence'] += (
        f' The original full-weight main run remains active on four original L40s at{progress["actual_updates"]}/35284updates. '
        'The new early contrastive check performs actual NN inference and original execution only; it adds no optimizer step. '
        'The completed40-update supervised/preference training states and separate full-size vision replay remain unchanged. '
        'The same8192held token NLL0.7746043680235744 remains a producer token-prediction observation, not evidence of new '
        'code/knowledge/image capability or high-end-model equivalence.')
    requirements['gpu_runtime']['evidence'] += (
        ' Main owner and all four original worker birth identities are freshly live. Human/code/full-text/full-visual successors '
        'remain childless with GPUs hidden. Text/image audits use at most one CPU thread each; the early contrastive probe used '
        'two CPU threads and is now terminal. The native output reader executes constrained source programs with two workers '
        'and waits CPU-hidden without children between completed endpoints. No running training job was preempted or restarted.')
    requirements['human_feedback']['evidence'] += (' Execution-labelled contrastive fit and literal source-overlap checks are machine feedback/data '
        'audits, not new human judgments of our own responses. The original complete public human-ranking comparison remains pending.')
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('Every original requested final requirement remains incomplete')
    OUT.mkdir(); files = [Path(__file__), previous, *paths.values(), phase_path, code_queue_path, reader_plan_path, *audit_paths,
        *(ROOT / 'reports' / (n + '-queue.json') for n in queue_names)]
    report = dict(at=runtime['at'], requirements=requirements, actual_runtime=runtime, actual_matched_train_source_outcomes=paired,
        independent_initial48_case_code_outcomes=baseline, completed_current_and_future_text_overlap_scopes=source_scopes,
        latest_evidence={k: str(p.relative_to(ROOT)) for k, p in paths.items()},
        corrected_output_reader_v1_failed_tokenizer_accessor_v2_uses_original_corpus_provenance=True,
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True, no_remaining_scope_removed=True,
        no_contrastive_advantage_or_new_native_code_generalization_established=True,
        source_files_sha256={str(p): file_hash(p) for p in files}, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUT / 'review.json', report)
    protected = set(phase['source_files_sha256']) | set(code['source_files_sha256']) | set(reader['source_files_sha256'])
    protected.update(text_plan['source_files_sha256']); protected.update(image_plan['source_files_sha256'])
    for q in queues.values(): protected.update(q['source_files_sha256'])
    for v in evidence.values(): protected.update(v.get('source_files_sha256', {}))
    docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', ROOT / 'reports/completion-audit.json']
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT / 'review.json'), bound_root_documents_preserved=True)), flush=True); return
    master_path = docs[-1]; master = json.loads(master_path.read_text())
    (ROOT / 'reports/completion-audit-before-contrastive-fit-full-audits-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=runtime['at'], requirements={k: v['evidence'] for k, v in requirements.items()},
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)), latest_actual_main_runtime=runtime,
        latest_contrastive_source_fit_readback=str(paths['contrast_source'].relative_to(ROOT)),
        latest_native_code_outcome_reader='reports/own-execution-contrast-full-CPU-v2-outcome-readback/plan.json',
        current_AI2D_exposure_audits={k: v['owner'] for k, v in audit_status.items()},
        current_goal_turn_classification='progress', all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (f'Actual contrastive fit and complete-source audits, **{runtime["at"]}**: both40-update own coding students solve '
        '**3/4 original training problems** unassisted; their same three successful programs exactly match the human reference/EOS. '
        'A fresh independent execution of all four contrastive and four retained CE programs reproduces the tie. **No preference-loss '
        'advantage or generalization gain is established.** The initial model’s full48 local code deliveries also freshly execute: '
        '**0/24 completion and0/24 instruction correct**, with8/24 and14/24 syntax passes. The trained48-case endpoint evaluations '
        'continue; an exact-owner CPU-hidden reader will execute every original output as its endpoint completes.\n\n'
        'The full AI2D text-overlap audit is live with **three signatures per question** over all current/queued text and three older '
        'raw pools. Completed current/future scopes show9current and1human **question-only** matches; no ordered-option/full-native-prompt '
        'match, and none in future code. Older pools remain unfinished. A separate live image audit reconstructs actual source/native448 '
        'pixels for all **302,627 selected image/word positions**; it is still in the word family, so no whole-image-pool clean claim is '
        'made. These exact checks cannot establish semantic/perceptual or complete historical independence. '
        f'The original four-L40 main run is active at **{progress["actual_updates"]:,}/35,284updates**; original human/code/text/visual '
        'successors remain queued. **All ten final requirements remain incomplete.** '
        '[Review](reports/contrastive-fit-and-full-audits-v1-review/review.json), '
        '[paired actual source executions](reports/own-execution-contrast-early-contrast-source-greedy-CPU-v1-readback/complete.json), '
        '[complete initial native code readback](reports/own-execution-contrast-full-CPU-v2-outcome-readback/initial/complete.json).\n\n')
    for path in docs[:2]:
        first, rest = path.read_text().split('\n', 1); path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), CE40_source_passes=3, contrast40_source_passes=3,
        initial_full_native_code_passes='0/48', main_updates=progress['actual_updates'], goal_achieved=False)), flush=True)


if __name__ == '__main__': main()
