"""Record complete visual evaluation preparation and exact live training state."""
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
OUT = ROOT / 'reports/full-AI2D-queued-v1-review'


def identity(value, waiting=False):
    proc = Path('/proc') / str(value['pid']); fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
    if fields[19] != value['ticks'] or fields[0] in ['Z', 'X']: raise ValueError('Exact original PID/start ticks required')
    children = (proc / 'task' / str(value['pid']) / 'children').read_text().strip()
    visibility = next((v.split('=', 1)[1] for v in (proc / 'environ').read_bytes().decode().split('\0') if v.startswith('CUDA_VISIBLE_DEVICES=')), None)
    if waiting and (children or visibility != ''): raise ValueError('Waiting successor must be CPU-hidden and childless')
    return dict(pid=value['pid'], ticks=fields[19], state=fields[0], children=children, CUDA_VISIBLE_DEVICES=visibility)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden complete-scope review required')
    previous = ROOT / 'reports/full-size-vision-and-code-fit-v1-review/review.json'
    parent = json.loads(previous.read_text()); requirements = parent['requirements']
    main_plan = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(main_plan.read_text())
    progress = json.loads((ROOT / 'reports/phase-joint-v3/progress.json').read_text())
    queues = {}
    for name in ['phase-human-preference-v4', 'broad-anchored-code-v3', 'full-held-QA-pipeline-v1', 'full-AI2D-pipeline-v1']:
        path = ROOT / 'reports' / (name + '-queue.json'); queues[name] = json.loads(path.read_text())
    prep_path = ROOT / 'reports/ai2d-full-native-evaluation-v1-preparation/complete.json'; prep = json.loads(prep_path.read_text())
    visual = queues['full-AI2D-pipeline-v1']; text = queues['full-held-QA-pipeline-v1']
    if (visual['predecessor'] != text['owner'] or visual['original_population_sha256'] != prep['original_population_sha256']
            or visual['original_questions_each_endpoint'] != 3088 or visual['actual_neural_assessments'] != 0
            or prep['actual_native448_variants_checked'] != 1555 or prep['original_nonempty_diagrams'] != 814):
        raise ValueError('Complete exact original visual inputs and exact original text-QA dependency required')
    FileBindings(visual['source_files_sha256']).check(full=True)
    held = {}
    held_paths = []
    for step in ['initial', '2048', '4096', '6144', '8192']:
        path = ROOT / 'reports/phase-joint-v3' / ('held-packed-' + step + '.json')
        value = json.loads(path.read_text()); held[step] = value; held_paths.append(path)
        for key in ['original_split', 'selected_original_positions', 'original_valid_population', 'evaluated_windows',
                    'complete_causal_targets', 'maximum_original_context', 'original_manifest_sha256']:
            if value[key] != held['initial'][key]: raise ValueError('Every observed held loss must use the same actual declared validation protocol')
    initial = held['initial']['token_mean_NLL']; latest = held['8192']['token_mean_NLL']
    loss = dict(initial_NLL=initial, step8192_NLL=latest, step8192_perplexity=math.exp(latest),
        relative_NLL_drop=(initial - latest) / initial, fixed_windows=128, complete_targets=524160,
        neural_losses_are_producer_observations_not_fresh_reader_forwards=True,
        no_new8192_weight_Adam_or_CUDA_replay_claim=True, not_code_knowledge_or_visual_accuracy=True)
    children = json.loads((ROOT / 'reports/phase-joint-v3/children.json').read_text())
    workers = [identity(row) for row in children if row['subject'].startswith('train-rank-')]
    if len(workers) != 4: raise ValueError('Every exact original main GPU worker required')
    code_queue_path = ROOT / 'reports/own-execution-contrast-full-CPU-v1-queue.json'; code = json.loads(code_queue_path.read_text())
    baseline_path = ROOT / 'reports/own-execution-contrast-full-CPU-v1/initial/code-completion.json'
    baseline = json.loads(baseline_path.read_text())
    if baseline['task_count'] != 24 or baseline['functional_pass_at_1'] != 0 or baseline['efficiency_stress_pass_rate'] != 0:
        raise ValueError('Exact completed initial code baseline must remain explicit')
    runtime = dict(at=datetime.now(timezone.utc).isoformat(), main_owner=identity(phase['owner']), actual_four_main_workers=workers,
        main_published_updates=progress['actual_updates'], main_planned_updates=35284,
        original_successor_owners={name: identity(q['owner'], True) for name, q in queues.items()},
        code40_CPU_owner=identity(code['owner']), code40_native_all_endpoints_complete=(ROOT / 'reports/own-execution-contrast-full-CPU-v1/complete.json').exists(),
        actual_GPU_observation=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'], text=True))
    for item in requirements.values():
        item['evidence'] = item['evidence'].replace('905with questions,76without, plus ID4325 absent',
            '814with nonempty questions,91with empty question files,76with no question file, plus ID4325 absent')
    requirements['broad_independent_benchmarks']['evidence'] += (
        ' The full AI2D native evaluation implementation and exact four-L40 successor are now complete and live CPU-hidden/childless, '
        'waiting for exact full text-QA owner3116287/start ticks39439809 and every owned worker/assessment/restoration. '
        'All3088 original questions, whole203-token-max prompts, original options and fixed32-action greedy budgets survive. '
        'Every question uses proper, gray and source-fixed wrong diagrams with the same ABC role and distinct source/pixel hashes; '
        'gold labels never select controls. Six fixed original main/human/code endpoints require18528 question-endpoint and55584 '
        'question-endpoint-condition assessments, sequentially using all four L40s, with actual pixel hashes, all generated actions/logps, '
        'and spaced/direct full-vocabulary letter likelihoods retained. No score-based winner is selected. Strict index and exact '
        'duplicate-equivalent correctness remain separate. Actual model assessment and text/pixel/semantic exposure audits remain pending; '
        'this font/native448 protocol is not an official leaderboard score or full independence certificate. '
        'Fresh inventory reconciliation corrects older wording:905 existing question files include91empty files;814 diagrams have3088 '
        'questions,76 present diagrams have no question file, and publisher ID4325 is absent. All1555 available original/ABC native448 '
        'pixel tensors were checked. No scored case was omitted, invented or changed. The earlier immutable evidence is preserved.')
    requirements['text_and_images']['evidence'] += (
        ' Three meaningful actual small-own-NN tests verify correct/gray/wrong images reach native prefill, changing gold leaves all '
        'generated tokens/choice probabilities unchanged, cached greedy actions equal native generate, every model value stays fixed, '
        'and image/score/token/source-role tampering rejects. Four driver tests verify all six exact endpoint paths, exact live owner/worker '
        'exclusion, complete24-shard text-QA/restoration receipts, and full3088-case paired image/prompt identity checks. These are mechanics '
        'and scheduling checks, not full-size model capability results or new visual training.')
    requirements['serious_full_model_training']['evidence'] += (
        f' The original full-weight run remains active at{progress["actual_updates"]}/35284 published updates on four original L40s. '
        f'Its same128window/524160target held producer NLL at8192 is{latest}, versus{initial} initially '
        f'({100 * loss["relative_NLL_drop"]:.4f}% reduction; perplexity{loss["step8192_perplexity"]}). '
        'This review verifies the unchanged source protocol/report bytes and recomputes arithmetic, not new NN loss forwards or8192 '
        'Adam/next-update CUDA state. It does not establish correct generated code, independent knowledge accuracy or visual ability.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' The40-update trial has now completed the initial24-case completion baseline:0/24 functional/efficiency passes and8/24 syntax '
        'passes in actual CPUFP32 native inference. Initial instruction and both trained CE/contrastive endpoints remain active/pending. '
        'This is an initial baseline, not regression caused by the new objectives. The separate independently re-executed0/4->3/4 '
        'train-source fit remains true, but there is no new generalization or RL/contrastive advantage demonstrated.')
    requirements['gpu_runtime']['evidence'] += (
        ' The new exact visual assessment owner is live with no children and GPUs hidden while original main training keeps all four '
        'L40s occupied. It can acquire the original global lease only after the full text-QA predecessor and all owned workers are terminal, '
        'restoration is recorded and every GPU is idle. Original-waiter hold/restoration, source immutability, no retry and all-loaded-values '
        'checks are retained. No running training process was stopped or restarted and no GPU model was substituted.')
    requirements['public_data']['evidence'] += (
        ' The visual preparation uses the same original publicly cached AI2D archive, original question bytes and unchanged packaged '
        'license for private evaluation only. No training admission, redistribution, pretrained weights or live teacher was introduced.')
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('All ten original requested final requirements remain incomplete')
    OUT.mkdir()
    files = [Path(__file__), previous, prep_path, main_plan, code_queue_path, baseline_path, *held_paths,
        *(ROOT / 'reports' / (name + '-queue.json') for name in queues)]
    review = dict(at=runtime['at'], requirements=requirements, actual_runtime=runtime, observed_held_loss=loss,
        latest_visual_input_preparation=str(prep_path.relative_to(ROOT)),
        latest_full_visual_assessment_queue='reports/full-AI2D-pipeline-v1-queue.json',
        corrected_literal_visual_input_inventory=prep['legacy_disposition_label_correction'],
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True, no_remaining_scope_removed=True,
        new_full_visual_neural_results_or_new_coding_generalization_not_established=True,
        source_files_sha256={str(p): file_hash(p) for p in files}, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUT / 'review.json', review)
    protected = set(phase['source_files_sha256']) | set(code['source_files_sha256'])
    protected.update(prep['source_files_sha256'])
    for q in queues.values(): protected.update(q['source_files_sha256'])
    docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', ROOT / 'reports/completion-audit.json']
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT / 'review.json'), bound_root_documents_preserved=True)), flush=True); return
    master_path = docs[-1]; master = json.loads(master_path.read_text())
    (ROOT / 'reports/completion-audit-before-full-AI2D-queued-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=runtime['at'], requirements={k: v['evidence'] for k, v in requirements.items()},
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)), latest_actual_main_runtime=runtime,
        queued_full_visual_QA_assessment='reports/full-AI2D-pipeline-v1-queue.json',
        latest_full_visual_QA_native_preparation=str(prep_path.relative_to(ROOT)),
        latest_held8192_token_loss_observation=loss, all_ten_final_requirements_incomplete=True,
        current_goal_turn_classification='progress', candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (f'Full visual QA assessment queued, **{runtime["at"]}**: all **3,088 original AI2D questions** will be assessed at six '
        'fixed original main/human/code checkpoints with correct, gray and fixed wrong images, full original choices and32-action budgets '
        '(55,584 question/endpoint/image-condition assessments). The exact CPU-hidden owner waits for the complete original full text-QA '
        'pipeline and restoration; it takes no GPU resources while waiting. Three actual small-own-NN mechanics tests and four scheduling/'
        'reconciliation tests pass. Every one of1,555 available native448 image variants was checked. Corrected inventory: **814 diagrams '
        'contain3,088 questions**;91 more have empty question files,76 have no question file, and publisher ID4325 is absent. No scored '
        'question was dropped. Native font/448/original-option choices differ from published leaderboard protocols, and image/text exposure '
        'audit remains pending. **No full visual neural score yet.**\n\n'
        f'The original four-L40 main run is active at **{progress["actual_updates"]:,}/35,284 updates**. Matched held token NLL at8,192 '
        f'is **{latest:.6f}**, down **{100 * loss["relative_NLL_drop"]:.2f}%** from{initial:.6f}; these are producer loss observations, '
        'with unchanged input/report arithmetic checked here, not new code/knowledge/vision accuracy. The40-update CPU code trial’s initial '
        'completion baseline is0/24 correct; trained endpoint assessments remain active. The separate0/4->3/4 source-fit result is still '
        '**training-data fit**. **All ten final requirements remain incomplete.** '
        '[Review](reports/full-AI2D-queued-v1-review/review.json), '
        '[visual queue](reports/full-AI2D-pipeline-v1-queue.json), '
        '[complete native image preparation](reports/ai2d-full-native-evaluation-v1-preparation/complete.json).\n\n')
    for path in docs[:2]:
        first, rest = path.read_text().split('\n', 1); path.write_text(first + '\n\n' + paragraph + rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), new_visual_assessment_queued=True,
        main_updates=progress['actual_updates'], matched8192_NLL=latest, goal_achieved=False)), flush=True)


if __name__ == '__main__': main()
