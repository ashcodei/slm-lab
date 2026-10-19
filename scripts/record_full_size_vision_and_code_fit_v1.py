"""Record actual separate full-size trials and preserve every final requirement."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/full-size-vision-and-code-fit-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden full-scope review required')
    parent_path = ROOT / 'reports/own-negative-queued-and-step6144-v1-review/review.json'
    parent = json.loads(parent_path.read_text()); requirements = parent['requirements']
    paths = dict(vision=ROOT / 'reports/own-actor-vision-replay-full-CPU-v1-readback/complete.json',
        code_fit=ROOT / 'reports/own-execution-contrast-early-source-greedy-CPU-v1-readback/complete.json',
        code_training=ROOT / 'reports/own-execution-contrast-full-CPU-v1-training-readback/complete.json',
        old_conditional=ROOT / 'reports/conditional-prefix-full-CPU-v1-endpoint-readback/complete.json',
        AI2D=ROOT / 'reports/ai2d-all-original-test-eval-v2-readback/complete.json')
    evidence = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if (evidence['vision']['actual_native_values_counted'] != 999706442
            or not evidence['vision']['independent_actual_private_state_reconstruction_vs_self_contained_cold_text_image_word_NN_logits_bitexact']
            or evidence['code_fit']['actual_outcomes']['initial']['full_finite_case_passes'] != 0
            or evidence['code_fit']['actual_outcomes']['anchor_CE']['full_finite_case_passes'] != 3
            or evidence['AI2D']['independently_reconstructed_original_questions'] != 3088):
        raise ValueError('Actual full-size native/fit and full original image-question evidence required')
    for proof in evidence.values():
        for name, digest in proof['source_files_sha256'].items():
            if file_hash(name) != digest: raise ValueError('New accepted evidence input changed')
    phase_path = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(phase_path.read_text())
    code_path = ROOT / 'reports/own-execution-contrast-full-CPU-v1/plan.json'; code = json.loads(code_path.read_text())
    vision_path = ROOT / 'reports/own-actor-vision-replay-full-CPU-v1/plan.json'; vision = json.loads(vision_path.read_text())
    progress = json.loads((ROOT / 'reports/phase-joint-v3/progress.json').read_text())
    def identity(owner, waiting=False):
        proc = Path('/proc') / str(owner['pid']); fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
        if fields[19] != owner['ticks'] or fields[0] in ['Z', 'X']: raise ValueError('Exact live PID/start ticks required')
        children = (proc / 'task' / str(owner['pid']) / 'children').read_text().strip()
        visible = next((s.split('=', 1)[1] for s in (proc / 'environ').read_bytes().decode().split('\0') if s.startswith('CUDA_VISIBLE_DEVICES=')), None)
        if waiting and (children or visible != ''): raise ValueError('Original successor must remain CPU-hidden/childless while waiting')
        return dict(pid=owner['pid'], ticks=fields[19], state=fields[0], children=children, CUDA_VISIBLE_DEVICES=visible)
    queues = {name: json.loads((ROOT / 'reports' / name).read_text()) for name in [
        'phase-human-preference-v4-queue.json', 'broad-anchored-code-v3-queue.json', 'full-held-QA-pipeline-v1-queue.json']}
    workers = [identity(row) for row in json.loads((ROOT / 'reports/phase-joint-v3/children.json').read_text()) if row['subject'].startswith('train-rank-')]
    if len(workers) != 4: raise ValueError('All four original main GPU workers required')
    runtime = dict(at=datetime.now(timezone.utc).isoformat(), main_owner=identity(phase['owner']), actual_four_main_workers=workers,
        main_published_updates=progress['actual_updates'], main_planned_updates=35284,
        code40_evaluation_owner=identity(code['owner']), code40_actual_training_updates_each_arm=40,
        code40_full_native_development_assessments_complete=(ROOT / 'reports/own-execution-contrast-full-CPU-v1/complete.json').exists(),
        original_human_code_text_QA_successors={name: identity(queue['owner'], True) for name, queue in queues.items()},
        actual_GPU_observation=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'], text=True))
    requirements['own_architecture']['evidence'] += (' A new actual full-size vision/matching/word replay graph now updates only130 existing tensors '
        '(52,818,498 values), with original native caption/EOS CE plus image-text matching and complete original real-word CTC/native CE. The entire '
        'language/actor/critic and other heads stay bit exact. A new self-contained native bundle contains every updated own base/head/actor value and '
        'requires no old foundation weights or teacher at inference. Independent reconstruction from the original6144 source plus selected private state '
        'matches a fresh full bundle load on actual text/image/word NN logits. Twelve fresh initial/trained word-head NN forwards reproduce all paths. '
        'This is a separate experimental fork from the code40 model, not a merged or promoted final model.')
    requirements['under_1b_parameters']['evidence'] += (' Every actual native base, tied vocabulary alias, auxiliary head, actor and separate critic value '
        'is independently counted in the self-contained vision bundle:999,706,442. Its130 selected native/private tensors agree, every selected Adam clock '
        'is2, and all original fixed base/language/actor/critic/head values are exact. Both separate40-update code students also count999,706,442 values '
        'and have all561 Adam clocks40. Their own initial reference is a cached output distribution, not another neural model inside inference.')
    requirements['text_and_images']['evidence'] += (' The actual full-size vision fork completed two logical/three physical updates on two human-caption '
        'photos and two original real words, with exact private weights/Adam/CPU-RNG next-update replay and source-tamper rejection. On four fixed development '
        'word crops, independent actual NN readout and raw-path replay reproduce byte edits104->18 over19 reference bytes (5.473684->0.947368 error rate), '
        'but exact reading remains0/4. Shorter/blank-like outputs can reduce this error without reliable reading. Eight dev photo retrieval results are '
        'mixed: image-to-text1->2correct, text-to-image2->1correct. Native dev caption NLL2.101255->2.095682 is a producer observation; blank/wrong image '
        'gains slightly decrease, not evidence of stronger general grounding. These tiny dev cohorts do not establish broad visual ability. Full-size '
        'new GPU admission and a complete larger image/word curriculum are still required.')
    requirements['serious_full_model_training']['evidence'] = (f'The original full-weight984,399,690-value main model remains active on all four L40s '
        f'at{progress["actual_updates"]:,}/35,284 published updates. The verified6144 snapshot has212,107,871 actual supervised targets and matched held '
        'NLL0.782702675787732 (10.0285% lower than initial0.8699451996362768), not a new code or knowledge score. Both separate own-negative CPU arms '
        'actually completed40logical/41physical updates, all20 pairs twice,5,700 full original positive/EOS targets and16,626 observed negative actions '
        'per arm. Their full native/private actor tensors, finite Adam moments/clocks40, actual frozen own base hash, complete pair/source/mask/EOS '
        'accounting and CE/preference/objective arithmetic independently reconcile. Their final actors differ in561 tensors. The full-size vision '
        'fork also has two actual updates plus independent full native IO/state proof. None is a complete replacement for the live main/human/code source epochs.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (' New genuine unassisted coding fit evidence: on every one of the four predeclared '
        'public train problems, the initial student passes0/4 and the40-update supervised CE control passes3/4. All three successful generated action '
        'sequences exactly match the unchanged human reference including genuine EOS, without supplying a reference prefix at inference. A separate '
        'reader reconstructs every prompt/action/budget/EOS and freshly re-executes all eight actual programs with the same outcomes. This is fit to '
        'four training problems, not75% general coding accuracy, an RL/contrastive advantage, algorithmic efficiency proof or a result on new problems. '
        'The residual competition source has276/283 teacher-forced target argmaxes correct yet its actual free-running output diverges and is invalid. '
        'The three-update conditional trial is now fully read back: all30matched source endpoints remain zero reward. The new40-update study’s '
        'complete48-case native syntax/function/efficiency assessments per initial/CE/contrastive endpoint are live; no new held correctness result yet.')
    requirements['broad_independent_benchmarks']['evidence'] += (' All3,088 original AI2D publisher-test questions are now prepared and independently '
        'reconstructed, preserving every original option/gold index, including17duplicate/null-placeholder cases. The982official IDs comprise981present '
        'images:905with questions,76without, plus ID4325 absent from the original archive. Every disposition is explicit; no scored question/image is '
        'invented or omitted. All1,555 original/ABC native448 image variants, geometry/replacement labels, full prompts/token IDs (max203tokens), '
        '32-token budgets and original question bytes are rederived. ABC inputs use pinned LiberationSans and unpruned choices; published leaderboard '
        'protocol equivalence is not claimed. No new model was scored on this set. Exact/semantic image/text exposure audit is still pending. This '
        'expands future visual assessment; the complete16961question text-QA pipeline and broader code/vision retention checks remain required.')
    requirements['public_data']['evidence'] += (' AI2D uses the already cached original archive and its unchanged packaged license for private evaluation '
        'only; its old listing/license conflict remains recorded and it is not admitted to training or public redistribution. No external model weights '
        'or live teacher were used. The failed first preparation stopped on absent publisher image4325; its partial output/log/source are preserved '
        'and v2 explicitly accounts for the unavailable ID while retaining all3,088 original scored questions.')
    requirements['gpu_runtime']['evidence'] += (' The original four exact main ranks remain live. New code, vision, input preparation and independent '
        'checks used CPU with GPUs hidden; all NN source/model/cache profiles remain explicit. The original human/code/full-text-QA owners still wait '
        'unchanged. Full-size CPU success does not establish new GPU/distributed capacity for vision replay or contrastive coding.')
    requirements['human_feedback']['evidence'] += (' These new source-fit, machine-preference, OCR and dataset checks do not count as human RLHF. '
        'The full public-human-ranking chosen-CE/human-DPO successor and its original held preference assessments remain pending.')
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('Every original final requirement must remain explicitly incomplete')
    OUT.mkdir(); files = [Path(__file__), parent_path, *paths.values(), phase_path, code_path, vision_path,
        ROOT / 'data/ai2d-all-original-test-eval-v2/manifest.json',
        ROOT / 'data/ai2d-all-original-test-eval-v2/evaluation-input-reconstruction-admission.json',
        *(ROOT / 'reports' / name for name in queues)]
    report = dict(at=runtime['at'], requirements=requirements, actual_runtime=runtime,
        latest_actual_evidence={name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        separate_experimental_code_and_vision_models_not_a_merged_final_candidate=True,
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True, no_remaining_scope_removed=True,
        new_unassisted_training_source_code_fit_measured=True, new_held_code_generalization_or_broad_visual_gain_not_established=True,
        source_files_sha256={str(p): file_hash(p) for p in files}, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUT / 'review.json', report)
    protected = set(phase['source_files_sha256']) | set(code['source_files_sha256']) | set(vision['source_files_sha256'])
    for q in queues.values(): protected.update(q['source_files_sha256'])
    for proof in evidence.values(): protected.update(proof['source_files_sha256'])
    docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', ROOT / 'reports/completion-audit.json']
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT / 'review.json'), source_bound_root_documents_preserved=True))); return
    master_path = docs[-1]; master = json.loads(master_path.read_text())
    with (ROOT / 'reports/completion-audit-before-full-size-vision-code-fit-v1.json').open('xb') as stream: stream.write(master_path.read_bytes())
    master.update(updated_at=runtime['at'], requirements={k: v['evidence'] for k, v in requirements.items()},
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)), latest_actual_main_runtime=runtime,
        latest_own_training_source_code_fit=str(paths['code_fit'].relative_to(ROOT)),
        latest_full_size_vision_replay_readback=str(paths['vision'].relative_to(ROOT)),
        latest_full_original_visual_test_input_readback=str(paths['AI2D'].relative_to(ROOT)),
        all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (f'Actual full-size code fit and vision replay, **{runtime["at"]}**: the40-update supervised coding control now passes **3/4 '
        'predeclared public train problems unassisted**, versus0/4initially. The three successful generated programs exactly match the original human '
        'reference/EOS; independent re-execution of all eight actual episodes reproduces the outcomes. This is **training-data fit**, not general '
        'coding accuracy, a contrastive/RL advantage or efficiency proof. Both40-update arms’ complete actual actor/Adam/source/objective states '
        'reconcile, and their full48-case development code assessments remain live.\n\n'
        'A **separate999,706,442-value vision fork** actually completed two photo/word replay updates. Independent private-state reconstruction '
        'matches its **self-contained native text/image/word bundle** bit exactly; all frozen language/actor/critic/head values are unchanged. '
        'Four dev words reduce byte edits104->18 over19bytes, but exact reading stays **0/4**. Eight dev image retrieval scores are mixed, and '
        'image intervention gains slightly fall. **Broad visual improvement is unproven.** All **3,088 original AI2D test questions** and their '
        'declared native image inputs now independently reconstruct; training remains forbidden and exposure audit/model assessment are pending. '
        f'The original main run is active at **{progress["actual_updates"]:,}/35,284 updates** on four L40s; human/code/full-text-QA successors '
        'continue waiting. The experimental code and vision forks have not been merged or promoted. **All ten final requirements remain incomplete.** '
        '[Review](reports/full-size-vision-and-code-fit-v1-review/review.json), '
        '[unassisted source fit](reports/own-execution-contrast-early-source-greedy-CPU-v1-readback/complete.json), '
        '[full-size vision readback](reports/own-actor-vision-replay-full-CPU-v1-readback/complete.json), '
        '[complete diagram test inputs](reports/ai2d-all-original-test-eval-v2-readback/complete.json).\n\n')
    for path in docs[:2]:
        first, tail = path.read_text().split('\n', 1); path.write_text(first + '\n\n' + paragraph + tail.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual_unassisted_train_source_passes='0/4->3/4',
        full_size_vision_native_readback_complete=True, full_visual_original_test_questions=3088, goal_achieved=False)), flush=True)


if __name__ == '__main__': main()
