"""Record the new vision mechanism without promoting it or changing queues."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/vision-alignment-mechanics-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-only full-scope review required')
    previous_path = ROOT / 'reports/conditional-CPU-trial-and-step4096-v1-review/review.json'
    previous = json.loads(previous_path.read_text())
    proof_path = ROOT / 'reports/own-actor-vision-alignment-CPU-v2/complete.json'
    readback_path = ROOT / 'reports/own-actor-vision-alignment-CPU-v2-readback/complete.json'
    proof = json.loads(proof_path.read_text()); reader = json.loads(readback_path.read_text())
    if (not reader['actual_fresh_next_update_losses_and_all_final_weights_bitexact'] or proof['actual_counts']['actual_unique'] != 1116866
            or not proof['actual_text_only_logits_bitexact_after_both_updates'] or proof['actual_full_size_student_trained']
            or not proof['word_head_weights_unchanged_but_shared_vision_output_changed']):
        raise ValueError('Actual small vision update/replay proof and disclosed OCR retention risk required')
    for name, digest in reader['source_files_sha256'].items():
        if file_hash(name) != digest: raise ValueError('Immutable accepted vision evidence changed')
    def identity(owner, waiting=False):
        path = Path('/proc') / str(owner['pid']); fields = (path / 'stat').read_text().split(') ', 1)[1].split()
        if fields[19] != owner['ticks'] or fields[0] in ('Z', 'X'): raise ValueError('Exact live PID/start ticks required')
        children = (path / 'task' / str(owner['pid']) / 'children').read_text().strip()
        visible = next((line.split('=', 1)[1] for line in (path / 'environ').read_bytes().decode().split('\0')
                        if line.startswith('CUDA_VISIBLE_DEVICES=')), None)
        if waiting and (children or visible != ''): raise ValueError('Queued successor must remain CPU-hidden and childless')
        return dict(pid=owner['pid'], ticks=fields[19], state=fields[0], children=children, CUDA_VISIBLE_DEVICES=visible)
    phase_path = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(phase_path.read_text())
    trial_root = ROOT / 'reports/conditional-prefix-full-CPU-v1'
    trial_path = trial_root / 'plan.json'; trial = json.loads(trial_path.read_text())
    queue_names = ['phase-human-preference-v4-queue.json', 'broad-anchored-code-v3-queue.json', 'full-held-QA-pipeline-v1-queue.json']
    queues = {name: json.loads((ROOT / 'reports' / name).read_text()) for name in queue_names}
    children = json.loads((ROOT / 'reports/phase-joint-v3/children.json').read_text())
    workers = [identity(r) for r in children if r['subject'].startswith('train-rank-')]
    if len(workers) != 4: raise ValueError('All four original broad-training ranks required')
    journal = []
    raw = (ROOT / 'reports/phase-joint-v3/training-metrics.jsonl').read_bytes()
    for line in raw.splitlines():
        try: journal.append(json.loads(line))
        except json.JSONDecodeError: break
    if [r['step'] for r in journal] != list(range(1, len(journal) + 1)): raise ValueError('Committed journal prefix must be contiguous')
    updates = {}; endpoints = {}
    for arm in ['anchor_CE', 'anchor_execution']:
        path = trial_root / (arm + '-progress.json')
        updates[arm] = json.loads(path.read_text())['actual_logical_updates'] if path.exists() else 0
    for stage in ['initial', 'anchor_CE-final', 'anchor_execution-final']:
        path = trial_root / (stage + '-actual-evaluation-prefix.json')
        if not path.exists(): continue
        rows = json.loads(path.read_text())
        endpoints[stage] = dict(observed_episodes=len(rows), complete_ten_episode_evaluation=len(rows) == 10,
            source_is_public_train_not_held=True, results=[dict(quarters=r['quarters'], repetition=r['repetition'],
                reward=r['actual_execution_reward']['reward'], stopped_on_generated_EOS=r['actual_rollout']['stopped_on_generated_EOS']) for r in rows])
    runtime = dict(at=datetime.now(timezone.utc).isoformat(), main_owner=identity(phase['owner']), main_workers=workers,
        main_actual_committed_updates=len(journal), main_planned_updates=35284,
        main_actual_committed_targets=int(sum(r['response_EOS_targets'] for r in journal)),
        CPU_conditional_trial_owner=identity(trial['owner']), CPU_conditional_logical_updates=updates,
        original_waiting_successors={name: identity(q['owner'], True) for name, q in queues.items()},
        current_GPU_observation=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'], text=True))
    requirements = previous['requirements']
    requirements['own_architecture']['evidence'] += (' A new own-actor image training view now reuses its exact existing encoder and pointing projections '
        'with no new neural values. It combines full native human-caption/EOS CE with symmetric image-text matching using every observed exact-pixel/caption '
        'association, including repeated-caption positives without incorrectly merging all connected images. Only explicit vision/pointing values receive gradients.')
    requirements['text_and_images']['evidence'] += (' New actual small-architecture proof: four original public human train photos at full448 resolution, '
        '181 complete caption/EOS targets, two logical/three physical optimizer updates, exact next-update weights/Adam/Python/NumPy/TorchCPU RNG recovery, '
        'bit-exact native cold text/image reload, and bit-exact text-only outputs after both updates. A separate fresh reader reconstructs all source images, '
        'pixels, prompts, captions, masks and EOS, then independently executes the actual saved next update and reproduces every final weight exactly. '
        'This model has1,116,866 values and is not the current full-size student. Shared-vision OCR outputs actually change despite fixed word-head weights: '
        'OCR/source replay and held retention assessment remain required before a larger multimodal stage. No new held visual gain, full-size export or four-GPU admission exists.')
    requirements['public_data']['evidence'] += (' Actual source inspection finds12,319 train/835 valid caption rows:11,971/800 original human spoken descriptions '
        'and348/35 Wikimedia metadata captions. Their exact group/stored-image/448pixel hash overlap is zero. Some metadata rows omit a per-row split; only the '
        'frozen manifest file role defines those rows. All four exercised human-photo rows independently have explicit original train roles. This is not a semantic '
        'independence or public-domain license certificate.')
    requirements['serious_full_model_training']['evidence'] = (f'Original full-weight984,399,690-value main run remains live at{len(journal):,}/35,284 '
        f'committed updates and{runtime["main_actual_committed_targets"]:,} supervised targets. The full-size999,706,442-value conditional CPU comparison remains '
        f'live with actual observed updates{updates}. Its matched source-only endpoint observations are retained explicitly; they are not held capability evidence. '
        'The new vision proof exercises only a1,116,866-value small own initialization, not the current full-size model. Full human/code source epochs and final assessments remain queued.')
    requirements['gpu_runtime']['evidence'] += (' All four exact original broad-training GPU ranks are still live. The new vision proof and independent next-update '
        'reexecution used CPU only, initialized no CUDA devices, and did not modify/interrupt any main, human, code or full-QA job. No new vision GPU admission is claimed.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (' Latest conditional source-only baseline/CE endpoint observations are separately retained '
        'without treating partial/incomplete RL comparison results as improvements. The last independently completed full code suite remains0/24completion and '
        '1/24instruction in both old CE/RL arms. General unassisted arbitrary-code correctness/efficiency is not achieved.')
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('Preserve all ten incomplete final requirements')
    files = [Path(__file__), previous_path, proof_path, readback_path, trial_path, phase_path,
             ROOT / 'reports/own-actor-vision-alignment-CPU-v1-preflight-failure.json',
             *(ROOT / 'reports' / name for name in queue_names)]
    OUT.mkdir(); atomic_json(OUT / 'observed-conditional-train-source-endpoints.json', endpoints)
    report = dict(at=runtime['at'], requirements=requirements, actual_runtime=runtime,
        new_actual_small_vision_mechanism_proof=str(proof_path.relative_to(ROOT)),
        independent_actual_next_update_readback=str(readback_path.relative_to(ROOT)),
        conditional_source_only_observations=endpoints, previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        source_files_sha256={str(p): file_hash(p) for p in files}, new_unassisted_held_coding_or_all_domain_gain_measured=False,
        new_full_size_vision_training_or_GPU_admission=False, all_ten_final_requirements_incomplete=True,
        no_remaining_scope_removed=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(OUT / 'review.json', report)
    docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', ROOT / 'reports/completion-audit.json']
    protected = set(phase['source_files_sha256']) | set(trial['source_files_sha256'])
    for queue in queues.values(): protected.update(queue['source_files_sha256'])
    if any(str(p) in protected for p in docs): raise ValueError('Root documents unexpectedly bound by a live/queued job; review retained without modifying them')
    master_path = docs[-1]; master = json.loads(master_path.read_text())
    with (ROOT / 'reports/completion-audit-before-vision-alignment-mechanics-v1.json').open('xb') as stream: stream.write(master_path.read_bytes())
    master.update(updated_at=runtime['at'], requirements={k: v['evidence'] for k, v in requirements.items()},
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)), live_joint_training=runtime,
        latest_small_vision_mechanism_proof=str(proof_path.relative_to(ROOT)),
        latest_small_vision_independent_next_update=str(readback_path.relative_to(ROOT)),
        actual_conditional_trial_observation=endpoints, all_ten_final_requirements_incomplete=True,
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress', candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (f'Own-actor vision alignment mechanism, **{runtime["at"]}**: a new training view combines native caption/EOS CE with '
        'image-text matching through our existing image encoder/pointing projections, adding **zero neural parameters**. Two mechanism tests pass. '
        'A **1,116,866-value small own model** actually processes **four original human train photos at448×448**, keeps all **181 caption/EOS '
        'targets**, and passes two logical updates plus exact next-update/Adam/all-three-CPU-RNG replay. Text-only outputs remain bit exact; native '
        'cold text/image reloads match. A separate reader reconstructs all original pixels/tokens and independently reproduces every final value '
        'by executing the saved next update. **This is not full-size vision training, held visual improvement, a final export, or four-GPU admission.** '
        'Fixed word-head weights still produce changed OCR outputs because vision is shared; OCR replay and independent retention checks remain required.\n\n'
        f'The original full-weight main run remains active at **{len(journal):,}/35,284 updates**; human/code/full-QA successors still wait. '
        f'The full-size conditional coding CPU trial is live, with **{updates}** actual logical updates in this snapshot. The verified4,096 main '
        'checkpoint still supplies the latest paired held packed NLL:0.793504, down8.79% from0.869945; this is token prediction, not a new '
        'unassisted code or broad knowledge result. Last completed full code correctness remains0/24completion and1/24instruction in both old arms. '
        '**All ten final requirements remain incomplete.** Evidence: [review](reports/vision-alignment-mechanics-v1-review/review.json), '
        '[actual vision mechanism](reports/own-actor-vision-alignment-CPU-v2/complete.json), '
        '[fresh next-update readback](reports/own-actor-vision-alignment-CPU-v2-readback/complete.json).\n\n')
    for path in docs[:2]:
        first, tail = path.read_text().split('\n', 1); path.write_text(first + '\n\n' + paragraph + tail.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual_main_updates=len(journal),
        actual_conditional_CPU_updates=updates, new_full_size_vision_training=False, goal_achieved=False)), flush=True)


if __name__ == '__main__': main()
