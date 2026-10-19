"""Preserve full goal scope while recording actual new coding evidence."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/own-negative-contrast-queued-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden full-scope review required')
    previous_path = ROOT / 'reports/vision-alignment-mechanics-v1-review/review.json'
    previous = json.loads(previous_path.read_text())
    readback_path = ROOT / 'reports/conditional-prefix-full-CPU-v1-training-readback/complete.json'
    readback = json.loads(readback_path.read_text())
    data_path = ROOT / 'data/own-execution-negative-pairs-CPU-v1/manifest.json'; data = json.loads(data_path.read_text())
    proof_path = ROOT / 'reports/own-execution-contrast-CPU-v1-mechanics/complete.json'; proof = json.loads(proof_path.read_text())
    if (readback['actual_logical_updates_each_arm'] != 3 or readback['actual_original_training_candidates_freshly_reexecuted'] != 24
            or readback['actual_final_differing_actor_tensors'] != 561 or data['pairs'] != 20
            or data['genuine_EOS_negatives'] != 6 or data['censored_full512_action_negatives'] != 14
            or not proof['next_actual_update_weights_Adam_and_private_CPU_RNG_bitexact']):
        raise ValueError('Actual full-size state readback, real20-pair data and small recovery proof required')
    for evidence in [readback, proof]:
        for path, digest in evidence['source_files_sha256'].items():
            if file_hash(path) != digest: raise ValueError('Accepted new evidence changed')
    def identity(owner, waiting=False):
        proc = Path('/proc') / str(owner['pid']); fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
        if fields[19] != owner['ticks'] or fields[0] in ['Z', 'X']: raise ValueError('Exact live PID/start ticks required')
        children = (proc / 'task' / str(owner['pid']) / 'children').read_text().strip()
        visible = next((s.split('=', 1)[1] for s in (proc / 'environ').read_bytes().decode().split('\0') if s.startswith('CUDA_VISIBLE_DEVICES=')), None)
        if waiting and (children or visible != ''): raise ValueError('Waiting owner must be CPU-hidden and childless')
        return dict(pid=owner['pid'], ticks=fields[19], state=fields[0], children=children, CUDA_VISIBLE_DEVICES=visible)
    phase_path = ROOT / 'reports/phase-joint-v3/plan.json'; phase = json.loads(phase_path.read_text())
    cpu_path = ROOT / 'reports/conditional-prefix-full-CPU-v1/plan.json'; cpu = json.loads(cpu_path.read_text())
    queue_names = ['phase-human-preference-v4-queue.json', 'broad-anchored-code-v3-queue.json',
                   'full-held-QA-pipeline-v1-queue.json', 'own-execution-contrast-full-CPU-v1-queue.json']
    queues = {name: json.loads((ROOT / 'reports' / name).read_text()) for name in queue_names}
    workers = [identity(row) for row in json.loads((ROOT / 'reports/phase-joint-v3/children.json').read_text()) if row['subject'].startswith('train-rank-')]
    if len(workers) != 4: raise ValueError('All four exact broad-training workers required')
    progress = json.loads((ROOT / 'reports/phase-joint-v3/progress.json').read_text())
    runtime = dict(at=datetime.now(timezone.utc).isoformat(), main_owner=identity(phase['owner']), main_workers=workers,
        main_actual_published_updates=progress['actual_updates'], main_planned_updates=35284,
        full_size_conditional_owner=identity(cpu['owner']), actual_conditional_updates_each_arm=3,
        full_size_conditional_final_endpoint_assessment_complete=(ROOT / 'reports/conditional-prefix-full-CPU-v1/complete.json').exists(),
        waiting_successors={name: identity(queue['owner'], True) for name, queue in queues.items()},
        original_GPU_observation=subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used', '--format=csv,noheader'], text=True))
    requirements = previous['requirements']
    requirements['own_architecture']['evidence'] += (' A new CPU-only contrastive code kernel and distinct native-policy format now combine always-on original '
        'human-code CE, an own-initial-policy cached full-context KL anchor, and length-normalized execution preferences on actual own wrong unassisted outputs. '
        'It is inspired by SimPO with explicitly declared extra CE/KL and censored negatives; novelty or paper-result replication is not claimed.')
    requirements['public_data']['evidence'] += (' All20 actual unassisted failures from the fixed four-source neural diagnostic are now preserved as negative '
        'training sequences, paired with unchanged public human reference programs. All20wrong deliveries and all four original references were freshly executed: '
        'references receive full finite-case credit and negatives zero. Six wrong sequences genuinely generate EOS; fourteen retain their exact512-action censoring '
        'without fabricated EOS or omitted failure rows. There are2,850 positive/EOS targets with pair repetitions and8,313 actual negative actions. Only four '
        'original train problems are represented; this is not new human preference collection, full open-source coverage or a complete3,721-source epoch.')
    requirements['human_feedback']['evidence'] += (' The new contrastive pairs use machine execution preferences, explicitly not human rankings or genuine '
        'human RLHF. The existing full public-human-ranking controlled GPU successor still waits for the complete main run.')
    requirements['serious_full_model_training']['evidence'] = (f'The original984,399,690-value broad main run remains active at '
        f'{progress["actual_updates"]:,}/35,284 published updates, with all four original L40 workers live. Both999,706,442-value conditional CPU arms have '
        'actually completed three logical updates each. Independent raw-state readback verifies all561 selected Adam clocks3, finite moments and nonnegative '
        'second moments, every native/private actor tensor, valid private CPU RNG, fixed base/critic, all498 original reference/EOS targets per arm, and all24 '
        'actual sampled candidate rewards/advantages/action denominators/policy surrogates by fresh original program execution. The first flat-reward updates '
        'match across arms; all561 final actor tensors differ after the nonflat RL update. Endpoint quality is still incomplete. A new exact CPU-hidden owner '
        'is queued to compare40updates per arm across two complete passes over20 fixed own-negative pairs, then assess all48 native local development code cases '
        'per endpoint and all four original train problems unassisted. It has no production updates yet and must wait for the exact earlier CPU owner to terminate. '
        'Original complete main/human/code GPU curricula and final assessments remain required.')
    requirements['under_1b_parameters']['evidence'] += (' The queued contrastive trial declares999,706,442 values per complete exported student, including '
        'all base/actor/critic/vision/auxiliary heads. Its KL target uses cached distributions from the actual own initial student, with no additional neural '
        'reference model and no teacher or training examples in inference. Adam/recovery copies and distribution caches are training-only; this is not a '
        'claim that all training memory contains fewer than1B numeric values. Full-size new execution of this method is still pending.')
    requirements['gpu_runtime']['evidence'] += (' The additional own-negative trial is an exact live, CPU-hidden, childless waiting process and owns no GPUs. '
        'Its meaningful kernel/IO/order tests and actual small next-update recovery proof passed. It does not establish new four-L40 capacity or alter the '
        'frozen main/human/code/full-QA pipeline. A separate CPU-only observer is waiting for the main6144 checkpoint; no6144 held result is claimed here.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (' Actual matched full-size conditional CE baseline/final source-only evaluations '
        'remain all-zero rewards across their ten episodes, with unchanged unassisted greedy output. The RL endpoint assessment is still running. Distinct '
        'weights/valid RL gradients do not prove better coding. The last completed local full code suite still has0/24completion and1/24instruction in both '
        'old arms. New40-update contrastive endpoints schedule complete native syntax/function/efficiency development assessments, with no automatic promotion.')
    if len(requirements) != 10 or any(r['status'] != 'incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('Preserve all ten incomplete final requirements')
    files = [Path(__file__), previous_path, readback_path, data_path, proof_path, phase_path, cpu_path,
        ROOT / 'scripts/train_own_execution_contrast_full_CPU_v1.py', ROOT / 'scripts/read_phase_joint_6144_v1.py',
        *(ROOT / 'reports' / name for name in queue_names)]
    OUT.mkdir(); report = dict(at=runtime['at'], requirements=requirements, actual_runtime=runtime,
        actual_full_size_conditional_training_readback=str(readback_path.relative_to(ROOT)),
        new_own_execution_negative_pair_manifest=str(data_path.relative_to(ROOT)),
        actual_small_contrastive_update_recovery_proof=str(proof_path.relative_to(ROOT)),
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        new_unassisted_held_coding_or_all_domain_gain_measured=False, all_ten_final_requirements_incomplete=True,
        no_remaining_scope_removed=True, candidate_promoted=False, goal_achieved=False,
        source_files_sha256={str(p): file_hash(p) for p in files})
    atomic_json(OUT / 'review.json', report)
    docs = [ROOT / 'README.md', ROOT / 'RESEARCH.md', ROOT / 'reports/completion-audit.json']
    protected = set(phase['source_files_sha256']) | set(cpu['source_files_sha256'])
    for queue in queues.values(): protected.update(queue['source_files_sha256'])
    if any(str(p) in protected for p in docs): raise ValueError('Root documents unexpectedly source-bound; review retained without changes')
    master_path = docs[-1]; master = json.loads(master_path.read_text())
    with (ROOT / 'reports/completion-audit-before-own-negative-contrast-v1.json').open('xb') as stream: stream.write(master_path.read_bytes())
    master.update(updated_at=runtime['at'], latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)),
        requirements={k: v['evidence'] for k, v in requirements.items()}, live_joint_training=runtime,
        latest_full_size_conditional_training_readback=str(readback_path.relative_to(ROOT)),
        own_execution_contrastive_trial_queue='reports/own-execution-contrast-full-CPU-v1-queue.json',
        all_ten_final_requirements_incomplete=True, previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        candidate_promoted=False, goal_achieved=False)
    atomic_json(master_path, master)
    paragraph = (f'Own execution-negative contrastive trial queued, **{runtime["at"]}**: all20 actual unassisted wrong deliveries from four '
        'original public train problems are paired with freshly verified human solutions. Six negatives end with generated EOS; fourteen preserve their '
        'actual512-action censoring. The new CPU-hidden owner waits for the exact existing CPU trial, then compares **40updates per arm**, full original '
        'CE versus CE plus length-normalized preferences, with an own-initial cached KL anchor. All **48 native local development code cases per endpoint** '
        'and four source problems will be evaluated after training. Kernel/IO/order tests and an actual small-model next-update recovery proof pass; '
        '**new full-size contrastive updates have not started**. This is machine feedback, not human RLHF.\n\n'
        'Independent inspection now confirms both existing **999,706,442-value** conditional actors actually completed three updates: all561 Adam '
        'clocks3, native/private weights exact, fixed base/critic, and all24 sampled training candidates freshly re-executed. The final actors differ '
        'in561 tensors. The CE arm’s matched source evaluations remain zero; RL endpoint assessment is still running. **Distinct weights do not prove '
        f'improved coding.** The original four-GPU main run is active at **{progress["actual_updates"]:,}/35,284 published updates**. All ten final '
        'requirements remain incomplete. [Review](reports/own-negative-contrast-queued-v1-review/review.json), '
        '[full-size training readback](reports/conditional-prefix-full-CPU-v1-training-readback/complete.json), '
        '[own negative corpus](data/own-execution-negative-pairs-CPU-v1/manifest.json).\n\n')
    for path in docs[:2]:
        first, tail = path.read_text().split('\n', 1); path.write_text(first + '\n\n' + paragraph + tail.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT / 'review.json'), main_published_updates=progress['actual_updates'],
        actual_three_update_RL_actor_differs=True, new40_update_trial_waiting=True, goal_achieved=False)), flush=True)


if __name__ == '__main__': main()
