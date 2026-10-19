"""Record completed native comparisons and an exact current training snapshot."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/completed173-and-phase3-review-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():
        raise ValueError('Fresh CPU-only status snapshot required')
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    OUT.mkdir()
    bindings = {}

    def read(path):
        path = ROOT / path
        bindings[str(path)] = file_hash(path)
        return json.loads(path.read_text())

    at = datetime.now(timezone.utc).isoformat()
    outcomes = read('reports/completed-execution-outcomes-readback-v1/complete.json')
    state = read('reports/execution-final173-readback-v1/complete.json')
    reward = read('reports/execution-final173-reward-replay-v1/complete.json')
    capacity = read('reports/phase-joint-v3/capacity.json')
    full_plan = read('reports/phase-joint-v3/plan.json')
    queue = read('reports/phase-human-preference-v4-queue.json')
    experts = read('reports/own-task-experts-full-CPU-v1/complete.json')
    read('reports/completed-execution-tokenizer-packaging-v1.json')
    read('reports/completed-execution-assessment-v4/restored.json')
    read('reports/control160-original-vs-recovered-v1.json')
    if (not capacity['passed'] or outcomes['native_code_outputs_reexecuted'] != 96
            or experts['actual_counts']['actual_unique'] != 999924554):
        raise ValueError('Actual completed native and full-sized architecture evidence required')

    def identity(record):
        path = Path('/proc') / str(record['pid'])
        fields = (path / 'stat').read_text().split(') ', 1)[1].split()
        if fields[19] != record['ticks'] or fields[0] in ['Z', 'X']:
            raise ValueError('Exact live owner identity required')
        return dict(pid=record['pid'], ticks=fields[19], state=fields[0],
            argv=(path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(),
            CUDA_VISIBLE_DEVICES=next((x.split('=', 1)[1] for x in (path / 'environ').read_bytes().decode().split('\0')
                                      if x.startswith('CUDA_VISIBLE_DEVICES=')), None),
            children=(path / 'task' / str(record['pid']) / 'children').read_text().split())

    current_owner = identity(full_plan['owner']); human_owner = identity(queue['owner'])
    journal_path = ROOT / 'reports/phase-joint-v3/training-metrics.jsonl'
    raw = journal_path.read_bytes(); lines = raw.splitlines()
    rows = []
    for line in lines:
        try: rows.append(json.loads(line))
        except json.JSONDecodeError: break  # Only a complete committed prefix is evidence.
    if not rows or [r['step'] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('Contiguous actual production journal required')
    atomic_json(OUT / 'phase-production-journal-prefix.json', rows)
    progress = json.loads((ROOT / 'reports/phase-joint-v3/progress.json').read_text())
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
                          '--format=csv,noheader'], text=True, capture_output=True, check=True).stdout
    observation = dict(at=at, owner=current_owner, actual_committed_updates=len(rows),
        planned_updates=full_plan['planned_steps'], actual_targets=int(sum(r['response_EOS_targets'] for r in rows)),
        planned_targets=467581587, phase=rows[-1]['phase'], progress=progress,
        GPU_observation=gpu, queued_human_owner=human_owner,
        queue_status=queue['status'], full_training_complete=False, goal_achieved=False)
    atomic_json(OUT / 'runtime-snapshot.json', observation)
    requirements = dict(
        own_architecture='Own scratch-origin backbone, tokenizer and vision remain. Separate code/assistant/reasoning adapters now instantiate at full size with one shared backbone and critic. Two meaningful small-model tests and one full-size complete training example demonstrate isolation and cold reload; automatic routing and generalization remain unproven.',
        public_data='28 selected source families contain 669,863 family records and 467,581,587 next/response/EOS targets. The active full-source pass includes books, multilingual code, OS/network/security code, factual QA, ordinary photographs, image text and other existing audited sources. Explicit repeated exercise references and semantic overlap remain disclosed; this is not all public or open-source data.',
        under_1b_parameters='Completed single-adapter code policies count 999,706,442 unique values. Active broad policy counts 984,399,690. The separate full-size three-adapter experiment counts 999,924,554, including every shared head/router and one critic. A later training-only own-foundation reference is excluded from the exported policy. No final capable model is selected.',
        text_and_images='Completed control/RL endpoints remain weak: OCR exact 0/20 each, object proper 3/15 with blank 4/15 and wrong 3/15, caption retrieval producer observations I2T 1/32 and T2I 2/32. The active later phase trains shared vision and word recognition through a fixed decoder. New task adapters freeze the same vision encoder and do not add image learning.',
        human_feedback='Execution feedback is machine verification, not human RLHF. The live successor waits for the complete broad foundation, then compares chosen-CE and genuine human-DPO on all 85,786 original public human-ranked pairs per arm, with 4,642 held pairs. Those rankings concern public other-model answers; fresh human judgments of our own outputs are still absent. Full-size human production updates remain zero.',
        serious_full_model_training=f'Both controlled code arms completed 173 updates, including explicit numerical-method migration after 138. All four final actor/Adam/private-state readbacks reconcile. The broader full-weight run passed actual four-L40 extremal-context/recovery admission and has {len(rows):,}/35,284 committed updates and {observation["actual_targets"]:,}/467,581,587 targets in this snapshot. Its complete source epoch remains outstanding.',
        high_end_llm_capability='NOT ACHIEVED. Both code endpoints recover exactly one local instruction task, sum_even, from a warm-up score of 0/24. Both completion scores remain 0/24; the older pre-warm-up initial model had 1/24 completion. Broad QA, math and image results are poor. There is no evidence of GPT6.1-class or general high-end capability.',
        broad_independent_benchmarks='All 96 completed control/RL code outputs and 48 CPU-FP32 comparison outputs were independently executed. Generated knowledge, exact-rational math, OCR edits and object intervention verdicts were recalculated against original sources. Saved NLL aggregates were recalculated; neural generation and retrieval embeddings were not replayed. These repeatedly observed development subsets are not independent final benchmarks.',
        gpu_runtime='The exact phase-v3 owner and four production workers use the four local NVIDIA L40s after successful actual full-size capacity and recovery controls. The exact human-v4 waiting owner is CPU-only until the full-source run, native assessments and restoration complete. No external GPU, pretrained model, tokenizer, vision weights or live teacher has been acquired.',
        any_code_correctness_and_efficiency='NOT ACHIEVED. Supervised-control and execution-hybrid each pass correctness and bounded efficiency stress on only 1/24 instruction tasks; completion is 0/24. This one recovered function passed independent fresh candidate/reference execution. Runtime errors, unsupported-subset errors, invalid syntax and incorrect algorithms remain common. RL has no coding advantage over CE; finite stress measurements do not prove optimal algorithms or arbitrary-code competence.')
    review = dict(at=at, objective_preserved=True, previous_goal_turn_classification='progress',
        current_goal_turn_classification='progress', requirements={k: dict(status='incomplete_for_requested_final_model', evidence=v) for k, v in requirements.items()},
        completed_code_updates_each_arm=173, native_code=outcomes['code'], broad=outcomes['broad'],
        prequential_training_reward_diagnostics=reward['summaries'],
        actual_runtime=observation, actual_full_size_task_adapter_experiment=experts,
        original_completed_training_not_repeated_for_tokenizer_packaging_repair=True,
        numerical_transport_migration_preserved_original_control160_values_Adam_RNG=True,
        single_local_instruction_recovery_proven=True, RL_coding_advantage_over_CE_proven=False,
        broad_or_frontier_capability_improvement_proven=False, no_remaining_scope_removed=True,
        all_ten_final_requirements_incomplete=True, candidate_promoted=False, goal_achieved=False)
    for p in [OUT / 'runtime-snapshot.json', OUT / 'phase-production-journal-prefix.json', Path(__file__)]:
        bindings[str(p)] = file_hash(p)
    review['source_files_sha256'] = bindings
    atomic_json(OUT / 'review.json', review)

    audit_path = ROOT / 'reports/completion-audit.json'
    audit = json.loads(audit_path.read_text())
    backup = ROOT / 'reports/completion-audit-before-completed173-and-phase3-v1.json'
    with backup.open('xb') as f: f.write(audit_path.read_bytes())
    audit.update(goal_achieved=False, updated_at=at, requirements=requirements,
        previous_goal_turn_classification='progress', current_goal_turn_classification='progress',
        latest_full_scope_review=str((OUT / 'review.json').relative_to(ROOT)),
        latest_native_development_readback='reports/completed-execution-outcomes-readback-v1/complete.json',
        current_posttraining=dict(status='both_arms_training_and_native_evaluation_completed', actual_updates_each_arm=173,
            instruction_correct_each_arm='1/24', completion_correct_each_arm='0/24', RL_coding_advantage_proven=False),
        live_joint_training=observation, queued_full_source_phase_training=None,
        queued_human_preference_comparison=dict(owner=human_owner, train_pairs_each_arm=85786,
            held_pairs=4642, planned_updates_each_arm=5362, production_updates=0,
            queue='reports/phase-human-preference-v4-queue.json'),
        new_task_adapter_experiment=dict(report='reports/own-task-experts-full-CPU-v1/complete.json',
            actual_unique_parameters=999924554, full_size_disposable_updates=1,
            held_capability_gain_measured=False, automatic_routing_trained=False),
        all_ten_final_requirements_incomplete=True, candidate_promoted=False)
    atomic_json(audit_path, audit)
    text = f'''Current measured status, **{at}**: both supervised-control and execution-hybrid code arms completed **173 updates**. Independent execution of every completed output confirms **1/24 correct and efficient instruction tasks each**, up from warm-up 0/24; both completion scores remain **0/24**. Both recover the same Python `sum_even` function. RL shows **no coding advantage over CE**. The older pre-warm-up model had 1/24 completion, so this is a narrow recovery, not a general capability breakthrough. Generated knowledge remains 1/56; math is control 0/16 and RL 1/16; OCR exact remains 0/20; object answers with real images score 3/15, below blank-image 4/15. Retrieval producer observations remain I2T 1/32 and T2I 2/32. Full final independent benchmarks remain required.

The **984,399,690-parameter** broad foundation is now actively training on all four local L40s after successful actual full-size capacity/recovery checks: this snapshot records **{len(rows):,}/35,284 updates** and **{observation['actual_targets']:,}/467,581,587 targets**. The 28-family curriculum covers books, broad and systems code, factual QA, ordinary photographs, OCR and other audited public sources, with declared repeated references and overlap. This complete selected-source pass must still finish. The live human-v4 successor then compares chosen-CE and genuine human-DPO on **85,786 public human-ranked pairs per arm**, including all **4,642 held pairs**. Human production has not started; these public rankings are not fresh judgments of our own answers.

A separate custom task-adapter experiment fits **999,924,554 total unique parameters**: one own backbone, separate code/assistant/reasoning adapters, all original heads/router and one shared critic. Two focused tests and an actual full-size complete train-example update verify selected-expert learning, unchanged base/other experts, and exact cold reload. **Automatic routing is untrained and no capability gain has been measured for this architecture.** It does not change the active baseline run.

All **ten final-model requirements remain incomplete**, and no candidate is promoted. See the [current full review](reports/completed173-and-phase3-review-v1/review.json), [independent outcomes](reports/completed-execution-outcomes-readback-v1/complete.json), [live full-source progress](reports/phase-joint-v3/progress.json), [human successor](reports/phase-human-preference-v4-queue.json), and [full-size adapter experiment](reports/own-task-experts-full-CPU-v1/complete.json). Entries below are historical snapshots and may describe owners that have since ended.

'''
    for name in ['README.md', 'RESEARCH.md']:
        path = ROOT / name
        header, rest = path.read_text().split('\n\n', 1)
        path.write_text(header + '\n\n' + text + rest)
    print(json.dumps(dict(review=str(OUT / 'review.json'), actual_phase_updates=len(rows),
        actual_phase_targets=observation['actual_targets'], actual_expert_parameters=999924554,
        all_ten_final_requirements_incomplete=True)), flush=True)


if __name__ == '__main__':
    main()
