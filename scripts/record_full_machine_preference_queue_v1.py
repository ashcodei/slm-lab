"""Record the actual full-source machine preference queue and unfinished scope."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/full-machine-preference-v1-queue-review'


def read(path):return json.loads(Path(path).read_text())


def live(identity):
    if terminal(identity):raise ValueError('The exact original controller/worker birth must remain live')
    fields=(Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=identity['pid'],ticks=fields[19],state=fields[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh GPU-hidden review required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path)
    previous_path=ROOT/master['latest_full_scope_review'];requirements=deepcopy(read(previous_path)['requirements'])
    queue_path=ROOT/'reports/full-machine-preference-v1-queue.json';queue=read(queue_path);queued_owner=live(queue['owner'])
    proof_path=ROOT/'reports/parallel-machine-preference-four-CPU-v2/complete.json';proof=read(proof_path)
    tests={'parallel-machine-preference-v1-tests':7,'machine-preference-policy-v1-tests':2,'full-machine-preference-v1-driver-tests':7}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('Actual complete target tests required')
    if (not proof['all_four_CPU_workers_passed'] or not proof['both_matched_arms']
            or proof['actual_unique_original_pairs_per_arm']!=5 or proof['full_source_epoch'] is not False
            or queue['original_machine_pairs_each_arm']!=1620 or queue['planned_updates_each_arm']!=102
            or queue['production_updates']!=0 or queue['human_preferences'] is not False
            or not queue['all_existing_eight_endpoint_populations_preserved']):
        raise ValueError('The actual whole-source queue and limited four-worker proof must remain distinct')
    for worker in proof['proofs']:
        if len(worker['arms'])!=2 or any(not a['exact_next_global_update_weights_Adam_and_private_RNG']
                or a['final_replica_pair_counts']!=[1,0,0,0] for a in worker['arms']):
            raise ValueError('Both actual uneven-tail recovery arms on all four independent workers must pass')
    stage=ROOT/'reports/code-first-continuation-v3';plan=read(stage/'plan.json');progress=read(stage/'progress.json')
    parent=live(plan['owner']);ranks=[p for p in read(stage/'children.json') if p['subject'].startswith('train-rank-')]
    if len(ranks)!=4 or (stage/'failure.json').exists():raise ValueError('All four same production GPU workers must remain successful')
    workers=[live(p) for p in ranks]
    names=['code-first-continuation-v3','phase-human-preference-v7','broad-anchored-code-v6',
        'code-first-final-standalone-CPU-v3','full-held-QA-pipeline-v4','full-AI2D-pipeline-v4',
        'full-original-MBPP-pipeline-v4','AI2D-fixed-exposure-strata-v4','full-machine-preference-v1']
    protected=set();controllers=[]
    for name in names:
        path=ROOT/'reports'/name/'plan.json'
        if not path.exists():path=ROOT/'reports'/(name+'-queue.json')
        item=read(path);protected.update(item.get('source_files_sha256',{}));controllers.append(dict(stage=name,owner=live(item['owner'])))
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,code_first_parent=parent,actual_four_same_training_workers=workers,
        actual_code_first_updates=progress['actual_updates'],planned_code_first_updates=7558,
        new_machine_preference_owner=queued_owner,new_machine_preference_production_updates=0,all_nine_live_controllers=controllers,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        current_GPU_run_and_eight_existing_controllers_not_preempted_or_replaced=True)
    notes={
        'own_architecture':' A new complete-program machine-preference trainer now supports four actual replicas with a single global pair/token normalization and gradient SUM, a frozen snapshot of the actual nonzero own starting actor, and a matched chosen-code control. No new student parameters are added. The code is a custom implementation, not a claim of global architectural novelty.',
        'public_data':' The full1620original admitted execution preferences are now wired to a real production controller, not just the two-pair CPU source-learning experiment. Each arm uses every whole native chosen/rejected program and EOS exactly once, with global batch16and102updates including a four-pair final batch. Rejected programs remain preference-only and never positive CE targets. The separate public-source admission and original finite-case qualifications remain unchanged.',
        'under_1b_parameters':' The source-bound loader requires the actual999706442-value own code/RL actor in the production driver. Its private own-reference adapters are training-only and excluded from student exports. The new loader unit tests cold-load complete small-network source epochs with exact text/image outputs and reject partial counts, wrong base/source identities, changed unused critics and human-label substitutions. This does not replace the queued full-size GPU cold-load gate or later standalone dense export.',
        'text_and_images':' The new full production comparison schedules all16961original held QA questions, all3088original visual questions under proper/gray/fixed-wrong native448images, and all600original MBPP problems at the initial own code actor and both new machine-training endpoints. The original small OCR, object and generic-photo checks are also retained. These are queued, not measured visual/reading gains; a fixed encoder alone does not guarantee retained image-conditioned answers.',
        'human_feedback':' Machine execution labels remain explicitly nonhuman in data, update metrics, recovery, export, queue and reports. The original85786human-pair chosenCE/DPOstage stays unchanged. Each new machine endpoint also schedules all4642held human-ranked preference pairs for retention. There are still no new human judgments on the new model outputs.',
        'serious_full_model_training':f' The same current full dense code/state/replay production run remains live at{progress["actual_updates"]}/7558updates. The new machine-preference stage is now implemented, tested and actually queued, but has0production updates and fresh full-size GPU capacity is still required. It starts from the full completed broad-anchored-code-v6anchor_executionactor after the intact eight-endpoint pipeline; neither an older partial actor nor a zero-base reference is substituted. The new objective updates all561actor tensors, while current pretraining updates the full selected dense/vision/word groups.',
        'high_end_llm_capability':' No new generated-code or held capability gain is measured in this queueing turn. The previous full-size two-pair CPU trial still has0correct-program ranking flips, and the latest completed main still has0/48local functional coding passes. Correct global normalization, source coverage and exact recovery are implementation evidence, not evidence of frontier-level capability. The original all-round and GPT6.1-level aspirations remain unachieved.',
        'broad_independent_benchmarks':' All original eight full-assessment endpoints and all original source populations/budgets are preserved. The new machine branch adds three complete fixed endpoints and reuses the full original16961QA,3088AI2Dtimes3conditions,600MBPP/1816case and4642human-pair protocols, plus the small native development checks. Existing source-exposure disclosures remain; no all-history semantic-independence certificate or official leaderboard result is claimed. Independent outcome readback and new standalone finalization remain pending.',
        'gpu_runtime':' The new CPU-only waiting owner is live with its exact birth and predecessor identities recorded. It acquires none of the currently occupied L40s. Before production it must observe the exact last original GPU owner and every registered child terminal, verify all eight complete original MBPP endpoints and all3721source code/RL receipts, then acquire the actual kernel GPU lock with all four GPUs idle. The current four GPU rank births and all eight original controllers remain live unchanged. No signal or restart is performed in this turn.',
        'any_code_correctness_and_efficiency':' Sixteen targeted tests pass across the existing machine objective, new replica gradients, own-policy cold loading and the new production driver. An actual four-worker Gloo experiment on five whole admitted pairs per arm checks global gradients, a one-pair final batch with three empty replicas, identical final worker/Adam values, corrupt-state rejection and exact actual next-update/private-RNG replay. Its first version failed because it passed a dataset folder to a constructor requiring the repository root; that failed script/log are preserved, and version2completes. No full1620epoch or arbitrary-code/efficiency gain is inferred from this small-network proof.'}
    for key,note in notes.items():requirements[key]['evidence']+=note
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('All ten original final requirements must remain explicitly incomplete')
    OUT.mkdir();atomic_json(OUT/'runtime.json',runtime)
    sources=[Path(__file__),previous_path,queue_path,proof_path,OUT/'runtime.json',
        ROOT/'scripts/train_full_machine_preference_v1.py',ROOT/'src/slm_lab/parallel_machine_preference_v1.py',
        ROOT/'src/slm_lab/machine_preference_policy_v1.py',*[ROOT/'reports'/(name+'.log') for name in tests]]
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,actual_runtime=runtime,
        previous_full_scope_review=str(previous_path.relative_to(ROOT)),source_files_sha256={str(p):file_hash(p) for p in sources},
        all_ten_final_requirements_incomplete=True,no_original_scope_removed=True,
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):raise ValueError('An actual bound source document must remain unchanged')
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_full_machine_preference_four_worker_proof=str(proof_path.relative_to(ROOT)),
        queued_full_machine_preference_comparison=str(queue_path.relative_to(ROOT)),
        next_full_machine_preference_experiment=str(queue_path.relative_to(ROOT)),
        next_required_work=['Finish the original7558update code/state/replay epoch and all scheduled native assessments.',
            'Complete all85786human preference pairs and all3721original code/RL sources in both matched arms, with the original standalone export and all eight full benchmark endpoints.',
            'When its exact predecessor finishes, run the queued fresh four-L40 machine-preference capacity/recovery/cold checks, then all1620original pairs once in each102update arm and all full original branch assessments.',
            'Independently read actual generated outcomes, compare broad/visual/human/code retention and finalize a standalone own student before any promotion or high-end capability claim.'],
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Full-source machine preference comparison queued, **{at}**: the same four-L40 dense code/state run remains live at '
        f'**{progress["actual_updates"]}/7,558 updates**. A new production stage now schedules both correct-code CE and machine DPO '
        'on **all 1,620 admitted whole-program execution preferences, once per arm, in 102 updates each**. Both start from our '
        'complete own code/RL actor. The original eight-endpoint pipeline stays intact; the new stage waits after its last GPU owner.\n\n'
        '**Sixteen targeted tests pass**, and four actual CPU workers pass both matched five-pair recovery/uneven-tail experiments. '
        '**New production updates: 0; full-size four-L40 capacity remains pending. No new coding or held capability gain is established.** '
        'The branch also schedules complete original QA, visual QA, MBPP and held human-preference assessments. All ten final '
        'requirements remain incomplete. [Actual queue](reports/full-machine-preference-v1-queue.json), '
        '[four-worker proof](reports/parallel-machine-preference-four-CPU-v2/complete.json), '
        '[full review](reports/full-machine-preference-v1-queue-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),live_code_first_updates=progress['actual_updates'],
        actual_new_machine_queue_owner=queued_owner,new_machine_production_updates=0,all_ten_final_requirements_incomplete=True,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
