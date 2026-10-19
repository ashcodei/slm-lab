"""Record the actual complete standalone evaluation queue without claiming gains."""
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
OUT=ROOT/'reports/standalone-full-assessment-v1-queue-review'


def read(path):return json.loads(Path(path).read_text())


def live(identity):
    if terminal(identity):raise ValueError('The exact original controller/worker birth must remain live')
    f=(Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=identity['pid'],ticks=f[19],state=f[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden full-scope review required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous_path=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous_path)['requirements']);queue_path=ROOT/'reports/machine-standalone-full-assessment-v1-queue.json';queue=read(queue_path)
    assessment_owner=live(queue['owner']);tests={'standalone-held-human-v1-tests':4,'standalone-word-shards-v1-tests':3,'machine-standalone-full-v1-driver-tests':11}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('Every actual final standalone scoring/dispatch test must pass')
    if (queue['expected_student_values']!=984399690 or set(queue['suites'])!={'code','broad','preferences','QA','AI2D','MBPP','words'}
            or queue['full_QA_questions']!=16961 or queue['full_AI2D_questions']!=3088 or queue['full_MBPP_problems']!=600
            or queue['held_human_pairs']!=4642 or queue['held_word_crops']!=3429 or queue['native_word_generation_budget']!=64
            or queue['word_image_conditions']!=['proper','blank','wrong'] or queue['model_updates']!=0
            or not queue['old_actor_or_reference_weights_not_loaded_for_neural_inference']):
        raise ValueError('The actual full original standalone assessment scope must remain fixed')
    source=ROOT/'reports/code-first-continuation-v3';plan=read(source/'plan.json');progress=read(source/'progress.json')
    parent=live(plan['owner']);ranks=[p for p in read(source/'children.json') if p['subject'].startswith('train-rank-')]
    if len(ranks)!=4 or (source/'failure.json').exists():raise ValueError('All same original production workers must remain healthy')
    workers=[live(r) for r in ranks];protected=set();controllers=[]
    for name in ['code-first-continuation-v3','phase-human-preference-v7','broad-anchored-code-v6',
        'code-first-final-standalone-CPU-v3','full-held-QA-pipeline-v4','full-AI2D-pipeline-v4','full-original-MBPP-pipeline-v4',
        'AI2D-fixed-exposure-strata-v4','full-machine-preference-v1','machine-preference-final-standalone-CPU-v1','machine-standalone-full-assessment-v1']:
        path=ROOT/'reports'/name/'plan.json'
        if not path.exists():path=ROOT/'reports'/(name+'-queue.json')
        row=read(path);protected.update(row['source_files_sha256']);controllers.append(dict(stage=name,owner=live(row['owner'])))
    neural=ROOT/'reports/code-first-step2048-native-code-CPU-v1';neural_plan=read(neural/'plan.json');protected.update(neural_plan['source_files_sha256'])
    native_owner=live(neural_plan['owner']);native_progress=read(neural/'progress.json');baseline_path=neural/'initial/completion-complete.json';baseline=read(baseline_path)
    if (baseline['actual_results']['original_native_tasks']!=24 or baseline['actual_results']['native_functionally_correct']!=0
            or (neural/'failure.json').exists()):raise ValueError('The actual partial starting-model frame must not be substituted for trained-model outcomes')
    at=datetime.now(timezone.utc).isoformat();runtime=dict(at=at,actual_code_first_owner=parent,actual_four_same_GPU_workers=workers,
        actual_code_first_updates=progress['actual_updates'],planned_code_first_updates=7558,all_eleven_live_pipeline_controllers=controllers,
        actual_standalone_assessment_owner=assessment_owner,actual_CPU_coding_owner=native_owner,actual_CPU_coding_progress=native_progress,
        new_standalone_assessment_deliveries=0,new_model_updates=0,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        original_GPU_training_and_all_previous_controllers_preserved=True)
    notes={
        'own_architecture':' A new full evaluation controller is now implemented and actually queued for the complete984399690-value standalone machine-DPOstudent. It loads only the packaged own dense weights and tokenizer for neural inference, without attaching an actor or an older backbone/reference model. No new architecture parameters or globally novel model design are claimed by this evaluation work.',
        'public_data':' The new standalone endpoint retains all original16961QA questions,3088visual questions under3conditions,600MBPP problems/1816original cases,4642held human pairs and3429held word crops under3conditions. A real-data scope test rederives all3429crop positions,68147native context tokens and16712response/EOS targets and the same full-population one-to-one wrong-image mapping across four replicas. No new training source or benchmark filtering is introduced.',
        'under_1b_parameters':' The standalone loader counts every actually loaded dense/vision/word/auxiliary tensor and rejects a mismatch with the fixed984399690-value endpoint, wrong packaged-tokenizer identity, wrong source receipt or incomplete102update cursor. A real small-network cold-load test verifies every parameter digest and actual image-conditioned outputs without an older actor. The full-size export and its actual production assessment remain pending; small-network tests do not prove full-size quality.',
        'text_and_images':' The final assessment now explicitly evaluates both the learned byte-CTC word head and native image-conditioned text generation on all3429held crops, with proper/blank/wrong images and a fixed64token native budget. Four replicas preserve the original full-population donors and all source crops/targets exactly once. Tests compare both neural paths with the original serial evaluator, reconstruct raw byte paths/literal scores and reject altered donors or dropped/duplicated sources. These are word crops, not full-photo detection; generic photos, OCR and full visual QA remain separate checks.',
        'human_feedback':' A new standalone-specific evaluator runs only the dense student on every complete held human chosen/rejected response and EOS mask. It reuses the exact original frozen-own-base reference scores from the completed machine-actor held assessment, with explicit source/runtime/profile/SHA bindings and full source readback. It loads no reference neural weights and clearly reports cached scores rather than new reference forwards. Four actual small-model scoring/corruption tests pass. This is retention on human-ranked other-model answers, not fresh human judgments of new student outputs.',
        'serious_full_model_training':f' The actual four-L40 dense code/state/replay training continues at{progress["actual_updates"]}/7558updates with the same parent and rank births. The new controller performs no updates and waits for both complete1620pair machine arms, all original machine endpoint assessments, every original GPU child terminal/restored, and the complete standalone CPU export. The original full-model/human/code/RL pipeline stays unchanged; no narrow test or queued stage is substituted for those pending full source epochs.',
        'high_end_llm_capability':' This turn adds no measured trained-model coding or all-round capability gain. The existing step2048matched held-code prediction result remains a4.213711percent NLL reduction. The real CPU generated-code comparison is still live. Its completed initial-model completion frame has12/24syntax,0/24native functional and0/12secondary ordinary-Python passes; these are starting-model partial-frame results, not an after-training score. Frontier-level or arbitrary-code capability remains unproved.',
        'broad_independent_benchmarks':' The new queue scores the final standalone model on every full original QA, visual QA, MBPP, held human and word population plus the unchanged local coding/small broad checks. It compares full text/visual/code outcomes with the pre-export machine-DPOactor without assuming folding preserves greedy trajectories or accuracy. Held-word results are absolute final-model scores because no matched pre-export word run is promised. Existing source-exposure and nonleaderboard limitations remain; independent neural-score/generated-outcome readbacks and actual execution of these queued assessments are still required.',
        'gpu_runtime':' All four same original training workers and all eleven current pipeline controllers are live. The new assessment owner waits with allCUDAdevices hidden and acquires no current GPU. After exact predecessor/finalizer completion, it must take the actual exclusive kernel GPU lock and verify all four original L40s idle before launching evaluation workers. Eighteen targeted tests pass across standalone human scoring, full held-word sharding, source ownership, actual native cold loading and all suite dispatches. No current process is preempted, stopped or restarted.',
        'any_code_correctness_and_efficiency':' The final standalone code checks retain all original Python/JavaScript tasks, completion/instruction frames and512token budgets, with all actual generated token actions saved. All600original MBPP problems remain, with500publisher-test problems primary and90validation/10prompting reported separately; no reference qualification drops a failed case. A new dispatch test exercises all six actual original code deliveries per shard/frame. Timing stress checks remain empirical, and no arbitrary-code correctness or asymptotic efficiency proof is inferred from implementation tests.'}
    for key,note in notes.items():requirements[key]['evidence']+=note
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):raise ValueError('All ten original final requirements remain explicitly incomplete')
    OUT.mkdir();atomic_json(OUT/'runtime.json',runtime)
    evidence=[Path(__file__),previous_path,queue_path,baseline_path,neural/'plan.json',OUT/'runtime.json',
        ROOT/'scripts/assess_machine_standalone_full_v1.py',ROOT/'src/slm_lab/standalone_held_human_v1.py',
        ROOT/'src/slm_lab/standalone_word_shards_v1.py',*[ROOT/'reports'/(name+'.log') for name in tests]]
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,actual_runtime=runtime,
        previous_full_scope_review=str(previous_path.relative_to(ROOT)),source_files_sha256={str(p):file_hash(p) for p in evidence},
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,no_original_scope_removed=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):raise ValueError('Actual fixed scientific source documents must remain unchanged')
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        queued_full_standalone_assessment=str(queue_path.relative_to(ROOT)),
        latest_standalone_human_preference_implementation='src/slm_lab/standalone_held_human_v1.py',
        latest_full_held_word_sharding='src/slm_lab/standalone_word_shards_v1.py',
        next_required_work=['Finish the live7558update dense code/state/replay epoch and complete the existing96case CPU coding comparison with its source/outcome readback.',
            'Complete the unchanged85786pair human preference arms,3721source code/RL arms, original standalone export and all eight original full benchmark endpoints.',
            'Complete the queued four-L40 machine capacity/recovery controls, both1620pair source epochs, all original branch assessments and standalone dense export.',
            'Run the newly queued full standalone assessment, including every3429held word crop under3conditions, then independently read actual final-model outcomes before any promotion or high-end capability claim.'],
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Full standalone assessment queued, **{at}**: the same four-L40 training continues at '
        f'**{progress["actual_updates"]}/7,558 updates**. A new owner now waits for the complete machine comparison and its '
        'standalone export, then scores the actual **984,399,690-value** dense student on all **16,961 QA questions, '
        '3,088 visual questions × 3 image conditions, 600 MBPP problems, 4,642 held human pairs and 3,429 held word crops × '
        '3 controls**, plus the unchanged local coding and small broad checks.\n\n'
        '**Eighteen targeted tests pass.** Standalone preference inference loads no older actor/reference weights; the original '
        'own-reference scalar scores are explicitly cached and source-bound. Word checks measure both byte-CTC and native text '
        'without describing supplied word crops as full-photo detection. **No new trained-model capability gain is measured in '
        'this queueing turn; all ten final requirements remain incomplete.** '
        '[Actual standalone assessment queue](reports/machine-standalone-full-assessment-v1-queue.json), '
        '[full review](reports/standalone-full-assessment-v1-queue-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_code_first_updates=progress['actual_updates'],
        actual_standalone_assessment_owner=assessment_owner,targeted_tests=18,new_trained_quality_gain=False,
        all_ten_final_requirements_incomplete=True,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
