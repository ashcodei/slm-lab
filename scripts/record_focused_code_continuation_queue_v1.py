"""Record the actual frozen successor queue without declaring GPU/quality gain."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/focused-code-continuation-v1-queue-review'


def identity(value,*,waiting=False):
    proc=Path('/proc')/str(value['pid']);fields=(proc/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=value['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Exact live process birth required')
    children=(proc/'task'/str(value['pid'])/'children').read_text().strip()
    visible=next((s.split('=',1)[1] for s in (proc/'environ').read_bytes().decode().split('\0') if s.startswith('CUDA_VISIBLE_DEVICES=')),None)
    if waiting and (visible!='' or children):raise ValueError('Waiting owner must remain CPU-hidden and childless')
    return dict(pid=value['pid'],ticks=fields[19],state=fields[0],children=children,CUDA_VISIBLE_DEVICES=visible)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden queue review required')
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text())
    previous_path=ROOT/master['latest_full_scope_review'];previous=json.loads(previous_path.read_text());requirements=previous['requirements']
    queue_path=ROOT/'reports/focused-code-continuation-v1-queue.json';queue=json.loads(queue_path.read_text())
    FileBindings(queue['source_files_sha256']).check(full=True)
    queued=identity(queue['owner'],waiting=True)
    for path in [ROOT/'reports/focused-code-continuation-v1-parent-failure.json',ROOT/'reports/focused-code-continuation-v1',
        ROOT/'reports/focused-code-continuation-v1-initializer',ROOT/'checkpoints/focused-code-continuation-v1']:
        if path.exists():raise ValueError('This review is for the live waiting queue, not a started/failed production run')
    tests=[ROOT/'reports/focused-continuation-run-contract-v1-final-tests.log',ROOT/'reports/focused-code-continuation-driver-v1-final-tests.log']
    if '1 passed' not in tests[0].read_text() or '2 passed' not in tests[1].read_text():raise ValueError('Actual new recovery and resource controls required')
    main_folder=ROOT/'reports/phase-joint-v3';main_plan=json.loads((main_folder/'plan.json').read_text())
    progress=json.loads((main_folder/'progress.json').read_text())
    workers=[identity(row) for row in json.loads((main_folder/'children.json').read_text()) if row['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('All original main GPU workers required')
    protected={**main_plan['source_files_sha256'],**queue['source_files_sha256']}
    successors={}
    for name in ['phase-human-preference-v4','broad-anchored-code-v3','full-held-QA-pipeline-v1','full-AI2D-pipeline-v1']:
        path=ROOT/'reports'/(name+'-queue.json');value=json.loads(path.read_text())
        successors[name]=identity(value['owner'],waiting=True);protected.update(value['source_files_sha256'])
    source_jobs={}
    for name in ['CodeContests-full-Python-references-v1','CodeContests-public-execution-traces-candidate-v1',
        'CodeContests-reference-outputs-v1-readback','public-execution-targets-v1-readback-admission',
        'main-foundation-instruction14336-code-CPU-v2','AI2D-fixed-exposure-strata-v1']:
        path=ROOT/'reports'/name/'plan.json';value=json.loads(path.read_text())
        if path.with_name('failure.json').exists():raise ValueError('Original source/assessment job has failed: '+name)
        source_jobs[name]=dict(owner=identity(value['owner']))
        for filename in ['progress.json','waiting.json']:
            if path.with_name(filename).exists():source_jobs[name][filename]=json.loads(path.with_name(filename).read_text())
        protected.update(value['source_files_sha256'])
    first_frame=ROOT/'reports/main-foundation-instruction14336-code-CPU-v2/foundation12853/completion-complete.json'
    frame=json.loads(first_frame.read_text())
    if frame['actual_results']['original_native_tasks']!=24:raise ValueError('The entire original completion frame required')
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,original_main_owner=identity(main_plan['owner']),original_four_main_workers=workers,
        actual_main_updates=progress['actual_updates'],planned_main_updates=progress['planned_updates'],phase=progress['phase'],
        original_successors=successors,new_frozen_continuation_queue=queued,CPU_source_and_observer_jobs=source_jobs,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
    requirements['serious_full_model_training']['evidence'] += (
        ' A new frozen actual waiting four-L40 successor now connects the prepared full58488example code epoch, '
        'every independently admitted public-output example and all12selected raw/general/visual/word replay families. '
        'Its fixed initializer is the complete human-DPO plus3721-source own anchor_execution policy, chosen before '
        'its future scores; no older prototype or score-selected fallback is allowed. It will fold into984399690 '
        'dense values, calibrate full native code/output/photo inputs and exact real-word bytes on CPU, then run a '
        'fresh four-L40 seven-logical/nine-physical-update capacity/recovery gate before any production update. '
        'It uses full317-language-tensor gradients, original vision/word groups, whole contexts, raw replay and a '
        'frozen copy of its own complete starting policy for text KL. The new actual small-model CPU control has '
        'zero foundation steps, two full-language updates and one image update. It rejects changed original source '
        'families/positions before model or Adam mutation, restores the source cursor/complete state and reproduces '
        'the exact next image update while the decoder remains fixed. One actual neural test passes. This is not '
        'a full-size new epoch, a GPU gate result or measured quality gain. The new successor is only waiting.')
    requirements['under_1b_parameters']['evidence'] += (
        ' The new successor implements standalone dense model plus tokenizer export for the future final984399690 '
        'value student. Its capacity and final paths require actual cold reload of every parameter, complete4096 '
        'logits and real-word byte outputs with no old base/actor/checkpoint/optimizer/reference loaded for inference. '
        'Those future cold checks have not run; the earlier full-size CPU standalone proof remains the actual '
        'completed evidence. No new final model bundle is claimed to exist.')
    requirements['broad_independent_benchmarks']['evidence'] += (
        ' The new continuation queue fixes initial/trained full16961-question MMLU-Pro/GSM8K/NQ and3088-question '
        'AI2D proper/gray/fixed-wrong image assessments, all on four original L40s after its complete source epoch. '
        'Every original input, native budget, delivered action/choice evidence and raw per-question verdict remains '
        'retained; before/after paired comparisons use the existing original full-population reconciliation. '
        'Local coding and broader OCR/object/visual pulses also run. These are future assessments, not measured '
        'scores, semantic-independence certificates or frontier equivalence. Human retention and broader full '
        'held-code assessment of this future model remain required.')
    requirements['gpu_runtime']['evidence'] += (
        ' The new actual successor owner is live, CPU-hidden and childless. It checks exact original dependency '
        'birth identities, all complete original visual output readback, complete code source epochs, admitted '
        'public-output targets and worker restoration before GPU ownership. Two tests use an actual live process '
        'birth and reused-PID rejection, plus live-worker/partial-epoch/unadmitted-source/missing-restoration '
        'controls. Both pass. The original four main ranks and all original queues remain live and unchanged. '
        'No new GPUs are attached, no original work is preempted, no fresh capacity success or throughput gain is '
        'claimed, and there is no new production optimizer call yet.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' The exact main foundation12853completion frame has now completed all24native deliveries:10syntax-valid, '
        '0functionally correct and0empirical-efficiency passes. All12Python deliveries also remain0under the '
        'secondary ordinary runtime. This is one entire frame at the older completed-foundation endpoint, '
        'not the full matched two-endpoint/four-frame result or a score of the currently training checkpoint. '
        'The remaining original observer work continues without checkpoint/budget substitution. No new functional '
        'coding improvement is established; new source supervision and actual future assessments are still needed.')
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('Every original final requirement remains incomplete')
    OUT.mkdir();evidence=[Path(__file__),previous_path,queue_path,first_frame,*tests,
        ROOT/'scripts/train_focused_code_continuation_v1.py',ROOT/'src/slm_lab/focused_continuation_run_contract_v1.py']
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        authoritative_frozen_continuation_queue=str(queue_path.relative_to(ROOT)),actual_runtime=runtime,
        completed_old_foundation_one_frame_only=frame,new_capacity_and_full_source_training_and_scores_pending=True,
        source_files_sha256={str(p):file_hash(p) for p in evidence},previous_goal_turn_classification='progress',
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,
        no_original_scope_removed=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(path) in protected for path in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),bound_root_documents_preserved=True)),flush=True);return
    (ROOT/'reports/completion-audit-before-focused-code-continuation-queue-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:r['evidence'] for k,r in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_queued_new_code_continuation=str(queue_path.relative_to(ROOT)),current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Complete code continuation queued, **{at}**: a live, CPU-hidden successor now connects all '
        '**58,488 new code examples**, every admitted public execution target and **12 replay families** to real '
        'full-weight training. It will start from the fixed complete own human-DPO and execution-code policy, '
        'folded into one 984,399,690-parameter dense student. Fresh four-L40 capacity and exact recovery must pass '
        'first. Its planned final standalone checkpoint and tokenizer require cold checks of every parameter, '
        'full 4,096-token logits and real-word outputs without older model dependencies. **It is waiting; no new '
        'capacity gate, source epoch, standalone final bundle or quality gain has been completed.**\n\n'
        'The queue includes original local coding/broad assessments plus all **16,961 text QA questions** and '
        '**3,088 visual questions with proper, gray and fixed wrong images** at initial/trained endpoints. '
        'The existing main-model observer has finished the older foundation completion frame: **0/24 functional '
        'passes, 10/24 syntax passes**; its full matched comparison is still running. The original four-GPU '
        'main run and all earlier queues continue. **All ten final requirements remain incomplete.** '
        '[Queue review](reports/focused-code-continuation-v1-queue-review/review.json), '
        '[frozen successor](reports/focused-code-continuation-v1-queue.json).\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_new_waiting_owner=queued,
        main_updates=progress['actual_updates'],planned_new_code_examples=58488,new_production_updates=0,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
