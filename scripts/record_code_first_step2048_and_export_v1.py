"""Record the actual dense continuation milestone and queued standalone export."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/code-first-step2048-and-machine-export-v1-review'


def read(path):return json.loads(Path(path).read_text())


def live(identity):
    if terminal(identity):raise ValueError('The same actual original process birth must remain live')
    f=(Path('/proc')/str(identity['pid'])/'stat').read_text().split(') ',1)[1].split()
    return dict(pid=identity['pid'],ticks=f[19],state=f[0])


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('A fresh GPU-hidden full-scope review is required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous_path=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous_path)['requirements'])
    state_path=ROOT/'reports/code-first-step2048-CPU-readback-v2/complete.json';state=read(state_path)
    state_plan=read(state_path.parent/'plan.json')
    if (not terminal(state_plan['owner']) or not state['complete'] or state['actual_updates']!=2048
            or state['actual_full_Adam_states']!=445 or state['actual_unique_student_values']!=984399690
            or state['changed_parameter_tensors']!=dict(language=317,vision=112,word=16)
            or not state['every_committed_source_position_and_target_budget_rederived']
            or not state['held_input_positions_and_bytes_rederived']):
        raise ValueError('The actual completed full-state/source/held-input milestone readback is required')
    FileBindings(state['source_files_sha256'])
    final_queue_path=ROOT/'reports/machine-preference-final-standalone-CPU-v1-queue.json';final_queue=read(final_queue_path)
    final_owner=live(final_queue['owner']);tests=ROOT/'reports/machine-preference-finalizer-v1-driver-tests.log'
    if ('4 passed' not in tests.read_text() or final_queue['fixed_source_arm']!='machine_DPO'
            or final_queue['fixed_full_machine_updates']!=102 or final_queue['original_machine_pairs']!=1620
            or not final_queue['both_matched_source_epochs_required'] or not final_queue['all_eight_final_private_worker_files_required']):
        raise ValueError('The actual whole-source finalizer queue and controls are required')
    neural=ROOT/'reports/code-first-step2048-native-code-CPU-v1';neural_plan=read(neural/'plan.json');neural_owner=live(neural_plan['owner'])
    neural_progress=read(neural/'progress.json');neural_reader_path=ROOT/'reports/code-first-step2048-native-code-CPU-v1-readback-queue.json'
    neural_reader=read(neural_reader_path);reader_owner=live(neural_reader['owner'])
    if neural_plan['total_native_deliveries']!=96 or neural_reader['exact_neural_assessment_owner']!=neural_plan['owner'] or (neural/'failure.json').exists():
        raise ValueError('The exact all96case neural comparison and its own readback owner must remain successful')
    source=ROOT/'reports/code-first-continuation-v3';plan=read(source/'plan.json');progress=read(source/'progress.json')
    current_owner=live(plan['owner']);ranks=[r for r in read(source/'children.json') if r['subject'].startswith('train-rank-')]
    if len(ranks)!=4 or (source/'failure.json').exists():raise ValueError('All four original GPU workers must remain healthy')
    workers=[live(r) for r in ranks];protected=set();controllers=[]
    for name in ['code-first-continuation-v3','phase-human-preference-v7','broad-anchored-code-v6',
        'code-first-final-standalone-CPU-v3','full-held-QA-pipeline-v4','full-AI2D-pipeline-v4',
        'full-original-MBPP-pipeline-v4','AI2D-fixed-exposure-strata-v4','full-machine-preference-v1',
        'machine-preference-final-standalone-CPU-v1']:
        path=ROOT/'reports'/name/'plan.json'
        if not path.exists():path=ROOT/'reports'/(name+'-queue.json')
        item=read(path);protected.update(item['source_files_sha256']);controllers.append(dict(stage=name,owner=live(item['owner'])))
    protected.update(neural_plan['source_files_sha256']);at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,actual_training_owner=current_owner,actual_four_original_GPU_workers=workers,
        actual_code_first_updates=progress['actual_updates'],planned_code_first_updates=7558,all_ten_pipeline_controllers=controllers,
        actual_CPU_native_code_owner=neural_owner,actual_CPU_native_code_progress=neural_progress,
        actual_CPU_native_code_readback_owner=reader_owner,new_machine_finalizer_owner=final_owner,
        new_machine_preference_production_updates=0,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        original_training_not_preempted_or_restarted=True)
    notes={
        'own_architecture':' The first actual2048update continuation snapshot is now read independently without constructing a model or optimizer. It contains the full own984399690-value dense/vision/word/auxiliary student. All317language,112vision and16word selected tensors have changed, while all16unused tensors remain exact. This is full selected-weight learning; no new architecture or frontier capability is inferred from the state inventory.',
        'public_data':' Every one of the first2048actual source-family updates, complete original positions, context counts and response/EOS denominators is rederived against the frozen7558update schedule. The snapshot is a partial curriculum, not completion of the58488code/2616stdout/1310STATEepoch or all selected replay presentations. The matched held128original windows and524160causal targets rebuild from their exact valid-source byte hashes.',
        'under_1b_parameters':' A real CPU-only finalizer is newly queued for the fixed complete machine_DPOendpoint. After both1620pair source epochs and all eight final private states are committed, it must load the actual999706442-value actor, fold its own matrices/norms/critic into a984399690-value dense student, package its own tokenizer, measure whole4096code/native448photo FP32 drift and cold-load every parameter plus text/photo/real-word outputs exactly. Four source-owner/full-epoch/private-file corruption tests pass. No new standalone model has been exported yet, and the fixed endpoint is not a chosen quality winner.',
        'text_and_images':' Actual early training includes329vision-active updates, including82word-active updates, and changes every112vision/16word selected tensor. These are optimizer/state observations, not measured reading or general visual-answer improvements. The new finalizer explicitly retains all original visual/word/auxiliary values and requires actual photo/real-word cold probes; broad image, OCR and held visual quality remain required.',
        'human_feedback':' The original85786pair human preference comparison and the subsequent3721source code/RL arms remain queued unchanged. The new machine finalizer accepts explicitly nonhuman machine-feedback epochs and refuses human-label substitutions. Its candidate retains the full intended own human/code lineage through the source loader; new human judgments on generated outputs and measured alignment quality remain absent.',
        'serious_full_model_training':f' The actual dense run is now live at{progress["actual_updates"]}/7558updates. At immutable step2048the full private/native weights agree across every461unique tensor, all445full Adam leaves are finite with source-active clocks(language1719,vision329,word82), every source record reconstructs, and all four private CPU RNG states parse without invokingCUDA. CUDA RNG bytes are structure-checked, not replayed by this reader; original four-GPU next-update recovery remains the earlier capacity evidence. Full7558training and all planned post-training are still incomplete.',
        'high_end_llm_capability':f' On the identical128held code windows/524160targets, original GPU validation NLL changes from{state["held_initial_NLL"]}to{state["held_step2048_NLL"]}, a{100*state["relative_held_NLL_reduction"]:.6f}percent reduction. The static reader rechecks source identities and denominators; it does not rerun neural loss forwards or supply per-window uncertainty. This is a code next-token prediction gain, not generated-code correctness, all-domain mastery or high-end LLM equivalence. The latest fully completed main functional result remains0/48until a new complete generation comparison is available.',
        'broad_independent_benchmarks':' A real CPU_FP32comparison now generates all96original Python/JavaScript task/frame deliveries from the exact same own starting model and fixed step2048checkpoint, retaining actual input/generated token IDs and unchanged512token budgets. All24canonical native and12ordinary Python reference controls pass before generation. The comparison and its exact-owner all96case source/outcome reader are live, not complete. These remain local development tasks; the existing full16961QA,3088AI2Dtimes3conditions and600MBPP original benchmark pipeline remains unchanged.',
        'gpu_runtime':' The same original parent and all four training rank births remain live. The new neural coding comparison uses two CPU threads, the state reader uses one CPU thread and completes without constructing a neural model, and the new finalizer/readback owners wait withCUDAhidden. No GPU process, original waiting controller or scientific bound source is restarted or replaced. All ten current pipeline controllers remain live and are recorded with exact process births.',
        'any_code_correctness_and_efficiency':' The current measured improvement is4.213711percent lower matched held-code NLL; generated-program syntax, finite-case correctness and empirical efficiency at step2048are still being measured. No partial successful case is promoted to a full-suite gain. The new coding reader preserves both original native and secondary ordinary-Python protocols without repairing programs. An initial static state reader used the wrong warmup/cosine readback formula and failed; its script/error are preserved. Version2uses the actual ceil warmup and total-minus-warmup-minus-one cosine denominator and successfully verifies all2048actual learning rates, without changing training.'}
    for key,note in notes.items():requirements[key]['evidence']+=note
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):raise ValueError('Every original final requirement remains explicitly incomplete')
    OUT.mkdir();atomic_json(OUT/'runtime.json',runtime)
    evidence=[Path(__file__),previous_path,state_path,final_queue_path,tests,neural/'plan.json',neural_reader_path,OUT/'runtime.json']
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),
        actual_runtime=runtime,actual_step2048_held_NLL=state['held_step2048_NLL'],relative_held_code_NLL_reduction=state['relative_held_NLL_reduction'],
        source_files_sha256={str(p):file_hash(p) for p in evidence},all_ten_final_requirements_incomplete=True,
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',no_original_scope_removed=True,candidate_promoted=False,goal_achieved=False))
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):raise ValueError('An actual immutable bound source document must remain unchanged')
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_actual_code_first_milestone_readback=str(state_path.relative_to(ROOT)),
        latest_code_first_matched_native_code_assessment=str((neural/'plan.json').relative_to(ROOT)),
        latest_code_first_matched_native_code_readback_queue=str(neural_reader_path.relative_to(ROOT)),
        queued_machine_preference_standalone_export=str(final_queue_path.relative_to(ROOT)),
        next_required_work=['Finish the live7558update code/state/replay epoch; complete the actual96case CPU initial-versus2048code comparison and its independent source/outcome readback.',
            'Retain all85786human preference pairs, all3721code/RL sources, the original standalone export and all eight full original benchmark endpoints.',
            'Complete the queued fresh four-L40 machine capacity, both1620pair source epochs, all complete original branch assessments and the newly queued standalone dense export.',
            'Measure the final standalone candidate on full code/knowledge/math/visual/word/human criteria and independently read outcomes before any promotion; all-round and high-end capability remain unproved.'],
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'First code/state continuation milestone, **{at}**: the same four-L40 run is live at '
        f'**{progress["actual_updates"]}/7,558 updates**. At immutable **step 2,048**, loss on the identical **128 held code windows / '
        f'524,160 targets** falls **0.868549 → 0.831951 (4.21%)**. The completed CPU state reader reconstructs every source cursor and '
        'learning rate, all 445 full Adam leaves, every native/private model value and the original held-input bytes. '
        '**This is improved code prediction; generated-program correctness is still being measured.**\n\n'
        'A real matched CPU comparison now generates all **96 original task/frame/endpoint deliveries**, retaining actual tokens and '
        'unchanged 512-token budgets. The complete machine-DPO standalone export is also newly queued, with all full-epoch/private-state '
        'gates and four passing source tests. No new machine production updates or standalone export have occurred. '
        '**All ten final requirements remain incomplete.** '
        '[Step-2048 state and held metric](reports/code-first-step2048-CPU-readback-v2/complete.json), '
        '[live coding comparison](reports/code-first-step2048-native-code-CPU-v1/plan.json), '
        '[standalone export queue](reports/machine-preference-final-standalone-CPU-v1-queue.json), '
        '[full review](reports/code-first-step2048-and-machine-export-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_code_first_updates=progress['actual_updates'],
        step2048_held_NLL=state['held_step2048_NLL'],relative_held_NLL_reduction=state['relative_held_NLL_reduction'],
        new_generated_code_gain_not_established=True,all_ten_final_requirements_incomplete=True,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
