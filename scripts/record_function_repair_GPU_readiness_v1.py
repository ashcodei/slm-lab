"""Review runnable, admitted GPU research separately from pending GPU capacity."""
from datetime import datetime,timezone
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.function_repair_GPU_contract_v1 import QUEUE,REPORT,ADMISSION,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/function-repair-GPU-controller-readiness-v1'


def read(path):return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CUDA-hidden readiness review required')
    queue=read(ROOT/QUEUE);admission=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,queue['owner']) or (ROOT/REPORT/'plan.json').exists():
        raise ValueError('Actual admitted live research queue must still wait without GPU capacity/optimization')
    complete=ROOT/'reports/function-repair-GPU-lineage-v1/complete.json'
    if read(complete)['GPU_study_owner']!=queue['owner']:raise ValueError('Actual complete handoff must name the same research owner')
    tests={'function-repair-evaluation-GPU-v1-tests':3,'function-repair-GPU-controller-v1-tests':7,'function-repair-GPU-lineage-v1-tests':5}
    paths=[Path(__file__),ROOT/ADMISSION,complete,ROOT/'reports/function-repair-GPU-lineage-v1/retired.json',
        ROOT/'reports/function-repair-GPU-lineage-v1-file-map.json',ROOT/'reports/parallel-repair-recovery-CPU-v1/complete.json']
    for name,count in tests.items():
        path=ROOT/'reports'/(name+'.log')
        if str(count)+' passed' not in path.read_text():raise ValueError('All actual numerical/gating/scope controls must pass')
        paths.append(path)
    files={**queue['source_files_sha256'],**admission['source_files_sha256'],**{str(p):file_hash(p) for p in paths}}
    guard=FileBindings(files)
    runtime=ROOT/'reports/public-pretraining-recovery-v1';math=ROOT/'reports/public-pretraining-v1'
    runtime_plan=read(runtime/'plan.json');cpu=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')
    if terminal(runtime_plan['owner']) or terminal(cpu['owner']):raise ValueError('Original neural runs must remain live')
    raw=(runtime/'training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1];rows=list(map(json.loads,prefix.splitlines()))
    if [r['step'] for r in rows]!=list(range(1,len(rows)+1)):raise ValueError('All complete original public update records must be contiguous')
    main_progress=dict(owner=runtime_plan['owner'],complete_actual_updates=len(rows),planned_updates=read(math/'plan.json')['planned_steps'],
        response_EOS_targets=sum(int(r['response_EOS_targets']) for r in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest(),
        actual_four_GPU_ranks=read(runtime/'children.json'))
    cpu_progress=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
    scope=dict(source_train_functions=826,second_views=791,epochs=3,updates_each_profile=306,presentations_each_profile=4851,
        response_EOS_targets_each_profile=2287110,profiles=queue['profiles'],validation_functions=15,source_generation_budget=1024,
        local_code_tasks=48,local_code_generation_budget=512,knowledge=56,math=16,OCR=20,objects=15,photos=32,image_size=448)
    OUT.mkdir();at=datetime.now(timezone.utc).isoformat()
    proof=dict(complete=True,at=at,meaning='runnable controller, CPU mechanics and actual waiting-lineage admission only',
        actual_GPU_study_owner=queue['owner'],actual_new_downstream_controllers=admission['new_controllers'],
        source_files_sha256=files,new_small_CPU_tests=tests,previous_small_parallel_stage_IO_tests=16,
        original_live_main=main_progress,original_live_CPU_comparison=cpu_progress,actual_GPU_rows=gpu,
        matched_GPU_study_scope=scope,GPU_capacity_profiles_completed=0,GPU_production_updates=0,new_generated_quality_assessments=0,
        all_original_neural_functions_and_full_benchmark_populations_preserved=True,canonical_human_RL_initializer_unchanged=True,
        fresh_full_size_four_L40_capacity_recovery_cold_gates_still_required=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',proof)
    previous=read(ROOT/'reports/parallel-repair-recovery-CPU-v1/review.json');review=copy.deepcopy(previous);review['at']=at
    review['requirements']['serious_full_model_training']['evidence']=(
        f"The same public pretraining has {len(rows)}/4447actual complete updates; the unchanged full-size CPU comparison is at "
        f"{cpu_progress['actual_complete_updates']}/306in its current {cpu_progress['profile']}arm. The new GPU controller is runnable and "
        "its actual CUDA-hidden process is admitted behind the full public predecessor.15additional small CPU tests pass:3native observer/cohort,"
        "7actual neural experiment/recovery/durable/gating and5scientific-scope/retirement-precondition tests, following16parallel-kernel/stage/IO tests."
        "All three planned GPU arms retain826original functions,791second views,306updates and2287110targets each. Actual new four-L40 capacity "
        "and production remain0; no quality gain is inferred from software tests.")
    review['requirements']['human_feedback']['evidence']+=(
        " A complete actual operational handoff now replaces only the ten confirmed unstarted hidden childless controllers with fresh tested queues."
        " Every original mathematical function,85786human pairs,3721code sources/arm and1620machine pairs/arm remains. The first human gate waits "
        "for optional research restoration; human/RL initialization stays the complete original public-pretrained model. No research actor is adopted.")
    review['requirements']['broad_independent_benchmarks']['evidence']+=(
        " All ten replacements independently publish exactly matching original numerical queue scopes, including all nine endpoints and original full populations."
        " The optional GPU comparison predeclares the same complete15source-validation and48local-code plus broad text/image development checks for baseline and all three arms."
        " Every arm finishes training before those assessments; source public examples remain visible and are not certified independent benchmarks.")
    review['requirements']['gpu_runtime']['evidence']=(
        "The same actual original four L40 compute PIDs and live public-training birth remain unchanged through the completed CPU-only scheduler handoff."
        " No neural worker received a signal. The added GPU controller waits with CUDA hidden, no capacity/model/optimizer worker and no GPU claim."
        " After complete4447step public training, native controls and verified restoration, all3profiles must pass fresh actual full-size four-GPU "
        "complete-source/longest-backward/all561Adam/private-RNG replay/4096text448photo real-word cold gates before any production repair update.")
    review['requirements']['any_code_correctness_and_efficiency']['evidence']+=(
        " The same complete-source comparison now also has a runnable admitted four-GPU controller, using the later complete public-data base "
        "in a separate experiment. Its capacity and model-quality results remain pending. Native code scoring retains original syntax, execution "
        "and efficiency stress checks, complete model action logs and full original local task populations; no generated program is repaired.")
    for requirement in review['requirements'].values():
        if requirement['status']!='incomplete_for_requested_final_model':raise ValueError('No final requirement may be promoted by readiness alone')
    review['current_actual_runtime']=dict(main=main_progress,CPU=cpu_progress,GPU_queue_owner=queue['owner'])
    review['previous_full_scope_review']=str(ROOT/'reports/parallel-repair-recovery-CPU-v1/review.json')
    review['current_goal_turn_classification']='new runnable GPU experiment, tested operational handoff and actual admitted waiting lineage; full-size GPU capacity and quality pending'
    review['source_files_sha256']={**files,str(OUT/'complete.json'):file_hash(OUT/'complete.json')}
    review['new_runnable_controller_proof']=str(OUT/'complete.json');review['goal_achieved']=False
    atomic_json(OUT/'review.json',review);guard.check(full=True)
    audit_path=ROOT/'reports/completion-audit.json';audit=read(audit_path)
    # This root document is a mutable review index, never a training input.
    audit['latest_full_scope_review']=str(OUT/'review.json');audit['requirements']=review['requirements']
    audit['latest_actual_main_runtime']=main_progress;audit['latest_repair_GPU_readiness']=str(OUT/'complete.json')
    audit['goal_achieved']=False;audit['at']=at;atomic_json(audit_path,audit)
    paragraph=(f"GPU repair controller and verified scheduling handoff, **{at}**: the added CUDA-hidden GPU controller is now live and "
        f"admitted behind complete public pretraining. All ten downstream controllers have fresh actual queues with unchanged numerical functions "
        f"and full benchmark populations; no neural worker was signalled. **15new small CPU tests pass**, in addition to16parallel/recovery/IO controls. "
        f"Actual GPU capacity and repair training remain pending. Each planned arm retains306updates and2,287,110targets; baseline/all3arms retain "
        f"all15source-validation and48local-code plus broad text/image development checks. The unchanged main is at{len(rows)}/4,447updates and "
        f"the CPU comparison at{cpu_progress['actual_complete_updates']}/306in its current arm. **No new accuracy gain is established; all ten final "
        f"requirements remain incomplete.** [Readiness proof](reports/function-repair-GPU-controller-readiness-v1/complete.json), "
        f"[actual handoff](reports/function-repair-GPU-lineage-v1/complete.json), [full review](reports/function-repair-GPU-controller-readiness-v1/review.json). "
        "Earlier entries are historical snapshots.\n\n")
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;text=path.read_text()
        if text.startswith('#'):
            first,rest=text.split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
        else:path.write_text(paragraph+text)
    print(json.dumps(dict(readiness=str(OUT/'complete.json'),new_small_CPU_tests=15,actual_GPU_capacity_passed=0,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
