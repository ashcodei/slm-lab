"""Reconcile real source-quality evidence without promoting model capability."""
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.function_repair_GPU_contract_v1 import QUEUE,ADMISSION,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/function-semantic-source-quality-v1-review'


def read(path):return json.loads(Path(path).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CUDA-hidden semantic source review required')
    source=ROOT/'reports/function-semantic-stress-CPU-v1/complete.json'
    admitted=ROOT/'reports/public-semantic-source-contrast-v1-readback-admission/complete.json'
    stress=read(source);admission=read(admitted);summary=stress['actual_matched_post_hoc_source_train_results']
    if (not stress.get('complete') or not admission.get('complete') or admission['actual_admitted_correction_examples']!=3
            or set(stress['original_public_implementations_with_counterexamples'])!={'perfect_cube','is_prime','is_sq'}):
        raise ValueError('Actual complete source/model stress and three public correction admissions required')
    files={**stress['source_files_sha256'],**admission['source_files_sha256']}
    for path in [Path(__file__),source,admitted,ROOT/'reports/function-repair-GPU-controller-readiness-v1/review.json']:
        files[str(path)]=file_hash(path)
    guard=FileBindings(files);queue=read(ROOT/QUEUE)
    if not handoff_admitted(ROOT,queue['owner']):raise ValueError('Actual full-scope GPU and downstream waiting lineage must remain live')
    actual=read(ROOT/'reports/public-pretraining-recovery-v1/plan.json');cpu_plan=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json')
    if terminal(actual['owner']) or terminal(cpu_plan['owner']):raise ValueError('Actual original neural owners must remain live')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=list(map(json.loads,prefix.splitlines()))
    if [v['step'] for v in rows]!=list(range(1,len(rows)+1)):raise ValueError('Contiguous complete actual public journal prefix required')
    current=dict(owner=actual['owner'],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(v['response_EOS_targets']) for v in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    cpu=read(ROOT/'reports/full-function-repair-comparison-CPU-v1/progress.json');at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    table=['# Public-source semantic diagnostic','',
        'This is a post-hoc source-train diagnostic. It is not an independent benchmark or a new neural-generation run.','',
        '| Endpoint | Original public examples, trained functions | Extra numerical/domain checks, trained functions | Unoptimized controls, extra checks | Trained function bodies matching original AST |',
        '|---|---:|---:|---:|---:|']
    public_scores={'original_public':10,'initial':0,'chosen_CE':9,'trajectory_DPO':9}
    for endpoint in ['original_public','initial','chosen_CE','trajectory_DPO']:
        values=summary[endpoint];train=values['optimization_source_train'];control=values['unoptimized_source_train_control']
        table.append(f"| {endpoint} | {public_scores[endpoint]}/10 | {train['all_domain_cases_passed']}/10 | {control['all_domain_cases_passed']}/4 | {train['body_AST_matches_original_public']}/10 |")
    table += ['',
        'Nine trained functions have 64 fresh argument cases each. The binary NOT function has only its two already-published binary inputs, so no fresh-input claim is made for it. The four controls have 222 fresh argument cases in total, including all 30 unused nonnegative factorial inputs through 32.',
        '', 'All 14 references and all 42 original native model deliveries are retained. Syntax failures remain failures. Existing code is never repaired during scoring. Both full native code and original source/control roles reconstruct from frozen artifacts.',
        '', 'Three original public implementations fail the numeric specifications: perfect_cube (floating cube-root rounding), is_prime (0 and 1), and is_sq (large exact integer squares). This does not estimate the defect rate of the complete 826-function corpus.',
        '', 'Three complete correction examples use unchanged alternative implementations already in the pinned public MIT repository, plus a static positional alias when names differ. Every original and donor public example and all fixed fresh numeric cases pass. Independent readback performs 66 actual confined program runs, preserves both complete response/EOS sequences and all prompt masks, and loads all three rows for training (1,527 positive response/EOS targets). They are machine source contrasts, not human preferences or sampled on-policy trajectories.',
        '', 'No correction training or native accuracy gain is claimed. The last completed broader coding result remains 0/48 functional. The original live training and all complete final benchmark populations remain unchanged.','']
    (OUT/'metrics.md').write_text('\n'.join(table))
    proof=dict(complete=True,at=at,source_files_sha256=files,actual_matched_source_train_semantics=summary,
        original_public_reference_semantic_failures=stress['original_public_implementations_with_counterexamples'],
        original_source_diagnostic_actual_confined_program_runs=stress['actual_fresh_confined_program_runs'],
        independent_readback_actual_confined_program_runs=admission['actual_fresh_confined_program_runs'],
        actual_admitted_public_correction_rows=3,actual_positive_response_EOS_targets=admission['actual_response_EOS_targets'],
        correction_training_updates=0,original_main=current,original_CPU_study=cpu,
        GPU_study_owner=queue['owner'],actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        previous_goal_turn_classification='progress: actual runnable GPU controller and complete tested ten-controller handoff',
        current_goal_turn_classification='progress: actual new numerical/source evidence and independently admitted public correction targets',
        source_train_diagnostic_not_independent_benchmark=True,no_new_neural_quality_result=True,
        all_original_training_and_full_benchmark_populations_preserved=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',proof)
    previous=ROOT/'reports/function-repair-GPU-controller-readiness-v1/review.json';review=copy.deepcopy(read(previous));review['at']=at
    message=(" New post-hoc fresh-input diagnostics retain all14original source functions and42native model programs. CEandDPOeach pass7/10trained "
        "functions under the fixed extra numerical/domain checks, compared with9/10on original public examples; all4unoptimized source-train controls "
        "remain0/4.9/10trained function bodies reproduce original ASTs.3public references themselves fail finite mathematical boundaries. "
        "The binary NOTdomain was already exhausted by its public examples; only13functions have fresh arguments.3public alternative implementations "
        "and their native masks/pairs pass independent fresh readback and are admitted as correction data with1527positive targets. "
        "These are public source contrasts, not human or on-policy feedback. No new model training, unseen-code quality gain or broad corpus defect rate is established.")
    for key in ['public_data','human_feedback','high_end_llm_capability','any_code_correctness_and_efficiency']:
        review['requirements'][key]['evidence']+=message
    review['requirements']['serious_full_model_training']['evidence']+=(f" In this latest captured prefix the same public run has{len(rows)}/4447updates "
        f"and the same full-size CPU comparison has{cpu['actual_complete_updates']}/306in its current{cpu['profile']}arm. "
        "The semantic-source diagnostic constructs no model and adds no optimization. The added GPU comparison is still waiting for its unchanged complete public predecessor.")
    if any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):raise ValueError('Source qualification cannot complete a final capability requirement')
    review['previous_full_scope_review']=str(previous);review['source_files_sha256']={**files,str(OUT/'complete.json'):file_hash(OUT/'complete.json')}
    review['current_actual_runtime']=dict(main=current,CPU=cpu,GPU_queue_owner=queue['owner'])
    review['current_goal_turn_classification']=proof['current_goal_turn_classification'];review['goal_achieved']=False
    atomic_json(OUT/'review.json',review);guard.check(full=True)
    # Review documents are mutable indices; they are absent from the live
    # controller source maps, checked here before any update.
    live=read(ROOT/ADMISSION)['new_controllers'];live_maps=[read(ROOT/v['queue'])['source_files_sha256'] for v in live]
    live_maps += [queue['source_files_sha256'],actual['source_files_sha256'],cpu_plan['source_files_sha256']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in mapping for mapping in live_maps):raise ValueError('Never modify a live source-bound review file')
    audit=read(ROOT/'reports/completion-audit.json');audit.update(at=at,latest_full_scope_review=str(OUT/'review.json'),requirements=review['requirements'],
        latest_actual_main_runtime=current,latest_semantic_source_quality=str(OUT/'complete.json'),goal_achieved=False)
    atomic_json(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f"Public-source semantic qualification, **{at}**: fresh-input diagnostics show CEandDPOat**7/10trained functions**, with "
        "**0/4unoptimized controls**, versus9/10on the original public examples. Nine learned bodies copy their original source ASTs; three public "
        "reference implementations themselves fail mathematical boundary cases. Three complete correction examples now use already-public alternate "
        "implementations and pass independent readback (**66fresh confined program runs;1,527positive targets**). They have not been trained. "
        f"The original main continues at{len(rows)}/4,447updates and CPU comparison at{cpu['actual_complete_updates']}/306in its current arm; all full "
        "human/RL/evaluation scopes remain. **This is source-train diagnosis and data qualification, not a new independent accuracy gain. All ten final "
        "requirements remain incomplete.** [Matched semantic results](reports/function-semantic-source-quality-v1-review/metrics.md), "
        "[correction admission](reports/public-semantic-source-contrast-v1-readback-admission/complete.json), "
        "[full review](reports/function-semantic-source-quality-v1-review/review.json). Earlier entries are historical snapshots.\n\n")
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;text=path.read_text()
        if text.startswith('#'):
            first,rest=text.split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
        else:path.write_text(paragraph+text)
    print(json.dumps(dict(review=str(OUT/'review.json'),admitted_public_corrections=3,new_neural_updates=0,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
