"""Update full-scope status after the actual7558-update epoch completes."""
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
OUT=ROOT/'reports/completed-code-epoch-pretraining-queue-v1-review'


def read(p):return json.loads(Path(p).read_text())


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden final-epoch status snapshot required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous)['requirements']);source=ROOT/'reports/code-first-continuation-v3'
    plan=read(source/'plan.json');trained=read(source/'training-complete.json');before=read(source/'held-packed-initial.json');after=read(source/'held-packed-7558.json')
    if (trained['actual_updates']!=7558 or plan['planned_steps']!=7558
            or not trained['all58488new_code_and_all_admitted_public_execution_targets_once']
            or not trained['all_admitted_public_state_examples_once'] or terminal(plan['owner'])):
        raise ValueError('The actual complete original code/output/state epoch and ongoing native assessment owner required')
    for key in ['selected_original_positions','complete_causal_targets','original_manifest_sha256','evaluated_windows']:
        if before[key]!=after[key]:raise ValueError('Matched original held loss inputs required')
    active=[r for r in read(source/'children.json') if not terminal(r)]
    if len(active)!=4 or any(not r['subject'].startswith('assess-') for r in active):raise ValueError('All four original-owned assessment ranks must be live')
    q_path=ROOT/'reports/public-pretraining-v1-queue.json';queue=read(q_path)
    replay_path=ROOT/'reports/code-first-complete-native-program-reexecution-CPU-v1-queue.json';replay=read(replay_path)
    if terminal(queue['owner']) or terminal(replay['owner']):raise ValueError('Actual new pretraining and independent program replay queues must remain live')
    readback_path=ROOT/'reports/public-billion-causal-candidate-v2-full-readback-v2/progress.json';reader=read(readback_path)
    ratio=1-after['token_mean_NLL']/before['token_mean_NLL']
    requirements['serious_full_model_training']['evidence']=(
        'The same original full code/state/replay epoch has now completed all7558updates,58488whole new code references and all admitted original execution/state targets. '
        'Both immutable trained and self-contained native models are written; actual held code loss is0.8192751234164461versus0.8685494615347125initial. '
        'Original GPU training ranks end normally, and all four currently active owned workers run the predeclared final native comparisons. '
        'The new public-pretraining controller, all ten source-preserving downstream replacements and complete exact unstarted-queue handoff are implemented/tested/queued. '
        'Full1Boriginal raw admission, CPUv3/v4preparation and actual fresh four-L40capacity/recovery/native cold load remain mandatory before new corpus optimization. '
        'New raw/function corpus neural updates remain0; the human/execution/machine full epochs remain pending on that new foundation.')
    requirements['gpu_runtime']['evidence']=(
        'The same exact original GPU parent retains the kernel lease and runs all four final native assessment workers after completing its7558updates. '
        'All ten retired old controllers were exact unstarted hidden childless CPU births, and every original training worker was untouched. '
        'The new pretraining parent remains live and waits for every predecessor child terminal, restored ownership and complete source admission/preparation. '
        'Its actual full-size four-L40nine-batch recovery/cold-output capacity proof has not run; a queued plan or small CPU controls are not that proof. '
        'Persistent current+pending full-state snapshots and initial/final native packages must preserve26GiBfree reserve; the34GiBvolatile capacity upper bound and96GiBworking headroom are charged to actual256GiBcgroup RAM. No new capacity scratch is allocated yet.')
    requirements['high_end_llm_capability']['evidence']=(
        f'NOT ACHIEVED. Completed7558update matched held code NLL is{after["token_mean_NLL"]:.12f}from{before["token_mean_NLL"]:.12f}, '
        f'a{100*ratio:.6f}%reduction on the same128windows/524160original targets. These are producer neural observations, not fresh loss recomputation, generated programs or independent all-domain accuracy. '
        'The latest completed independently checked matched generated-code result remains0/48functional at both initial andstep2048, with29to14syntax passes. '
        'The actual complete-epoch initial/trained final native comparisons are now running; no final correctness result is inferred from the loss. '
        'A separate actual hidden-GPU CPU queue will reconstruct every96stored decoded program source byte and rerun original finite cases, plus48ordinaryPython executions. '
        'Original GPU assessment token-ID/EOS journals are unavailable and are explicitly not certified by that code-source check. No GPT-level capability claim is supported.')
    for value in requirements.values():
        if value['status']!='incomplete_for_requested_final_model':raise ValueError('All ten final requirements remain incomplete')
    protected=set()
    for name in subprocess.check_output(['rg','--files','reports','-g','plan.json','-g','*-queue.json'],cwd=ROOT,text=True).splitlines():
        p=read(ROOT/name)
        if isinstance(p,dict):protected.update(p.get('source_files_sha256',{}))
    for p in [master_path,ROOT/'README.md',ROOT/'RESEARCH.md']:
        if str(p) in protected:raise ValueError('Immutable source-bound summary cannot be edited')
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    runtime=dict(at=at,actual_original_code_updates=7558,planned_updates=7558,original_GPU_parent=plan['owner'],
        active_four_owned_native_assessment_workers=active,new_public_pretraining_owner=queue['owner'],
        independent_decoded_program_reexecution_owner=replay['owner'],current_raw_reader_progress=reader,
        new_raw_and_function_corpus_neural_updates=0,final_native_correctness_results_pending=True,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
    atomic_json(OUT/'runtime.json',runtime)
    paths=[Path(__file__),previous,source/'training-complete.json',source/'held-packed-initial.json',source/'held-packed-7558.json',
        q_path,replay_path,OUT/'runtime.json',ROOT/'reports/decoded-native-code-reexecution-v1-tests.log']
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,previous_full_scope_review=str(previous.relative_to(ROOT)),
        source_files_sha256={str(p):file_hash(p) for p in paths},previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,original_full_scope_preserved=True,candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={n:r['evidence'] for n,r in requirements.items()},latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),
        latest_actual_main_runtime=runtime,latest_complete_code_epoch_training='reports/code-first-continuation-v3/training-complete.json',
        latest_complete_code_epoch_held_NLL=after['token_mean_NLL'],latest_complete_code_epoch_relative_held_NLL_reduction=ratio,
        queued_complete_code_epoch_program_reexecution=str(replay_path.relative_to(ROOT)),all_ten_final_requirements_incomplete=True,
        candidate_promoted=False,goal_achieved=False,current_goal_turn_classification='progress')
    atomic_json(master_path,master)
    paragraph=(f'Completed code epoch, **{at}**: all **7,558updates** and the full declared code/output/state/replay exposure are complete. '
        'The trained and self-contained model files are written; all four owned GPUs now run the predeclared final native comparisons. '
        f'Same-window held code NLL is **{after["token_mean_NLL"]:.6f}**, down **{100*ratio:.2f}%** from **{before["token_mean_NLL"]:.6f}** '
        '(128windows/524160targets; producer observations). **Final generated correctness is pending**; the last independently checked earlier '
        'coding result remains0/48functional atstep2048. A separate CPU queue will freshly rerun all96final decoded native programs '
        'and48ordinaryPython programs. This does not certify unavailable original token-action/EOS journals.\n\n'
        'The completed1B-target raw corpus is still undergoing full independent source admission; verified826train/15validfunctions are admitted. '
        'The new complete public-pretraining and full human/RL/evaluation pipeline is actually queued after the completed tested handoff. '
        '**New raw/function corpus updates:0. All ten final requirements remain incomplete.** '
        '[Complete source epoch](reports/code-first-continuation-v3/training-complete.json), '
        '[fresh program-reexecution queue](reports/code-first-complete-native-program-reexecution-CPU-v1-queue.json), '
        '[full review](reports/completed-code-epoch-pretraining-queue-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for p in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_original_updates=7558,final_held_NLL=after['token_mean_NLL'],
        relative_reduction=ratio,new_corpus_updates=0,all_ten_final_requirements_incomplete=True)),flush=True)


if __name__=='__main__':main()
