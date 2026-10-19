"""Record actual source/kernel work and the completed negative coding result."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from record_public_billion_candidate_progress_v1 import STAGES,live,read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/public-pretraining-kernel-progress-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists():raise ValueError('Fresh CPU-only full-scope review required')
    master_path = ROOT/'reports/completion-audit.json';master = read(master_path)
    previous = ROOT/master['latest_full_scope_review']
    requirements = deepcopy(read(previous)['requirements'])
    protected = set();controllers = []
    for name in STAGES+['public-billion-causal-candidate-CPU-v2','public-billion-causal-candidate-v2-full-readback-v1','public-pretraining-sources-CPU-v1']:
        path = ROOT/'reports'/name/'plan.json'
        if not path.exists():path = ROOT/'reports'/(name+'-queue.json')
        plan = read(path);protected.update(plan['source_files_sha256'])
        controllers.append(dict(stage=name,owner=live(plan['owner'])))
        if (ROOT/'reports'/name/'failure.json').exists():raise ValueError('No silent source/production failure may be reported as progress')
    source = ROOT/'reports/code-first-continuation-v3'
    progress = read(source/'progress.json')
    workers = [live(p) for p in read(source/'children.json') if p['subject'].startswith('train-rank-')]
    if len(workers) != 4:raise ValueError('All four exact original training workers required')
    code = ROOT/'reports/code-first-step2048-native-code-CPU-v1'
    coding_reader = ROOT/'reports/code-first-step2048-native-code-CPU-v1-readback/complete.json'
    observed = read(coding_reader)
    if (not observed['complete'] or not terminal(read(code/'plan.json')['owner'])
            or not terminal(read(ROOT/'reports/code-first-step2048-native-code-CPU-v1-readback-queue.json')['owner'])):
        raise ValueError('The exact coding producer and independent reader must actually finish')
    outcome = observed['actual_matched_endpoint_totals']
    if (outcome['initial']['tasks'] != 48 or outcome['step2048']['tasks'] != 48
            or outcome['initial']['syntax'] != 29 or outcome['step2048']['syntax'] != 14
            or outcome['initial']['functional'] != 0 or outcome['step2048']['functional'] != 0):
        raise ValueError('Completed negative coding outcome must remain explicit')
    diagnostic_path = ROOT/'reports/code-first-repetition-and-prompt-diagnostic-CPU-v1/complete.json'
    diagnostic = read(diagnostic_path)
    profiles = diagnostic['actual_all96_action_profiles']
    if not diagnostic['complete'] or sum(r['deliveries'] for r in profiles) != 96:
        raise ValueError('Every actual generation action profile required')
    EOS = {endpoint:sum(r['stopped_on_EOS'] for r in profiles if r['endpoint']==endpoint) for endpoint in ['initial','step2048']}
    if EOS != dict(initial=25,step2048=7):raise ValueError('Actual stop-action regression must reconstruct')
    parallel_path = ROOT/'reports/public-pretraining-four-CPU-v1/complete.json'
    parallel = read(parallel_path)
    if (not parallel['complete'] or not parallel['all_four_actual_CPU_workers_passed']
            or parallel['serial_full_update_maximum_parameter_difference'] > 2e-6
            or parallel['final_replica_examples'] != [1,0,0,0]
            or parallel['production_updates'] != 0):raise ValueError('Actual four-worker empty-tail/recovery proof required')
    tests = {'public-pretraining-data-update-checkpoint-v1-tests':7,'public-pretraining-source-preparation-v1-tests':2}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('All nine new targeted tests must pass')
    intake = read(ROOT/'reports/public-billion-causal-candidate-CPU-v2/progress.json')
    preparation = ROOT/'reports/public-pretraining-sources-CPU-v1/plan.json'
    notes = {
        'own_architecture':' A new full-weight public-pretraining objective now removes the own-reference forward from raw/retained text CE, while image/word updates explicitly keep the language decoder fixed. It uses the original own neural computation without new parameters, external model weights or live teachers. Two modality tests and a direct independent token-loss gradient check cover this behavior; no globally novel architecture or quality gain is claimed.',
        'public_data':f' The corrected public collector now has{intake["candidate"]["actual_train_targets"]}retained unadmitted train targets. A new strict optimization data adapter requires the actual full1Bsource admission and exact own BPE, then adds every selected previous code/output/state/12family replay presentation. Compute bucketing preserves their full multiset, including prior replay repetitions, without clipping source tails or admitting held roles. A new CPU preparation controller is live and waiting for the exact full independent reader before constructing the complete real source schedule and family-extreme tensor batches.',
        'under_1b_parameters':' This data/objective work adds0student parameters. The four-CPU control uses an explicitly small randomly initialized own model, not the full984399690-value student. Its evidence cannot substitute for a fresh full-size four-L40 capacity/cold-load experiment, which remains required.',
        'text_and_images':' The new objective has actual separate text, image and byte/native word optimizer tests. Text changes language leaves while retaining every vision/word/auxiliary value; image and word updates retain every language/unused auxiliary value. The preparation controller retains all selected original photo, visual reasoning, OCR and real-word sources and checks their actual complete tensors after full raw-data admission. No new production visual score is measured.',
        'human_feedback':' The original human chosen-CE/DPO, code execution-RL and machine preference arms remain unchanged and queued. The reference-free raw-pretraining kernel is not RLHF. Its five real public code examples have already-admitted publisher CE labels, not new human preferences or newly executed correctness rewards.',
        'serious_full_model_training':f' Same original four-L40 training continues at{progress["actual_updates"]}/7558updates. A new complete-source update/Adam/private-RNG checkpoint implementation passes7targeted data/objective/recovery tests. Four actual CPU workers process five distinct already-admitted full code references in2logical updates/3physical calls, including a1/0/0/0tail. Full parameters, Adam moments and each private RNG reproduce bit exactly on the real next update; corrupt source positions, optimizer clocks and another worker RNG reject before mutation. Their serial full-update maximum parameter difference is{parallel["serial_full_update_maximum_parameter_difference"]}. This is a small-network mechanism proof, not new full-size pretraining. Fresh GPU capacity, a concrete GPU production controller and actual training on the new corpus remain unfinished.',
        'high_end_llm_capability':' The matched step2048generated-code comparison and independent source/outcome readback are now COMPLETE. Both endpoints score0/48functional/empirical-efficiency passes and0/24ordinaryPython passes; syntax regresses29/48to14/48. Thus the token-loss improvement does not establish a functional coding improvement. All96original input/action/program deliveries and48secondaryPythonexecutions are preserved. Frontier-level and all-round capability are still unachieved.',
        'broad_independent_benchmarks':' The finished96-delivery CPU experiment covers the original12tasks×2languages×2frames×2endpoints with unchanged512token greedy budgets. Its reader reconstructs original input/actions/decoded programs and retained actual outcomes without a new neural generation or program rerun. The original broad/full QA, visual, MBPP, held human and final-word populations remain scheduled; no narrow mechanism proof replaces their pending actual scores.',
        'gpu_runtime':' All11original pipeline controllers and four original GPU ranks remain live. The corrected source collector, full source reader and new source-preparation controller are also live and GPU-hidden. No existing process is interrupted or restarted. The four-worker Gloo experiment finished successfully on CPU and never owned a GPU. Its complete-source recovery proof and nine targeted tests are preserved independently of the unchanged training jobs.',
        'any_code_correctness_and_efficiency':' Completed results remain0/48functional after2048updates, with syntax29to14. A new static diagnostic identifies a stopping/repetition regression: EOS-terminated generations fall25/48to7/48; unchanged512token budget exhaustions rise23/48to41/48; samples with a majority of repeated16grams rise22/48to39/48. Eight evenly spaced real admitted training prompt prefixes match the evaluation User/Assistant wrapper, so no observed wrapper mismatch explains these samples. This is a limited format check, not an all-history certificate. Repetition is a diagnostic statistic, not correctness. These findings motivate source-grounded function-contract/EOS instruction examples and controlled objective experiments before a claim that raw-data scale or RL alone fixes generation.'}
    if set(notes) != set(requirements):raise ValueError('All ten original final requirements must remain in scope')
    for name,note in notes.items():
        requirements[name]['evidence'] += note
        if requirements[name]['status'] != 'incomplete_for_requested_final_model':raise ValueError('No final requirement is established by these mechanisms')
    at = datetime.now(timezone.utc).isoformat()
    runtime = dict(at=at,actual_code_first_updates=progress['actual_updates'],planned_code_first_updates=7558,
        actual_same_four_GPU_workers=workers,all_fourteen_live_controllers=controllers,
        completed_coding_endpoint_totals=outcome,actual_stop_action_totals=EOS,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        new_billion_source_neural_updates=0,original_jobs_not_signalled_or_restarted=True)
    if any(str(p) in protected for p in [master_path,ROOT/'README.md',ROOT/'RESEARCH.md']):
        raise ValueError('Live source-bound documentation must remain unchanged')
    OUT.mkdir();atomic_json(OUT/'runtime.json',runtime);atomic_json(OUT/'captured-source-progress.json',intake)
    evidence = [Path(__file__),previous,coding_reader,diagnostic_path,parallel_path,preparation,
        OUT/'runtime.json',OUT/'captured-source-progress.json',
        *[ROOT/'reports'/(name+'.log') for name in tests]]
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,actual_runtime=runtime,
        previous_full_scope_review=str(previous.relative_to(ROOT)),source_files_sha256={str(p):file_hash(p) for p in evidence},
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,original_scope_preserved=True,candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:v['evidence'] for k,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_public_pretraining_kernel='src/slm_lab/public_pretraining_update_v1.py',
        latest_public_pretraining_four_CPU_proof='reports/public-pretraining-four-CPU-v1/complete.json',
        queued_public_pretraining_source_preparation=str(preparation.relative_to(ROOT)),
        latest_completed_coding_result=str(coding_reader.relative_to(ROOT)),
        latest_coding_failure_diagnostic=str(diagnostic_path.relative_to(ROOT)),
        next_required_work=[
            'Finish the original7558update code/state/replay run; the completed step2048comparison proves no functional gain and a stopping/syntax regression.',
            'Finish all1Bpublic source quotas, the queued independent full original-source/native/rights/held reader and the new real-source compute-batched preparation.',
            'Implement the actual full-size own-model public-pretraining GPU controller with fresh four-L40 capacity, complete Adam/private-RNG recovery, native cold export, explicit storage/GPU ownership and an orderly lineage integration. No new billion-source optimization has occurred.',
            'Prepare independently verified whole-function public doc/contract instruction and EOS supervision to address the measured degeneration; compare against existing objectives without changing held inputs, repairing generated programs or inventing benchmark wins.',
            'Complete the original full human preference, execution-RL, machine comparison, standalone exports and all full QA/visual/code/human/word assessments. Verify all ten final requirements before promotion.'],
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph = (f'Reference-free pretraining kernel and completed coding regression, **{at}**: the same four-L40 run is at '
        f'**{progress["actual_updates"]}/7,558 updates**. The complete matched step2,048coding result remains **0/48functional**, '
        '**29/48→14/48syntax**, and **0/24ordinaryPython** at both endpoints. Actual EOS stops fall **25/48→7/48**. '
        'Eight sampled real training prompts match the native instruction wrapper; the observed issue includes repetition and stopping.\n\n'
        '**Nine new targeted tests and four actual CPU workers pass** for a reference-free raw/text objective with decoder-frozen '
        'image/word replay, complete source/optimizer/private-RNG recovery and a1/0/0/0tail. The serial parameter difference is '
        f'**{parallel["serial_full_update_maximum_parameter_difference"]:.2g}** on the small own mechanics model. '
        f'The public collector has **{intake["candidate"]["actual_train_targets"]:,}candidate train targets**; a new source-preparation '
        'controller waits for full1Bdata admission, then retains every previous code/state/12family replay presentation in '
        'compute-bounded batches. **New billion-source training updates:0. Full-size GPU production remains to implement. '
        'All ten final requirements remain incomplete.** '
        '[Completed coding readback](reports/code-first-step2048-native-code-CPU-v1-readback/complete.json), '
        '[repetition diagnostic](reports/code-first-repetition-and-prompt-diagnostic-CPU-v1/complete.json), '
        '[four-worker proof](reports/public-pretraining-four-CPU-v1/complete.json), '
        '[full review](reports/public-pretraining-kernel-progress-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest = path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_code_first_updates=progress['actual_updates'],
        actual_source_targets=intake['candidate']['actual_train_targets'],completed_coding=outcome,
        new_public_source_training_updates=0,all_ten_final_requirements_incomplete=True,goal_achieved=False)),flush=True)


if __name__ == '__main__':main()
