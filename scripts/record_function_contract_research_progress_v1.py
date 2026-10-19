"""Record completed full-size controls and direct public source candidates."""
from copy import deepcopy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from record_public_billion_candidate_progress_v1 import STAGES,live,read

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/function-contract-research-progress-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden original full-scope review required')
    master_path=ROOT/'reports/completion-audit.json';master=read(master_path);previous=ROOT/master['latest_full_scope_review']
    requirements=deepcopy(read(previous)['requirements']);protected=set();controllers=[]
    for name in STAGES+['public-billion-causal-candidate-CPU-v2','public-billion-causal-candidate-v2-full-readback-v1','public-pretraining-sources-CPU-v1']:
        path=ROOT/'reports'/name/'plan.json'
        if not path.exists():path=ROOT/'reports'/(name+'-queue.json')
        plan=read(path);protected.update(plan['source_files_sha256']);controllers.append(dict(stage=name,owner=live(plan['owner'])))
        if (ROOT/'reports'/name/'failure.json').exists():raise ValueError('No original live production/source failure may be hidden')
    stage=ROOT/'reports/code-first-continuation-v3';progress=read(stage/'progress.json')
    workers=[live(p) for p in read(stage/'children.json') if p['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('All four exact original training workers must remain live')
    probe_path=ROOT/'reports/function-termination-full-CPU-v1/complete.json';probe=read(probe_path)
    probe_plan=read(probe_path.parent/'plan.json')
    proof_path=ROOT/'reports/function-termination-full-CPU-v1-readback/complete.json';proof=read(proof_path)
    git_path=ROOT/'reports/direct-algorithm-function-contracts-CPU-v1/complete.json';git=read(git_path)
    pilot_path=ROOT/'reports/public-function-contracts-pilot-CPU-v1/complete.json';pilot=read(pilot_path)
    for path in [probe_path,git_path,pilot_path]:
        if not read(path)['complete'] or not terminal(read(path.parent/'plan.json')['owner']):
            raise ValueError('Every completed control and source job must actually be terminal')
        if (path.parent/'failure.json').exists():raise ValueError('No failed source/control may be declared complete')
        protected.update(read(path.parent/'plan.json')['source_files_sha256'])
    roles=git['actual_verified_roles']
    if (not proof['complete'] or probe['actual_unique_student_values']!=984399690
            or roles['train_functions']!=826 or roles['valid_functions']!=15
            or git['actual_original_regular_files']!=1737 or git['actual_original_Python_files']!=1608
            or git['optimization_admitted'] is not False):raise ValueError('Every actual full model/source/role count must remain explicit')
    metrics=proof['arithmetic_from_original_observed_neural_likelihoods']
    tests={'public-function-contracts-and-termination-v1-final-tests':7,'public-git-function-sources-v1-final-tests':4}
    for name,count in tests.items():
        if str(count)+' passed' not in (ROOT/'reports'/(name+'.log')).read_text():raise ValueError('All eleven actual source/native/EOS/Git controls required')
    intake=read(ROOT/'reports/public-billion-causal-candidate-CPU-v2/progress.json')
    notes={
        'own_architecture':' A new source-grounded function-contract builder preserves each complete original Python function as its response, derives instructions from its original signature/literal docstring and supplies an explicit stdlib context. A separate auxiliary reweights only the existing true terminal EOS while retaining the full original code CE. Neither adds model parameters, imports external weights nor uses a neural teacher. These are controlled training hypotheses, not globally novel architecture claims.',
        'public_data':f' The whole first pinned public Parquet was inspected:126925rows/8028Pythonrows yield14static function candidates and8passing complete-example candidates. Rather than treating that tiny yield as sufficient, a separate direct public GitHub source is pinned atTheAlgorithms/Python commit25c712f2692449e91a57748d328ef34fccf25654/tree85d4f12cb691afddb21e3a54e4a69583570e41eb. Every1737ordinary archive file matches its original Git blob; all1608Pythonfiles and their top-level functions are observed. The826train/15valid complete functions pass all4289original public examples twice, with389458train/6563validresponseEOS targets. All original source notices and complete repository MIT license remain. There is no publisher train partition: the source roles are our existing deterministic document-hash rule, explicitly distinguished from legacy runtime adapter fields. Both new corpora remain unadmitted pending full independent source/native/runtime readback. The larger collector is now at{intake["candidate"]["actual_train_targets"]}candidate train targets, with0new corpus optimization updates.',
        'under_1b_parameters':' The full-size matched EOS experiment loads the exact immutable own984399690-value step2048snapshot, including all text/image/word/auxiliary heads. Each two-update arm changes all317language tensors and retains all144other tensors bit exactly. The complete standalone models cold-load every value and all four complete-function logit tensors exactly. The separate static reader independently checks every461unique native tensor and actual source masks. These diagnostic models start from an explicitly partial continuation snapshot and are not final production models.',
        'text_and_images':' Both full-size function control arms preserve every original vision, word and unused auxiliary value. This proves modality isolation for those text updates, not improved image quality. The original generic photos, visual QA, OCR, matching and held-word assessment scope remains intact; no new production image score or full-photo detection claim is introduced.',
        'human_feedback':' The new labels are original public code/docstrings and actual machine observations on their finite published examples, not human RLHF. All original human preference, execution-RL and machine-preference arms remain queued unchanged. Expected public answers are visible inside the unchanged original function docstrings: the new wrapper explicitly corrects the legacy expected-text flag and distinguishes withholding separate comparison artifacts from hiding source-contained answers.',
        'serious_full_model_training':f' The original four-L40 full code/state/replay epoch continues at{progress["actual_updates"]}/7558updates. A separate actual full-size CPU comparison performs two complete admitted function updates per arm from identical original parameters, source order, target masks and optimizer settings, comparing plain CE withCEplus0.05true-EOS auxiliary. It optimizes2previously admitted sources and observes2other unoptimized train-pool sources. They are not new independent benchmarks or all-history unseen data. Complete native models and source/value readback are preserved. This is not a full source epoch, fresh four-L40 capacity proof or durable Adam recovery test; no new1Bcorpus production update has occurred.',
        'high_end_llm_capability':f' NOT ACHIEVED. The new actual four-reference experiment finds source-gold endings already have meanEOS probability{metrics["whole_CE"]["before_optimized"]["mean_true_EOS_probability"]:.6f}on the optimized pair and{metrics["whole_CE"]["before_unoptimized_train_pool"]["mean_true_EOS_probability"]:.6f}on the unoptimized pair before updates. The EOS auxiliary improves observed terminal likelihood, but both arms worsen the unoptimized pair tokenNLL from0.538898toabout0.570. No fresh generated-code gain is measured, so the auxiliary is not promoted as a repetition fix. Actual completed native coding remains0/48functional with29to14syntax regression atstep2048. These results direct work toward source-function instruction coverage and behavior under self-generated contexts, not a claim that EOS weighting solves the failure.',
        'broad_independent_benchmarks':' The new unoptimized function controls are explicitly two previously admitted train-pool records and are not relabelled as independent held benchmarks. Public source example checks are finite verification, not hidden benchmark accuracy. All original full QA, AI2D, MBPP, held human, held words and local coding endpoints remain scheduled; the complete96-delivery source/outcome reader still establishes0/48functional at both fixed earlier endpoints. No benchmark population, case or action budget is removed.',
        'gpu_runtime':' All four exact original L40workers and all14original/current production/source/preparation controllers remain live; no process is signalled or restarted. The new function-source collectors and matched full-size EOS experiment finish on CPU with GPUs hidden. Eleven new source/native/EOS/Git controls pass. Read-only storage observation finds82GiBpersistent free at the end of this work and a256GiBtmpfs at/dev/shm; tmpfs consumes cgroup RAM and is not new persistent capacity or an allocation guarantee. It is a possible location for future disposable capacity snapshots after an actual memory check, not a change made by this turn.',
        'any_code_correctness_and_efficiency':' The new direct source corpus contains841whole functions/4289published finite examples, with826eligible source-train candidates and15source-valid candidates. Function bodies are not repaired, expected outputs are not fabricated, all examples must run twice and the unchanged complete native source must fit4096tokens. Unsupported context, decorators, syntax, embedded Markdown, duplicates and held-text signals retain explicit dispositions. This is candidate data quality evidence, not generated any-code correctness or optimal complexity. The full-size EOS control produces no fresh programs and establishes no algorithmic/efficiency gain; plain CE slightly outperforms the auxiliary on its two optimized-source tokenNLLs while the auxiliary better predicts EOS, so no general superiority is claimed.'}
    if set(notes)!=set(requirements):raise ValueError('All ten original final requirements must remain in scope')
    for name,note in notes.items():
        requirements[name]['evidence']+=note
        if requirements[name]['status']!='incomplete_for_requested_final_model':raise ValueError('No final capability requirement is complete')
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,actual_code_first_updates=progress['actual_updates'],planned_code_first_updates=7558,
        actual_same_four_GPU_workers=workers,all_fourteen_live_original_pipeline_and_source_controllers=controllers,
        original_jobs_not_signalled_or_restarted=True,new_public_corpus_neural_updates=0,
        actual_GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True),
        actual_filesystem_observation=subprocess.check_output(['df','-h',str(ROOT),'/tmp','/dev/shm'],text=True))
    if any(str(p) in protected for p in [master_path,ROOT/'README.md',ROOT/'RESEARCH.md']):raise ValueError('No live source-bound documentation may be changed')
    OUT.mkdir();atomic_json(OUT/'runtime.json',runtime);atomic_json(OUT/'captured-source-progress.json',intake)
    paths=[Path(__file__),previous,probe_path,proof_path,git_path,pilot_path,OUT/'runtime.json',OUT/'captured-source-progress.json',
        *[ROOT/'reports'/(n+'.log') for n in tests]]
    atomic_json(OUT/'review.json',dict(at=at,requirements=requirements,actual_runtime=runtime,
        previous_full_scope_review=str(previous.relative_to(ROOT)),source_files_sha256={str(p):file_hash(p) for p in paths},
        previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,original_full_scope_preserved=True,candidate_promoted=False,goal_achieved=False))
    (OUT/'previous-completion-audit.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={n:v['evidence'] for n,v in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_full_size_termination_control=str(probe_path.relative_to(ROOT)),latest_full_size_termination_value_readback=str(proof_path.relative_to(ROOT)),
        latest_direct_git_function_candidates=str(git_path.relative_to(ROOT)),
        next_required_work=[
            'Complete the live7558update full code/state/replay epoch and its unchanged final native assessments. Do not infer final coding gains from token loss or source verification.',
            'Finish all1Bpublic source quotas, the existing full original-source/native/held/rights reader and the queued complete source/retention preparation.',
            'Independently rebuild the full new direct-Git source/native/function/example corpus and reexecute every original qualified candidate before data admission; retain the declared841candidate outcome and all source exclusions.',
            'Implement/test the actual full-size public pretraining production controller, fresh four-L40capacity/private-RNG recovery/cold export and explicit GPU/storage ownership. Integrate an orderly complete lineage without mutating live scientific work.',
            'Use verified whole-function instruction coverage and controlled feedback on self-generated code to address the actual generation failure. The tiny EOS-only experiment is not a demonstrated fix.',
            'Finish all original full human preference, execution-RL, machine comparison, standalone and QA/visual/code/human/word assessments; verify all ten original final requirements before promotion.'],
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Whole-function public data and full-size EOS control, **{at}**: original four-L40 training continues at '
        f'**{progress["actual_updates"]}/7,558 updates**. A direct pinned GitHub source now yields **841complete function candidates '
        '(826train/15valid)** with **4,289original public examples passed twice**, after all1,608Pythonfiles and1,737regular '
        'Git blobs are inspected. Source-contained example answers are explicitly visible; these are finite public checks. '
        '**Data admission still requires full independent readback.**\n\n'
        'An actual **984,399,690-value** matched CPU experiment compares two complete-function CE updates with and without0.05true-EOS '
        'weight. All317language tensors change,144other tensors remain exact, and both complete native models cold-load. '
        'Correct reference endings already have99.2–99.8%meanEOSconfidence. Extra EOS weight improves that likelihood, but both '
        'arms worsen the two unoptimized train-pool tokenNLLs; **no generated correctness gain or production EOS change is established**. '
        f'The larger raw candidate has **{intake["candidate"]["actual_train_targets"]:,}targets**, with0new neural updates. '
        '**All ten final requirements remain incomplete.** '
        '[Direct source candidate](reports/direct-algorithm-function-contracts-CPU-v1/complete.json), '
        '[full-size source/value readback](reports/function-termination-full-CPU-v1-readback/complete.json), '
        '[full review](reports/function-contract-research-progress-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for path in [ROOT/'README.md',ROOT/'RESEARCH.md']:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),actual_code_first_updates=progress['actual_updates'],
        actual_direct_Git_function_candidates=841,source_targets=intake['candidate']['actual_train_targets'],
        new_function_or_raw_corpus_admitted=False,new_generated_code_quality_gain=False,goal_achieved=False)),flush=True)


if __name__=='__main__':main()
