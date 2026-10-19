"""Record full-size continuation mechanics and the ongoing main-model check."""
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/folded-dense-continuation-v1-review'


def current(owner,waiting=False):
    proc=Path('/proc')/str(owner['pid']);fields=(proc/'stat').read_text().split(') ',1)[1].split()
    if fields[19]!=owner['ticks'] or fields[0] in {'Z','X'}:raise ValueError('Exact original live owner required')
    children=(proc/'task'/str(owner['pid'])/'children').read_text().strip()
    visible=next((s.split('=',1)[1] for s in (proc/'environ').read_bytes().decode().split('\0') if s.startswith('CUDA_VISIBLE_DEVICES=')),None)
    if waiting and (children or visible!=''):raise ValueError('Waiting original GPU successor must remain hidden and childless')
    return dict(pid=owner['pid'],ticks=fields[19],state=fields[0],children=children,CUDA_VISIBLE_DEVICES=visible)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-only whole-scope review required')
    master_path=ROOT/'reports/completion-audit.json';master=json.loads(master_path.read_text())
    previous_path=ROOT/master['latest_full_scope_review'];previous=json.loads(previous_path.read_text());requirements=previous['requirements']
    paths=dict(fold=ROOT/'reports/full-actor-fold-dense-update-CPU-v1/complete.json',
        fold_readback=ROOT/'reports/full-actor-fold-dense-update-CPU-v1-readback/complete.json',
        data=ROOT/'reports/focused-code-continuation-sources-v1-preparation/complete.json')
    reports={name:json.loads(path.read_text()) for name,path in paths.items()}
    for name in ['fold_readback','data']:FileBindings(reports[name]['source_files_sha256']).check(full=True)
    fold,readback,data=[reports[name] for name in ['fold','fold_readback','data']]
    if (fold['actual_language_updates']!=2 or fold['physical_optimizer_calls']!=3
            or readback['actual_unique_native_student_values']!=984399690 or readback['actual_changed_dense_language_tensors']!=317
            or readback['actual_fixed_vision_critic_auxiliary_tensors']!=144
            or not fold['actual_next_native_update_all_language_weights_Adam_private_CPU_RNG_bit_exact']
            or not fold['cold_final_student_full4096logits_and_word_outputs_bit_exact']):
        raise ValueError('Every actual full-size fold, full-gradient, recovery and cold native evidence required')
    if (not data['every58488original_code_presentation_rebuilt_in_complete_real_native_batches']
            or data['actual_code_examples']!=58488 or data['actual_complete_code_context_positions']!=88610948
            or not data['execution_source_missing_so_optimization_forbidden']):
        raise ValueError('Complete explicitly preparation-only code/replay data path required')
    main_path=ROOT/'reports/phase-joint-v3/plan.json';main=json.loads(main_path.read_text())
    progress=json.loads(main_path.with_name('progress.json').read_text())
    reference_path=main_path.with_name('train-reference-binding.json');reference=json.loads(reference_path.read_text())
    if reference['foundation_completed_updates']!=12853 or progress['phase']!='instruction_and_vision':
        raise ValueError('The original main run must have actually entered its later phase with its own completed foundation')
    workers=[current(r) for r in json.loads(main_path.with_name('children.json').read_text()) if r['subject'].startswith('train-rank-')]
    if len(workers)!=4:raise ValueError('All four original main workers required')
    names=['phase-human-preference-v4','broad-anchored-code-v3','full-held-QA-pipeline-v1','full-AI2D-pipeline-v1']
    queues={name:json.loads((ROOT/'reports'/(name+'-queue.json')).read_text()) for name in names}
    interim_path=ROOT/'reports/main-foundation-instruction14336-code-CPU-v2/plan.json';interim=json.loads(interim_path.read_text())
    if interim_path.with_name('failure.json').exists():raise ValueError('The new exact main-model interim assessment has failed')
    source_jobs={}
    for name in ['CodeContests-full-Python-references-v1','CodeContests-public-execution-traces-candidate-v1']:
        path=ROOT/'reports'/name/'plan.json';plan=json.loads(path.read_text())
        source_jobs[name]=dict(owner=current(plan['owner']),progress_snapshot=json.loads(path.with_name('progress.json').read_text()))
    at=datetime.now(timezone.utc).isoformat()
    runtime=dict(at=at,main_owner=current(main['owner']),original_main_workers=workers,main_actual_updates=progress['actual_updates'],
        main_planned_updates=35284,main_phase=progress['phase'],original_waiting_GPU_successors={n:current(q['owner'],True) for n,q in queues.items()},
        current_main_interim_code_owner=current(interim['owner']),source_jobs=source_jobs,
        GPU_observation=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],text=True))
    initial_path=main_path.with_name('held-packed-initial.json');initial=json.loads(initial_path.read_text())
    candidates=[p for p in main_path.parent.glob('held-packed-*.json') if p.stem.rsplit('-',1)[-1].isdigit()]
    latest_path=max(candidates,key=lambda p:int(p.stem.rsplit('-',1)[-1]));latest=json.loads(latest_path.read_text())
    for key in ['original_split','selected_original_positions','original_valid_population','evaluated_windows','complete_causal_targets','maximum_original_context','original_manifest_sha256']:
        if initial[key]!=latest[key]:raise ValueError('Unchanged held token-loss population and native budget required')
    held=dict(step=int(latest_path.stem.rsplit('-',1)[-1]),initial_NLL=initial['token_mean_NLL'],latest_NLL=latest['token_mean_NLL'],
        relative_reduction=(initial['token_mean_NLL']-latest['token_mean_NLL'])/initial['token_mean_NLL'],perplexity=math.exp(latest['token_mean_NLL']),
        producer_neural_observation_input_protocol_and_arithmetic_checked=True,no_fresh_weight_Adam_or_neural_loss_recomputation_claim=True,
        not_generated_code_knowledge_or_visual_accuracy=True)
    worst=max(r['max_absolute_logit_difference'] for r in fold['actual_FP32_full_size_fold_calibrations'])
    requirements['own_architecture']['evidence'] += (
        ' A new own-policy folding mechanism merges245own low-rank updates into their actual dense matrices, copies71policy '
        'normalizations and replaces the unused base critic with the actual policy critic. It copies every own visual and '
        'auxiliary head and retains one tied vocabulary. Four meaningful tests cover nonzero policy/norm/critic math, '
        'text/image/word outputs, source/RNG/storage independence, full dense gradients, cold loading and finite factors '
        'whose product overflows. The actual999706442-value old step2048+40update own machine-contrast actor folds to one '
        f'984399690-value student, removing15306752values. FP32 full-size probes use complete4096and1301new-code contexts '
        f'and a real448human train photo; maximum logit drift is{worst},with0next-token argmax differences. Actual word-byte '
        'logits remain bit-exact. This is numerical calibration on an explicitly old experiment, not a production initializer '
        'or generated-code/all-head-function/GPU-BF16 parity claim. No outside pretrained model or teacher is used.')
    requirements['under_1b_parameters']['evidence'] += (
        ' The actual new independently stored folded native bundle has984399690unique neural values including all '
        'vision/reward/word/pointing/router/detector/critic weights. The static reader counts all461unique tensor entries, '
        'verifies the one stored vocabulary/head tie, and compares every final native tensor to the full private state. '
        'The native loader reconstructs from that bundle alone; it does not need old source checkpoints, actor weights, '
        'optimizer moments or a training reference for inference. The requested final high-end model is still unproved.')
    requirements['serious_full_model_training']['evidence'] += (
        ' The full-size folded CPU prototype actually trains all317dense language tensors on two different whole admitted '
        'new-code examples, one4096context/3603outputEOS targets and one1301context/740targets. It performs2logical/3physical '
        'updates, restores the first full model/Adam/private Python/NumPy/TorchCPU RNG and replays the exact next update. '
        'Changed-source-cursor and malformed CPU RNG states reject before mutation. The independent reader verifies all '
        '317language tensors changed, all144fixed vision/critic/auxiliary tensors stayed exact, all317Adam clocks/moments '
        'are finite and match1then2steps, and every final private/native tensor agrees. Actual cold student full4096logits '
        'and original word outputs are bit-exact. The reader does not repeat the neural replay. This is a mechanism '
        'proof, not a whole58488example epoch, useful coding result, human alignment or four-L40 admission. '
        f'The original main run separately completed its12853update foundation, bound its own new reference, and remains '
        f'live at{progress["actual_updates"]}/35284in instruction/vision training on its original four L40workers.')
    requirements['public_data']['evidence'] += (
        ' A real preparation-only continuation data path rebuilds every58488admitted new-code presentation, all88610948 '
        'context positions and54063335outputEOS targets in complete native batches. Its fixed schedule interleaves '
        'conversations, facts, math, varied code, JavaScript headers, Python exercises, real photos, visual reasoning, '
        'image text and word replay. All10actual replay family extremes, including native448images and separate64x448word '
        'CTC inputs, reconstruct. Every new code/execution record is scheduled once; replay uses explicitly repeated '
        'selected batches and never claims all original sources once. Four tests verify native masks/padding, partial '
        'distributed tails, deterministic source-complete scheduling, missing auxiliary admission rejection and length '
        'buckets. On the complete code population, bucketed full-batch padded token slots are91986968versus199262144 '
        'for same-width seeded random batches,53.83620483376913%fewer. This is not measured GPU speedup. The current '
        'preparation instance cannot optimize and omits the still-unadmitted execution candidate; production cannot '
        'silently omit that task. Public reference checks and actual-output candidate preparation continue separately.')
    requirements['any_code_correctness_and_efficiency']['evidence'] += (
        ' A newly live CPU-only matched observer fixes the main model completed foundation12853and step14336before '
        'seeing latter code outputs. It retains all48local native deliveries per endpoint, both unchanged frames and512 '
        'greedy cached FP32 actions, and now captures every real input/generated token ID without altering the returned '
        'tensor. Two real-small-neural tests verify transparent observation and failure restoration. It freshly '
        'reexecutes every native program and separately checks all48Python deliveries under the broader ordinary '
        'runtime, with12main-guard canonical controls. The ordinary runtime remains secondary to the original prompt '
        'contract, and timing stress is not asymptotic proof. The earlier failed fixture/setup is preserved; the '
        'guard rejected it before any real checkpoint generation, and the corrected version keeps identical fixed '
        'endpoints. Results remain pending. No code gain is inferred from a parameter fold or two source updates; '
        'the prior old contrast trial still has0/48functional correctness despite31/48syntax passes.')
    requirements['gpu_runtime']['evidence'] += (
        ' Fresh process-birth checks confirm allfour original main workers, every original human/code/text/visual '
        'GPU successor, the new exact CPU interim observer and both original public reference/data owners. No GPU '
        'lease, frozen training source or fixed final assessment was preempted. The new full-size CPU mechanism '
        'finished terminally; its source-preparation and static readers also completed. New four-GPU capacity, '
        'complete new-source continuation training and generated assessments remain required.')
    if len(requirements)!=10 or any(r['status']!='incomplete_for_requested_final_model' for r in requirements.values()):
        raise ValueError('Every original requested final requirement must remain explicitly incomplete')
    OUT.mkdir();evidence=[Path(__file__),previous_path,*paths.values(),main_path,reference_path,interim_path,initial_path,latest_path,
        ROOT/'reports/own-actor-dense-fold-v1-final-tests.log',ROOT/'reports/focused-code-continuation-data-v1-final-tests.log',
        ROOT/'reports/native-greedy-observer-v1-final-tests.log',ROOT/'reports/main-foundation-instruction14336-code-CPU-v1-failed-setup.json']
    report=dict(at=at,requirements=requirements,previous_full_scope_review=str(previous_path.relative_to(ROOT)),actual_runtime=runtime,
        actual_full_size_fold_and_language_learning=fold,actual_independent_full_state_readback=readback,
        actual_complete_continuation_source_preparation=data,latest_same_protocol_held_token_loss=held,
        interim_main_model_developmental_code_assessment_plan=str(interim_path),
        source_files_sha256={str(p):file_hash(p) for p in evidence},previous_goal_turn_classification='progress',current_goal_turn_classification='progress',
        all_ten_final_requirements_incomplete=True,no_original_scope_removed=True,
        new_generated_functional_knowledge_or_visual_gain_not_established=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'review.json',report)
    protected=set(main['source_files_sha256'])
    for q in [*queues.values(),interim]:protected.update(q['source_files_sha256'])
    for name in ['CodeContests-full-Python-references-v1','CodeContests-public-execution-traces-candidate-v1','AI2D-fixed-exposure-strata-v1']:
        protected.update(json.loads((ROOT/'reports'/name/'plan.json').read_text())['source_files_sha256'])
    docs=[ROOT/'README.md',ROOT/'RESEARCH.md',master_path]
    if any(str(p) in protected for p in docs):
        print(json.dumps(dict(review=str(OUT/'review.json'),bound_root_documents_preserved=True)),flush=True);return
    (ROOT/'reports/completion-audit-before-folded-dense-continuation-v1.json').write_bytes(master_path.read_bytes())
    master.update(updated_at=at,requirements={k:r['evidence'] for k,r in requirements.items()},
        latest_full_scope_review=str((OUT/'review.json').relative_to(ROOT)),latest_actual_main_runtime=runtime,
        latest_own_full_size_dense_fold_readback=str(paths['fold_readback'].relative_to(ROOT)),
        latest_prepared_complete_continuation_data=str(paths['data'].relative_to(ROOT)),
        latest_main_foundation_instruction_interim_code_plan=str(interim_path.relative_to(ROOT)),
        current_goal_turn_classification='progress',all_ten_final_requirements_incomplete=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(master_path,master)
    paragraph=(f'Full dense continuation mechanics, **{at}**: an old own trained actor folds from **999,706,442 to '
        '984,399,690 parameters**, retaining all own visual/auxiliary heads and one tied vocabulary. Full-size FP32 '
        f'text/real-photo calibration has maximum logit drift **{worst:.8f}**, no next-token choice differences and exact '
        'word-byte outputs. This is an old experimental source, not the production initializer or proof of generated '
        'quality. It actually completes **two full-backbone updates**, including a whole **4,096-token** admitted new-code '
        'example. The independent reader confirms all **317 language tensors changed**, all **144 fixed tensors** stayed '
        'exact and full Adam states reached steps1/2. Actual next-update recovery and cold native4,096logits/word outputs '
        'are exact. This is not a full source epoch or GPU capacity gate.\n\n'
        'The next data path rebuilds every **58,488 complete coding example** and all10broad/visual/word replay family '
        'extremes. Length buckets use **91,986,968 full-batch padded slots versus199,262,144** under seeded random batching '
        '(53.84% fewer); GPU speedup is unmeasured. Production rejects a missing/unadmitted execution task. '
        f'The original main run completed its12,853update foundation and is at **{progress["actual_updates"]:,}/35,284 '
        'updates in instruction/vision training**. A CPU observer is assessing the fixed main12853/14336checkpoints '
        'on all unchanged local code tasks, retaining actual action IDs and fresh execution outcomes. All original '
        'GPU assessment queues remain intact. **No new functional coding, knowledge or visual gain is claimed; '
        'all ten final requirements remain incomplete.** '
        '[Review](reports/folded-dense-continuation-v1-review/review.json), '
        '[full-state reader](reports/full-actor-fold-dense-update-CPU-v1-readback/complete.json), '
        '[continuation data](reports/focused-code-continuation-sources-v1-preparation/complete.json).\n\n')
    for path in docs[:2]:
        first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(review=str(OUT/'review.json'),folded_student_values=984399690,actual_dense_language_updates=2,
        new_code_examples=58488,main_updates=progress['actual_updates'],main_phase=progress['phase'],goal_achieved=False)),flush=True)


if __name__=='__main__':main()
