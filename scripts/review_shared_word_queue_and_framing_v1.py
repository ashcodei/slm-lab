"""Full-scope snapshot of the live word queue and completed framing diagnosis."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/shared-word-production-queue-and-framing-diagnostic-v1-review.json'


def read(name):return json.loads((ROOT/name).read_text())


def actor(identity):
    try:
        p=Path('/proc')/str(identity['pid']);fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=identity['pid'],start_ticks=fields[19],state=fields[0],birth_matches=fields[19]==str(identity['ticks']),
            live=fields[19]==str(identity['ticks']) and fields[0] not in ('Z','X'),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
    except FileNotFoundError:return dict(pid=identity['pid'],live=False,handle_missing=True)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Read-only CPU review must explicitly hide GPUs')
    if OUTPUT.exists():raise ValueError('Preserve the original review snapshot')
    diagnostic=read('reports/knowledge-chat-diagnostic-v1/complete.json')
    if not diagnostic['all2236_original_question_frame_policy_deliveries_and_scores_reconciled']:
        raise ValueError('Complete actual four-cell diagnostic reconciliation required')
    diag_plan=read('reports/knowledge-chat-diagnostic-v1/plan.json');FileBindings(diag_plan['source_files_sha256']).check(full=True)
    diag_actors=[diag_plan['owner'],*read('reports/knowledge-chat-diagnostic-v1/children.json')]
    if any(actor(a)['live'] for a in diag_actors):raise ValueError('Diagnostic must actually end before final result readback')
    summaries=diagnostic['matched_CPU_FP32_model_and_prompt_factorial'];paired={}
    files=['reports/knowledge-chat-diagnostic-v1/plan.json','reports/knowledge-chat-diagnostic-v1/complete.json',
        'reports/shared-words-v1-proposed-exposure.json','reports/shared-words-full-CPU-v1/complete.json',
        'reports/shared-words-full-CPU-v1/independent-complete-readback.json','data/shared-byte-words-v1/admission.json',
        'src/slm_lab/shared_word_stage_v1.py','src/slm_lab/shared_word_evaluation_v1.py','src/slm_lab/shared_word_inference_v1.py',
        'src/slm_lab/shared_word_evidence_v1.py','scripts/train_shared_words_v1.py',
        'tests/test_shared_word_evaluation_v1.py','tests/test_shared_word_inference_v1.py','tests/test_shared_word_driver_v1.py',
        'reports/shared-word-stage-v1-tests.log','reports/shared-word-evaluation-v1-tests.log',
        'reports/shared-word-inference-v1-tests.log','reports/shared-word-driver-v1-tests.log',
        'reports/whole-human-SFT-v1/independent-public-comparison-v1.json']
    for frame in ('plain','chat'):
        names=[f'reports/knowledge-chat-diagnostic-v1/{arm}-{frame}.json' for arm in ('initial','rate_5e5')];files+=names
        rows=[{r['id']:r for r in read(n)['predictions']} for n in names]
        if len(rows[0])!=559 or rows[0].keys()!=rows[1].keys():raise ValueError('Same complete original diagnostic question population required')
        result={}
        for metric in ('generated_correct','likelihood_correct'):
            result[metric]=dict(gained=sum(not rows[0][k][metric] and rows[1][k][metric] for k in rows[0]),
                lost=sum(rows[0][k][metric] and not rows[1][k][metric] for k in rows[0]),
                retained=sum(rows[0][k][metric] and rows[1][k][metric] for k in rows[0]))
        paired[frame]=result
    queue_paths=['reports/shared-words-v1-queue.json','reports/code-initialized-alignment-v1-queue.json','reports/whole-human-reward-v1-queue.json']
    queues={};doc_paths={str(ROOT/n) for n in ['README.md','RESEARCH.md','reports/completion-audit.json']}
    for name in queue_paths:
        q=read(name)
        if doc_paths&set(q['source_files_sha256']):raise ValueError('Never edit documentation bound by a live queue')
        if not actor(q['owner'])['live']:raise ValueError('Actual live queue birth identity required')
        FileBindings(q['source_files_sha256']).check(full=True)
        queues[name]=dict(status=q['status'],owner=actor(q['owner']),predecessor=actor(q['predecessor']),
            production_updates=q.get('production_model_updates',q.get('model_updates')),all_bound_source_files_rehashed_unchanged=True)
    proposal=read('reports/shared-words-v1-proposed-exposure.json')
    if (proposal['source_exposure_budget']['source_population']!=91585 or proposal['source_exposure_budget']['source_exposures']!=91585
            or proposal['source_exposure_budget']['repeated_rounding_exposures']!=0 or proposal['complete_held_words']!=3429):
        raise ValueError('Every complete original word exposure and full held control budget required')
    sft_plan=read('reports/whole-human-SFT-v1/plan.json');registered=[]
    for who in read('reports/whole-human-SFT-v1/children.json'):
        observed=actor(who)
        if observed['live']:registered.append(dict(observed,subject=who['subject']))
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),
        antecedent_full_scope_review='reports/code-initialized-model-and-shared-word-admission-v1-review.json',
        previous_goal_turn_classification=dict(classification='progress',reason='Implemented actual shared word gradients/recovery and completed full-size source-bound CPU learning/readback plus all-public-program reexecution.'),
        current_goal_turn_classification=dict(classification='progress',reason='Implemented and tested full held-word controls/raw-score audit/inference and owned full-source driver, proved all91585unrepeated source exposures/all3429wrong controls, launched a live source-bound production queue and completed four-cell framing diagnosis.'),
        actual_model_unique_parameters=999860418,external_pretrained_teacher_or_OCR_weights_loaded=False,
        completed_CPU_FP32_format_diagnostic=dict(questions_per_cell=559,total_original_deliveries=2236,
            results=summaries,paired_trained_minus_initial=paired,all_original_raw_sources_tokens_and_scores_reconciled=True,
            numerical_neural_forwards_not_independently_replayed=True,
            interpretation='Chat framing improves each policy relative to plain, but training regresses within both matched frames; format alone does not remove the observed regression.',
            native_CUDA_BF16_benchmark_not_replaced=True,held_training_or_model_selection=False,model_updates=0,all_actual_registered_actors_ended=True),
        shared_word_training=dict(source_budget=proposal['source_exposure_budget'],steps_per_arm=5725,
            unrepeated_final_batch_words=1,complete_held_crops=3429,
            actual_all_held_wrong_controls_one_to_one_within_family_different_original_photo_pixels_transcript=True,
            learning_rates=[1e-4,3e-4,1e-3],physical_training_GPUs=[0,1,2],initial_held_GPU=3,
            actual_optimized_values=2714178,actual_optimized_tensors=211,
            fixed_language_reward_critic_and_original_values=True,complete_byte_CTC_and_native_response_EOS_CE=True,
            native_generation_budget=64,native_eval_batch=16,all_held_proper_blank_wrong_CTC_and_native_generations=True,
            native_OCR170_object164_caption128_comparisons_for_initial_and_all_arms=True,
            actual_source_extreme_four_L40_capacity_and_exact_recovery_required_before_production=True,
            full_CPU_two_disposable_updates_already_proven=True,
            language_anchor='original own trained CE_control code computation; later mixed language factors not attached',
            final_mixed_policy_word_attachment_and_broad_native_validation_still_required=True,
            source_bound_actual_live_queue='reports/shared-words-v1-queue.json',production_updates=0,
            new_focused_checks_passed=7,prior_kernel_checks_passed=3,
            no_automatic_held_based_promotion=True,queue_expiry_hours=36,owned_GPU_lease_hours=14,automatic_retry=False,
            full_photo_text_detection_not_established_by_crop_learning=True),
        actual_live_source_bound_queues=queues,actual_live_original_GPU_owner=actor(sft_plan['owner']),registered_live_GPU_workers=registered,
        GPU_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip().splitlines(),
        original_work=dict(data_waiter=actor(dict(pid=2754743,ticks='27602681')),photo_collector=actor(dict(pid=130149,ticks='20300752')),
            photo_progress=read('data/vqa-real-pairs-v1/progress.json'),human_feedback_waiter=actor(dict(pid=130151,ticks='20300753')),
            existing_work_not_signaled_or_restarted_by_this_turn=True),
        full_scope_requirements_remaining=dict(high_end_all_round_or_GPT6_1_parity='not achieved or demonstrated',
            correct_efficient_broad_code='public2/175after human SFT, local functional/efficiency zero; new mixed code training and full evaluations pending',
            useful_generic_multimodal_reasoning='native image metrics weak; word production training, final assembly and full-photo integration unfinished',
            genuine_human_RLHF='full human-ranked reward learning queued; fresh own-policy PPO/critic training and useful aligned behavior unverified',
            complete_broad_public_curriculum='education pack admission/training, original photo finalization and full later curriculum incomplete',
            independent_final_capability_validation='existing development and exposure qualifications retained; no fully independent high-end result'),
        evidence_files_sha256={name:file_hash(ROOT/name) for name in files},candidate_promoted=False,goal_achieved=False))
    print(OUTPUT)


if __name__=='__main__':main()
