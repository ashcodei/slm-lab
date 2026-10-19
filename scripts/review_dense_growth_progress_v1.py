"""Full-scope evidence snapshot for the actual dense-growth CPU prototype."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/dense-fold-growth-and-full-plasticity-v1-review.json'


def read(name):return json.loads((ROOT/name).read_text())


def actor(identity):
    try:
        p=Path('/proc')/str(identity['pid']);fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
        return dict(pid=identity['pid'],start_ticks=fields[19],state=fields[0],birth_matches=fields[19]==str(identity['ticks']),
            live=fields[19]==str(identity['ticks']) and fields[0] not in ('Z','X'),command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip())
    except FileNotFoundError:return dict(pid=identity['pid'],live=False,handle_missing=True)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU evidence review must explicitly hide every GPU')
    if OUTPUT.exists():raise ValueError('Preserve the original dense progress snapshot')
    proof=read('reports/own-dense-growth-full-CPU-v1/complete.json');reader=read('reports/own-dense-growth-full-CPU-v1/independent-complete-readback.json')
    if (proof['actual_counts']['total_unique']!=984399690 or proof['actual_dense_language_values']!=931244544
            or proof['actual_dense_language_Adam_states']!=317 or not proof['actual_next_update_all_weights_Adam_RNG_source_cursor_and_metrics_exact']
            or not proof['actual_trained_full_size_cold_reload_exact']
            or reader['original_complete_sha256']!=file_hash(ROOT/'reports/own-dense-growth-full-CPU-v1/complete.json')
            or not reader['every168_original_dense_matrix_sum_independently_recomputed']):
        raise ValueError('Actual full dense/head count, next update/cold reload and independent all-matrix readback required')
    FileBindings(read('reports/own-dense-growth-full-CPU-v1/plan.json')['source_files_sha256']).check(full=True)
    queues={};doc_paths={str(ROOT/n) for n in ('README.md','RESEARCH.md','reports/completion-audit.json')}
    for name in ['reports/whole-human-reward-v1-queue.json','reports/code-initialized-alignment-v1-queue.json','reports/shared-words-v1-queue.json']:
        q=read(name)
        if doc_paths&set(q['source_files_sha256']):raise ValueError('Never edit live-bound documentation')
        queues[name]=dict(status=q['status'],owner=actor(q['owner']),predecessor=actor(q['predecessor']),
            production_updates=q.get('production_model_updates',q.get('model_updates')))
        if not queues[name]['owner']['live']:raise ValueError('Exact queued process birth must actually remain live')
    sft=read('reports/whole-human-SFT-v1/plan.json');live_workers=[]
    for who in read('reports/whole-human-SFT-v1/children.json'):
        current=actor(who)
        if current['live']:live_workers.append(dict(current,subject=who['subject']))
    files=['src/slm_lab/own_dense_growth_v1.py','src/slm_lab/own_dense_checkpoint_v1.py','tests/test_own_dense_growth_v1.py',
        'reports/own-dense-growth-v1-tests.log','scripts/probe_own_dense_growth_full_CPU_v1.py',
        'scripts/review_own_dense_growth_CPU_v1.py','reports/own-dense-growth-full-CPU-v1/plan.json',
        'reports/own-dense-growth-full-CPU-v1/complete.json','reports/own-dense-growth-full-CPU-v1/independent-complete-readback.json',
        'reports/shared-word-production-queue-and-framing-diagnostic-v1-review.json',
        'reports/knowledge-chat-diagnostic-v1/complete.json','reports/whole-human-SFT-v1/independent-public-comparison-v1.json']
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),
        antecedent_full_scope_review='reports/shared-word-production-queue-and-framing-diagnostic-v1-review.json',
        previous_goal_turn_classification=dict(classification='progress',reason='Built and tested full held-word scoring/source audit and owned full-source controller, launched actual word queue and completed all2236frame/model knowledge deliveries.'),
        current_goal_turn_classification=dict(classification='progress',reason='Implemented and instantiated own folded35-layer984Mmodel with all heads, executed actual931Mfull-language Adam/RNG recovery/cold reload and independently reconstructed all168warm matrix sums, both optimizer snapshots and complete used sources.'),
        new_experimental_architecture=dict(actual_counts=proof['actual_counts'],original_own_code_dense_layers=24,
            grown_dense_layers=35,added_initial_identity_blocks=11,
            original_own_trained_code_merged_into_single_dense_backbone=True,external_pretrained_or_teacher_weights_loaded=False,
            original_general_branch_not_retained_in_new_candidate=True,
            complete_counted_heads=['critic','reward','word','pointing','router','detector'],
            current_controlled_aligned_models_remain_999860418_parameters=True,
            prototype_does_not_replace_current_default_candidate_or_running_jobs=True),
        full_CPU_proof=dict(seconds=proof['seconds'],full_FP32_threads=2,
            actual_dense_language_values=931244544,actual_dense_language_Adam_states=317,
            actual_initial_sampled_FP32_parity=proof['actual_initial_fold_and_growth_sampled_FP32_parity'],
            all11added_blocks_initial_outputs_zero=True,all_initial_saved_dense_matrices_independently_rebuilt_from_own_durable_sources=True,
            complete_actual_train_source_metrics=proof['complete_source_metrics'],
            complete_source_choice='shortest whole original training conversation18tokens/6targets and verified program89tokens/12targets; CPU mechanics only',
            logical_disposable_updates=2,physical_optimizer_calls=3,
            all_full_dense_language_gradient_tensors_finite_and_no_fixed_grads=True,
            exact_actual_next_weights_Adam_RNG_cursor_and_metrics=True,
            CPU_restore_privately_copies_Adam_to_preserve_input_payload=True,
            all_fixed_candidate_vision_critic_and_heads_values_versions_pointers_flags_exact=True,
            all_source_original_weights_metadata_and_preexisting_code_cache_exact=True,
            actual_full_size_trained_neural_cold_reload_exact=True,
            both_complete_Adam_updates_and_initial_model_retained=True,
            independent_reader_did_not_replay_neural_updates=True,
            independent_reader_checked_each_actual_weight_shape_finiteness_tied_storage_source_target_and_optimizer_state=True,
            every_probe_update_discarded_before_production=True),
        proposed_future_actor=proof['proposed_future_actor_budget'],
        new_dense_model_limitations=dict(no_actual_proposed_actor_instantiated_or_trained=True,
            full_corpus_dense_training_not_started=True,source_extreme_GPU_capacity_not_yet_proven=True,
            CUDA_BF16_native_code_math_knowledge_and_image_baselines_not_yet_run=True,
            numeric_fold_equivalence_is_FP32_tolerance_not_bitwise_arithmetic=True,
            old_general_grounding_and_human_RM_encoder_functions_not_preserved_by_code_folding=True,
            copied_or_fresh_auxiliary_heads_do_not_establish_useful_reading_reward_routing_or_detection=True,
            no_new_held_general_or_coding_capability_gain_measured=True),
        relevant_primary_prior_work=[dict(title='Net2Net: Accelerating Learning via Knowledge Transfer',url='https://arxiv.org/abs/1511.05641',
            relevance='Function-preserving deeper/wider transfer is established prior work; our zero-output pre-norm block construction is tested directly.'),
            dict(title='LoRA: Low-Rank Adaptation of Large Language Models',url='https://arxiv.org/abs/2106.09685',
            relevance='Low-rank updates are prior work; merging own learned linear factors is not a new-method or quality claim.')],
        live_original_GPU_owner=actor(sft['owner']),live_registered_GPU_workers=live_workers,actual_live_source_bound_queues=queues,
        actual_GPU_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip().splitlines(),
        original_work=dict(photo_collector=actor(dict(pid=130149,ticks='20300752')),photo_progress=read('data/vqa-real-pairs-v1/progress.json'),
            data_waiter=actor(dict(pid=2754743,ticks='27602681')),feedback_waiter=actor(dict(pid=130151,ticks='20300753')),
            nothing_restarted_preempted_or_signaled_by_dense_prototype=True),
        full_original_scope_remaining=dict(high_end_GPT6_1_level_coding_and_all_round='not achieved or verified',
            current_correct_efficient_code='local functional/efficiency zero; selected public2/175; mixed code experiment pending',
            broad_multimodal_reasoning='native image results weak; full word experiment queued, final integration and full-photo work incomplete',
            genuine_human_RLHF='full ranking reward learning queued; fresh own-policy PPO/critic and useful preference behavior unfinished',
            full_public_curriculum='education pack not admitted/trained, original paired-photo finalization/later continuation unfinished',
            broad_independent_final_benchmarks='development/source-exposure qualifications remain; complete current math and new dense native comparisons pending'),
        evidence_files_sha256={n:file_hash(ROOT/n) for n in files},candidate_promoted=False,goal_achieved=False))
    print(OUTPUT)


if __name__=='__main__':main()
