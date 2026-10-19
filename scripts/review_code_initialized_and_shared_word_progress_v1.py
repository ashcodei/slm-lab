"""Current full-scope evidence snapshot without changing any live experiment."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'reports/code-initialized-model-and-shared-word-admission-v1-review.json'


def read(name): return json.loads((ROOT/name).read_text())


def actor(pid, expected_ticks=None):
    try:
        values = Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        cmd = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        matches = expected_ticks is None or str(expected_ticks)==values[19]
        return dict(pid=pid,start_ticks=values[19],state=values[0],parent=int(values[1]),
                    birth_matches=matches,live=matches and values[0] not in ('Z','X'),command=cmd)
    except FileNotFoundError: return dict(pid=pid,live=False,handle_missing=True)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('Read-only review must explicitly hide GPUs from model code')
    if OUTPUT.exists(): raise ValueError('Preserve prior authoritative snapshots')
    code = read('reports/own-aligned-assistant-v3-full-CPU-v1/complete.json')
    word = read('reports/shared-words-full-CPU-v1/complete.json')
    word_reader = read('reports/shared-words-full-CPU-v1/independent-complete-readback.json')
    admission = read('data/shared-byte-words-v1/admission.json')
    word_manifest = read('data/shared-byte-words-v1/manifest.json')
    queue_names = ['reports/whole-human-reward-v1-queue.json','reports/code-initialized-alignment-v1-queue.json']
    queues = {n:read(n) for n in queue_names}
    docs = {str(ROOT/n) for n in ['README.md','RESEARCH.md','reports/completion-audit.json']}
    plan_names = ['reports/whole-human-SFT-v1/plan.json','reports/knowledge-chat-diagnostic-v1/plan.json',
                  'reports/shared-words-full-CPU-v1/plan.json']
    for name in plan_names+queue_names:
        d = read(name)
        if docs & set(d['source_files_sha256']): raise ValueError('Documentation must not mutate any bound running source')
        FileBindings(d['source_files_sha256']).check(full=True)
    if (code['actual_counts']['total_unique']!=999860418 or word['actual_counts']!=code['actual_counts']
            or word['actual_vision_word_Adam_states']!=211 or word['actual_optimized_values']!=2714178
            or word_reader['original_complete_sha256']!=file_hash(ROOT/'reports/shared-words-full-CPU-v1/complete.json')
            or admission['training_admitted'] is not True or admission['manifest_sha256']!=file_hash(ROOT/'data/shared-byte-words-v1/manifest.json')
            or admission['independent_readback_sha256']!=file_hash(ROOT/'data/shared-byte-words-v1/independent-source-readback.json')):
        raise ValueError('Completed full-size/source-bound code, word and admission evidence required')
    files = [
        'reports/own-aligned-assistant-v3-full-CPU-v1/complete.json',
        'reports/own-aligned-assistant-v3-full-CPU-v1/independent-complete-readback.json',
        'reports/shared-words-full-CPU-v1/plan.json','reports/shared-words-full-CPU-v1/complete.json',
        'reports/shared-words-full-CPU-v1/independent-complete-readback.json',
        'reports/shared-word-stage-v1-tests.log','reports/shared-word-stage-v1-tests-initial-roundoff-failure.log',
        'tests/test_shared_word_stage_v1.py','src/slm_lab/shared_word_stage_v1.py',
        'scripts/probe_shared_words_full_CPU_v1.py','scripts/review_shared_words_full_CPU_v1.py',
        'data/shared-byte-words-v1/manifest.json','data/shared-byte-words-v1/admission.json',
        'data/shared-byte-words-v1/independent-source-readback.json',
        'reports/code-initialized-alignment-v1-proposed-exposure.json',
        'reports/whole-human-SFT-v1/independent-complete-training-readback-v1.json',
        'reports/whole-human-SFT-v1/independent-complete-held-targets-v1.json',
        'reports/whole-human-SFT-v1/independent-native-knowledge-readback-v1.json',
        'reports/whole-human-SFT-v1/initial/math-complete.json',
        'reports/dense-code-fold-and-depth-growth-parameter-budget-v1.json',
        'reports/dense-code-fold-and-depth-growth-parameter-budget-v1-readback.log']
    for arm in ('initial','rate_5e5'):
        files += [f'reports/whole-human-SFT-v1/{arm}/{name}' for name in
                  ['knowledge.json','code-completion.json','code-instruction.json','C++.json','ocr.json','objects.json','vision.json']]
    native = {}
    for arm in ('initial','rate_5e5'):
        base = f'reports/whole-human-SFT-v1/{arm}/'
        knowledge = read(base+'knowledge.json'); ocr = read(base+'ocr.json'); objects = read(base+'objects.json'); vision = read(base+'vision.json')
        native[arm] = dict(knowledge=knowledge['overall'],OCR=ocr['overall'],objects=objects['overall'],
            caption_retrieval=vision['caption_retrieval'],
            code={n:{k:v for k,v in read(base+n).items() if k in ('task_count','syntax_pass_rate','functional_pass_at_1','efficiency_stress_pass_rate')}
                  for n in ['code-completion.json','code-instruction.json','C++.json']})
        math = ROOT/base/'math-complete.json'
        native[arm]['full_native_math'] = ({k:v for k,v in json.loads(math.read_text()).items() if k in
            ('official_test_examples','evaluated_examples','correct','numeric_exact_accuracy','generation_budget','inference_precision',
             'all_original_IDs_prompts_generated_tokens_numeric_answers_and_scores_reconciled')} if math.exists() else dict(status='pending'))
        public = sorted((ROOT/base).glob('public-*.json'))
        native[arm]['public'] = dict(status='pending',completed_shards=len(public))
        if len(public)==4:
            data = [json.loads(p.read_text()) for p in public]
            native[arm]['public'] = dict(supported_problems=sum(d['supported_problems'] for d in data),
                syntax=sum(round(d['syntax_rate_on_supported_subset']*d['supported_problems']) for d in data),
                finite_case_passes=sum(round(d['pass_at_1_on_supported_subset']*d['supported_problems']) for d in data),
                final_independent_pass_reexecution_pending=not (ROOT/'reports/whole-human-SFT-v1/independent-comparison.json').exists())
            files += [str(p.relative_to(ROOT)) for p in public]
    diagnostic = {}
    for frame in ('plain','chat'):
        for arm in ('initial','rate_5e5'):
            name = f'reports/knowledge-chat-diagnostic-v1/{arm}-{frame}.json'
            if (ROOT/name).exists():
                diagnostic[arm+'-'+frame] = read(name)['overall']; files.append(name)
            else: diagnostic[arm+'-'+frame] = dict(status='pending')
    queue_states = {}
    for name,q in queues.items():
        who = q['owner']; previous = q['predecessor']
        queue_states[name] = dict(status=q['status'],owner=actor(who['pid'],who['ticks']),
            predecessor=actor(previous['pid'],previous['ticks']),production_model_updates_at_queue_snapshot=q['model_updates'],
            all_bound_source_files_rehashed_unchanged=True)
    sft = actor(2989330,'34058845'); children = read('reports/whole-human-SFT-v1/children.json')
    live_workers = []
    for child in children:
        observed = actor(child['pid'],child['ticks'])
        if observed['live']: live_workers.append(dict(observed,subject=child['subject']))
    atomic_json(OUTPUT,dict(observed_at=datetime.now(timezone.utc).isoformat(),
        antecedent_full_scope_review='reports/native-knowledge-and-format-diagnostic-v1-review.json',
        previous_goal_turn_classification=dict(classification='progress',reason='Completed actual code-initialized full-size mixed-update proof, source-bound full-training queue and full word-source admission.'),
        current_goal_turn_classification=dict(classification='progress',reason='Implemented complete-source shared word learner, tested an independent joint gradient oracle and actual restart, and executed/reviewed full-size CPU word learning with both optimizer states retained.'),
        actual_parameters=999860418,external_pretrained_or_teacher_weights_loaded=False,
        code_initialized_model=dict(full_CPU_proof_complete=True,disposable_logical_updates=2,production_updates=0,
            same_unique_parameter_count=True,new_held_coding_gain_established=False,
            full_source_queue='reports/code-initialized-alignment-v1-queue.json',
            source_exposure=dict(human_unique=12287,human_exposures=14192,human_repeated_exposures=1905,
                                 verified_program_unique=3548,verified_program_exposures=3548,program_repeats=0),
            arms=dict(mixed_CE=0.,mixed_reference_KL=.1),updates_per_arm=3548),
        shared_words=dict(training_admitted=True,counts=word_manifest['splits'],real_training_crops_not_photos=41585,
            all_complete_original_pixel_transcript_byte_and_BPE_target_readback=True,
            actual_vision_word_optimized_values=2714178,actual_vision_word_Adam_states=211,
            full_CPU_proof_seconds=word['seconds'],disposable_logical_updates=2,physical_optimizer_calls=3,
            actual_next_update_all_weights_Adam_RNG_metrics_exact=True,
            retained_both_step1_and_step2_Adam_and_export=True,
            all_original_and389unused_new_tensors_and_metadata_fixed=True,
            aligned_text_only_and_original_routes_codecache_exact=True,actual_trained_cold_reload_exact=True,
            independent_reader_reconstructed_used_full_pixels_and_complete_sources=True,
            independent_reader_did_not_replay_neural_updates=True,production_full_word_epochs=0,
            held_reading_gain_measured=False,full_photo_detection_and_reading_not_proven=True,
            publisher_annotation_mirror_equivalence_not_independently_certified=True),
        previous_human_SFT=dict(complete_training=True,updates_per_arm=3072,held_human_examples=677,
            held_human_response_EOS_targets=292093,initial_token_NLL=2.4938758787850976,
            selected_token_NLL=2.0493155266839134,teacher_gain_not_generated_capability=True,
            native_results_so_far=native,full_comparison_complete=(ROOT/'reports/whole-human-SFT-v1/complete.json').exists()),
        post_hoc_CPU_FP32_prompt_diagnostic=dict(complete=(ROOT/'reports/knowledge-chat-diagnostic-v1/complete.json').exists(),
            complete_cells=diagnostic,original_CUDA_BF16_native_results_not_replaced=True,no_held_training_or_selection=True),
        live_GPU_owner=sft,live_registered_GPU_workers=live_workers,live_source_bound_queues=queue_states,
        original_work=dict(photo_collector=actor(130149,'20300752'),photo_progress=read('data/vqa-real-pairs-v1/progress.json'),
            human_feedback_waiter=actor(130151,'20300753'),data_waiter=actor(2754743,'27602681'),
            separate_original_CPU_math=[actor(2852035),actor(2904126)],none_restarted_or_signalled_by_this_review=True),
        current_GPU_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip().splitlines(),
        dense_fold_and_depth_growth_research=dict(meta_shapes_only=True,actual_folded_or_grown_weights=False,
            function_preservation_GPU_capacity_training_and_quality_unproven=True),
        requirements_remaining=dict(high_end_GPT6_1_level_all_round_and_coding='not achieved or verified',
            syntactically_algorithmically_correct_and_efficient_broad_code='local functional/efficiency remain zero; broad variety prepared, exhaustive all-code coverage not established',
            complete_public_broad_domain_training='education pack not admitted/trained; complete original paired-photo finalization/continuation unfinished',
            shared_generic_images_text_in_images_and_visual_reasoning='native image results weak; new word production training and full-photo integration unfinished',
            genuine_human_RLHF='full ranking head still queued, fresh own-policy PPO and useful aligned behavior unverified',
            independent_broad_final_evaluations='existing development/source-exposure qualifications remain; current full comparison incomplete'),
        evidence_files_sha256={n:file_hash(ROOT/n) for n in files},
        live_plan_and_queue_source_bindings_verified_unchanged=True,documentation_not_bound_by_checked_live_plans=True,
        no_candidate_promoted=True,goal_achieved=False))
    print(OUTPUT)


if __name__=='__main__': main()
