"""Record full native-source coverage; GPU training remains explicitly pending."""
import copy
from datetime import datetime,timezone
import os
from pathlib import Path
import json
import subprocess

from record_public_code_instruction_screen_v1 import read,sha,atomic,live
from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/full-multimodal-source-epoch-CPU-v1'
READER=ROOT/'reports/full-multimodal-source-epoch-v1-readback'
OUT=ROOT/'reports/full-multimodal-source-epoch-v1-review'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.nice(0)<19 or OUT.exists():raise ValueError('Fresh low-priority CPU review required')
    source=read(SOURCE/'complete.json');reader=read(READER/'complete.json')
    old=read(ROOT/'reports/own-actor-vision-replay-full-CPU-v1/complete.json')
    old_reader=read(ROOT/'reports/own-actor-vision-replay-full-CPU-v1-readback/complete.json')
    if (not source.get('complete') or not reader.get('complete') or source['training_admitted'] or reader['training_admitted']
            or source['all_original_train_views']!=302627 or reader['actual_original_views_rechecked']!=302627
            or source['exact_single_epoch_updates']!=4733 or reader['actual_epoch_updates']!=4733
            or not reader['all91585word_native_sequences_inverse_decoded_without_clipping']
            or reader['actual_complete_CTC_UTF8_bytes']!=644063 or reader['canonical_exact_text_photo_positive_masks_rechecked']!=1223
            or reader['actual_empty_rank_tail_steps']!=[3205] or reader['photo_batches_without_observed_negatives']
            or old['logical_updates']!=2 or old['actual_counts']['actual_unique']!=999706442
            or not old_reader['all_development_OCR_exact_scores_still_zero']):
        raise ValueError('Complete full native source/readback and correctly scoped prior mechanics required')
    audit=read(ROOT/'reports/completion-audit.json');previous=Path(audit['latest_full_scope_review']);review=copy.deepcopy(read(previous))
    gpu=read(ROOT/QUEUE);handoff=read(ROOT/ADMISSION)
    if not handoff_admitted(ROOT,gpu['owner']):raise ValueError('All ten exact original successors must remain admitted')
    paths=[ROOT/'reports/public-pretraining-recovery-v1/plan.json',ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json',
        ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json',ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json',
        ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json',ROOT/QUEUE,ROOT/'reports/full-public-code-shard-qualification-CPU-v1/plan.json',
        *[ROOT/item['queue'] for item in handoff['new_controllers']]]
    plans=[read(p) for p in paths];maps=[p['source_files_sha256'] for p in plans]+[source['source_files_sha256'],reader['source_files_sha256']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in m for m in maps):raise ValueError('Never edit an active source-bound index or document')
    files={**source['source_files_sha256'],**reader['source_files_sha256']}
    for p in [Path(__file__),previous,SOURCE/'complete.json',READER/'complete.json',
            ROOT/'reports/own-actor-vision-replay-full-CPU-v1/complete.json',ROOT/'reports/own-actor-vision-replay-full-CPU-v1-readback/complete.json',*paths]:
        files[str(p)]=sha(p)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every source/native/schedule/reader byte must remain exact')
    owners=[live(p['owner']) for p in plans[:7]]
    if any(item['state'] in {'T','t'} for item in owners[:3]):raise ValueError('No original neural worker may be stopped')
    raw=(ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix=raw[:raw.rfind(b'\n')+1]
    rows=[json.loads(line) for line in prefix.split(b'\n') if line]
    if [row['step'] for row in rows]!=list(range(1,len(rows)+1)):raise ValueError('Actual contiguous main update prefix required')
    current=dict(owner=plans[0]['owner'],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(row['response_EOS_targets']) for row in rows))
    image_names=set(source['actual_family_budgets']);actual={}
    for row in rows:
        if row['family'] not in image_names:continue
        item=actual.setdefault(row['family'],dict(committed_updates=0,source_presentations=0,source_response_EOS_targets=0))
        item['committed_updates']+=1;item['source_presentations']+=len(row['global_original_positions'])
        item['source_response_EOS_targets']+=int(row['response_EOS_targets'])
    at=datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof=dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: all24older initial code programs diagnosed; cropping improves syntax but not functional correctness; actual calculation errors observed',
        current_goal_turn_classification='progress: all302627admitted visual/word source views and4733step whole-source epoch independently reconstructed; full pixel/GPU/training admission pending',
        actual_all_original_train_views=302627,actual_distinct_stored_image_and_crop_paths=195121,
        actual_duplicate_native_views_across_source_and_loss_recipes=51595,
        actual_family_budgets=source['actual_family_budgets'],actual_native_response_EOS_targets=sum(b['response_EOS_targets'] for b in source['actual_family_budgets'].values()),
        actual_complete_CTC_UTF8_bytes=644063,all91585word_sequences_inverse_decoded=True,
        actual_complete_source_epoch_updates=4733,actual_family_updates=reader['actual_family_updates'],
        all_original_positions_once=True,partial_batches_kept=True,global_width=64,world_size=4,
        all1223photo_positive_masks_rechecked_with_negatives=True,
        actual_empty_rank_tail=dict(step=3205,family='words',rank_counts=[1,0,0,0],must_be_supported_before_distributed_admission=True),
        main_current_pass_planned_image_presentations=19488,main_current_pass_planned_image_optimizer_updates=21,
        counts_are_current_pass_not_total_historical_image_training=True,actual_current_committed_image_exposures=actual,
        source_word_crops_are_not_full_photo_text_detection=True,
        original_RGB448_and_float_word_pixel_hash_protocols_kept_separate=True,
        source_metadata_and_native_masks_verified_but_full_pixel_bytes_and_geometry_not_freshly_certified=True,
        prior_full_size_vision_replay_mechanics=dict(own_unique_parameters=999706442,logical_updates=2,
            selected_neural_values=52818498,selected_tensors=130,added_parameters=0,full_GPU_admission=False,
            prior_small_development_OCR_exact_scores_still_zero=True),
        new_GPU_stage_distributed_gradient_recovery_capacity_and_full_visual_quality_pending=True,
        new_training_or_GPU_job_not_started=True,neural_forwards_from_this_work=0,new_production_optimizer_updates_from_this_work=0,
        actual_original_main=current,actual_original_live_owners=owners,
        actual_new_code_data_progress=read(ROOT/'reports/full-public-code-shard-qualification-CPU-v1/progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        latest_complete_local_functional_correctness='0/48_on_older_checkpoint',newer_code_and_visual_quality_pending=True,
        training_admitted=False,candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    lines=[f'Full multimodal source epoch prepared and independently read back, {at}. '
        'Every302,627original training view from nine already admitted image/word families retains its original source role, native prompt, complete response/EOS mask and original source reference. '
        'The fresh reader checks committed raw row/index hashes and inverse-decodes all91,585packed word prompts/transcripts; all644,063UTF8byte targets remain complete.\n',
        '| Source family | Original views | Distinct stored image paths | Complete response/EOS targets | Proposed updates |',
        '|---|---:|---:|---:|---:|']
    for name,budget in source['actual_family_budgets'].items():
        lines.append(f'| {name} | {budget["original_views"]:,} | {budget["unique_stored_image_paths"]:,} | {budget["response_EOS_targets"]:,} | {reader["actual_family_updates"][name]:,} |')
    lines += ['', 'There are195,121distinct stored image/crop paths across the pools, not302,627unique photographs. '
        '51,595native views share physical image/text inputs across source/loss recipes; those occurrences are retained, including the extra CTC objective. '
        'The78,255additional real photographs carry original human Localized Narratives descriptions. The pool also includes real word crops, public document text, rendered public text, CLEVR and a small legacy Fashion-MNIST component.\n',
        'The exact proposed one-pass schedule has4,733updates at global width64/world4. Every original position occurs once, including the final partial batches. '
        'All1,223photo batches have actual observed contrastive negatives under the canonical exact-caption positive rule. '
        'Step3,205contains one original word example, with rank counts[1,0,0,0]; a distributed trainer must implement and validate this case instead of dropping it.\n',
        'The current public-pretraining pass plans19,488image/word presentations across21updates; these counts do not describe the complete historical training. '
        'Preserving those replay exposures does not amount to a full visual source epoch. The new source inventory makes the additional coverage and update schedule concrete; it does not prove that a full pass will improve vision.\n',
        'An earlier complete full-size CPU replay proof trains52,818,498existing vision/pointing/word values in130tensors within the999,706,442parameter own model. '
        'It verifies two logical updates and recovery/cold/text preservation on tiny development cohorts, whose exact OCR scores remain zero. '
        'It is a useful existing mechanism, not broad quality or four-GPU admission.\n',
        '**No new vision training or GPU job starts here.** Full physical pixel/geometry revalidation, deterministic CUDA/CTC/distributed gradients and recovery, '
        'largest-source capacity, safe runtime handoff and complete matched visual/word/text/code assessments remain required. '
        'Metadata RGB448 and normalized float word/native pixel hash protocols stay separate; this readback does not silently certify pixel equivalence across protocols. '
        'Word crops do not establish full-photo text detection.\n',
        f'The original main continues at{len(rows):,}/4,447updates and{current["response_EOS_targets"]:,}targets. '
        'All original source/human/RL/nine-endpoint scopes remain. Latest completed local code correctness remains0/48 on an older checkpoint; '
        'newer matched code and image results remain incomplete. No model-quality gain or frontier parity is established. All ten final requirements remain incomplete.','']
    (OUT/'metrics.md').write_text('\n'.join(lines))
    if len(review['requirements'])!=10 or any(v['status']!='incomplete_for_requested_final_model' for v in review['requirements'].values()):
        raise ValueError('Native source preparation establishes none of the requested final capabilities')
    review['current_actual_runtime']['main']=current
    review.update(at=at,previous_full_scope_review=str(previous),previous_goal_turn_classification=proof['previous_goal_turn_classification'],
        current_goal_turn_classification=proof['current_goal_turn_classification'],latest_actual_full_multimodal_source_epoch=proof,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(p)!=digest for p,digest in files.items()):raise ValueError('Every fixed source/native/readback byte must stay exact before index update')
    audit.update(at=at,requirements=review['requirements'],latest_full_scope_review=str(OUT/'review.json'),latest_actual_main_runtime=current,
        latest_actual_full_multimodal_source_epoch=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',audit)
    paragraph=(f'Full multimodal source epoch prepared, **{at}**: all**302,627training views** across nine existing families independently reconstruct their original source/native masks; '
        'all91,585word sequences inverse-decode and all644,063byte targets remain. An exact**4,733update**schedule retains every position and partial batch. '
        'All1,223photo positive masks have negatives; the one-example word tail requires empty-rank support. '
        '**This is source preparation, not new image training or quality evidence**; full pixel/GPU/state/capacity and matched evaluation remain pending. '
        f'The original main continues at**{len(rows)}/4,447**. **Latest complete code correctness remains0/48 on an older checkpoint; all ten final requirements remain incomplete.** '
        '[Full coverage and remaining gates](reports/full-multimodal-source-epoch-v1-review/metrics.md), '
        '[full review](reports/full-multimodal-source-epoch-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path=ROOT/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=True,all_original_views=302627,source_epoch_updates=4733,actual_main_updates=len(rows),training_admitted=False,goal_achieved=False)))


if __name__=='__main__':main()
