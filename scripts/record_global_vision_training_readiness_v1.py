"""Record actual four-process native mechanics and the public PDF label defect."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from record_public_code_instruction_screen_v1 import read,sha,atomic,live
from slm_lab.function_repair_GPU_contract_v1 import ADMISSION,QUEUE,handoff_admitted
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/global-vision-training-readiness-v1-review'
MATCHING = ROOT/'reports/global-visual-matching-v2-pytest/test_actual_four_process_globa0'
NATIVE = ROOT/'reports/global-photo-prefix-stage-v1-pytest/test_actual_four_own_replicas_0'
DOC = ROOT/'reports/public-document-blank-supervision-CPU-v1/complete.json'
REGION = ROOT/'reports/public-document-blank-region-CPU-v1/complete.json'


def main():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19:
        raise ValueError('Fresh CUDA-hidden low-priority readiness recorder required')
    matching_log = ROOT/'reports/global-visual-matching-gradients-v2-tests.log'
    native_log = ROOT/'reports/global-photo-prefix-stage-kernel-v1-tests.log'
    if '11 passed' not in matching_log.read_text() or '1 passed' not in native_log.read_text():
        raise ValueError('Actual full selected global-feature and integrated native test runs required')
    matching = [read(MATCHING/f'{rank}.json') for rank in range(4)]
    native = [read(NATIVE/f'{rank}.json') for rank in range(4)]
    gradient_errors = [];weight_errors = []
    for rank,rows in enumerate(matching):
        if len(rows) != 15:raise ValueError('All six exact-feature, three learned-projection and six coordinated-rejection observations required per rank')
        features = [r for r in rows if r['kind'] == 'exact_complete_feature_derivatives']
        projections = [r for r in rows if r['kind'] == 'actual_projection_gradient_SUM_and_Adam']
        rejected = [r for r in rows if r['kind'] == 'all_ranks_rejected']
        if (len(features) != 6 or len(projections) != 3 or len(rejected) != 6
                or {r['source_count'] for r in features} != {2,7,64}
                or {r['dtype'] for r in features} != {'torch.float32','torch.bfloat16'}
                or any(r['rank'] != rank for r in rows)):
            raise ValueError('Both feature precisions, every complete global batch and exact ranks must remain represented')
        for row in projections:
            if (row['actual_total_parameters'] != 78 or row['selected_parameter_values'] != 60
                    or not row['replicas_bitexact'] or not row['fixed_head_unchanged']
                    or row['source_count'] == 2 and rank >= 2 and row['actual_local_neural_projection_calls'] != 0
                    or row['serial_max_gradient_absolute_difference'] >= 1e-6
                    or row['serial_max_Adam_parameter_absolute_difference'] >= 1e-6):
                raise ValueError('Actual tiny global-normalized gradient SUM/Adam comparisons and empty-rank behavior required')
        current = native[rank]
        if not current['complete'] or not current['missing_EOS_on_one_rank_rejected_everywhere_before_forward_and_Adam']:
            raise ValueError('All four actual own-native workers must finish and coordinate invalid target rejection')
        if [r['global_sources'] for r in current['proofs']] != [3,2]:raise ValueError('Both actual integrated whole-caption updates required')
        for row in current['proofs']:
            if (row['actual_own_unique_parameters'] != 64194 or row['actual_selected_values'] != 36576
                    or not row['replicas_bitexact'] or not row['text_logits_and_fixed_weights_bitexact']
                    or row['actual_image_encoder_calls'] != bool(row['local_sources'])
                    or row['serial_max_gradient_absolute_difference'] >= 5e-6
                    or row['serial_max_Adam_weight_absolute_difference'] >= 5e-6):
                raise ValueError('Actual own-native caption/matching/one-prefix/fixed-text update observations required')
            gradient_errors.append(row['serial_max_gradient_absolute_difference']);weight_errors.append(row['serial_max_Adam_weight_absolute_difference'])
    document,region = read(DOC),read(REGION)
    if (not document.get('complete') or not region.get('complete')
            or document['actual_source_PDF']['actual_target_characters'] != 39
            or not region['both_requested_visible_regions_have_no_nonwhite_pixels']
            or len(region['actual_bad_original_positive_views']) != 2 or region['actual_bad_native_response_EOS_targets'] != 40
            or not region['full_source_4733_update_plan_requires_explicit_requalification_without_silent_source_removal']):
        raise ValueError('Original PDF rendering, white glyphs and both complete defective native targets must be reproduced')
    existing = read(ROOT/'reports/completion-audit.json');previous = Path(existing['latest_full_scope_review'])
    requirements = read(previous)['requirements']
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('Distributed mechanics and data diagnostics establish none of the final high-end capabilities')
    if not handoff_admitted(ROOT,read(ROOT/QUEUE)['owner']):raise ValueError('The full original ten-controller handoff must remain admitted')
    handoff = read(ROOT/ADMISSION)
    paths = [ROOT/'reports/public-pretraining-recovery-v1/plan.json',
        ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json',
        ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json',
        ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json',
        ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json',ROOT/QUEUE,
        ROOT/'reports/full-public-code-shard-qualification-CPU-v1/plan.json',
        ROOT/'reports/full-multimodal-pixel-validation-CPU-v1/plan.json']
    plans = [read(p) for p in paths];maps = [p['source_files_sha256'] for p in plans]
    maps += [read(ROOT/item['queue'])['source_files_sha256'] for item in handoff['new_controllers']]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in m for m in maps):raise ValueError('Never edit a source-bound status index or document')
    jobs = [live(p['owner']) for p in plans]
    if any(p['state'] in {'T','t'} for p in jobs[:3]):raise ValueError('No existing neural worker may be stopped')
    if os.sched_getscheduler(plans[-1]['owner']['pid']) != os.SCHED_IDLE:raise ValueError('Original pixel auditor must remain at its declared idle priority')
    files = {**document['source_files_sha256'],**region['source_files_sha256']}
    artifact_paths = [DOC,REGION,previous,matching_log,native_log,ROOT/ADMISSION,
        ROOT/'reports/public-document-blank-supervision-CPU-v1/semantic-veto.json',
        ROOT/'reports/public-document-blank-region-CPU-v1/semantic-vetoes.json',
        ROOT/'reports/global-visual-matching-first-fixture-failure/receipt.json',
        ROOT/'reports/global-visual-matching-first-fixture-failure/original-test.py',
        ROOT/'reports/global-visual-matching-first-fixture-failure/original-implementation.py',
        ROOT/'reports/global-visual-matching-gradients-v1-tests.log',
        *[folder/f'{rank}.json' for folder in [MATCHING,NATIVE] for rank in range(4)],
        *python_sources(ROOT,[Path(__file__),ROOT/'tests/test_global_visual_matching_gradients_v1.py',
            ROOT/'tests/test_global_photo_prefix_stage_kernel_v1.py'])]
    files.update({str(p):sha(p) for p in artifact_paths})
    if any(sha(p) != h for p,h in files.items()):raise ValueError('Every original data byte, observed source defect, actual test and kernel must remain exact')
    raw = (ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes();prefix = raw[:raw.rfind(b'\n')+1]
    rows = [json.loads(line) for line in prefix.split(b'\n') if line]
    if [r['step'] for r in rows] != list(range(1,len(rows)+1)):raise ValueError('Actual complete contiguous main update prefix required')
    current = dict(owner=plans[0]['owner'],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(r['response_EOS_targets']) for r in rows),complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    at = datetime.now(timezone.utc).isoformat();OUT.mkdir()
    proof = dict(complete=True,at=at,source_files_sha256=files,
        previous_goal_turn_classification='progress: naive prefix reuse rejected; seven small separate-backward numerical/recovery controls pass',
        current_goal_turn_classification='progress: actual four-process global and own-native matching/caption updates validated; two invisible public document targets independently reproduced and vetoed for future positive admission',
        actual_global_feature_controls_passed=11,actual_integrated_own_native_four_process_control_passed=1,
        first_fixture_failure_preserved_and_corrected_without_kernel_or_tolerance_changes=True,
        actual_global_feature_batches=[7,2,64],actual_feature_precisions=['FP32','BF16'],
        global_pairs_and_normalization_and_one_parameter_gradient_SUM_checked=True,
        actual_integrated_own_fixture_parameters=64194,actual_selected_fixture_values=36576,
        actual_integrated_global_updates=2,actual_integrated_global_batch_sizes=[3,2],
        actual_integrated_max_gradient_absolute_difference=max(gradient_errors),
        actual_integrated_max_Adam_weight_absolute_difference=max(weight_errors),
        actual_all_four_replicas_weights_bitexact=True,actual_fixed_text_and_other_heads_unchanged=True,
        actual_empty_ranks_no_fabricated_forward=True,
        actual_defective_public_document_positive_views=2,actual_defective_native_response_EOS_targets=40,
        actual_hidden_target_PDF_white_filled_characters=39,
        proposed_4733_update_whole_source_epoch_requires_explicit_source_requalification=True,
        no_automatic_source_removal_or_relabeling_or_new_training_admission=True,
        actual_original_main=current,actual_original_live_jobs=jobs,
        actual_pixel_audit_progress=read(ROOT/'reports/full-multimodal-pixel-validation-CPU-v1/progress.json'),
        actual_code_data_progress=read(ROOT/'reports/full-public-code-shard-qualification-CPU-v1/progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        full_size_CUDA_capacity_recovery_and_whole_source_training_and_matched_quality_pending=True,
        original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        latest_completed_local_functional_correctness='0/48 on older checkpoint',newer_matched_quality_pending=True,
        training_admitted=False,candidate_promoted=False,new_production_optimizer_updates=0,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    (OUT/'metrics.md').write_text(
        f'Global vision training mechanics and public-source defect, {at}. **11 global-feature controls and one '
        'actual four-process own-native integration control pass.** These are CPU mechanics tests, not production '
        'GPU training or evidence of better visual/coding capability.\n\n'
        'The new primitive gathers small detached FP32 image/text feature tables through Gloo, reconstructs the '
        'complete batch and both directions of the canonical multi-positive matching loss, and sends each rank '
        'its actual feature derivatives. Parameter gradients need one SUM, with no world average. Exact feature '
        'derivatives match the serial reference in FP32/BF16 for 7, 2 and 64 global sources. Four actual 78-parameter '
        'projection replicas also match serial gradients/Adam updates, including two empty ranks. Six adversarial '
        'source/order/precision/value cases reject on every rank. The first run failed because its BF16 reference '
        'was not a leaf tensor; the exact failed fixture, implementation and log are archived. The fixture alone '
        'was corrected; kernel and tolerances were unchanged.\n\n'
        'The integrated kernel runs four actual 64,194-parameter own text/image replicas, selecting 36,576 existing '
        'vision/pointing values. It combines globally normalized complete caption CE with full-batch matching '
        'through one visual prefix, with separate backwards. Two real Adam updates use global batches of 3 and 2 '
        'without manufacturing examples on empty ranks. All replicas agree bit exactly; fixed text/actor/critic/other '
        'heads and text-only logits stay exact. The largest serial gradient difference is '
        f'{max(gradient_errors):.9g}; the largest serial Adam weight difference is {max(weight_errors):.9g}. '
        'A missing EOS on one rank rejects everywhere before any encoder forward or Adam mutation. Integrated '
        'BF16/CUDA, full-size capacity/recovery, the singleton word/CTC stage and full visual quality remain pending.\n\n'
        '**A separate actual data defect is established.** The existing crop PLAW-117publ169-0-2 is entirely white, '
        'yet its OCR target is nonempty. The original cached public PDF reproduces every crop pixel and shows '
        '39 opaque white text-layer characters on that white region. Its paired full-page region-reading view '
        'requests the same invisible text; the actual requested 6x103-pixel region is also entirely white. Both '
        'original native views retain 40 complete response/EOS targets. The source-derived labels therefore '
        'cannot be certified merely by matching file or input-pixel hashes.\n\n'
        'Exact evidence and semantic vetoes are recorded for future positive admission. The existing source files, '
        'live main and queued recipes remain unchanged. The proposed 302,627-view/4,733-update visual epoch now '
        'explicitly requires requalification of both defective pairs before admission; no source is silently '
        'removed or relabeled. The source-level diagnosis does not measure historical model impact or explain '
        'the coding score. The full 195,121-path pixel audit continues.\n\n'
        f'The original main continues at {len(rows):,}/4,447 committed updates and '
        f'{current["response_EOS_targets"]:,} response/EOS targets. All canonical source/human/RL/nine-endpoint '
        'scopes remain. Latest completed local functional correctness remains 0/48 on an older checkpoint; '
        'newer matched quality is pending. All ten final requirements remain incomplete.\n')
    review = dict(at=at,requirements=requirements,previous_full_scope_review=str(previous),
        current_goal_turn_classification=proof['current_goal_turn_classification'],
        current_actual_runtime=dict(main=current,original_live_jobs=jobs),
        latest_actual_global_vision_training_readiness=str(OUT/'complete.json'),
        original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(p) != h for p,h in files.items()):raise ValueError('Every original source and actual test observation must stay exact before updating indices')
    existing.update(at=at,requirements=requirements,latest_full_scope_review=str(OUT/'review.json'),
        latest_actual_main_runtime=current,latest_actual_global_vision_training_readiness=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',existing)
    paragraph = (f'Global native vision mechanics verified, **{at}**: **11 feature/gradient controls plus one actual four-process own-native control pass**. '
        'Global matching/caption gradients use one visual prefix, complete target normalization and no fabricated empty-rank examples. '
        '**These are small CPU controls, not full-size/GPU quality proof.** A fresh original-PDF readback identifies **two invisible OCR positive targets** '
        '(white text on white pixels); the full visual source epoch requires explicit requalification before admission. '
        f'The main continues at **{len(rows)}/4,447**. **Latest completed local correctness remains0/48 on an older checkpoint; all ten final requirements remain incomplete.** '
        '[Actual controls, source defect and limits](reports/global-vision-training-readiness-v1-review/metrics.md), '
        '[full review](reports/global-vision-training-readiness-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path = ROOT/name;first,rest = path.read_text().split('\n',1);path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=True,actual_controls_passed=12,defective_public_document_views=2,
        actual_main_updates=len(rows),training_admitted=False,goal_achieved=False)))


if __name__ == '__main__':main()
