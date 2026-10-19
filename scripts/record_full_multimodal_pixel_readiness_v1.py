"""Record an actual source-only pixel audit launch without claiming model gains."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

from record_public_code_instruction_screen_v1 import read, sha, atomic, live
from slm_lab.function_repair_GPU_contract_v1 import ADMISSION, QUEUE, handoff_admitted

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/full-multimodal-pixel-readiness-v1-review'
AUDIT = ROOT/'reports/full-multimodal-pixel-validation-CPU-v1'
RECEIPT = ROOT/'reports/full-multimodal-pixel-validation-CPU-v1-launch.json'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19 or OUT.exists():
        raise ValueError('A fresh low-priority CPU status recorder is required')
    plan, receipt = read(AUDIT/'plan.json'), read(RECEIPT)
    source = read(ROOT/'reports/full-multimodal-source-epoch-CPU-v1/complete.json')
    reader = read(ROOT/'reports/full-multimodal-source-epoch-v1-readback/complete.json')
    owner = {k:receipt[k] for k in ['pid','ticks']}
    if (owner != plan['owner'] or sha(receipt['source']) != receipt['source_sha256']
            or os.sched_getscheduler(owner['pid']) != os.SCHED_IDLE or plan['training_admitted']
            or plan['expected_distinct_physical_paths'] != 195121
            or not source.get('complete') or not reader.get('complete') or source['training_admitted'] or reader['training_admitted']):
        raise ValueError('Exact live idle-scheduled physical source auditor and complete source/readback required')
    pixel_owner = live(owner)
    if (AUDIT/'failure.json').exists() or (AUDIT/'complete.json').exists():
        raise ValueError('This launch review must not replace a terminal pixel audit finding')
    canonical = read(ROOT/QUEUE)
    if not handoff_admitted(ROOT, canonical['owner']):
        raise ValueError('The exact original full ten-controller handoff must remain admitted')
    handoff = read(ROOT/ADMISSION)
    plan_paths = [ROOT/'reports/public-pretraining-recovery-v1/plan.json',
        ROOT/'reports/full-function-repair-comparison-CPU-v1/plan.json',
        ROOT/'reports/public-pretraining-interim-CPU-v2/evaluation-plan.json',
        ROOT/'reports/public-edge-preference-comparison-CPU-v1-queue.json',
        ROOT/'reports/public-reasoning-warmup-CPU-v1-queue.json',ROOT/QUEUE,
        ROOT/'reports/full-public-code-shard-qualification-CPU-v1/plan.json',AUDIT/'plan.json']
    plans = [read(p) for p in plan_paths]
    # Queue metadata may be rewritten while preserving its bytes. Inspect its
    # source map here; do not bind the mutable human queue itself as evidence.
    downstream = [read(ROOT/item['queue']) for item in handoff['new_controllers']]
    maps = [p['source_files_sha256'] for p in plans+downstream]
    for name in ['reports/completion-audit.json','README.md','RESEARCH.md']:
        if any(str(ROOT/name) in m for m in maps):
            raise ValueError('Never edit a document or status index bound by an active source map')
    existing = read(ROOT/'reports/completion-audit.json')
    previous = Path(existing['latest_full_scope_review']); previous_review = read(previous)
    requirements = previous_review['requirements']
    if len(requirements) != 10 or any(v['status'] != 'incomplete_for_requested_final_model' for v in requirements.values()):
        raise ValueError('Source auditing establishes none of the requested final model capabilities')
    files = dict(plan['source_files_sha256'])
    for p in [Path(__file__),previous,AUDIT/'plan.json',RECEIPT,ROOT/ADMISSION,
            ROOT/'reports/full-multimodal-source-epoch-CPU-v1/complete.json',
            ROOT/'reports/full-multimodal-source-epoch-v1-readback/complete.json']:
        files[str(p)] = sha(p)
    if any(sha(p) != digest for p,digest in files.items()):
        raise ValueError('Every fixed original source, native readback and audit runtime must match')
    live_jobs = [live(p['owner']) for p in plans[:7]]
    if any(p['state'] in {'T','t'} for p in live_jobs[:3]):
        raise ValueError('No original neural worker may be stopped')
    raw = (ROOT/'reports/public-pretraining-recovery-v1/training-metrics.jsonl').read_bytes()
    prefix = raw[:raw.rfind(b'\n')+1]
    rows = [json.loads(line) for line in prefix.split(b'\n') if line]
    if [row['step'] for row in rows] != list(range(1,len(rows)+1)):
        raise ValueError('Actual complete contiguous main metrics prefix required')
    main_runtime = dict(owner=plans[0]['owner'],complete_actual_updates=len(rows),planned_updates=4447,
        response_EOS_targets=sum(int(row['response_EOS_targets']) for row in rows),
        complete_prefix_sha256=hashlib.sha256(prefix).hexdigest())
    at = datetime.now(timezone.utc).isoformat(); OUT.mkdir()
    proof = dict(complete=True,at=at,source_files_sha256=files,
        current_goal_turn_classification='progress: five pixel-protocol tests pass and an exact idle-scheduled full physical source auditor is live; pixel audit and new model quality remain pending',
        original_all_train_views=302627,original_physical_image_or_crop_paths=195121,original_word_views=91585,
        original_whole_epoch_updates=4733,actual_test_controls_passed=5,
        actual_new_pixel_audit_owner=pixel_owner,actual_new_pixel_audit_scheduler='SCHED_IDLE',
        actual_pixel_audit_progress=read(AUDIT/'progress.json'),
        pixel_audit_is_not_complete=True,pixel_integrity_not_yet_certified=True,
        actual_original_main=main_runtime,actual_original_live_jobs=live_jobs,
        actual_new_code_data_progress=read(ROOT/'reports/full-public-code-shard-qualification-CPU-v1/progress.json'),
        actual_GPU_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        every_existing_neural_worker_and_queued_neural_stage_unmodified=True,
        latest_completed_local_functional_correctness='0/48 on older checkpoint',newer_matched_endpoint_scores_pending=True,
        semantic_label_quality_not_certified=True,new_training_or_GPU_job_started=False,
        neural_forwards_from_this_work=0,new_production_optimizer_updates_from_this_work=0,
        training_admitted=False,candidate_promoted=False,goal_achieved=False)
    atomic(OUT/'complete.json',proof)
    metrics = (
        f'Physical multimodal source audit launched, {at}. The exact live worker {owner["pid"]}/{owner["ticks"]} '
        'uses CUDA-hidden one-thread data processing with SCHED_IDLE and nice19. Scheduling changes apply only '
        'to this newly launched source worker. All existing neural workers and canonical stages remain unchanged.\n\n'
        'Five meaningful controls pass: canonical native/word transforms and source preservation, cross-protocol '
        'hash substitution rejection, conflicting/invalid metadata rejection, raw constant-color detection despite '
        'padding contrast, and changed/corrupt physical source rejection. These are input-integrity tests, not model scores.\n\n'
        'The full auditor covers 195,121 distinct stored image/crop paths behind all 302,627 original train views, '
        'including all 91,585 whole word crops. It merges separate original expectations, streams the actual stored bytes, '
        'decodes RGB, reconstructs native RGB448 and normalized float CHW inputs, and checks word-reader geometry '
        'where required. Raw RGB and padded RGB and normalized float hashes never substitute for each other. '
        'Per-source states are checked during observations and again at the end; frozen metadata is rehashed.\n\n'
        '**The physical audit is still running.** Source constancy is a separate content anomaly; exact bytes and '
        'transforms do not establish correct captions/transcriptions. Fresh output readback, full-size distributed '
        'capacity/gradient/recovery checks, actual visual training and matched quality assessment remain pending. '
        'The prepared 4,733-update full-source epoch is not a new admitted vision training job.\n\n'
        f'The main public-pretraining run has committed {len(rows):,}/4,447 updates and '
        f'{main_runtime["response_EOS_targets"]:,} response/EOS targets. The newer matched CPU assessment remains live. '
        'Latest completed local functional correctness is still 0/48 on an older checkpoint. No new accuracy gain '
        'or frontier parity is established. All ten final requirements remain incomplete.\n')
    (OUT/'metrics.md').write_text(metrics)
    review = dict(at=at,requirements=requirements,previous_full_scope_review=str(previous),
        current_goal_turn_classification=proof['current_goal_turn_classification'],
        current_actual_runtime=dict(main=main_runtime,original_live_jobs=live_jobs,pixel_audit=pixel_owner),
        latest_actual_full_multimodal_pixel_readiness=str(OUT/'complete.json'),
        all_original17family_human_RL_and_nine_endpoint_scopes_preserved=True,
        source_files_sha256={**files,str(OUT/'complete.json'):sha(OUT/'complete.json')},goal_achieved=False)
    atomic(OUT/'review.json',review)
    if any(sha(p) != digest for p,digest in files.items()):
        raise ValueError('Fixed audit/source bytes must remain exact before updating unbound indices')
    existing.update(at=at,requirements=requirements,latest_full_scope_review=str(OUT/'review.json'),
        latest_actual_main_runtime=main_runtime,latest_actual_full_multimodal_pixel_readiness=str(OUT/'complete.json'),goal_achieved=False)
    atomic(ROOT/'reports/completion-audit.json',existing)
    paragraph = (f'Full physical pixel audit active, **{at}**: five separate-protocol/tamper/geometry controls pass. '
        'A new CUDA-hidden SCHED_IDLE data worker checks all **195,121stored image/crop paths** behind **302,627train views**, '
        'including all91,585word crops. **The pixel audit is incomplete; this is not new vision training or model-quality evidence.** '
        f'The original main continues at **{len(rows)}/4,447** with all canonical source/human/RL/nine-endpoint scopes. '
        '**Latest complete local code correctness remains0/48 on an older checkpoint; all ten final requirements remain incomplete.** '
        '[Actual launch, scope and remaining checks](reports/full-multimodal-pixel-readiness-v1-review/metrics.md), '
        '[full review](reports/full-multimodal-pixel-readiness-v1-review/review.json). Earlier entries are historical snapshots.\n\n')
    for name in ['README.md','RESEARCH.md']:
        path = ROOT/name; first,rest = path.read_text().split('\n',1)
        path.write_text(first+'\n\n'+paragraph+rest.lstrip('\n'))
    print(json.dumps(dict(complete=True,pixel_audit_live_owner=owner,actual_main_updates=len(rows),
        pixel_audit_complete=False,training_admitted=False,goal_achieved=False)))


if __name__ == '__main__':
    main()
