"""Reaudit original photo questions after collector-only runtime changes, preserving old evidence."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.evaluation_provenance import capture
from slm_lab.textvqa_input_audit import audit
from slm_lab.textvqa_admission import admit
from slm_lab.own_ocr_question_admission import admit as admit_context

ROOT=Path(__file__).resolve().parents[1]


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Keep all GPUs dedicated to training')
    torch.set_num_threads(1)
    old_path=ROOT/'reports/textvqa-original-input-audit-v1.json'
    fresh_path=ROOT/'reports/textvqa-original-input-audit-runtime-refresh-v2.json'
    preserved=ROOT/'reports/textvqa-original-input-audit-before-runtime-refresh-v2.json'
    receipt_path=ROOT/'reports/textvqa-runtime-audit-refresh-v2.json'
    if any(p.exists() for p in [fresh_path,preserved,receipt_path]):raise ValueError('Preserve every earlier audit/refresh attempt')
    old_bytes=old_path.read_bytes();old_sha=hashlib.sha256(old_bytes).hexdigest();old=json.loads(old_bytes)
    files=[ROOT/name for name in old['implementation_runtime']['files_sha256']]
    actual=capture(files);wanted=old['implementation_runtime']
    if any(actual[k]!=wanted[k] for k in wanted if k!='files_sha256'):
        raise ValueError('Collector-only source refresh cannot authorize a different Python/package runtime')
    changed=[p for p,h in wanted['files_sha256'].items() if actual['files_sha256'].get(p)!=h]
    if changed!=['src/slm_lab/vqa_pairs.py']:raise ValueError('Only reviewed collector source changed; inspect any different mismatch')
    snapshot=ROOT/'reports/source-snapshots/vqa-pairs-before-storage-reserve-v1.py'
    current=ROOT/'src/slm_lab/vqa_pairs.py'
    if file_hash(snapshot)!=wanted['files_sha256'][changed[0]]:raise ValueError('Exact original collector implementation snapshot required')
    def functions(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
    before,after=functions(snapshot),functions(current)
    changed_functions=sorted(k for k in before.keys()|after.keys() if before.get(k)!=after.get(k))
    if changed_functions!=['collect','main','source_identity'] or before['known_photo_aliases']!=after['known_photo_aliases']:
        raise ValueError('Original photo-alias screening must remain identical')
    bound={str(p):file_hash(p) for p in [Path(__file__),*files,snapshot]}
    atomic_json(ROOT/'reports/textvqa-runtime-audit-refresh-v2-plan.json',dict(
        started_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,prior_audit_sha256=old_sha,
        prior_audit=str(old_path),new_complete_audit=str(fresh_path),changed_functions=changed_functions,
        known_photo_aliases_ast_identical=True,training_inputs_or_model_weights_changed=False,goal_achieved=False))
    fresh=audit(fresh_path)
    for p,h in bound.items():
        if file_hash(p)!=h:raise ValueError('Audit refresh source/runtime changed')
    if (file_hash(old_path)!=old_sha or fresh['passed'] is not True or fresh['implementation_runtime']!=actual
            or any(fresh[k]!=old[k] for k in ['manifest_sha256','files_sha256','original_source_and_inventory_sha256',
                'tokenizer_sha256','splits','actual_full_photos_rehashed','actual_model_pixel_sizes_rechecked','validation_group_overlap'])):
        raise ValueError('Complete fresh source/question/pixel/mask audit must preserve all original inputs/counts')
    # The canonical report is not a live controller's bound implementation/input;
    # preserve its exact earlier bytes before publishing the completed recheck.
    with preserved.open('xb') as stream:stream.write(old_bytes);stream.flush();os.fsync(stream.fileno())
    pending=old_path.with_suffix('.runtime-refresh-v2.pending')
    with pending.open('xb') as stream:stream.write(fresh_path.read_bytes());stream.flush();os.fsync(stream.fileno())
    if file_hash(old_path)!=old_sha or file_hash(pending)!=file_hash(fresh_path):
        raise ValueError('Earlier audit or full fresh replacement changed before publication')
    os.replace(pending,old_path)
    rows,reading_receipt=admit(ROOT);context_rows,context_receipt=admit_context(ROOT)
    if len(rows)!=5858 or len(context_rows)!=5858:raise ValueError('All original plain/own-context training questions required')
    atomic_json(receipt_path,dict(completed_at=datetime.now(timezone.utc).isoformat(),passed=True,
        prior_audit_preserved=str(preserved),prior_audit_sha256=old_sha,fresh_audit=str(fresh_path),
        fresh_audit_sha256=file_hash(fresh_path),canonical_audit_sha256=file_hash(old_path),
        changed_collector_functions=changed_functions,known_photo_alias_helper_ast_identical=True,
        complete_original_question_and_pixel_reaudit=True,original_plain_training_questions_readmitted=len(rows),
        original_own_context_training_questions_readmitted=len(context_rows),
        reading_admission=reading_receipt,own_context_admission=context_receipt,
        live_bound_controller_sources_changed=False,model_updates=0,goal_achieved=False,
        scope='Repair stale runtime evidence through full original-source/pixel/mask reaudit. Old evidence preserved; no guard bypass or model capability gain.'))
    print('complete_reading_runtime_reaudit_and_actual_admissions_passed',len(rows),len(context_rows),flush=True)


if __name__=='__main__':main()
