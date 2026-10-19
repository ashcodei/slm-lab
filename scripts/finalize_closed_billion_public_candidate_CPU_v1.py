"""Finalize metadata after the fully collected producer's duplicate-field error.

No source collection, tokenization or training is repeated. All original failed
reports and every closed data byte are preserved. The independent full-source
reader still must verify every original row/native target before admission.
"""
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.public_billion_candidate_v1 import QUOTAS
from slm_lab.public_billion_metadata_v1 import manifest,completion,validate_summary

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'reports/public-billion-causal-candidate-CPU-v2'
OUT=ROOT/'reports/public-billion-causal-candidate-CPU-v3'
DATA=ROOT/'data/public-billion-causal-candidate-v2'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists() or (DATA/'manifest.json').exists():
        raise ValueError('Fresh GPU-hidden metadata-only finalizer for the closed existing corpus required')
    old_plan=json.loads((OLD/'plan.json').read_text());failure=json.loads((OLD/'failure.json').read_text())
    progress=json.loads((OLD/'progress.json').read_text());summary=failure['partial_candidate']
    if (old_plan['owner']!=dict(pid=3815469,ticks='43593948') or not terminal(old_plan['owner'])
            or failure['error']!='TypeError("dict() got multiple values for keyword argument \'optimization_admitted\'")'
            or "atomic_json(DATA/'manifest.json'" not in failure['traceback']
            or not failure['partial_original_data_retained'] or progress['candidate']!=summary):
        raise ValueError('Only the verified final-metadata exception after a fully closed original producer is eligible')
    validate_summary(summary,QUOTAS)
    tests=ROOT/'reports/public-billion-metadata-v1-tests.log'
    if '2 passed' not in tests.read_text():raise ValueError('Actual duplicate-field completion and corrupt-count controls required')
    files=dict(old_plan['source_files_sha256'])
    paths=[Path(__file__),ROOT/'src/slm_lab/public_billion_metadata_v1.py',ROOT/'tests/test_public_billion_metadata_v1.py',tests,
        OLD/'plan.json',OLD/'failure.json',OLD/'progress.json',OLD/'source-catalogs.json']
    files.update({str(p):file_hash(p) for p in paths});guard=FileBindings(files)
    receipts=[json.loads(line) for line in (DATA/'original-source-file-receipts.jsonl').open()]
    rows=sum(r['original_prefix_rows_observed'] for r in receipts);dispositions=Counter()
    for r in receipts:dispositions.update(r['source_prefix_dispositions'])
    if (rows!=progress['original_rows_observed'] or len(receipts)!=progress['actual_source_files']
            or dict(dispositions)!=progress['actual_prefix_dispositions']):
        raise ValueError('All closed original source receipts must reconcile with the exact completed prefix')
    native=DATA/'whole-native-documents'
    for role in ['train','valid']:
        if (native/(role+'.tokens.u16')).stat().st_size!=2*summary['native_context_token_counts'][role]:
            raise ValueError('Every closed original native role must retain its complete declared byte size')
    db=sqlite3.connect('file:'+str(native/'exact-original-body-shas.sqlite3')+'?mode=ro',uri=True)
    try:count=db.execute('SELECT count(*) FROM documents').fetchone()[0]
    finally:db.close()
    if count!=sum(sum(n.values()) for n in summary['original_documents'].values()):
        raise ValueError('The closed exact-body source database must retain every accepted original document')
    OUT.mkdir();owner=dict(pid=os.getpid(),ticks=Path('/proc/self/stat').read_text().split(') ',1)[1].split()[19])
    new_plan=dict(old_plan,at=datetime.now(timezone.utc).isoformat(),owner=owner,source_files_sha256=files,
        job_kind='Metadata-only finalizer after closed full-quota original source collection',
        original_collection_owner=old_plan['owner'],original_collection_failure_preserved=True,
        metadata_fix_does_not_change_any_source_native_or_role_bytes=True,full_independent_source_readback_still_required=True)
    atomic_json(OUT/'plan.json',new_plan);shutil.copyfile(OLD/'source-catalogs.json',OUT/'source-catalogs.json')
    artifacts={str(p):file_hash(p) for p in DATA.rglob('*') if p.is_file()}
    raw_manifest=manifest(summary,QUOTAS,file_hash(ROOT/'data/tokenizer-corpus/tokenizer.json'),artifacts)
    raw_manifest.update(metadata_completed_after_original_final_record_exception=True,original_collection_failure_sha256=file_hash(OLD/'failure.json'))
    atomic_json(DATA/'manifest.json',raw_manifest)
    result=completion(summary,QUOTAS,rows=rows,source_files=len(receipts),dispositions=dict(dispositions),bindings={
        str(OUT/'plan.json'):file_hash(OUT/'plan.json'),str(OUT/'source-catalogs.json'):file_hash(OUT/'source-catalogs.json'),
        str(DATA/'manifest.json'):file_hash(DATA/'manifest.json')})
    guard.check(full=True)
    result.update(at=datetime.now(timezone.utc).isoformat(),original_collection_owner=old_plan['owner'],
        original_collection_seconds=progress['seconds'],metadata_only_completion=True,
        every_closed_original_data_file_sha256_preserved=artifacts,original_failure_and_all_previous_attempts_preserved=True)
    atomic_json(OUT/'complete.json',result)
    print(json.dumps(dict(complete=str(OUT/'complete.json'),owner=owner,actual_train_targets=summary['actual_train_targets'],
        original_data_not_recollected_or_retrained=True,optimization_admitted=False)),flush=True)


if __name__=='__main__':main()
