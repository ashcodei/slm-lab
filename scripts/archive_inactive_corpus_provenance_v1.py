"""Losslessly archive inactive raw provenance copies; keep every packed/live input intact."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from prepare_broad_code_continuation import verify_packed
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.nq_input_audit import admit
from slm_lab.source_gzip_archive_v1 import archive,restore
from slm_lab.source_segment_archive_v1 import no_readers

ROOT=Path(__file__).resolve().parents[1]


def main():
    reports=ROOT/'reports/inactive-corpus-provenance-archive-v1'
    if reports.exists():raise ValueError('Preserve every prior provenance-archive attempt')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes']+384*1024**2
    if shutil.disk_usage(ROOT).free<reserve+384*1024**2:raise ValueError('Remaining writers/assessments/indexing plus full archive output required')
    token=file_hash(ROOT/'data/tokenizer-corpus/tokenizer.json')
    # These are the actual immutable packed views used by later continuation.
    for name in ['code-search-expanded-v1','wikipedia-expanded-v1']:
        verify_packed(ROOT/'data'/name,token)
    exposure=json.loads((ROOT/'reports/nq-development-exposure-v1/audit.json').read_text())['source_files_sha256']
    protected=set(exposure)
    for name in ['reports/codenet-image-replay-v1/plan.json','reports/code-instruction-12000-codenet-v1/plan.json']:
        protected.update(json.loads((ROOT/name).read_text())['source_files_sha256'])
    jobs=[];bound={str(Path(__file__)):file_hash(Path(__file__))}
    for directory in ['code-search-expanded-v1','wikipedia-expanded-v1']:
        for part in sorted((ROOT/'data'/directory/'parts').iterdir()):
            if not part.is_dir():continue
            manifest=part/'corpus.json';original=json.loads(manifest.read_text());bound[str(manifest)]=file_hash(manifest)
            names=['documents.jsonl','provided-licenses.jsonl'] if directory=='code-search-expanded-v1' else ['documents.jsonl']
            for name in names:
                source=part/name
                digest=original['files_sha256'][name] if directory=='code-search-expanded-v1' else original['document_provenance_sha256']
                if str(source) in protected or file_hash(source)!=digest or source.stat().st_nlink!=1:
                    raise ValueError('Inactive exact source copy outside every current input binding required')
                no_readers(source)
                target=source.with_name(source.name+'.lossless-v1.json')
                if target.exists():raise ValueError('Preserve existing source archive')
                jobs.append(dict(source=str(source),source_sha256=digest,source_bytes=source.stat().st_size,
                    original_corpus_manifest=str(manifest),archive_manifest=str(target)))
    if len(jobs)!=25:raise ValueError('Exact reviewed25-file raw provenance inventory required')
    files=[ROOT/'src/slm_lab/source_gzip_archive_v1.py',ROOT/'src/slm_lab/source_segment_archive_v1.py',
        ROOT/'scripts/prepare_broad_code_continuation.py',ROOT/'src/slm_lab/nq_input_audit.py']
    bound.update({str(p):file_hash(p) for p in files})
    reports.mkdir();atomic_json(reports/'plan.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound,jobs=jobs,occupied_verified_writer=occupied,minimum_remaining_free_bytes=reserve,
        protected_input_paths=sorted(protected),all_packed_inputs_verified_before_archival=True,
        source_preprocessing_requires_restore=True,model_weights_or_current_training_inputs_changed=False,goal_achieved=False))
    results=[];free_before=shutil.disk_usage(ROOT).free
    for job in jobs:
        for p,h in bound.items():
            if file_hash(p)!=h:raise ValueError('Bound raw archival source/runtime changed')
        result=archive(job['source'],job['archive_manifest'],job['source_sha256'],job['source_bytes']+1024**2,reserve)
        results.append({**result,'archive_manifest':job['archive_manifest'],'archive_manifest_sha256':file_hash(job['archive_manifest'])})
        atomic_json(reports/'status.json',dict(status='losslessly archiving inactive original provenance copies',
            updated_at=datetime.now(timezone.utc).isoformat(),completed=len(results),planned=len(jobs),
            net_reclaimed_allocated_bytes=sum(r['reclaimed_allocated_bytes'] for r in results),goal_achieved=False))
        print('lossless_original_provenance_archives',len(results),len(jobs),flush=True)
    # Actually restore the largest original file, independently hash it, then remove only that temporary copy.
    largest=max(results,key=lambda r:r['original_bytes'])
    with tempfile.TemporaryDirectory(prefix='slm-original-provenance-restore-') as tmp:
        path=Path(tmp)/'original.jsonl';restored=restore(largest['archive_manifest'],path)
        if file_hash(path)!=largest['original_sha256']:raise ValueError('Actual full original provenance restore differs')
        path.unlink();atomic_json(reports/'actual-restore-check.json',dict(**restored,
            archive_manifest=largest['archive_manifest'],original_source=largest['original_path'],
            independently_rehashed_original_bytes=True,temporary_output_removed_after_check=True,goal_achieved=False))
    for name in ['code-search-expanded-v1','wikipedia-expanded-v1']:
        verify_packed(ROOT/'data'/name,token)
    rows,receipt=admit(ROOT)
    if len(rows)!=87925 or receipt['passed'] is not True:raise ValueError('All original human-fact admission bindings must survive archival')
    atomic_json(reports/'post-archive-fact-admission.json',receipt)
    for p,h in bound.items():
        if file_hash(p)!=h:raise ValueError('Bound archival runtime or original manifests changed')
    report=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,archives=results,
        original_files_archived=len(results),original_bytes=sum(r['original_bytes'] for r in results),
        compressed_bytes=sum(r['archive_bytes'] for r in results),
        net_reclaimed_allocated_bytes=sum(r['reclaimed_allocated_bytes'] for r in results),
        free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(ROOT).free,
        all_original_bytes_retained_losslessly=True,actual_largest_source_restored_and_independently_hashed=True,
        all_packed_training_and_validation_inputs_verified_after_archival=True,
        original_fact_training_questions_readmitted=87925,raw_source_preprocessing_requires_restore=True,
        model_updates=0,model_weights_changed=False,live_bound_sources_changed=False,goal_achieved=False,
        scope='Inactive raw provenance/license copies only. Every original byte retained and restorable; no packed source, frozen live SFT, held global validation, image or learned weight changed. Does not fund all remaining stages or establish model capability.')
    atomic_json(reports/'results.json',report)
    atomic_json(reports/'status.json',dict(status='all inactive provenance copies retained losslessly; live packed and fact inputs reverified',
        updated_at=datetime.now(timezone.utc).isoformat(),completed=25,net_reclaimed_allocated_bytes=report['net_reclaimed_allocated_bytes'],goal_achieved=False))
    print('complete_lossless_provenance_archival',25,report['net_reclaimed_allocated_bytes'],flush=True)


if __name__=='__main__':main()
