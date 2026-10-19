"""Reclaim inactive duplicate JSONL copies while retaining exact immutable inputs."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import tempfile

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.source_segment_archive_v1 import archive_duplicate,no_readers,restore,segment_hash,verify

ROOT=Path(__file__).resolve().parents[1]
LANGUAGES=['python','javascript','java','go','php','ruby']


def main():
    reports=ROOT/'reports/code-source-duplicate-archive-v1'
    code=ROOT/'data/code-search-expanded-v1';archives=code/'source-copy-archives-v1'
    if reports.exists() or archives.exists():raise ValueError('Preserve earlier archival plans and partial results')
    occupied=occupied_writer_reservation();remaining=26*1024**3-occupied['credited_bytes']
    if shutil.disk_usage(ROOT).free<remaining+4*1024**2:
        raise ValueError('Small metadata budget above remaining persistent writers required')
    frozen_path=ROOT/'reports/extended-curriculum-v1/plan.json';frozen=json.loads(frozen_path.read_text())
    spec=next(s for s in frozen['stages'] if s['name']=='multilingual-code-instruction')
    anchor=Path(spec['data']);corpus_path=code/'corpus.json';corpus=json.loads(corpus_path.read_text())
    validation=code/'sft.valid.jsonl'
    if (file_hash(anchor)!=spec['data_sha256']
            or file_hash(validation)!=corpus['sft_files_sha256']['valid']
            or [Path(s['path']).name for s in corpus['source_parts']]!=LANGUAGES):
        raise ValueError('Exact frozen active code input and original language/validation inventory required')
    with anchor.open('rb') as stream:
        for _ in range(9500):
            if json.loads(stream.readline()).get('source')!='HuggingFaceH4/no_robots':
                raise ValueError('Exact9500 original human rows before documented-code sources required')
        prefix_bytes=stream.tell()
    original_global=code/'sft.train.jsonl'
    if prefix_bytes+original_global.stat().st_size!=anchor.stat().st_size:
        raise ValueError('Whole original global code source must match the exact active suffix')
    items=[dict(target=str(original_global),anchor=str(anchor),offset=prefix_bytes,
                target_sha256=corpus['sft_files_sha256']['train'],anchor_sha256=spec['data_sha256'],
                archive_name='global-sft-train.json')]
    offsets={'train':prefix_bytes,'valid':0}
    bound={str(frozen_path):file_hash(frozen_path),str(corpus_path):file_hash(corpus_path),
           str(anchor):spec['data_sha256'],str(validation):corpus['sft_files_sha256']['valid']}
    for language,source in zip(LANGUAGES,corpus['source_parts']):
        part=code/'parts'/language;manifest_path=part/'corpus.json'
        if file_hash(manifest_path)!=source['manifest_sha256']:
            raise ValueError('Original per-language source inventory changed')
        bound[str(manifest_path)]=source['manifest_sha256'];manifest=json.loads(manifest_path.read_text())
        for split in ['train','valid']:
            target=part/('sft.'+split+'.jsonl');retained=anchor if split=='train' else validation
            items.append(dict(target=str(target),anchor=str(retained),offset=offsets[split],
                target_sha256=manifest['files_sha256'][target.name],anchor_sha256=bound[str(retained)],
                archive_name=language+'-sft-'+split+'.json'))
            offsets[split]+=target.stat().st_size
    if offsets!={'train':anchor.stat().st_size,'valid':validation.stat().st_size}:
        raise ValueError('Complete concatenation lengths must agree for every original source row')
    # These exact bytes are required by the waiting QA admission and remain full.
    exposure=json.loads((ROOT/'reports/nq-development-exposure-v1/audit.json').read_text())
    protected={Path(p).resolve() for p in exposure['source_files_sha256']}
    protected.update({anchor.resolve(),validation.resolve()})
    for item in items:
        target=Path(item['target'])
        if (target.is_symlink() or target.resolve() in protected
                or not target.resolve().is_relative_to(code) or target.stat().st_nlink!=1
                or file_hash(target)!=item['target_sha256']
                or segment_hash(item['anchor'],item['offset'],target.stat().st_size)!=item['target_sha256']):
            raise ValueError('Only exact inactive owned source duplicates may be archived')
        no_readers(target);item['bytes']=target.stat().st_size
    reports.mkdir();archives.mkdir()
    sources=[Path(__file__),ROOT/'src/slm_lab/source_segment_archive_v1.py']
    bound.update({str(p):file_hash(p) for p in sources})
    atomic_json(reports/'plan.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),items=items,
        retained_input_files_sha256=bound,protected_QA_source_files=sorted(map(str,protected)),
        expected_duplicate_bytes=sum(i['bytes'] for i in items),occupied_verified_writer=occupied,
        all13_source_segments_checksum_verified_before_any_removal=True,
        no_original_checkpoint_or_image_or_token_shard_selected=True,
        restoration_command='.venv/bin/python -m slm_lab.source_segment_archive_v1 restore --manifest <archive.json>',
        goal_achieved=False))
    # Exercise the actual restore path on the original smallest validation source.
    sample=min((i for i in items if i['anchor']==str(validation)),key=lambda i:i['bytes'])
    records=[];before=shutil.disk_usage(ROOT).free
    for item in items:
        for p,h in bound.items():
            if file_hash(p)!=h:raise ValueError('Retained original live input/runtime changed; inspect partial archival')
        path=archives/item['archive_name']
        result=archive_duplicate(item['target'],item['anchor'],item['offset'],path,item['target_sha256'],item['anchor_sha256'])
        record={**result,'archive_manifest':str(path),'archive_manifest_sha256':file_hash(path)};records.append(record)
        atomic_json(reports/'status.json',dict(status='archiving verified inactive source copies',
            updated_at=datetime.now(timezone.utc).isoformat(),completed=len(records),required=len(items),
            reclaimed_original_allocated_bytes=sum(r['original_allocated_bytes'] for r in records),goal_achieved=False))
        print('archived_duplicate',len(records),Path(item['target']).relative_to(ROOT),item['bytes'],flush=True)
        if item is sample:
            if shutil.disk_usage('/tmp').free<12*1024**3+sample['bytes']:
                raise ValueError('Keep compiler workspace reserve during actual restore verification')
            with tempfile.TemporaryDirectory(prefix='slm-source-segment-restore-') as work:
                restored=restore(path,Path(work)/'restored.jsonl')
                if restored['restored_sha256']!=sample['target_sha256']:
                    raise ValueError('Actual restored original source checksum differs')
                atomic_json(reports/'actual-restore-check.json',dict(**restored,
                    archived_source=item['target'],archive_manifest=str(path),temporary_output_removed_after_check=True))
    for p,h in bound.items():
        if file_hash(p)!=h:raise ValueError('Retained source changed after archival')
    for r in records:verify(r['archive_manifest'])
    atomic_json(reports/'results.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound,archives=records,original_duplicate_bytes=sum(r['original_bytes'] for r in records),
        reclaimed_original_allocated_bytes=sum(r['original_allocated_bytes'] for r in records),
        free_bytes_before=before,free_bytes_after=shutil.disk_usage(ROOT).free,
        free_space_measurements_share_the_filesystem_with_live_training=True,
        all_original_bytes_compared_exactly_and_restorable=True,
        retained_live_training_and_validation_bytes_unchanged=True,
        existing_protected_QA_exposure_source_paths_unchanged=True,
        model_weights_or_training_processes_changed=False,goal_achieved=False,
        scope='Lossless source-copy storage reclamation only. Raw-source reconstruction tools may require restoring their original individual paths first. This does not change any model capability or complete the pending CodeNet training gates.'))
    atomic_json(reports/'status.json',dict(status='all original duplicate source bytes retained in verified source-segment archives',
        updated_at=datetime.now(timezone.utc).isoformat(),completed=len(records),goal_achieved=False))


if __name__=='__main__':main()
