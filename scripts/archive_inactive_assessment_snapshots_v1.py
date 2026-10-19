"""Preserve all bytes of two terminal own assessment policies in durable XOR archives."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import torch

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.inference_snapshot_archive_v1 import archive_snapshot


ROOT=Path(__file__).resolve().parents[1]


def main():
    reports=ROOT/'reports/inactive-assessment-snapshot-archives-v1'
    if reports.exists():raise ValueError('Preserve earlier archive/restore evidence; no implicit restart')
    torch.set_num_threads(1)
    anchor=ROOT/'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
    anchor_sha='4c05d99b0c3fc775bf89e70c6cbac9c63017eb36f67e17c738b5be268fb191ca'
    targets=[(4000,'ceb71bbb459950667400632e4a9033d718df85ae7941f8f6398f4bcc8c8306a3'),
             (12000,'72513e608ce92c802f8b7cd89ed3ed31546610dbc56b29445fab268acaa5cbb8')]
    paths=[];receipts=[]
    for step,digest in targets:
        p=ROOT/f'checkpoints/code-instruction-{step}-assessment-v1/model.pt';receipt=p.parent/'extraction.json'
        if not p.is_file() or file_hash(p)!=digest:raise ValueError('Exact inactive assessment source bytes required')
        d=json.loads(receipt.read_text())
        if d['inference_checkpoint_sha256']!=digest or d['expected_metadata']['steps']!=step or d['expected_metadata']['planned_steps']!=28343:
            raise ValueError('Exact original intermediate extraction provenance required')
        paths.append(p);receipts.append(receipt)
    immutable=[Path(__file__),anchor,*receipts,ROOT/'src/slm_lab/inference_snapshot_archive_v1.py',
        ROOT/'src/slm_lab/checkpoint_xor.py',ROOT/'src/slm_lab/checkpoint_archive.py',
        ROOT/'src/slm_lab/source_segment_archive_v1.py',ROOT/'src/slm_lab/recurrent_ablation.py',
        ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'scripts/prepare_codenet_native_contract_pack_v2.py',
        ROOT/'tests/test_inference_snapshot_archive_v1.py',ROOT/'reports/inference-snapshot-archive-v1-focused-tests-v2.log']
    bound={str(p):file_hash(p) for p in immutable}
    if bound[str(anchor)]!=anchor_sha:raise ValueError('Keep the exact full shared own base checkpoint')
    guard=FileBindings(bound);occupied=occupied_writer_reservation()
    reserve=26*1024**3-occupied['credited_bytes']+256*1024**2
    reports.mkdir()
    def status(message,**fields):
        atomic_json(reports/'status.json',dict(status=message,updated_at=datetime.now(timezone.utc).isoformat(),
            model_training_started=False,goal_achieved=False,**fields))
    plan=dict(recorded_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        original_snapshots=[dict(path=str(p),steps=step,sha256=digest,bytes=p.stat().st_size,
            extraction_receipt=str(receipt),archive=str(p.with_name('model.pt.inference-v1.xor')))
            for p,(step,digest),receipt in zip(paths,targets,receipts)],
        retained_full_anchor=str(anchor),retained_full_anchor_sha256=anchor_sha,
        occupied_verified_original_writer=occupied,remaining_original_writer_reserve_bytes=reserve,
        maximum_payload_bytes_per_snapshot=int(2.25*1024**3),minimum_scratch_free_bytes=12*1024**3,
        physical_restore_required_before_original_removal=True,source_training_budgets_completed=False,
        no_current_pilot_model_or_live_core_controller_modified=True,goal_achieved=False)
    atomic_json(reports/'plan.json',plan);records=[];free_before=shutil.disk_usage(ROOT).free
    for p,(step,digest),receipt in zip(paths,targets,receipts):
        guard.check();current_writer=occupied_writer_reservation()
        current_reserve=max(reserve,26*1024**3-current_writer['credited_bytes']+256*1024**2)
        status('archiving exact terminal own intermediate inference snapshot',source_steps=step,
            remaining_original_writer_reserve_bytes=current_reserve)
        result=archive_snapshot(p,anchor,p.with_name('model.pt.inference-v1.xor'),receipt=receipt,
            checkpoint_root=ROOT/'checkpoints',expected_anchor_sha256=anchor_sha,
            maximum_payload_bytes=int(2.25*1024**3),minimum_free_bytes=current_reserve,
            scratch_directory=Path('/tmp'),minimum_scratch_free_bytes=12*1024**3,maximum_capacity_wait_seconds=60)
        if result['original_sha256']!=digest:raise ValueError('Original snapshot identity changed')
        guard.check(full=True)
        record={k:v for k,v in result.items() if k!='segments'};record['archive']=str(p.with_name('model.pt.inference-v1.xor'))
        records.append(record);atomic_json(reports/'completed-snapshots.json',dict(records=records,goal_achieved=False))
        print('exact_inactive_assessment_snapshot_archived',step,result['reclaimed_allocated_bytes'],flush=True)
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,archives=records,
        snapshots_archived=2,total_original_bytes_preserved=sum(r['original_bytes'] for r in records),
        payload_bytes=sum(r['payload_bytes'] for r in records),reclaimed_allocated_bytes=sum(r['reclaimed_allocated_bytes'] for r in records),
        free_home_bytes_before=free_before,free_home_bytes_after=shutil.disk_usage(ROOT).free,
        full_shared_anchor_retained=True,all_original_bytes_physically_restored_and_rehashed_before_retirement=True,
        all_original_intermediate_metadata_and_tokenizers_unchanged=True,original_paths_require_restore=True,
        live_training_or_assessment_inputs_modified=False,source_training_budgets_completed=False,
        full_future_training_capacity_funded=False,goal_achieved=False,
        scope='Only inactive4000/12000-update own inference snapshots. Every original file byte is retained through durable exact-anchor/XOR payloads with full physical reconstruction checked before retiring redundant full copies. Existing model evaluations remain historical; restore original model.pt paths before using them again. This does not improve model skill or fund all future stages.')
    atomic_json(reports/'results.json',result);status('two inactive own inference snapshots losslessly archived',reclaimed_allocated_bytes=result['reclaimed_allocated_bytes'])
    print('inactive_assessment_snapshot_archives_complete',result['reclaimed_allocated_bytes'],flush=True)


if __name__=='__main__':main()
