"""Index all admitted original human training facts for later coding rehearsal."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_fact_replay_v1 import FAMILY,FactReplayCorpus,fact_role
from slm_lab.codenet_replay_v1 import INDEX_DTYPE,FullReplayEncoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.nq_input_audit import admit

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'data/codenet-fact-replay-v1';reports=ROOT/'reports/codenet-fact-replay-v1'
    if output.exists() or reports.exists():raise ValueError('Preserve all prior fact-replay preparation')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes']
    if shutil.disk_usage(ROOT).free<reserve+64*1024**2:raise ValueError('Remaining writers plus64MiB fact index budget required')
    source=ROOT/'data/nq-human-questions-v1/train.jsonl';source_manifest=source.parent/'manifest.json'
    original=json.loads(source_manifest.read_text())
    admitted,receipt=admit(ROOT)
    if len(admitted)!=87925 or not receipt['validation_rows_not_returned_for_training']:
        raise ValueError('Actual complete original NQ training-only admission required')
    token_path=ROOT/'data/codenet-durable-pack-v2/tokenizer.json';config_path=ROOT/'configs/model.json'
    files=[Path(__file__),ROOT/'src/slm_lab/codenet_fact_replay_v1.py',ROOT/'src/slm_lab/codenet_replay_v1.py',
        ROOT/'src/slm_lab/training.py',ROOT/'src/slm_lab/model.py',ROOT/'src/slm_lab/nq_input_audit.py',
        ROOT/'src/slm_lab/codenet_checkpoint_v1.py',source,source_manifest,token_path,config_path,
        ROOT/'reports/nq-original-input-audit-v1.json',ROOT/'reports/nq-development-exposure-v1/audit.json']
    bound={str(p):file_hash(p) for p in files};guard=FileBindings(bound)
    if bound[str(source)]!=original['files_sha256']['train.jsonl'] or bound[str(token_path)]!=original['tokenizer_sha256']:
        raise ValueError('Exact original human facts and own tokenizer required')
    output.mkdir();reports.mkdir();atomic_json(output/'source-admission.json',receipt)
    atomic_json(reports/'plan.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        original_training_examples=87925,whole_response_masks_required=True,source_roles_not_resplit=True,
        no_new_images_or_teacher_or_model_weights=True,occupied_verified_writer=occupied,
        remaining_writer_reserve_bytes=reserve,maximum_artifact_bytes=64*1024**2,
        intended_final_checkpoint_match_verified=False,goal_achieved=False))
    encoder=FullReplayEncoder(Tokenizer.from_file(str(token_path)),ModelConfig.load(config_path),source.parent,448)
    entries=[];contexts=targets=0
    with source.open('rb') as f,(output/'records.jsonl').open('x') as records:
        for number,expected in enumerate(admitted):
            offset=f.tell();raw=f.readline();row=json.loads(raw)
            if row!=expected:raise ValueError('Original fact source/admission order or contents differ')
            fact_role(row);ids,labels=encoder.sequence(row,'response')
            target_count=sum(v!=-100 for v in labels)
            if any(v in {encoder.bos,encoder.eos,encoder.pad} for v in ids[1:-1]):
                raise ValueError('Literal reserved tokens need separate replay review')
            digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).digest()
            entries.append((offset,len(raw),0,len(ids),target_count,np.void(hashlib.sha256(raw).digest()),np.void(digest)))
            contexts+=len(ids);targets+=target_count
            records.write(json.dumps(dict(id=row['id'],group_id=row['group_id'],source_line=row['source_line'],
                offset=offset,bytes=len(raw),source_row_sha256=hashlib.sha256(raw).hexdigest(),
                full_input_ids_sha256=digest.hex(),context_tokens=len(ids),target_tokens=target_count))+'\n')
            if (number+1)%2000==0:
                guard.check();records.flush();print('complete_original_fact_replay_rows',number+1,flush=True)
        if f.read():raise ValueError('Original fact source has extra rows')
    if len(entries)!=87925 or targets!=434256 or encoder.truncated:
        raise ValueError('Every complete original human fact target required')
    np.save(output/'index.npy',np.asarray(entries,dtype=INDEX_DTYPE),allow_pickle=False)
    artifacts=['index.npy','records.jsonl','source-admission.json']
    size=sum((output/n).stat().st_size for n in artifacts)
    if size>64*1024**2 or shutil.disk_usage(ROOT).free<reserve:raise ValueError('Fact-replay storage reservation exceeded')
    guard.check(full=True)
    manifest=dict(format='slm-complete-human-fact-replay-v1',completed_at=datetime.now(timezone.utc).isoformat(),
        original_training_source=str(source),original_training_source_sha256=bound[str(source)],
        original_human_fact_manifest_sha256=bound[str(source_manifest)],family_names=[FAMILY],family_counts={FAMILY:len(entries)},
        complete_fit_rows=len(entries),complete_context_tokens=contexts,complete_target_tokens=targets,
        image_size=448,tokenizer_sha256=bound[str(token_path)],original_roles_preserved=True,
        artifact_files_sha256={n:file_hash(output/n) for n in artifacts},stored_artifact_bytes=size,
        intended_final_checkpoint_match_verified=False,remaining_queued_streams_automatically_covered=False,
        new_model_training_started=False,goal_achieved=False,
        scope='Complete original human fact-QA source component for the prospective CodeNet replay. This supplements existing rehearsal; dialogue, own-context, paired photos, later systems code and final-policy matching remain separate.')
    atomic_json(output/'manifest.json',manifest)
    replay=FactReplayCorpus(output)
    if replay.size(FAMILY)!=87925:raise ValueError('Actual fact replay reader population differs')
    atomic_json(reports/'status.json',dict(status='complete original fact replay indexed; independent readback and combined coverage pending',
        updated_at=datetime.now(timezone.utc).isoformat(),source_rows=87925,complete_response_targets=targets,
        original_human_fact_admission_reverified=True,new_model_training_started=False,goal_achieved=False))
    print('complete_fact_replay_component',len(entries),targets,flush=True)


if __name__=='__main__':main()
