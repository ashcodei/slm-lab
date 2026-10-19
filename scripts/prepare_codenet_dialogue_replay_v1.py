"""Index every complete original human dialogue without changing source text or roles."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from tokenizers import Tokenizer

from prepare_codenet_native_contract_pack_v2 import occupied_writer_reservation
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_dialogue_replay_v1 import FAMILY,DialogueReplayCorpus,dialogue_role
from slm_lab.codenet_replay_v1 import INDEX_DTYPE,FullReplayEncoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'data/codenet-dialogue-replay-v1';reports=ROOT/'reports/codenet-dialogue-replay-v1'
    if output.exists() or reports.exists():raise ValueError('Preserve every earlier dialogue-replay preparation')
    occupied=occupied_writer_reservation();reserve=26*1024**3-occupied['credited_bytes'];maximum=16*1024**2
    if shutil.disk_usage(ROOT).free<reserve+maximum:raise ValueError('Remaining writers plus bounded dialogue index budget required')
    source=ROOT/'data/human-dialogue-v1/train.jsonl';source_manifest=source.parent/'manifest.json'
    original=json.loads(source_manifest.read_text());config=original['configuration']
    token_path=ROOT/'data/codenet-durable-pack-v2/tokenizer.json';config_path=ROOT/'configs/model.json'
    files=[Path(__file__),ROOT/'src/slm_lab/codenet_dialogue_replay_v1.py',ROOT/'src/slm_lab/codenet_replay_v1.py',
        ROOT/'src/slm_lab/codenet_checkpoint_v1.py',ROOT/'src/slm_lab/training.py',ROOT/'src/slm_lab/model.py',
        ROOT/'src/slm_lab/human_dialogue_corpus.py',source_manifest,token_path,config_path]
    files.extend(source.parent/name for name in original['files_sha256'])
    files.extend(Path(item['path']) for item in config['protected_sources'])
    bound={str(p):file_hash(p) for p in files};guard=FileBindings(bound)
    if (config['implementation_sha256']!=bound[str(ROOT/'src/slm_lab/human_dialogue_corpus.py')]
            or config['tokenizer_sha256']!=bound[str(token_path)] or config['human_only'] is not True
            or config['reviewed_only'] is not True or config['no_truncation'] is not True
            or original['tree_overlap'] or original['prompt_overlap'] or original['synthetic_responses_admitted']):
        raise ValueError('Original frozen human-only reviewed dialogue source required')
    for name,h in original['files_sha256'].items():
        if bound[str(source.parent/name)]!=h:raise ValueError('Original dialogue file changed')
    for item in config['protected_sources']:
        if bound[item['path']]!=item['sha256']:raise ValueError('Original protected dialogue inputs changed')
    valid=[json.loads(s) for s in (source.parent/'valid.jsonl').open()]
    held_groups={r['group_id'] for r in valid};held_prompts={' '.join(r['prompt'].casefold().split()) for r in valid}
    output.mkdir();reports.mkdir()
    atomic_json(reports/'plan.json',dict(recorded_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        original_training_examples=11985,original_train_roles_preserved=True,complete_prompt_response_masks=True,
        occupied_verified_writer=occupied,remaining_writer_reserve_bytes=reserve,maximum_artifact_bytes=maximum,
        original_publisher_reconstruction_pending=True,final_starting_checkpoint_match_verified=False,goal_achieved=False))
    encoder=FullReplayEncoder(Tokenizer.from_file(str(token_path)),ModelConfig.load(config_path),source.parent,448)
    entries=[];contexts=targets=0;seen=set();groups=set()
    with source.open('rb') as stream,(output/'records.jsonl').open('x') as records:
        while raw:=stream.readline():
            offset=stream.tell()-len(raw);row=json.loads(raw);dialogue_role(row)
            if (row['id'] in seen or row['group_id'] in held_groups
                    or ' '.join(row['prompt'].casefold().split()) in held_prompts):
                raise ValueError('Original dialogue IDs or protected tree/prompt split differ')
            ids,labels=encoder.sequence(row,'response');count=sum(v!=-100 for v in labels)
            if count!=row['target_tokens'] or any(v in {encoder.bos,encoder.eos,encoder.pad} for v in ids[1:-1]):
                raise ValueError('Complete original dialogue answer mask/reserved IDs differ')
            digest=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).digest()
            entries.append((offset,len(raw),0,len(ids),count,np.void(hashlib.sha256(raw).digest()),np.void(digest)))
            seen.add(row['id']);groups.add(row['group_id']);contexts+=len(ids);targets+=count
            records.write(json.dumps(dict(id=row['id'],group_id=row['group_id'],source_message_id=row['source_message_id'],
                offset=offset,bytes=len(raw),source_row_sha256=hashlib.sha256(raw).hexdigest(),
                full_input_ids_sha256=digest.hex(),context_tokens=len(ids),target_tokens=count))+'\n')
            if len(entries)%1000==0:
                guard.check();records.flush();print('complete_original_dialogue_rows',len(entries),flush=True)
    expected=original['splits']['train']
    if len(entries)!=11985 or len(entries)!=expected['examples'] or targets!=expected['target_tokens'] or encoder.truncated:
        raise ValueError('All complete original dialogue examples/targets required')
    receipt=dict(passed=True,manifest_sha256=bound[str(source_manifest)],original_training_examples=len(entries),
        original_training_trees=len(groups),complete_response_targets=targets,validation_rows_not_returned_for_training=True,
        held_prepared_tree_and_prompt_overlap=0,original_publisher_reconstruction_pending=True,
        scope='Frozen reviewed-source inventory/roles and complete prompt/answer masks only; independent publisher-tree readback follows.',goal_achieved=False)
    atomic_json(output/'source-admission.json',receipt)
    np.save(output/'index.npy',np.asarray(entries,dtype=INDEX_DTYPE),allow_pickle=False)
    artifacts=['index.npy','records.jsonl','source-admission.json'];size=sum((output/n).stat().st_size for n in artifacts)
    if size>maximum:raise ValueError('Bounded dialogue index storage exceeded')
    guard.check(full=True)
    manifest=dict(format='slm-complete-reviewed-dialogue-replay-v1',completed_at=datetime.now(timezone.utc).isoformat(),
        original_training_source=str(source),original_training_source_sha256=bound[str(source)],
        original_source_manifest=str(source_manifest),original_source_manifest_sha256=bound[str(source_manifest)],
        tokenizer_sha256=bound[str(token_path)],image_size=448,family_names=[FAMILY],family_counts={FAMILY:len(entries)},
        complete_fit_rows=len(entries),complete_context_tokens=contexts,complete_target_tokens=targets,
        artifact_files_sha256={n:file_hash(output/n) for n in artifacts},stored_artifact_bytes=size,
        independent_original_publisher_reconstruction_pending=True,final_starting_checkpoint_match_verified=False,
        all_later_queued_streams_covered=False,new_model_training_started=False,goal_achieved=False)
    atomic_json(output/'manifest.json',manifest)
    if DialogueReplayCorpus(output).size(FAMILY)!=11985:raise ValueError('Actual dialogue reader population differs')
    atomic_json(reports/'status.json',dict(status='all original dialogues indexed; publisher readback and combined coverage pending',
        updated_at=datetime.now(timezone.utc).isoformat(),source_rows=len(entries),complete_response_targets=targets,
        new_model_training_started=False,goal_achieved=False))
    print('complete_dialogue_replay_component',len(entries),targets,flush=True)


if __name__=='__main__':main()
