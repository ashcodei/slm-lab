"""Read every later image replay source, complete target and actual image input back."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image,ImageOps
from tokenizers import Tokenizer
import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_dialogue_replay_v1 import DialogueReplayCorpus
from slm_lab.codenet_fact_replay_v1 import FactReplayCorpus,CombinedReplay
from slm_lab.codenet_image_replay_v1 import ImageReplayCorpus,FAMILIES,canonical_image
from slm_lab.codenet_replay_v1 import ReplayCorpus,FullReplayEncoder
from slm_lab.codenet_replay_schedule_v1 import coverage_replay_schedule
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT=Path(__file__).resolve().parents[1]


def wait_for_preparation(reports):
    launch=json.loads((ROOT/'reports/codenet-image-replay-v1-launch.json').read_text());own=file_hash(Path(__file__))
    while True:
        if file_hash(Path(__file__))!=own:raise ValueError('Waiting image auditor source changed')
        for p,h in launch['source_files_sha256'].items():
            if file_hash(p)!=h:raise ValueError('Bound original image preparation implementation changed')
        proc=Path('/proc')/str(launch['pid'])
        try:
            raw=(proc/'stat').read_text();a=raw[raw.rfind(')')+2:].split()
            live=a[19]==str(launch['start_ticks']) and a[0] not in {'Z','X'}
            if live and (proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()!=launch['observed_argv']:
                raise ValueError('Original image preparation argv changed')
        except FileNotFoundError:live=False
        if not live:
            if not (ROOT/'data/codenet-image-replay-v1/manifest.json').exists():
                raise ValueError('Preparation ended without a complete manifest; preserve and inspect partial output')
            return
        atomic_json(reports/'status.json',dict(status='waiting for exact live image preparation before independent readback',
            preparation_pid=launch['pid'],preparation_start_ticks=launch['start_ticks'],updated_at=datetime.now(timezone.utc).isoformat(),
            new_model_training_started=False,goal_achieved=False))
        time.sleep(10)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--wait-for-preparation',action='store_true');args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Keep all GPUs dedicated to the live curriculum')
    torch.set_num_threads(1);reports=ROOT/'reports/codenet-image-replay-v1-independent-audit'
    if reports.exists():raise ValueError('Preserve every original image readback/audit attempt')
    reports.mkdir()
    if args.wait_for_preparation:wait_for_preparation(reports)
    corpus=ImageReplayCorpus(ROOT/'data/codenet-image-replay-v1');m=corpus.manifest
    plan_path=ROOT/'reports/codenet-image-replay-v1/plan.json';plan=json.loads(plan_path.read_text())
    if file_hash(plan_path)!=m['preparation_plan_sha256']:raise ValueError('Exact complete preparation plan required')
    own_paths=[Path(__file__),ROOT/'src/slm_lab/codenet_image_replay_v1.py',ROOT/'src/slm_lab/codenet_replay_v1.py',
        ROOT/'src/slm_lab/codenet_fact_replay_v1.py',ROOT/'src/slm_lab/codenet_dialogue_replay_v1.py',
        ROOT/'src/slm_lab/codenet_replay_schedule_v1.py',ROOT/'src/slm_lab/codenet_stage_v1.py',
        corpus.folder/'manifest.json',plan_path]
    bound={**plan['source_files_sha256'],**{str(p):file_hash(p) for p in own_paths}}
    for name,h in m['artifact_files_sha256'].items():bound[str(corpus.folder/name)]=h
    guard=FileBindings(bound)
    atomic_json(reports/'plan.json',dict(source_files_sha256=bound,source_rows=m['source_rows'],
        complete_fit_rows=m['complete_fit_rows'],review_rows=m['review_rows'],image_size=448,
        no_model_loaded=True,no_GPU_context=True,goal_achieved=False))
    # Prove the own-context source view is exactly the original selected gzip lines.
    original_own=ROOT/'data/own-ocr-human-questions-v1';own_manifest=json.loads((original_own/'manifest.json').read_text())
    all_decoded=hashlib.sha256();selected_decoded=hashlib.sha256();own_rows=0
    with gzip.open(original_own/'train.jsonl.gz','rb') as original,(corpus.folder/'own-context-original-rows.jsonl').open('rb') as view:
        for raw in original:
            all_decoded.update(raw);row=json.loads(raw)
            if row['own_predicted_context_supplied']:
                if view.readline()!=raw:raise ValueError('Own-context view differs from literal original selected row bytes')
                selected_decoded.update(raw);own_rows+=1
        if view.readline():raise ValueError('Extra own-context source view row')
    if own_rows!=5858 or all_decoded.hexdigest()!=own_manifest['decoded_JSONL_sha256']['train.jsonl']:
        raise ValueError('Every original own-context training row and complete decoded source required')
    if selected_decoded.hexdigest()!=file_hash(corpus.folder/'own-context-original-rows.jsonl'):
        raise ValueError('Exact original own-context view checksum required')
    prior={json.loads(s)['group_id'] for s in (ROOT/'data/extended-curriculum-v1/joint.train.jsonl').open() if '"image"' in s}
    encoder=FullReplayEncoder(Tokenizer.from_file(str(ROOT/'data/codenet-durable-pack-v2/tokenizer.json')),
        ModelConfig.load(ROOT/'configs/model.json'),ROOT,448)
    fit=Counter();selected=Counter();reviewed=Counter();targets=Counter();contexts=Counter();actual_pixels={};seen=set();total=0
    with (corpus.folder/'records.jsonl').open() as records,(corpus.folder/'review.jsonl').open() as reviews:
        for source in m['sources']:
            name=source['family'];path=Path(source['path']);lines=0
            with path.open('rb') as stream:
                while raw:=stream.readline():
                    lines+=1;offset=stream.tell()-len(raw);row=json.loads(raw)
                    if name=='additional_commons' and (row.get('visual_source')!='commons' or row['group_id'] in prior):continue
                    if row['id'] in seen or any(row.get(k) not in {None,'train'} for k in ['split','source_split','source_subset']):
                        raise ValueError('Duplicate/held original image source in readback')
                    seen.add(row['id']);selected[name]+=1;total+=1
                    path_image=canonical_image(row,path,ROOT);actual_row={**row,'image':str(path_image)}
                    # Derive framing and target mask independently of FullReplayEncoder.sequence.
                    text=row['prompt'];prompt=text if text.rstrip().endswith('Assistant:') else 'User: '+text+'\nAssistant:'
                    prompt_ids=encoder.tokenizer.encode(prompt).ids;response_ids=encoder.tokenizer.encode(row['response']).ids
                    ids=[encoder.bos]+prompt_ids+response_ids+[encoder.eos]
                    labels=[-100]*(1+len(prompt_ids))+response_ids+[encoder.eos]
                    overflow=len(ids)>encoder.config.max_text_length
                    reserved=any(v in {encoder.bos,encoder.eos,encoder.pad} for v in ids[1:-1])
                    if overflow or reserved:
                        record=json.loads(next(reviews));reviewed[name]+=1
                        expected='original_whole_context_exceeds_window' if overflow else 'literal_reserved_model_ids_in_original_source'
                        if record['qualification']!=expected or record['complete_original_context_tokens']!=len(ids):
                            raise ValueError('Complete original nonfit source review differs')
                    else:
                        item=corpus.index[corpus.positions[name][fit[name]]];loaded=corpus.row(name,fit[name]);record=json.loads(next(records))
                        if ({k:v for k,v in loaded.items() if not k.startswith('_replay_')}!=actual_row
                                or int(item['offset'])!=offset or int(item['bytes'])!=len(raw)
                                or encoder.sequence(loaded,'response')!=(ids,labels)
                                or record['complete_input_ids_sha256']!=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
                                or record['complete_context_tokens']!=len(ids) or record['complete_target_tokens']!=len(response_ids)+1):
                            raise ValueError('Whole original image prompt/answer/metadata/mask differs through actual replay reader')
                        fit[name]+=1;contexts[name]+=len(ids);targets[name]+=len(response_ids)+1
                    if (record['id']!=row['id'] or record['group_id']!=row['group_id'] or record['family']!=name
                            or record['source_line']!=lines or record['source_offset']!=offset
                            or record['source_bytes']!=len(raw) or record['source_row_sha256']!=hashlib.sha256(raw).hexdigest()):
                        raise ValueError('Every original selected source row must be fit or explicitly reviewed exactly once')
                    if str(path_image) not in actual_pixels:
                        if file_hash(path_image)!=row['stored_image_sha256']:raise ValueError('Original image bytes changed')
                        tensor=encoder.pixels(actual_row)
                        if tensor.shape!=(3,448,448) or tensor.dtype!=torch.float32:raise ValueError('Actual original image input shape/precision differs')
                        actual=np.rint((tensor.permute(1,2,0).numpy()+1)*127.5).clip(0,255).astype(np.uint8)
                        with Image.open(path_image) as image:
                            direct=np.asarray(ImageOps.pad(image.convert('RGB'),(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128)),dtype=np.uint8)
                        if not np.array_equal(actual,direct):raise ValueError('Actual normalized input differs from direct original RGB rendering')
                        actual_pixels[str(path_image)]=hashlib.sha256(actual.tobytes()).hexdigest()
                    if actual_pixels[str(path_image)]!=row['model_pixel_sha256_448']:
                        raise ValueError('Actual original448px pixels differ from golden hashes')
                    if total%1000==0:
                        guard.check()
                        atomic_json(reports/'status.json',dict(status='independently reconstructing original image rows/masks/pixels',
                            updated_at=datetime.now(timezone.utc).isoformat(),source_rows=total,complete_fit_rows=sum(fit.values()),
                            reviewed_rows=sum(reviewed.values()),unique_actual_images=len(actual_pixels),new_model_training_started=False,goal_achieved=False))
                        print('independent_original_later_image_rows',total,'unique_pixels',len(actual_pixels),flush=True)
            if lines!=source['original_source_lines'] or selected[name]!=source['selected_rows'] or reviewed[name]!=source['review_rows']:
                raise ValueError('Every complete original source population must reconstruct')
        if records.readline() or reviews.readline():raise ValueError('Extra fit or nonfit source metadata')
    if (total!=154805 or total!=m['source_rows'] or dict(fit)!=m['family_counts'] or sum(fit.values())!=m['complete_fit_rows']
            or sum(reviewed.values())!=m['review_rows'] or len(actual_pixels)!=m['unique_original_image_files'] or encoder.truncated):
        raise ValueError('Full original image replay population/fit/review/pixels required')
    base=ReplayCorpus(ROOT/'data/codenet-replay-v1');facts=FactReplayCorpus(ROOT/'data/codenet-fact-replay-v1')
    dialogue=DialogueReplayCorpus(ROOT/'data/codenet-dialogue-replay-v1');combined=CombinedReplay(base,facts,dialogue,corpus)
    for component in [base,facts,dialogue]:
        bound[str(component.folder/'manifest.json')]=file_hash(component.folder/'manifest.json')
        bound[str(component.source)]=component.manifest['original_training_source_sha256']
        bound.update({str(component.folder/name):h for name,h in component.manifest['artifact_files_sha256'].items()})
    guard=FileBindings(bound)
    populations=tuple((name,combined.size(name)) for name in combined.families)
    if len(populations)!=19 or sum(n for _,n in populations)!=193328+sum(fit.values()):
        raise ValueError('Every original joint/fact/dialogue/later-image fit family required')
    try:coverage_replay_schedule(populations,5279,4*10)
    except ValueError:pass
    else:raise ValueError('Previous13-family replay capacity must reject expanded incomplete coverage')
    accumulation=next(a for a in range(1,65) if sum((n+4*a-1)//(4*a) for _,n in populations)<=5279)
    width=4*accumulation;schedule=coverage_replay_schedule(populations,5279,width)
    orders={name:EpochOrder(size,93042+1000+i) for i,(name,size) in enumerate(populations)}
    counts={name:np.zeros(size,dtype=np.uint32) for name,size in populations};draws=hashlib.sha256();exposed_targets=0
    token_counts={name:base.index[base.positions[name]]['target_tokens'].astype(np.uint64) for name in base.families}
    token_counts[facts.families[0]]=facts.index['target_tokens'].astype(np.uint64)
    token_counts[dialogue.families[0]]=dialogue.index['target_tokens'].astype(np.uint64)
    token_counts.update({name:corpus.index[corpus.positions[name]]['target_tokens'].astype(np.uint64) for name in corpus.families})
    for step in range(5279):
        name,occurrence=schedule.at(step)
        for rank in range(4):
            for group in orders[name].rank_groups(occurrence,rank,4,1,accumulation):
                for local in group:
                    counts[name][local]+=1;exposed_targets+=int(token_counts[name][local])
                    draws.update(name.encode()+b'\0'+int(local).to_bytes(4,'little'))
    by_family={name:dict(original_examples=len(c),distinct_reached=int((c>0).sum()),actual_exposures=int(c.sum()),
        minimum_exposures=int(c.min()),maximum_exposures=int(c.max())) for name,c in counts.items()}
    if any(v['original_examples']!=v['distinct_reached'] for v in by_family.values()):raise ValueError('Complete expanded source coverage required')
    guard.check(full=True)
    if torch.cuda.is_initialized():raise ValueError('No GPU context may be created by this source readback')
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        image_replay_manifest_sha256=file_hash(corpus.folder/'manifest.json'),original_source_rows_reconstructed=total,
        complete_fit_rows_reconstructed=sum(fit.values()),preserved_original_review_rows=sum(reviewed.values()),
        family_fit_counts=dict(fit),family_review_counts=dict(reviewed),family_context_tokens=dict(contexts),family_response_targets=dict(targets),
        original448px_image_exposures_verified=total,unique_original_image_inputs_rechecked=len(actual_pixels),
        actual_float32_tensor_matches_direct_original_RGB_rendering=True,all_original_fit_input_ids_and_masks_reconstructed=True,
        exact_original_selected_own_context_lines_verified=own_rows,source_rows_dropped_or_duplicated=0,
        encoder_truncated_sequences=0,protected_held_group_overlap=m['protected_held_group_overlap'],
        protected_recorded448px_pixel_overlap=m['protected_recorded448px_pixel_overlap'],
        combined_families=len(populations),combined_complete_fit_examples=sum(n for _,n in populations),
        total_original_joint_and_later_image_review_examples=2+sum(reviewed.values()),
        old_replay_accumulation10_rejected_as_insufficient=True,prospective_optimizer_steps=5279,
        prospective_replay_accumulation=accumulation,prospective_global_replay_width=width,
        prospective_example_exposures=5279*width,prospective_response_target_exposures=exposed_targets,
        source_draws_sha256=draws.hexdigest(),by_family=by_family,no_GPU_context_created=True,new_images_copied=0,
        final_starting_checkpoint_match_verified=False,all_later_queued_streams_covered=False,
        pending_later_sources=['finalized paired-photo questions','complete exercise/JS/Go/Ruby header variants','later systems code'],
        new_codenet_model_training_started=False,goal_achieved=False,
        scope='Every original selected later-image row, fit/review role, complete answer mask and actual448px input; expanded prospective source coverage. Final-base lineage, remaining queued streams, exact source/semantic/pretraining independence and retained/candidate model capability remain unproven.')
    atomic_json(reports/'results.json',result)
    atomic_json(reports/'status.json',dict(status='all original later-image replay sources/pixels independently reconstructed; expanded prospective coverage verified',
        updated_at=datetime.now(timezone.utc).isoformat(),source_rows=total,complete_fit_rows=sum(fit.values()),
        review_rows=sum(reviewed.values()),unique_actual_images=len(actual_pixels),new_model_training_started=False,goal_achieved=False))
    print('complete_original_later_image_readback_and_coverage',total,sum(fit.values()),sum(n for _,n in populations),accumulation,flush=True)


if __name__=='__main__':main()
