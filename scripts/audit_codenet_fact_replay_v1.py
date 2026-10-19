"""Reconstruct every original fact row, then enumerate combined source coverage."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_fact_replay_v1 import FactReplayCorpus,CombinedReplay,FAMILY
from slm_lab.codenet_replay_v1 import ReplayCorpus,FullReplayEncoder
from slm_lab.codenet_replay_schedule_v1 import coverage_replay_schedule
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.nq_input_audit import validate_row

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=ROOT/'data/codenet-fact-replay-v1';reports=ROOT/'reports/codenet-fact-replay-v1'
    output=reports/'independent-readback-and-coverage.json'
    if output.exists():raise ValueError('Preserve every earlier independent fact-replay audit')
    facts=FactReplayCorpus(data);manifest=facts.manifest
    original_manifest_path=ROOT/'data/nq-human-questions-v1/manifest.json'
    original_manifest=json.loads(original_manifest_path.read_text())
    original_source=ROOT/'data/nq-open-original-source-v1/nq_open/NQ-open.train.jsonl'
    development=ROOT/'data/nq-human-questions-v1/valid.jsonl'
    plan_path=reports/'plan.json';plan=json.loads(plan_path.read_text())
    for p,h in plan['source_files_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Original prepared fact replay/source/runtime changed')
    paths=[Path(__file__),ROOT/'src/slm_lab/codenet_fact_replay_v1.py',ROOT/'src/slm_lab/codenet_replay_v1.py',
        ROOT/'src/slm_lab/codenet_replay_schedule_v1.py',ROOT/'src/slm_lab/codenet_stage_v1.py',
        ROOT/'src/slm_lab/nq_input_audit.py',data/'manifest.json',original_manifest_path,original_source,development,plan_path]
    bound={str(p):file_hash(p) for p in paths};guard=FileBindings(bound)
    encoder=FullReplayEncoder(Tokenizer.from_file(str(ROOT/'data/codenet-durable-pack-v2/tokenizer.json')),
                              ModelConfig.load(ROOT/'configs/model.json'),ROOT,448)
    contexts=targets=0;groups=set();seen=set()
    with original_source.open() as source,(data/'records.jsonl').open() as records:
        for position in range(facts.size(FAMILY)):
            row=facts.row(FAMILY,position);original=json.loads(source.readline());record=json.loads(records.readline())
            targets+=validate_row(row,original,'train',position+1,encoder,original_manifest['source_revision'])
            ids,labels=encoder.sequence(row,'response');contexts+=len(ids)
            if (row['id'] in seen or row['group_id'] in groups or record['id']!=row['id']
                    or record['group_id']!=row['group_id'] or record['source_line']!=position+1
                    or record['full_input_ids_sha256']!=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
                    or record['context_tokens']!=len(ids) or record['target_tokens']!=sum(v!=-100 for v in labels)):
                raise ValueError('Every original fact identity/aliases/whole prompt and mask must reconstruct')
            seen.add(row['id']);groups.add(row['group_id'])
            if (position+1)%5000==0:
                guard.check();print('original_fact_rows_reconstructed',position+1,flush=True)
        if source.readline() or records.readline():raise ValueError('Extra original or replay metadata rows')
    dev_groups={json.loads(s)['group_id'] for s in development.open()}
    if (len(seen)!=87925 or groups & dev_groups or targets!=434256 or targets!=manifest['complete_target_tokens']
            or contexts!=manifest['complete_context_tokens'] or encoder.truncated):
        raise ValueError('All complete original training questions and held development roles required')
    base=ReplayCorpus(ROOT/'data/codenet-replay-v1');combined=CombinedReplay(base,facts)
    populations=tuple((name,combined.size(name)) for name in combined.families)
    if len(combined.families)!=12 or sum(n for _,n in populations)!=181343 or 'code:English' in combined.families:
        raise ValueError('All original source families plus explicit human-fact family required')
    try:coverage_replay_schedule(populations,5279,4*5)
    except ValueError:pass
    else:raise ValueError('Old replay capacity must not silently claim complete expanded coverage')
    accumulation=next(a for a in range(1,33) if sum((n+4*a-1)//(4*a) for _,n in populations)<=5279)
    width=4*accumulation;schedule=coverage_replay_schedule(populations,5279,width)
    orders={name:EpochOrder(size,93042+1000+i) for i,(name,size) in enumerate(populations)}
    counts={name:np.zeros(size,dtype=np.uint32) for name,size in populations};draw_hash=hashlib.sha256()
    base_target={name:base.index[base.positions[name]]['target_tokens'].astype(np.uint64) for name in base.families}
    base_target[FAMILY]=facts.index['target_tokens'].astype(np.uint64)
    target_exposures=0
    for step in range(5279):
        name,occurrence=schedule.at(step)
        for rank in range(4):
            for group in orders[name].rank_groups(occurrence,rank,4,1,accumulation):
                for local in group:
                    counts[name][local]+=1;target_exposures+=int(base_target[name][local])
                    draw_hash.update(name.encode()+b'\0'+int(local).to_bytes(4,'little'))
    by_family={name:dict(original_examples=len(c),distinct_reached=int((c>0).sum()),
        actual_exposures=int(c.sum()),minimum_exposures=int(c.min()),maximum_exposures=int(c.max())) for name,c in counts.items()}
    if any(r['distinct_reached']!=r['original_examples'] for r in by_family.values()):
        raise ValueError('Expanded proposal must reach every complete original source')
    guard.check(full=True)
    for p,h in plan['source_files_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Original fact source changed during independent audit')
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        fact_component_manifest_sha256=file_hash(data/'manifest.json'),
        original_human_fact_rows_reconstructed=87925,complete_original_fact_response_targets=targets,
        complete_original_fact_context_tokens=contexts,original_aliases_and_question_answer_fields_preserved=True,
        actual_training_input_ids_and_mask_reconstructed=True,training_development_group_overlap=0,
        original_complete_joint_replay_examples=93418,preserved_original_over_window_review_examples=2,
        base_replay_manifest_sha256=file_hash(ROOT/'data/codenet-replay-v1/manifest.json'),
        combined_families=len(combined.families),combined_complete_fit_examples=181343,
        old_replay_accumulation5_rejected_as_insufficient=True,prospective_replay_accumulation=accumulation,
        prospective_optimizer_steps=5279,prospective_global_replay_width=width,
        prospective_replay_example_exposures=5279*width,prospective_complete_response_target_exposures=target_exposures,
        by_family=by_family,source_draws_sha256=draw_hash.hexdigest(),
        final_starting_checkpoint_match_verified=False,all_later_queued_streams_covered=False,
        pending_later_sources=['reviewed dialogues','own-reader context questions','paired and additional real photos','later systems code and complete interface variants'],
        new_model_training_started=False,goal_achieved=False,
        scope='Complete original human-fact source component plus prospective old-joint/fact coverage enumeration. Not a final replay match, actual optimizer update, retained-knowledge gain or broad independent benchmark.')
    atomic_json(output,result);print('fact_replay_and_combined_coverage_verified',87925,181343,accumulation,5279*width,flush=True)


if __name__=='__main__':main()
