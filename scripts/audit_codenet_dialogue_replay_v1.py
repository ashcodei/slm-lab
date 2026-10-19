"""Independently rebuild human conversations from original publisher trees and full replay coverage."""
from collections import Counter
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_dialogue_replay_v1 import FAMILY,DialogueReplayCorpus
from slm_lab.codenet_fact_replay_v1 import FactReplayCorpus,CombinedReplay
from slm_lab.codenet_replay_v1 import ReplayCorpus,FullReplayEncoder
from slm_lab.codenet_replay_schedule_v1 import coverage_replay_schedule
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig

ROOT=Path(__file__).resolve().parents[1]


def eligible(node):
    if (node.get('synthetic') is not False or node.get('model_name') or node.get('deleted') is not False
            or node.get('review_result') is not True or node.get('role') not in {'prompter','assistant'}
            or not isinstance(node.get('text'),str) or not node['text'].strip() or len(node['text'])>100000):return False
    labels=node.get('labels') or {}
    if any(labels.get(k,{}).get('count',0)>0 and labels.get(k,{}).get('value',0)>0
           for k in ['spam','pii','lang_mismatch','fails_task']):return False
    if node['role']=='assistant':
        quality=labels.get('quality',{})
        if quality.get('count',0)<1 or quality.get('value',0)<.5:return False
    return True


def publisher_fields(tree,history,node,revision):
    if tree['tree_state']!='ready_for_export' or tree['prompt']['message_id']!=tree['message_tree_id']:
        raise ValueError('Original ready-for-export rooted tree required')
    chain=[*history,node]
    if len(chain)>65 or any(not eligible(n) or n['role']!=('prompter' if i%2==0 else 'assistant') for i,n in enumerate(chain)):
        raise ValueError('Every original conversation ancestor must retain reviewed human authorship/roles')
    for i in range(1,len(chain),2):
        parent,answer=chain[i-1],chain[i]
        approved=[c for c in parent.get('replies',[]) if c.get('role')=='assistant' and eligible(c)]
        if not (answer.get('rank')==0 or answer.get('rank') is None and len(approved)==1):
            raise ValueError('Only original top-ranked/single reviewed human replies required')
    if node['role']!='assistant' or not history:raise ValueError('Original assistant answer needs complete user history')
    prompt='\n'.join(('User: ' if n['role']=='prompter' else 'Assistant: ')+n['text'] for n in history)+'\nAssistant:'
    return dict(id='oasst2-'+node['message_id'],group_id='oasst2:'+tree['message_tree_id'],
        prompt=prompt,response=node['text'],language=node.get('lang'),conversation_turns=len(history),
        source='OpenAssistant/oasst2 original human conversation export',
        source_url='https://huggingface.co/datasets/OpenAssistant/oasst2/tree/'+revision,source_revision=revision,
        source_message_id=node['message_id'],source_message_tree_id=tree['message_tree_id'],license='Apache-2.0',
        response_origin='Platform-marked human message: synthetic=false and no model_name',
        human_review_count=node.get('review_count'),human_quality=node['labels']['quality'],human_rank=node.get('rank'),
        synthetic=False,source_text_sha256=hashlib.sha256(node['text'].encode()).hexdigest(),split='train')


def main():
    reports=ROOT/'reports/codenet-dialogue-replay-v1';output=reports/'independent-publisher-and-coverage.json'
    if output.exists():raise ValueError('Preserve earlier independent original-dialogue audits')
    dialogue=DialogueReplayCorpus(ROOT/'data/codenet-dialogue-replay-v1');m=dialogue.manifest
    plan_path=reports/'plan.json';plan=json.loads(plan_path.read_text())
    for p,h in plan['source_files_sha256'].items():
        if file_hash(p)!=h:raise ValueError('Prepared original dialogue source/runtime changed')
    original_manifest=json.loads(Path(m['original_source_manifest']).read_text());config=original_manifest['configuration']
    published=ROOT/'data/openassistant-source-v1'
    trees=published/'2023-11-05_oasst2_ready.trees.jsonl.gz'
    held=published/'validation-00000-of-00001-1deeef95c3248fe0.parquet'
    if file_hash(trees)!=config['tree_sha256'] or file_hash(held)!=config['official_validation_sha256']:
        raise ValueError('Exact original publisher tree export and official held tree table required')
    own_paths=[Path(__file__),ROOT/'src/slm_lab/codenet_dialogue_replay_v1.py',
        ROOT/'src/slm_lab/codenet_fact_replay_v1.py',ROOT/'src/slm_lab/codenet_replay_v1.py',
        ROOT/'src/slm_lab/codenet_replay_schedule_v1.py',ROOT/'src/slm_lab/codenet_stage_v1.py',
        ROOT/'src/slm_lab/training.py',ROOT/'configs/model.json',trees,held,plan_path,
        dialogue.folder/'manifest.json',ROOT/'data/codenet-fact-replay-v1/manifest.json',ROOT/'data/codenet-replay-v1/manifest.json']
    bound={**plan['source_files_sha256'],**{str(p):file_hash(p) for p in own_paths}};guard=FileBindings(bound)
    official_held=set(pq.read_table(held,columns=['message_tree_id']).column('message_tree_id').to_pylist())
    encoder=FullReplayEncoder(Tokenizer.from_file(str(ROOT/'data/codenet-durable-pack-v2/tokenizer.json')),
        ModelConfig.load(ROOT/'configs/model.json'),ROOT,448)
    wanted={};contexts=targets=0;languages=Counter();groups=set()
    with (dialogue.folder/'records.jsonl').open() as stream:
        for position in range(dialogue.size(FAMILY)):
            row=dialogue.row(FAMILY,position);record=json.loads(stream.readline());key=row['source_message_id']
            if key in wanted or row['source_message_tree_id'] in official_held:
                raise ValueError('Duplicate source answer or official held tree in replay')
            ids,labels=encoder.sequence(row,'response');count=sum(v!=-100 for v in labels)
            if (record['id']!=row['id'] or record['group_id']!=row['group_id'] or record['source_message_id']!=key
                    or record['full_input_ids_sha256']!=hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).hexdigest()
                    or record['context_tokens']!=len(ids) or record['target_tokens']!=count):
                raise ValueError('Every actual dialogue input/target mask and record identity must match')
            wanted[key]=row;contexts+=len(ids);targets+=count
            groups.add(row['group_id']);languages[row['language']]+=1
        if stream.readline():raise ValueError('Extra original dialogue replay records')
    rebuilt=set();source_trees=0
    def walk(tree,node,history):
        key=node['message_id']
        if key in wanted:
            if key in rebuilt:raise ValueError('Original publisher repeats a retained source message')
            row=wanted[key];expected=publisher_fields(tree,history,node,config['source_revision'])
            if any(row.get(k)!=v for k,v in expected.items()):
                raise ValueError('Original publisher question/history/answer/reviews/metadata differ')
            prompt_ids=encoder.tokenizer.encode(expected['prompt']).ids
            answer_ids=encoder.tokenizer.encode(expected['response']).ids+[encoder.eos]
            ids,labels=encoder.sequence(row,'response')
            if ids!=[encoder.bos]+prompt_ids+answer_ids or labels!=[-100]*(1+len(prompt_ids))+answer_ids:
                raise ValueError('Literal original multi-turn input framing and whole answer mask differ')
            rebuilt.add(key)
        for child in node.get('replies',[]):walk(tree,child,[*history,node])
    with gzip.open(trees,'rt') as source:
        for line in source:
            tree=json.loads(line);source_trees+=1;walk(tree,tree['prompt'],[])
    if (rebuilt!=set(wanted) or len(rebuilt)!=11985 or len(groups)!=9172 or targets!=4675705
            or contexts!=m['complete_context_tokens'] or targets!=m['complete_target_tokens'] or encoder.truncated
            or dict(languages)!=original_manifest['splits']['train']['languages'] or source_trees!=original_manifest['source_trees']):
        raise ValueError('Every complete original retained human answer/tree/language/target must reconstruct')
    base=ReplayCorpus(ROOT/'data/codenet-replay-v1');facts=FactReplayCorpus(ROOT/'data/codenet-fact-replay-v1')
    for component in [base,facts,dialogue]:
        bound[str(component.source)]=component.manifest['original_training_source_sha256']
        bound.update({str(component.folder/name):digest for name,digest in component.manifest['artifact_files_sha256'].items()})
    guard=FileBindings(bound)
    combined=CombinedReplay(base,facts,dialogue);populations=tuple((name,combined.size(name)) for name in combined.families)
    if len(populations)!=13 or sum(n for _,n in populations)!=193328:
        raise ValueError('All complete joint/fact/dialogue families required')
    try:coverage_replay_schedule(populations,5279,4*9)
    except ValueError:pass
    else:raise ValueError('Previous fact-only replay capacity must reject expanded incomplete coverage')
    accumulation=next(a for a in range(1,33) if sum((n+4*a-1)//(4*a) for _,n in populations)<=5279)
    width=4*accumulation;schedule=coverage_replay_schedule(populations,5279,width)
    orders={name:EpochOrder(size,93042+1000+i) for i,(name,size) in enumerate(populations)}
    counts={name:np.zeros(size,dtype=np.uint32) for name,size in populations};draws=hashlib.sha256()
    token_counts={name:base.index[base.positions[name]]['target_tokens'].astype(np.uint64) for name in base.families}
    token_counts[facts.families[0]]=facts.index['target_tokens'].astype(np.uint64)
    token_counts[FAMILY]=dialogue.index['target_tokens'].astype(np.uint64);exposed_targets=0
    for step in range(5279):
        name,occurrence=schedule.at(step)
        for rank in range(4):
            for group in orders[name].rank_groups(occurrence,rank,4,1,accumulation):
                for local in group:
                    counts[name][local]+=1;exposed_targets+=int(token_counts[name][local])
                    draws.update(name.encode()+b'\0'+int(local).to_bytes(4,'little'))
    by_family={name:dict(original_examples=len(c),distinct_reached=int((c>0).sum()),actual_exposures=int(c.sum()),
        minimum_exposures=int(c.min()),maximum_exposures=int(c.max())) for name,c in counts.items()}
    if any(v['original_examples']!=v['distinct_reached'] for v in by_family.values()):raise ValueError('Complete expanded rehearsal coverage required')
    guard.check(full=True)
    result=dict(completed_at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bound,
        dialogue_manifest_sha256=file_hash(dialogue.folder/'manifest.json'),original_publisher_rows_reconstructed=len(rebuilt),
        original_training_trees=len(groups),original_source_trees_read=source_trees,languages=dict(languages),
        complete_dialogue_context_tokens=contexts,complete_dialogue_response_targets=targets,
        all_original_human_history_answer_review_fields_preserved=True,actual_input_ids_and_whole_answer_masks_reconstructed=True,
        official_validation_tree_overlap=0,combined_families=13,combined_complete_fit_examples=193328,
        preserved_old_over_window_review_examples=2,old_replay_accumulation9_rejected_as_insufficient=True,
        prospective_optimizer_steps=5279,prospective_replay_accumulation=accumulation,prospective_global_replay_width=width,
        prospective_example_exposures=5279*width,prospective_response_target_exposures=exposed_targets,
        source_draws_sha256=draws.hexdigest(),by_family=by_family,
        final_starting_checkpoint_match_verified=False,all_later_queued_streams_covered=False,
        pending_later_sources=['own-reader context questions','additional and paired real photos/words','visual reasoning and complete exercise/header variants','later systems code'],
        new_codenet_model_training_started=False,goal_achieved=False,
        scope='Every retained original reviewed human dialogue and prospective expanded replay source coverage. Publisher review flags do not certify factual/code/medical correctness; no actual optimizer update, retained-knowledge gain, final-base match or independent broad capability claim.')
    atomic_json(output,result);print('original_publisher_dialogues_and_combined_coverage_verified',len(rebuilt),193328,accumulation,5279*width,flush=True)


if __name__=='__main__':main()
