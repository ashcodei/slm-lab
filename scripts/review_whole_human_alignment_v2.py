"""Reconstruct every admitted candidate source response and own token mask."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.human_dialogue_corpus import TREE_FILE, VALID_FILE
from slm_lab.whole_human_alignment_v2 import normalized, NamedHeldQuestions
from slm_lab.whole_human_data_v2 import WholeHumanData

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'data/whole-human-alignment-v2'


def native(tokenizer,row,field):
    prompt=row['prompt']
    framed=prompt if prompt.rstrip().endswith('Assistant:') else 'User: '+prompt+'\nAssistant:'
    prefix=[1,*tokenizer.encode(framed).ids]
    target=[*tokenizer.encode(row[field]).ids,2]
    return prefix+target,[-100]*len(prefix)+target


def main():
    destination=OUTPUT/'independent-source-readback.json'
    if destination.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh GPU-hidden original-source readback required')
    manifest=json.loads((OUTPUT/'manifest.json').read_text())
    files={**manifest['source_files_sha256'],**{str(OUTPUT/name):sha for name,sha in manifest['files_sha256'].items()},
        str(OUTPUT/'manifest.json'):file_hash(OUTPUT/'manifest.json'),str(OUTPUT/'plan.json'):manifest['plan_sha256']}
    guard=FileBindings(files);questions=NamedHeldQuestions(ROOT)
    if any(manifest['source_files_sha256'].get(path)!=sha for path,sha in questions.files.items()):
        raise ValueError('Every original named held question file must remain source-bound')
    tokenizer=Tokenizer.from_file(str(ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'))
    streams={(family,role):WholeHumanData(OUTPUT,family,role,'audit') for family in ('dialogue','preferences') for role in ('train','valid')}
    valid_ids=set(pq.read_table(ROOT/'data/openassistant-source-v1'/VALID_FILE,columns=['message_tree_id']).column('message_tree_id').to_pylist())
    wanted={row['source_message_id']: (role,position,row) for role in ('train','valid')
            for position,row in enumerate(streams['dialogue',role].rows)}
    if len(wanted)!=sum(len(streams['dialogue',r]) for r in ('train','valid')):
        raise ValueError('Every original human message must appear once')
    counts=Counter();groups={'train':set(),'valid':set()};prompts={'train':set(),'valid':set()}
    original_messages=set()
    def eligible(node):
        quality=(node.get('labels') or {}).get('quality',{})
        return (node.get('role')=='assistant' and node.get('synthetic') is False and not node.get('model_name')
            and node.get('deleted') is False and node.get('review_result') is True and quality.get('count',0)>0
            and quality.get('value',0)>=.5 and not any((node.get('labels') or {}).get(k,{}).get('count',0)>0
                and (node.get('labels') or {}).get(k,{}).get('value',0)>0 for k in ('spam','pii','lang_mismatch','fails_task')))
    def walk(tree,node,history):
        identifier=node['message_id']
        if identifier in wanted:
            role,position,row=wanted[identifier];original_messages.add(identifier)
            if (row['source_message_tree_id']!=tree['message_tree_id'] or row['group_id']!='oasst2:'+tree['message_tree_id']
                    or row['source_message_id']!=identifier or row['response']!=node['text']
                    or row['source_text_sha256']!=hashlib.sha256(node['text'].encode()).hexdigest()
                    or role!=('valid' if tree['message_tree_id'] in valid_ids else 'train') or not eligible(node)
                    or row['synthetic'] is not False or row['human_quality']!=node['labels']['quality'] or row['license']!='Apache-2.0'):
                raise ValueError('Every complete original human target, review label, tree role and provenance must reconstruct')
            for index,ancestor in enumerate(history):
                expected='prompter' if index%2==0 else 'assistant'
                if (ancestor.get('role')!=expected or ancestor.get('synthetic') is not False or ancestor.get('model_name')
                        or ancestor.get('deleted') is not False or ancestor.get('review_result') is not True
                        or any((ancestor.get('labels') or {}).get(k,{}).get('count',0)>0
                            and (ancestor.get('labels') or {}).get(k,{}).get('value',0)>0 for k in ('spam','pii','lang_mismatch','fails_task'))
                        or (expected=='assistant' and not eligible(ancestor))):
                    raise ValueError('Every actual human conversation ancestor must retain its original review and alternating role')
            if not history or history[-1]['role']!='prompter':
                raise ValueError('Original human assistant target needs its full original user history')
            approved=[child for child in history[-1].get('replies',[]) if eligible(child)]
            if node.get('rank')!=0 and not (node.get('rank') is None and len(approved)==1):
                raise ValueError('Original top-ranked or single reviewed response selection required')
            prompt='\n'.join(('User: ' if item['role']=='prompter' else 'Assistant: ')+item['text'] for item in history)+'\nAssistant:'
            if row['prompt']!=prompt or native(tokenizer,row,'response')!=tuple(streams['dialogue',role].sequence(position,'response')):
                raise ValueError('Every complete original conversation prefix, response/EOS and exact native answer mask required')
            if questions.matches(prompt+' '+node['text']):
                raise ValueError('Named held question text entered a candidate source stream')
            groups[role].add(row['group_id']);prompts[role].add(normalized(prompt))
            counts['dialogue.'+role]+=1
        for child in node.get('replies',[]):
            walk(tree,child,history+[node])
    with gzip.open(ROOT/'data/openassistant-source-v1'/TREE_FILE,'rt') as source:
        for line in source:
            tree=json.loads(line);walk(tree,tree['prompt'],[])
    if original_messages!=set(wanted) or groups['train']&groups['valid'] or prompts['train']&prompts['valid']:
        raise ValueError('Complete original messages and protected original held trees/prefixes required')
    rank_root=ROOT/'data/original-HH-alignment-v2';rank_manifest=json.loads((rank_root/'manifest.json').read_text())
    indexed={row['id']:(role,position,row) for role in ('train','valid') for position,row in enumerate(streams['preferences',role].rows)}
    reviews={row['id'] for row in map(json.loads,(OUTPUT/'reviews.jsonl').open()) if row['family']=='preferences'}
    seen=set();rank_prompts={'train':set(),'valid':set()};all_original_test_prompts=set()
    for blob in rank_manifest['sources']:
        if blob['original_split']=='test':
            with gzip.open(rank_root/blob['file'],'rt') as source:
                for line in source:
                    raw=json.loads(line)
                    for field in ('chosen','rejected'):
                        mark=raw[field].rfind('\n\nAssistant:')
                        if mark>=0:all_original_test_prompts.add(normalized(raw[field][:mark+len('\n\nAssistant:')]))
    for blob in rank_manifest['sources']:
        with gzip.open(rank_root/blob['file'],'rt') as source:
            for original_position,line in enumerate(source):
                raw=json.loads(line);identifier=f"HH:{blob['subset']}:{blob['original_split']}:{original_position}"
                if identifier not in indexed:
                    if identifier not in reviews:
                        raise ValueError('Every original ranking source row needs a candidate record or explicit review')
                    continue
                if identifier in reviews:
                    raise ValueError('A reviewed source row cannot simultaneously be admitted')
                role,position,row=indexed[identifier];seen.add(identifier)
                mark=raw['chosen'].rfind('\n\nAssistant:');other=raw['rejected'].rfind('\n\nAssistant:')
                prefix=raw['chosen'][:mark+len('\n\nAssistant:')]
                if (mark<0 or other<0 or prefix!=raw['rejected'][:other+len('\n\nAssistant:')]
                        or row['prompt']!=prefix or prefix+row['chosen']!=raw['chosen'] or prefix+row['rejected']!=raw['rejected']
                        or row['source_blob']!=blob['file'] or row['original_source_position']!=original_position
                        or row['source_record_sha256']!=hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                        or row['feedback_source']!='human' or row['response_origin']!=rank_manifest['response_origin']
                        or row['license']!='MIT' or role!=('valid' if blob['original_split']=='test' else 'train')):
                    raise ValueError('Every original ranked pair, full whitespace, source row, role and response origin must reconstruct')
                for field in ('chosen','rejected'):
                    if native(tokenizer,row,field)!=tuple(streams['preferences',role].sequence(position,field)):
                        raise ValueError('Every whole original chosen/rejected source token and response/EOS mask must reconstruct')
                if questions.matches(row['prompt']+' '+row['chosen']+' '+row['rejected']):
                    raise ValueError('Named held question entered a human-ranking candidate')
                rank_prompts[role].add(normalized(row['prompt']));counts['preferences.'+role]+=1
                if len(seen)%2000==0:
                    atomic_json(OUTPUT/'readback-progress.json',dict(status='reconstructing every original public human-ranked source pair and full native mask',pairs=len(seen),goal_achieved=False))
    if (seen!=set(indexed) or rank_prompts['train']&all_original_test_prompts
            or rank_prompts['train']&rank_prompts['valid'] or prompts['train']&all_original_test_prompts):
        raise ValueError('All original HH test prompt groups must remain outside every training stream')
    guard.check(full=True)
    report=dict(completed_at=datetime.now(timezone.utc).isoformat(),reader_sha256=file_hash(Path(__file__).resolve()),
        source_manifest_sha256=file_hash(OUTPUT/'manifest.json'),all_original_sources_indices_and_token_bytes_rehashed=True,
        every_retained_original_human_history_target_review_and_role_reconstructed=True,
        every_retained_original_ranked_pair_full_response_whitespace_and_role_reconstructed=True,
        every_native_BOS_complete_prompt_response_EOS_and_target_mask_reconstructed=True,
        every_original_ranking_row_retained_or_explicitly_reviewed=True,actual_retained_counts=dict(counts),
        all_original_HH_test_prompt_groups_protected=len(all_original_test_prompts),named_question_literal_protection=questions.counts,
        original_tree_and_prompt_training_validation_overlap=0,semantic_independence_not_certified=True,
        human_ranked_response_origin=rank_manifest['response_origin'],CUDA_initialized=False,neural_updates=0,
        candidate_promoted=False,goal_achieved=False)
    atomic_json(destination,report)
    admission=dict(created_at=datetime.now(timezone.utc).isoformat(),scope='whole-original-reviewed-human-dialogue-and-human-ranked-other-model-responses-only',
        training_admitted=True,source_manifest_sha256=file_hash(OUTPUT/'manifest.json'),
        independent_readback_sha256=file_hash(destination),original_held_gradients_forbidden=True,
        external_model_weights=False,source_provenance_and_semantic_overlap_qualifications_retained=True,goal_achieved=False)
    atomic_json(OUTPUT/'admission.json',admission)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
