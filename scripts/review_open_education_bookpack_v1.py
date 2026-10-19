"""Re-encode every original educational page and verify whole-book roles/tails."""
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.open_education_pack_v1 import WholeTokenWindows
from slm_lab.whole_human_alignment_v2 import NamedHeldQuestions

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'data/open-education-bookpack-v1'


def main():
    destination=OUTPUT/'independent-raw-source-and-token-readback.json'
    if destination.exists() or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Fresh GPU-hidden original educational source readback required')
    manifest=json.loads((OUTPUT/'corpus.json').read_text());plan=json.loads((OUTPUT/'plan.json').read_text())
    files={**manifest['source_files_sha256'],str(OUTPUT/'plan.json'):manifest['plan_sha256'],
        str(OUTPUT/'corpus.json'):file_hash(OUTPUT/'corpus.json'),str(OUTPUT/'original-pages.jsonl'):manifest['original_page_index_sha256']}
    guard=FileBindings(files);tokenizer=Tokenizer.from_file(str(ROOT/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'))
    index=[json.loads(line) for line in (OUTPUT/'original-pages.jsonl').open()]
    if len(index)!=40049 or [r['original_source_position'] for r in index]!=list(range(40049)):
        raise ValueError('Every original educational source position exactly once required')
    mapped={};ends={};cursors=dict(train=0,valid=0)
    for role in ('train','valid'):
        mapped[role]=[];ends[role]=[];total=0
        for shard in manifest['splits'][role]['shards']:
            path=OUTPUT/shard['path']
            if file_hash(path)!=shard['sha256'] or path.stat().st_size!=2*shard['tokens']:
                raise ValueError('Every complete actual token shard must match its retained bytes')
            mapped[role].append(np.memmap(path,dtype='<u2',mode='r'));total+=shard['tokens'];ends[role].append(total)
    def packed(role,start,length):
        result=[];remaining=length
        while remaining:
            shard=bisect.bisect_right(ends[role],start);before=0 if shard==0 else ends[role][shard-1]
            take=min(remaining,ends[role][shard]-start)
            result.extend(mapped[role][shard][start-before:start-before+take].astype(np.int64).tolist())
            start+=take;remaining-=take
        return result
    questions=NamedHeldQuestions(ROOT);books=defaultdict(set);texts=defaultdict(set);counts=Counter();missing=Counter();licenses=Counter()
    raw_texts={};question_components=set()
    with gzip.open(ROOT/'data/open-education-source-v1/libretexts-0000.json.gz','rt') as source:
        for position,line in enumerate(source):
            raw=json.loads(line);actual=index[position];body=raw['text'];meta=raw['metadata']
            text_hash=hashlib.sha256(body.encode()).hexdigest();raw_texts[position]=text_hash
            if (actual['id']!=raw['id'] or actual['metadata']!=meta or actual['source']!=raw['source']
                    or actual['created']!=raw['created'] or actual['added']!=raw['added']
                    or actual['full_original_text_sha256']!=text_hash or actual['full_original_text_bytes']!=len(body.encode())):
                raise ValueError('Every complete original author/license/title/URL/text-source field must reconstruct')
            hits=questions.matches(body)
            if hits!=actual['named_held_question_families']:
                raise ValueError('Every actual named held-question match must reconstruct')
            if hits:question_components.add(actual['book_component_id'])
            books[meta['book_url']].add(actual['original_book_role']);licenses[meta['license']]+=1
            for issue in actual['attribution_review']:missing[issue]+=1
            native=actual['native']
            if native is not None:
                role=native['role'];expected=[1,*tokenizer.encode(body).ids,2]
                if (native['offset']!=cursors[role] or native['length']!=len(expected)
                        or native['full_original_text_tokens']!=len(expected)-2 or actual['review_reason'] is not None
                        or packed(role,native['offset'],native['length'])!=expected):
                    raise ValueError('Every complete original own-BPE/BOS/text/EOS sequence and contiguous source offset must reconstruct')
                cursors[role]+=len(expected);counts[role]+=1;texts[text_hash].add(role)
            else:
                counts['review:'+actual['review_reason']]+=1
                if actual['review_reason']=='exact whole-text duplicate':
                    other=actual['duplicate_of_original_position']
                    if other not in raw_texts or raw_texts[other]!=text_hash:
                        raise ValueError('Every discarded exact duplicate must point to identical complete original text')
            if (position+1)%1000==0:
                atomic_json(OUTPUT/'readback-progress.json',dict(status='reconstructing every original educational page and native token byte',
                    original_pages=position+1,total_original_pages=40049,training_admitted=False,goal_achieved=False))
    if any(len(values)!=1 for values in books.values()) or any(len(values)!=1 for values in texts.values()):
        raise ValueError('A whole original book or identical text cannot enter both source roles')
    if any(row['native'] is not None and row['book_component_id'] in question_components for row in index):
        raise ValueError('An original book connected to any named held question entered packing')
    if dict(counts)!=manifest['roles'] or dict(missing)!=manifest['unresolved_publisher_attribution_counts'] or dict(licenses)!=manifest['original_license_counts']:
        raise ValueError('Every actual source role, unresolved attribution and literal license count must match')
    windows={}
    for role in ('train','valid'):
        if cursors[role]!=ends[role][-1]:raise ValueError('Every actual token byte must belong to a reconstructed source page')
        corpus=WholeTokenWindows(OUTPUT,manifest['tokenizer_sha256'],4096,role)
        next_target=1;targets=0
        for i in range(len(corpus)):
            row=corpus[i];start=row['original_first_target_position'];stop=row['original_last_target_position_exclusive']
            if start!=next_target or row['real_next_token_targets']!=stop-start:
                raise ValueError('Every real next-token position must be supervised once, including the source tail')
            mask=[v!=-100 for v in row['labels'][1:]]
            if sum(mask)!=stop-start or [v for v in row['labels'][1:] if v!=-100]!=packed(role,start,stop-start):
                raise ValueError('Actual window targets must equal every original real token; no padded target or silent tail loss')
            next_target=stop;targets+=stop-start
        if next_target!=corpus.total or targets!=corpus.total-1:
            raise ValueError('All original next-token targets must be covered')
        windows[role]=dict(windows=len(corpus),actual_next_token_targets=targets,
            final_real_targets=corpus[len(corpus)-1]['real_next_token_targets'],source_tokens=corpus.total)
    guard.check(full=True)
    report=dict(completed_at=datetime.now(timezone.utc).isoformat(),reader_sha256=file_hash(Path(__file__).resolve()),
        source_manifest_sha256=file_hash(OUTPUT/'corpus.json'),all_source_metadata_and_raw_text_hashes_reconstructed=40049,
        every_packed_page_own_BPE_BOS_text_EOS_and_token_byte_reconstructed=True,
        every_real_window_next_token_target_including_partial_tail_reconstructed=True,actual_windows=windows,
        original_book_and_full_text_cross_role_overlap=0,named_held_book_components_not_packed=True,
        literal_publisher_license_counts=dict(licenses),unresolved_publisher_attribution_counts=dict(missing),
        semantic_independence_not_certified=True,cross_corpus_source_overlap_audit_pending=True,
        attribution_license_version_admission_review_pending=True,training_admitted=False,preparation_model_updates=0,goal_achieved=False)
    atomic_json(destination,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
