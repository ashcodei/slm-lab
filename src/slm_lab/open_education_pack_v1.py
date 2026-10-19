"""Whole educational pages, duplicate-connected book roles and own BPE shards.

This produces a source-bound candidate, not training admission. Original author,
title and license metadata are retained including missing values; no attribution
is invented and unversioned source license declarations remain explicit review.
"""
import bisect
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from tokenizers import Tokenizer

from .codenet_checkpoint_v1 import FileBindings
from .corpus import ShardWriter, file_hash
from .diverse_images import atomic_json
from .whole_human_alignment_v2 import NamedHeldQuestions

FORMAT='whole-original-open-education-book-grouped-own-BPE-candidate-v1'


class BookComponents:
    def __init__(self,books):self.parent={book:book for book in books}
    def find(self,book):
        if self.parent[book]!=book:self.parent[book]=self.find(self.parent[book])
        return self.parent[book]
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)
    def groups(self):
        result=defaultdict(list)
        for book in self.parent:result[self.find(book)].append(book)
        return {key:sorted(value) for key,value in result.items()}


def book_roles(groups,hosts,seed):
    by_host=defaultdict(list)
    for group,books in groups.items():
        counts=Counter(hosts[book] for book in books)
        host=min(counts,key=lambda h:(-counts[h],h));by_host[host].append(group)
    result={}
    for host,items in by_host.items():
        ordered=sorted(items,key=lambda item:hashlib.sha256(f'{seed}:{host}:{item}'.encode()).digest())
        held=0 if len(items)<2 else min(len(items)-1,max(1,round(len(items)*.1)))
        result.update({group:('valid' if i<held else 'train') for i,group in enumerate(ordered)})
    return result


class WholeTokenWindows:
    """Cover every real next-token target, including the final partial window."""
    def __init__(self,root,tokenizer_sha256,length,role):
        self.root=Path(root);manifest=json.loads((self.root/'corpus.json').read_text())
        if role not in ('train','valid') or type(length) is not int or length<2 or manifest['tokenizer_sha256']!=tokenizer_sha256:
            raise ValueError('Explicit source role, native tokenizer and positive context required')
        self.length=length;self.stride=length-1;self.files=manifest['splits'][role]['shards'];self.ends=[];self.maps=[];total=0
        for item in self.files:
            path=self.root/item['path']
            if path.stat().st_size!=item['tokens']*2 or file_hash(path)!=item['sha256']:
                raise ValueError('Every actual complete original token shard required')
            total+=item['tokens'];self.ends.append(total);self.maps.append(np.memmap(path,dtype='<u2',mode='r'))
        if total<2:raise ValueError('At least two real original tokens required')
        self.total=total

    def __len__(self):return math.ceil((self.total-1)/self.stride)

    def __getitem__(self,index):
        if type(index) is not int or not 0<=index<len(self):raise IndexError(index)
        start=index*self.stride;real=min(self.length,self.total-start);remaining=real;offset=start;pieces=[]
        while remaining:
            shard=bisect.bisect_right(self.ends,offset);before=0 if shard==0 else self.ends[shard-1]
            count=min(remaining,self.ends[shard]-offset)
            pieces.extend(self.maps[shard][offset-before:offset-before+count].astype(np.int64).tolist())
            offset+=count;remaining-=count
        return dict(input_ids=pieces+[0]*(self.length-real),labels=pieces+[-100]*(self.length-real),
            attention_mask=[True]*real+[False]*(self.length-real),real_next_token_targets=real-1,
            original_first_target_position=start+1,original_last_target_position_exclusive=start+real)


def prepare(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Preserve existing educational packing; fresh candidate output required')
    blob=root/'data/open-education-source-v1/libretexts-0000.json.gz'
    source_audit=root/'reports/open-education-source-v2/complete-source.json';audit=json.loads(source_audit.read_text())
    if file_hash(blob)!=audit['source_sha256'] or blob.stat().st_size!=audit['stored_bytes']:
        raise ValueError('The complete actual pinned publisher educational source is required')
    token_path=root/'checkpoints/extended-curriculum-v1/joint/tokenizer.json';tokenizer=Tokenizer.from_file(str(token_path))
    if tokenizer.get_vocab_size()!=32768 or [tokenizer.token_to_id(k) for k in ('<pad>','<bos>','<eos>')]!=[0,1,2]:
        raise ValueError('Exact native own32768BPE framing required')
    protection=NamedHeldQuestions(root)
    sources={**protection.files,**{str(p):file_hash(p) for p in (blob,source_audit,token_path,Path(__file__).resolve(),
        root/'data/open-education-source-v1/publisher-metadata.json',root/'src/slm_lab/whole_human_alignment_v2.py',
        root/'src/slm_lab/corpus.py',root/'src/slm_lab/diverse_images.py')}}
    guard=FileBindings(sources)
    with gzip.open(blob,'rt') as stream:records=[json.loads(line) for line in stream]
    if len(records)!=40049 or len({r['id'] for r in records})!=40049:
        raise ValueError('Every unique original educational row required')
    books={r['metadata']['book_url'] for r in records};graph=BookComponents(books)
    hosts={r['metadata']['book_url']:urlsplit(r['metadata']['url']).hostname for r in records}
    primary={};duplicates={};hits={}
    for position,row in enumerate(records):
        digest=hashlib.sha256(row['text'].encode()).hexdigest();row['_text_sha256']=digest
        if row['text'].strip():
            if digest in primary:
                other=primary[digest];duplicates[position]=other
                graph.union(row['metadata']['book_url'],records[other]['metadata']['book_url'])
            else:primary[digest]=position
        found=protection.matches(row['text'])
        if found:hits[position]=found
    groups=graph.groups();roles=book_roles(groups,hosts,947181)
    protected_groups={graph.find(records[p]['metadata']['book_url']) for p in hits}
    group_id={group:hashlib.sha256(json.dumps(books,separators=(',',':')).encode()).hexdigest() for group,books in groups.items()}
    output.mkdir()
    atomic_json(output/'plan.json',dict(created_at=datetime.now(timezone.utc).isoformat(),format=FORMAT,source_files_sha256=sources,
        seed=947181,split='Duplicate-connected full original books; per-host deterministic10percent component holdout when at least2components.',
        original_rows=40049,original_books=len(books),duplicate_connected_book_components=len(groups),
        protected_named_questions=protection.counts,book_components_with_literal_held_question=len(protected_groups),
        all_page_author_license_title_URL_metadata_retained=True,no_invented_authors=True,training_admitted=False,goal_achieved=False))
    writers={role:ShardWriter(output,role,np.dtype('<u2'),4*1024**2) for role in ('train','valid')}
    counts=Counter();missing=Counter();licenses=Counter();role_domains=defaultdict(Counter);book_assignments={}
    index=output/'original-pages.jsonl'
    with index.open('x') as stream:
        for position,row in enumerate(records):
            meta=row['metadata'];book=meta['book_url'];group=graph.find(book);role=roles[group]
            attribution=[]
            if not meta['author']:attribution.append('publisher author missing')
            if not meta['title']:attribution.append('publisher title missing')
            if meta['license']=='GNU Free Documentation License':attribution.append('publisher GFDL version unrecorded')
            for issue in attribution:missing[issue]+=1
            licenses[meta['license']]+=1
            reason=('whole book component contains named held question' if group in protected_groups else
                    'exact whole-text duplicate' if position in duplicates else 'empty source text' if not row['text'].strip() else None)
            native=None
            if reason:counts['review:'+reason]+=1
            else:
                body=tokenizer.encode(row['text']).ids;tokens=[1,*body,2]
                offset=writers[role].tokens+len(writers[role].buffer);writers[role].append(tokens)
                native=dict(role=role,offset=offset,length=len(tokens),full_original_text_tokens=len(body))
                counts[role]+=1;role_domains[role][urlsplit(meta['url']).hostname]+=1
                book_assignments[book]=role
            record=dict(original_source_position=position,id=row['id'],source=row['source'],created=row['created'],added=row['added'],metadata=meta,
                full_original_text_sha256=row['_text_sha256'],full_original_text_bytes=len(row['text'].encode()),
                book_component_id=group_id[group],original_book_role=role,native=native,review_reason=reason,
                duplicate_of_original_position=duplicates.get(position),named_held_question_families=hits.get(position,[]),
                attribution_review=attribution,training_admitted=False)
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            if (position+1)%1000==0:
                atomic_json(output/'progress.json',dict(status='packing complete original educational pages with original book roles and metadata',original_pages=position+1,
                    total_pages=40049,packed_counts=dict(counts),training_admitted=False,goal_achieved=False))
    for writer in writers.values():writer.flush()
    guard.check(full=True)
    splits={}
    for role,writer in writers.items():
        total=writer.tokens;full,remainder=divmod(total-1,4095)
        splits[role]=dict(documents=counts[role],tokens=total,shards=writer.files,
            original_books=sum(v==role for v in book_assignments.values()),domains=dict(role_domains[role]),
            full4096token_windows=full,remaining_real_next_token_targets=remainder,
            whole_target_windows=full+bool(remainder),all_real_next_token_targets=total-1,
            final_window_right_padding_tokens=4095-remainder if remainder else 0)
    manifest=dict(completed_at=datetime.now(timezone.utc).isoformat(),format=FORMAT,plan_sha256=file_hash(output/'plan.json'),
        tokenizer_sha256=file_hash(token_path),dtype='<u2',splits=splits,source_files_sha256=sources,
        original_page_index_sha256=file_hash(index),original_rows=40049,original_books=len(books),
        duplicate_connected_book_components=len(groups),roles=dict(counts),original_license_counts=dict(licenses),
        unresolved_publisher_attribution_counts=dict(missing),no_invented_author_or_license_values=True,
        complete_unmodified_original_text_and_BOS_EOS_packed=True,no_tail_target_silently_dropped=True,
        train_valid_book_or_exact_text_overlap=0,named_held_book_components_outside_packing=True,
        source_author_license_version_and_cross_corpus_overlap_admission_review_still_required=True,
        source_semantic_independence_not_certified=True,training_admitted=False,preparation_model_updates=0,goal_achieved=False)
    atomic_json(output/'corpus.json',manifest);return manifest
