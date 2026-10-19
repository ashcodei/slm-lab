"""Pack a bounded public text stream into token shards for large-scale pretraining."""
import argparse
import bisect
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import numpy as np
from tokenizers import Tokenizer
from .data import stable_id


class ShardWriter:
    def __init__(self,root,split,dtype,shard_tokens):
        self.root,self.split,self.dtype,self.shard_tokens=root,split,dtype,shard_tokens
        self.buffer=[];self.files=[];self.tokens=0

    def append(self,tokens):
        for offset in range(0,len(tokens),self.shard_tokens):
            self.buffer.extend(tokens[offset:offset+self.shard_tokens])
            while len(self.buffer)>=self.shard_tokens:
                self.flush(self.shard_tokens)

    def flush(self,n=None):
        n=len(self.buffer) if n is None else n
        if not n:return
        path=self.root/f'{self.split}-{len(self.files):05d}.bin'
        arr=np.asarray(self.buffer[:n],dtype=self.dtype);arr.tofile(path)
        self.files.append({'path':path.name,'tokens':n,
            'sha256':hashlib.sha256(arr.tobytes()).hexdigest()})
        self.tokens+=n;del self.buffer[:n]


class PackedCorpus:
    def __init__(self,root,tokenizer_hash,length,split='train'):
        self.root=Path(root)
        self.manifest=json.loads((self.root/'corpus.json').read_text())
        if self.manifest['tokenizer_sha256']!=tokenizer_hash:
            raise ValueError('Packed corpus tokenizer checksum mismatch.')
        self.dtype=np.dtype(self.manifest['dtype'])
        self.files=self.manifest['splits'][split]['shards']
        self.ends=[];total=0
        for item in self.files:
            expected=item['tokens']*self.dtype.itemsize
            if (self.root/item['path']).stat().st_size!=expected:
                raise ValueError('Packed token shard has an invalid size.')
            total+=item['tokens'];self.ends.append(total)
        self.length=length;self.stride=length-1;self.total=total
        if length<2 or total<length:raise ValueError('Corpus is too small for the sequence length.')

    def __len__(self):return (self.total-1)//self.stride

    @lru_cache(maxsize=8)
    def mapped(self,index):
        return np.memmap(self.root/self.files[index]['path'],dtype=self.dtype,mode='r')

    def __getitem__(self,index):
        if not 0<=index<len(self):raise IndexError(index)
        start=index*self.stride;needed=self.length;pieces=[]
        while needed:
            shard=bisect.bisect_right(self.ends,start)
            preceding=self.ends[shard-1] if shard else 0
            local=start-preceding
            n=min(needed,self.ends[shard]-start)
            pieces.append(np.array(self.mapped(shard)[local:local+n],dtype=np.int64))
            start+=n;needed-=n
        return {'tokens':np.concatenate(pieces).tolist()}


def pack(args):
    if args.max_tokens<1 or args.shard_tokens<1:raise ValueError('Token limits must be positive.')
    root=Path(args.output)
    if root.exists() and any(root.iterdir()):raise ValueError('Use a new corpus output directory.')
    root.mkdir(parents=True,exist_ok=True)
    tokenizer=Tokenizer.from_file(args.tokenizer)
    eos=tokenizer.token_to_id('<eos>')
    if eos is None:raise ValueError('Tokenizer needs an EOS token.')
    if args.jsonl:
        def stream():
            with open(args.jsonl) as source:
                for line in source:
                    if line.strip():yield json.loads(line)
        records=stream()
        source={'local_jsonl':str(Path(args.jsonl).resolve()),
                'sha256':file_hash(args.jsonl)}
    else:
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision=args.revision or HfApi().dataset_info(args.dataset,token=False).sha
        records=load_dataset(args.dataset,args.subset,split=args.split,revision=revision,
                             streaming=True,token=False)
        source={'dataset':args.dataset,'subset':args.subset,'split':args.split,'revision':revision}
    dtype='<u2' if tokenizer.get_vocab_size()<=65536 else '<u4'
    writers={s:ShardWriter(root,s,dtype,args.shard_tokens) for s in ['train','valid']}
    seen=sqlite3.connect(root/'dedup.sqlite')
    seen.execute('CREATE TABLE hashes (id TEXT PRIMARY KEY)')
    total=0;documents=0;duplicates=0;coverage={};source_counts={}
    for row in records:
        text=row.get('text')
        if not isinstance(text,str) or not text.strip():continue
        key=stable_id(text)
        if seen.execute('INSERT OR IGNORE INTO hashes VALUES (?)',(key,)).rowcount==0:
            duplicates+=1;continue
        ids=tokenizer.encode(text).ids+[eos]
        # Stop before a document rather than putting a fragment in both splits.
        if total+len(ids)>args.max_tokens:
            break
        group_key=stable_id(str(row['group_id'])) if row.get('group_id') else key
        split='valid' if int(group_key[:8],16)%10==0 else 'train'
        writers[split].append(ids);total+=len(ids);documents+=1
        domain=row.get('domain','general');coverage.setdefault(domain,{'train':0,'valid':0})[split]+=len(ids)
        origin=row.get('source',args.dataset or 'local');source_counts[origin]=source_counts.get(origin,0)+len(ids)
    seen.commit();seen.close()
    for writer in writers.values():writer.flush()
    if not all(w.tokens for w in writers.values()):
        raise ValueError('Increase token budget to obtain both train and validation documents.')
    manifest={'source':source,'tokenizer_sha256':file_hash(args.tokenizer),'dtype':dtype,
        'documents':documents,'duplicates_removed':duplicates,'total_tokens':total,
        'domain_tokens':coverage,'source_tokens':source_counts,
        'splits':{s:{'tokens':w.tokens,'shards':w.files} for s,w in writers.items()},
        'split_rule':'SHA256 of source group_id when supplied, otherwise normalized document, modulo 10; all chunks of a source document stay together.',
        'requested_max_tokens':args.max_tokens}
    (root/'corpus.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='splits'},indent=2))


def file_hash(path):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        while chunk:=source.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--jsonl');source.add_argument('--dataset')
    p.add_argument('--subset',default='sample-10BT');p.add_argument('--split',default='train')
    p.add_argument('--revision');p.add_argument('--tokenizer',required=True)
    p.add_argument('--output',required=True);p.add_argument('--max-tokens',type=int,required=True)
    p.add_argument('--shard-tokens',type=int,default=64_000_000)
    a=p.parse_args()
    try:pack(a)
    except ValueError as e:p.error(str(e))


if __name__=='__main__':main()
