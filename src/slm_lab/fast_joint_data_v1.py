"""One complete joint curriculum of admitted text, code, words and images.

Families remain homogeneous within an update, so images never need fake
text-only placeholders. Length buckets reduce padding without clipping or
repeating the final partial batch. Every selected original train row is used.
"""
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from tokenizers import Tokenizer

from .codenet_token_pack_v1 import CompleteCodePack,packed_encoder
from .codenet_replay_v1 import ReplayCorpus
from .codenet_image_replay_v1 import ImageReplayCorpus
from .codenet_runner_v1 import VerifiedReplayEncoder
from .corpus import file_hash
from .dense_foundation_data_v1 import DenseFoundationSources
from .shared_word_data_v1 import SharedWordData


class CompleteBucketSchedule:
    """All original sources exactly once; seeded length buckets and interleave."""
    def __init__(self,lengths,widths,seed,bucket_updates=32):
        if (not lengths or set(lengths)!=set(widths) or type(seed) is not int or type(bucket_updates) is not int or bucket_updates<1
                or any(type(w) is not int or w<1 for w in widths.values())
                or any(not values or any(type(n) is not int or n<1 for n in values) for values in lengths.values())):
            raise ValueError('Explicit complete positive family lengths and global batch widths required')
        self.lengths=lengths;self.widths=widths;self.seed=seed;entries=[]
        for number,name in enumerate(sorted(lengths)):
            rng=random.Random(seed+number*1009);positions=list(range(len(lengths[name])));rng.shuffle(positions)
            width=widths[name];window=width*bucket_updates;groups=[]
            for start in range(0,len(positions),window):
                bucket=sorted(positions[start:start+window],key=lambda p:(lengths[name][p],p))
                groups.extend([bucket[i:i+width] for i in range(0,len(bucket),width)])
            rng.shuffle(groups)
            for occurrence,group in enumerate(groups):
                entries.append((Fraction(2*occurrence+1,2*len(groups)),number,name,group))
        self.entries=[(name,group) for _,_,name,group in sorted(entries)]

    def __len__(self):return len(self.entries)

    def at(self,step):
        if type(step) is not int or not 0<=step<len(self):raise IndexError('Complete source schedule exhausted')
        name,positions=self.entries[step];return name,list(positions)

    def rank_positions(self,step,rank,world):
        if type(world) is not int or world<1 or type(rank) is not int or not 0<=rank<world:
            raise ValueError('Exact distributed worker position required')
        name,positions=self.at(step);return name,positions[rank::world]


def collate(batches):
    if not batches:raise ValueError('Nonempty homogeneous complete source batch required')
    image=[b.get('pixel_values') is not None for b in batches]
    if len(set(image))!=1:raise ValueError('Image and text-only rows need distinct homogeneous batches')
    rows=sum(len(b['input_ids']) for b in batches);maximum=max(b['input_ids'].shape[1] for b in batches)
    result=dict(input_ids=torch.zeros(rows,maximum,dtype=torch.long),labels=torch.full((rows,maximum),-100,dtype=torch.long),
        attention_mask=torch.zeros(rows,maximum,dtype=torch.bool));offset=0
    for b in batches:
        n,length=b['input_ids'].shape
        for key in result:result[key][offset:offset+n,:length]=b[key]
        offset+=n
    if image[0]:result['pixel_values']=torch.cat([b['pixel_values'] for b in batches])
    return result


class FastJointSources:
    role='train';purpose='optimization'
    def __init__(self,root,config):
        self.root=Path(root).resolve();self.config=config
        tokenizer=self.root/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
        self.base=DenseFoundationSources(self.root,config,4096)
        folder=self.root/'data/documented-doctest-verified-v1'
        reader=json.loads((self.root/'reports/documented-doctest-verified-v1/independent-complete-source-and-runtime-readback-v1.json').read_text())
        if (not reader['complete'] or not reader['all_reexecuted_examples_passed']
                or reader['admitted_pack_sha256']!=file_hash(folder/'admission.json')
                or reader['manifest_sha256']!=file_hash(folder/'manifest.json')):
            raise ValueError('Every new complete function requires the independent full runtime replay gate')
        self.documented=CompleteCodePack(folder,'train','training')
        if file_hash(folder/'tokenizer.json')!=file_hash(tokenizer):raise ValueError('Exact original own tokenizer required')
        self.documented_encoder=packed_encoder(self.documented,Tokenizer.from_file(str(tokenizer)),config)
        self.words=SharedWordData(self.root/'data/shared-byte-words-v1','train','optimization')
        self.replay=ReplayCorpus(self.root/'data/codenet-replay-v1')
        self.images=ImageReplayCorpus(self.root/'data/codenet-image-replay-v1')
        self.replay_encoder=VerifiedReplayEncoder(Tokenizer.from_file(str(tokenizer)),config,self.replay.source.parent,448)
        self.lengths={};self.targets={};self.kinds={};self.locations={}
        for family in self.base.populations:
            i=list(self.base.populations).index(family);offset=0 if i==0 else self.base.ends[i-1]
            info=[self.base.information(offset+p) for p in range(self.base.populations[family])]
            self.lengths[family]=[r['complete_context_tokens'] for r in info]
            self.targets[family]=[r['response_EOS_targets'] for r in info];self.kinds[family]='text';self.locations[family]=offset
        self.lengths['documented']=[int(r['length']) for r in self.documented.index[self.documented.positions]]
        self.targets['documented']=[int(r['length'])-int(r['target_start']) for r in self.documented.index[self.documented.positions]]
        self.kinds['documented']='text'
        self.lengths['words']=[r['native']['length'] for r in self.words.rows]
        self.targets['words']=[r['native']['length']-r['native']['target_start'] for r in self.words.rows];self.kinds['words']='word'
        for name in ['math','generic_visual','image_reading']:
            family='replay:'+name;items=self.replay.index[self.replay.positions[name]]
            self.lengths[family]=[int(n) for n in items['context_tokens']];self.targets[family]=[int(n) for n in items['target_tokens']]
            self.kinds[family]='text' if name=='math' else 'image'
        for name in self.images.families:
            family='images:'+name;items=self.images.index[self.images.positions[name]]
            self.lengths[family]=[int(n) for n in items['context_tokens']];self.targets[family]=[int(n) for n in items['target_tokens']];self.kinds[family]='image'
        self.populations={name:len(values) for name,values in self.lengths.items()}

    def budget(self):
        return {n:dict(examples=self.populations[n],complete_context_tokens=sum(self.lengths[n]),response_EOS_targets=sum(self.targets[n]),kind=self.kinds[n]) for n in self.populations}

    def batch(self,name,positions,device='cpu'):
        if name not in self.populations or any(type(p) is not int or not 0<=p<self.populations[name] for p in positions):
            raise ValueError('Exact original admitted train family and source positions required')
        if not positions:return None
        if name in self.base.populations:
            batch=collate([self.base.batch(self.locations[name]+p,'cpu') for p in positions])
        elif name=='documented':batch=self.documented_encoder.batch([dict(packed_code_index=p) for p in positions],'response','cpu')
        elif name=='words':batch=self.words.batch(positions,'cpu')
        else:
            group,family=name.split(':',1);corpus=self.replay if group=='replay' else self.images
            rows=[corpus.row(family,p) for p in positions];batch=self.replay_encoder.batch(rows,'response','cpu')
        if int(batch['attention_mask'].sum())!=sum(self.lengths[name][p] for p in positions) or int((batch['labels'][:,1:]!=-100).sum())!=sum(self.targets[name][p] for p in positions):
            raise ValueError('Every complete original context and response/EOS target must match before optimization')
        if torch.device(device).type!='cpu':batch={k:(v.pin_memory().to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
        return batch
