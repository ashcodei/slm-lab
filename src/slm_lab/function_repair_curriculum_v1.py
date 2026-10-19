"""Complete826function curriculum with target-matched repair/control views."""
import gzip
import hashlib
import json
from pathlib import Path
import random

from .codenet_token_pack_v1 import CompleteCodePack,FILES
from .corpus import file_hash

ARMS=('plain_repeat','semantic_repair')


def target_pair(original,derived,plain,repaired):
    if (original['split']!='train' or original['source_candidate']['role']!='train'
            or derived['split']!='train' or derived['source_function_id']!=original['id']
            or derived['group_id']!=original['group_id'] or derived['response']!=original['response']
            or derived.get('complete_original_function_target_unchanged') is not True):
        raise ValueError('Same complete source-train function and sealed source group required for both views')
    a_ids,a_labels=plain;b_ids,b_labels=repaired
    def targets(ids,labels):
        if len(ids)!=len(labels) or not 3<=len(ids)<=4096 or ids[0]!=1 or ids[-1]!=2:
            raise ValueError('Complete bounded own native sequence required')
        start=next((i for i,v in enumerate(labels) if v!=-100),len(ids))
        if not 1<=start<len(ids)-1 or labels[:start]!=[-100]*start or labels[start:]!=ids[start:]:
            raise ValueError('Whole original response and true EOS must retain the exact prompt mask')
        return labels[start:]
    if targets(a_ids,a_labels)!=targets(b_ids,b_labels):
        raise ValueError('Both compared views must supervise identical original target/EOS IDs')
    return len(a_ids),len(b_ids),len(targets(a_ids,a_labels))


def schedule(lengths,*,epochs=3,width=16,seed=979113):
    if not lengths or any(type(n) is not int or not 3<=n<=4096 for n in lengths) or min(epochs,width,seed)<1:
        raise ValueError('Complete positive native lengths and explicit whole-epoch settings required')
    result=[]
    for epoch in range(epochs):
        rng=random.Random(seed+epoch);slots=list(range(len(lengths)));rng.shuffle(slots);batches=[]
        for begin in range(0,len(slots),width*32):
            bucket=sorted(slots[begin:begin+width*32],key=lambda p:(lengths[p],p))
            batches.extend([bucket[i:i+width] for i in range(0,len(bucket),width)])
        rng.shuffle(batches)
        if sorted(p for batch in batches for p in batch)!=list(range(len(lengths))):
            raise ValueError('Every full source presentation must occur exactly once per epoch')
        result.extend(batches)
    return result


def collate(sequences,device='cpu'):
    import torch
    if not sequences:raise ValueError('Nonempty complete native source batch required')
    maximum=max(len(ids) for ids,_ in sequences)
    ids=torch.zeros((len(sequences),maximum),dtype=torch.long,device=device)
    labels=torch.full_like(ids,-100);mask=torch.zeros_like(ids,dtype=torch.bool)
    for i,(values,targets) in enumerate(sequences):
        if len(values)!=len(targets) or len(values)>4096:raise ValueError('No source truncation or target loss permitted')
        ids[i,:len(values)]=torch.tensor(values,device=device);labels[i,:len(values)]=torch.tensor(targets,device=device);mask[i,:len(values)]=True
    return dict(input_ids=ids,attention_mask=mask,labels=labels)


class FunctionRepairCurriculum:
    role='train'
    def __init__(self,root,purpose):
        if purpose not in {'optimization','source_preparation'}:raise ValueError('Explicit complete-source purpose required')
        self.root=Path(root).resolve();self.purpose=purpose
        original=self.root/'data/direct-algorithm-function-contracts-admitted-v1'
        repair=self.root/'data/public-function-semantic-repair-admitted-v1'
        self.original=CompleteCodePack(original,'train','training');self.repairs=CompleteCodePack(repair,'train','training')
        admission=json.loads((repair/'admission.json').read_text());proof=Path(admission['independent_readback'])
        if (not admission.get('semantic_repair_SFT_admitted') or file_hash(proof)!=admission['independent_readback_sha256']
                or not json.loads(proof.read_text()).get('all_frozen_selected_native_bytes_masks_original_targets_rights_and_fresh_counterexamples_exact')):
            raise ValueError('Full independent source/mutation/native/public-execution repair admission required')
        all_rows=list(map(json.loads,gzip.open(original/'original-rows.jsonl.gz','rt')))
        self.original_rows=[all_rows[int(i)] for i in self.original.positions]
        self.held_rows=[r for r in all_rows if r['split']=='validation']
        self.repair_rows=list(map(json.loads,gzip.open(repair/'original-rows.jsonl.gz','rt')))
        if len(self.original)!=826 or len(self.held_rows)!=15 or len(self.repairs)!=791:
            raise ValueError('Every826original train function,791admitted repair and15original validation functions required')
        train_groups={r['group_id'] for r in self.original_rows};held_groups={r['group_id'] for r in self.held_rows}
        if train_groups & held_groups:raise ValueError('Original whole source-file roles must stay disjoint')
        self.plain=[self.original.sequence(i) for i in range(len(self.original))]
        self.alternate={};self.alternate_indices={}
        for i,row in enumerate(self.repair_rows):
            position=row['original_native_train_position'];native=self.repairs.sequence(i)
            if position in self.alternate or row['original_global_native_index']!=int(self.original.positions[position]):
                raise ValueError('One exact original train source per admitted second view required')
            target_pair(self.original_rows[position],row,self.plain[position],native)
            self.alternate[position]=native;self.alternate_indices[position]=i
        self.slots=[(i,False) for i in range(826)]+[(i,True) for i in sorted(self.alternate)]
        self.lengths=[];self.targets=[]
        for position,alternate in self.slots:
            plain=self.plain[position];other=self.alternate[position] if alternate else plain
            self.lengths.append(max(len(plain[0]),len(other[0])))
            self.targets.append(sum(v!=-100 for v in plain[1]))
        self.source_files_sha256={str(folder/name):file_hash(folder/name) for folder in [original,repair] for name in [*FILES,'manifest.json','admission.json']}
        self.source_files_sha256[str(proof)]=admission['independent_readback_sha256']
        self.tokenizer_path=original/'tokenizer.json'
        if file_hash(self.tokenizer_path)!=file_hash(repair/'tokenizer.json'):raise ValueError('Both views must share the identical own tokenizer')

    def native(self,arm,slot):
        if arm not in ARMS or type(slot) is not int or not 0<=slot<len(self.slots):raise ValueError('Exact matched arm and original train presentation required')
        position,alternate=self.slots[slot]
        return self.alternate[position] if alternate and arm=='semantic_repair' else self.plain[position]

    def batch(self,arm,slots,device='cpu',*,for_optimization=False):
        if for_optimization and self.purpose!='optimization':raise ValueError('Source preparation cannot supply optimization gradients')
        return collate([self.native(arm,slot) for slot in slots],device)

    def identity(self,entries):
        return hashlib.sha256(json.dumps(dict(source_files_sha256=self.source_files_sha256,slots=self.slots,batches=entries),sort_keys=True,separators=(',',':')).encode()).hexdigest()

    def budget(self,arm,entries):
        slots=[slot for batch in entries for slot in batch]
        return dict(original_plain_functions=826,second_view_source_functions=791,
            complete_presentations=len(slots),complete_context_tokens=sum(len(self.native(arm,p)[0]) for p in slots),
            original_response_EOS_targets=sum(self.targets[p] for p in slots),
            original_source_validation_functions_never_optimized=15)
