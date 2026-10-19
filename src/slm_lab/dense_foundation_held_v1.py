"""Complete held book/code/human teacher metrics for dense foundation arms."""
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from .admitted_education_v1 import EducationWindows
from .codenet_token_pack_v1 import CompleteCodePack,packed_encoder
from .corpus import file_hash
from .diverse_images import atomic_json
from .whole_human_data_v2 import WholeHumanData


def target_digest(values):return hashlib.sha256(np.asarray(values,dtype='<u2').tobytes()).hexdigest()


class DenseHeldSources:
    role='valid';purpose='evaluation';names=('education','variety','human')

    def __init__(self,root,config,education_context=4096):
        root=Path(root).resolve();token=root/'checkpoints/extended-curriculum-v1/joint/tokenizer.json'
        self.education=EducationWindows(root/'data/open-education-bookpack-v1',file_hash(token),education_context,'valid','evaluation')
        self.code=CompleteCodePack(root/'data/codenet-durable-pack-v2','validation',purpose='audit')
        self.encoder=packed_encoder(self.code,Tokenizer.from_file(str(token)),config)
        self.human=WholeHumanData(root/'data/whole-human-alignment-v2','dialogue','valid','evaluation')
        self.populations=dict(education=len(self.education),variety=len(self.code),human=len(self.human))

    def batch(self,name,position,device='cpu'):
        if name=='education':return self.education.batch([position],device)
        if name=='variety':return self.encoder.batch([dict(packed_code_index=position)],'response',device)
        if name=='human':return self.human.batch([position],'response',device)
        raise ValueError('Exact complete held original source family required')

    def information(self,name,position):
        batch=self.batch(name,position,'cpu');targets=batch['labels'][:,1:];targets=targets[targets!=-100].tolist()
        info=dict(family=name,original_held_position=position,complete_context_tokens=int(batch['attention_mask'].sum()),
                  actual_next_or_response_EOS_targets=len(targets),target_ids_sha256=target_digest(targets))
        if name=='education':
            row=self.education.window(position)
            info.update(original_first_target_position=row['original_first_target_position'],original_last_target_position_exclusive=row['original_last_target_position_exclusive'])
        if name=='human':info['original_source_id']=self.human.rows[position]['id']
        return info


def summary(records):
    targets=sum(r['actual_next_or_response_EOS_targets'] for r in records)
    return dict(examples=len(records),actual_next_or_response_EOS_targets=targets,
        token_mean_NLL=math.fsum(r['per_target_NLL_sum'] for r in records)/targets,
        teacher_argmax_accuracy=sum(r['teacher_correct'] for r in records)/targets,
        complete_context_tokens=sum(r['complete_context_tokens'] for r in records))


@torch.no_grad()
def evaluate(model,data,output,amp=None,checked=lambda:None):
    if type(data) is not DenseHeldSources or (data.role,data.purpose)!=('valid','evaluation'):
        raise ValueError('All original held sources with explicit evaluation-only role required')
    output=Path(output);journal=output.with_suffix('.jsonl')
    if output.exists() or journal.exists():raise ValueError('Preserve previous complete or partial held teacher deliveries')
    model.eval();amp=amp or nullcontext;device=next(model.parameters()).device;records=[]
    with journal.open('x') as stream:
        for name in data.names:
            for position in range(data.populations[name]):
                checked();batch=data.batch(name,position,device);inputs={k:v for k,v in batch.items() if k!='labels'}
                with amp():logits=model(**inputs)['logits'][:,:-1]
                target=batch['labels'][:,1:];mask=target!=-100;actual=target[mask].cpu().tolist()
                nll=F.cross_entropy(logits.float().reshape(-1,model.config.vocab_size),target.reshape(-1),ignore_index=-100,reduction='none').reshape(target.shape)[mask]
                if not len(actual) or not bool(torch.isfinite(nll).all()):raise ValueError('Every complete held target needs a finite actual neural likelihood')
                info=data.information(name,position)
                if info['actual_next_or_response_EOS_targets']!=len(actual) or info['target_ids_sha256']!=target_digest(actual):
                    raise ValueError('Actual held target masks must equal the original complete source tokens')
                row=dict(**info,per_target_NLL_sum=math.fsum(nll.cpu().tolist()),teacher_correct=int((logits.argmax(-1)[mask]==target[mask]).sum()))
                stream.write(json.dumps(row)+'\n');stream.flush();records.append(row);del logits,nll
                if (position+1)%100==0:print('complete_held_foundation',name,position+1,'/',data.populations[name],flush=True)
    report=dict(source_populations=data.populations,overall=summary(records),by_source={n:summary([r for r in records if r['family']==n]) for n in data.names},
        results=records,all_original_held_windows_responses_and_EOS=True,teacher_metrics_not_free_generated_capability=True,
        source_or_semantic_independence_not_certified=True,held_gradients=False,candidate_promoted=False,goal_achieved=False)
    atomic_json(output,report);return report


def readback(data,path):
    path=Path(path);report=json.loads(path.read_text());rows=report['results']
    if report['source_populations']!=data.populations or rows!=[json.loads(l) for l in path.with_suffix('.jsonl').open()]:
        raise ValueError('Every complete held source and flushed original delivery required')
    expected=[(name,p) for name in data.names for p in range(data.populations[name])]
    if [(r['family'],r['original_held_position']) for r in rows]!=expected:raise ValueError('Every original held position exactly once in fixed source order required')
    for row in rows:
        wanted=data.information(row['family'],row['original_held_position'])
        if (any(row[k]!=v for k,v in wanted.items()) or not math.isfinite(row['per_target_NLL_sum']) or row['per_target_NLL_sum']<0
                or not 0<=row['teacher_correct']<=row['actual_next_or_response_EOS_targets']):
            raise ValueError('Every held original complete target/hash/context and finite teacher aggregate must reconstruct')
    if report['overall']!=summary(rows) or report['by_source']!={n:summary([r for r in rows if r['family']==n]) for n in data.names}:
        raise ValueError('Every full held source and token-weighted teacher aggregate must reconstruct')
    return dict(overall=report['overall'],by_source=report['by_source'],all_actual_original_target_masks_hashes_and_record_aggregates_reconstructed=True,
                numerical_neural_logits_not_replayed=True,report_sha256=file_hash(path))
