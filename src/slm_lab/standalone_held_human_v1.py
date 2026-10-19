"""Held human preferences on the actual standalone dense student.

Only the student runs neural forwards. The original frozen-own-base reference
scores are reused from the completed actor assessment on the identical source
pairs, masks and precision, with explicit SHA/profile bindings. They are not
new reference forwards or judgments of newly generated model answers.
"""
import json
import math
from pathlib import Path

import torch

from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .dense_human_preference_evaluation_v1 import information,readback,scored,summary
from .dense_human_preference_v1 import sequence_logp
from .diverse_images import atomic_json
from .held_human_shards_v1 import positions,profile
from .own_dense_growth_v1 import OwnDenseAssistant

FORMAT='own-standalone-dense-held-human-cached-own-reference-v1'


class CachedHeldReference:
    def __init__(self,data,path,*,expected_sha256,expected_profile):
        positions(data,0,1)
        path=Path(path);journal=path.with_suffix('.jsonl')
        self.guard=FileBindings({str(path):expected_sha256,str(journal):file_hash(journal)})
        report=json.loads(path.read_text());p=dict(expected_profile);beta=p.pop('DPO_beta')
        if (profile(p,beta)!=expected_profile or report.get('profile')!=expected_profile
                or report.get('DPO_beta')!=beta or report.get('every_original_held_pair_once') is not True
                or report.get('held_gradients') is not False):
            raise ValueError('Exact completed original actor/reference source, runtime and full held-pair profile required')
        readback(data,path)
        self.data=data;self.path=path;self.sha256=expected_sha256;self.profile=dict(expected_profile)
        self.rows=report['results'];self.guard.check(full=True)

    def values(self,position):
        row=self.rows[position];expected=information(self.data,position)
        if any(row.get(k)!=v for k,v in expected.items()):
            raise ValueError('Every cached own-reference score must match the exact complete original held pair and target masks')
        return {f:row['reference_'+f+'_logp'] for f in ['chosen','rejected']}

    def provenance(self):
        return dict(original_actor_report=str(self.path.resolve()),original_actor_report_sha256=self.sha256,
            original_actor_profile=self.profile,reference_scores_reused_not_new_neural_forwards=True,
            old_actor_or_reference_weights_not_loaded=True)


def validate_row(data,row,cache,beta):
    expected=information(data,row['original_held_position'])
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Every complete original held identity and target fingerprint must reconstruct')
    reference=cache.values(row['original_held_position'])
    if (any(row.get('reference_'+f+'_logp')!=reference[f] for f in reference)
            or any(not math.isfinite(row[n+'_'+f+'_logp']) or row[n+'_'+f+'_logp']>0
                   for n in ['policy','reference'] for f in ['chosen','rejected'])
            or any(row.get(k)!=v for k,v in scored(row,beta).items())):
        raise ValueError('Every actual student score, original cached reference and preference/length verdict must reconstruct')


@torch.no_grad()
def evaluate_shard(model,data,path,cache,*,rank,world,provenance,beta=.1,checked=lambda:None):
    selected=positions(data,rank,world);declared=profile(provenance,beta)
    if (type(model) is not OwnDenseAssistant or type(cache) is not CachedHeldReference or cache.data is not data
            or model.counts()['total_unique']>=10**9 or any(p.grad is not None for p in model.parameters())
            or provenance['inference_device_type']!=next(model.parameters()).device.type
            or provenance['inference_precision']!='float32'
            or any(cache.profile[k]!=declared[k] for k in ['human_manifest_sha256','frozen_dense_sha256',
                'inference_device_type','inference_precision','DPO_beta'])):
        raise ValueError('Exact standalone own dense student, identical held source/reference and native FP32 runtime required')
    cache.guard.check(full=True);path=Path(path);journal=path.with_suffix('.jsonl')
    if path.exists() or journal.exists():raise ValueError('Preserve all earlier standalone held deliveries')
    model.eval();signature=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()]
    rows=[];device=next(model.parameters()).device
    with journal.open('x') as stream:
        for index,p in enumerate(selected):
            checked()
            if index%128==0:cache.guard.check()
            row=information(data,p);reference=cache.values(p)
            for field in ['chosen','rejected']:
                batch=data.batch([p],field,device);inputs={k:v for k,v in batch.items() if k!='labels'}
                logits=model(**inputs)['logits'];value,count=sequence_logp(logits,batch['labels'])
                if count!=row[field+'_response_EOS_targets']:raise ValueError('Every original response and EOS target must reach the standalone student')
                row['policy_'+field+'_logp']=float(value);row['reference_'+field+'_logp']=reference[field]
            row.update(scored(row,beta));validate_row(data,row,cache,beta);rows.append(row)
            stream.write(json.dumps(row)+'\n');stream.flush()
    if signature!=[(id(p),p._version,p.data_ptr(),p.requires_grad) for p in model.parameters()] or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Every actual standalone value and clean gradient must remain unchanged')
    cache.guard.check(full=True)
    result=dict(format=FORMAT,rank=rank,world=world,profile=declared,cached_reference=cache.provenance(),
        original_held_positions=selected,raw_journal=str(journal.resolve()),raw_journal_sha256=file_hash(journal),
        results=rows,overall=summary(rows),actual_student_values=model.counts()['total_unique'],
        student_only_neural_inference=True,held_gradients=False,implementation_sha256=file_hash(__file__),goal_achieved=False)
    atomic_json(path,result);return result


def reconcile(data,paths,output,cache):
    paths=[Path(p) for p in paths];output=Path(output)
    if not paths or output.exists() or output.with_suffix('.jsonl').exists():raise ValueError('Fresh complete standalone preference destination required')
    world=len(paths);seen=set();rows={};shared=None;inventory={};count=None
    cache.guard.check(full=True)
    for path in paths:
        shard=json.loads(path.read_text());rank=shard.get('rank');selected=positions(data,rank,world);journal=Path(shard['raw_journal'])
        if (shard.get('format')!=FORMAT or shard.get('world')!=world or rank in seen
                or shard.get('original_held_positions')!=selected or shard.get('cached_reference')!=cache.provenance()
                or shard.get('student_only_neural_inference') is not True or shard.get('held_gradients') is not False
                or shard.get('implementation_sha256')!=file_hash(__file__) or file_hash(journal)!=shard['raw_journal_sha256']
                or shard['results']!=[json.loads(line) for line in journal.open()]):
            raise ValueError('Every exact standalone shard, cached reference and original raw delivery is required')
        current=dict(shard['profile']);beta=current.pop('DPO_beta');current=profile(current,beta)
        if shared is None:shared=current;count=shard['actual_student_values']
        if (current!=shared or shard['actual_student_values']!=count
                or [r['original_held_position'] for r in shard['results']]!=selected
                or any(cache.profile[k]!=current[k] for k in ['human_manifest_sha256','frozen_dense_sha256',
                    'inference_device_type','inference_precision','DPO_beta'])):
            raise ValueError('Every standalone policy, cached original reference and source allocation must match')
        for row in shard['results']:
            validate_row(data,row,cache,beta);rows[row['original_held_position']]=row
        if shard['overall']!=summary(shard['results']):raise ValueError('Every original standalone shard summary must reconstruct')
        seen.add(rank);inventory[str(path.resolve())]=file_hash(path);inventory[str(journal)]=file_hash(journal)
    if seen!=set(range(world)) or set(rows)!=set(range(len(data))):raise ValueError('Every original held pair must occur exactly once')
    ordered=[rows[p] for p in range(len(data))];cache.guard.check(full=True)
    report=dict(DPO_beta=shared['DPO_beta'],overall=summary(ordered),results=ordered,
        by_source={n:summary([r for r in ordered if r['source_blob']==n]) for n in sorted({r['source_blob'] for r in ordered})},
        profile=shared,cached_reference=cache.provenance(),actual_student_values=count,
        every_original_held_pair_once=True,held_gradients=False,student_only_neural_inference=True,
        reference_scores_reused_not_new_reference_forwards=True,original_actor_or_reference_weights_not_loaded=True,
        teacher_other_model_responses_not_own_output_human_judgments=True,ties_receive_half_credit=True,
        source_files_sha256=inventory,neural_scores_not_independently_replayed_by_reconcile=True,
        candidate_promoted=False,goal_achieved=False)
    with output.with_suffix('.jsonl').open('x') as stream:
        for row in ordered:stream.write(json.dumps(row)+'\n')
    atomic_json(output,report);readback(data,output);return report
