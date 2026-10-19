"""Independent static shape/source/scalar readback of the completed999Mprobe.

No neural forward or optimizer replay. Stored actor tensors and actual whole
native sources are read directly; in-memory frozen-base integrity and neural
cold/recovery behavior remain producer observations, not repeated certificates.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exact_waiting_process_retirement_v1 import terminal
from slm_lab.execution_contrast_data_v1 import ExecutionContrastData
from slm_lab.own_code_delta_v1 import matrix_names, norm_names

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/full-machine-preference-CPU-v1'
SAVE=ROOT/'checkpoints/full-machine-preference-CPU-v1'
OUT=ROOT/'reports/full-machine-preference-CPU-v2-readback'


def digest(items):
    result=hashlib.sha256()
    for name,value in items:
        result.update(json.dumps([name,str(value.dtype),list(value.shape)]).encode())
        result.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('A fresh CPU-only static source/policy readback is required')
    torch.set_num_threads(1);plan_path=SOURCE/'plan.json';done_path=SOURCE/'complete.json'
    plan=json.loads(plan_path.read_text());done=json.loads(done_path.read_text())
    if (not terminal(plan['owner']) or not done['complete'] or done['actual_unique_exported_student_values']!=999706442
            or done['actual_selected_source_positions']!=[873,396,27] or not done['actual_nonzero_own_starting_actor_reference_retained']):
        raise ValueError('The exact complete isolated full-size matched probe must finish first')
    bindings={**plan['source_files_sha256'],**done['source_files_sha256'],str(plan_path):file_hash(plan_path),
              str(done_path):file_hash(done_path),str(Path(__file__)):file_hash(__file__)}
    guard=FileBindings(bindings);data=ExecutionContrastData(ROOT,purpose='audit')
    base=torch.load(plan['starting_main_inputs']['checkpoint'],map_location='cpu',weights_only=True,mmap=True)['model']
    named={group+'.'+name:value for group in ['network','heads'] for name,value in base[group].items() if name!='lm_head.weight'}
    if sum(v.numel() for v in named.values())!=984399690:raise ValueError('Every unique original frozen backbone/head value must remain counted')
    actual_base_digest=digest(named.items());reference=torch.load(SAVE/'private-own-starting-actor-reference.pt',map_location='cpu',weights_only=True,mmap=True)
    if reference['binding']['frozen_dense_sha256']!=actual_base_digest:
        raise ValueError('The recorded frozen base must match every actual source parameter')
    expected={};rank=plan['own_actor_settings']['rank'];layers=base['config']['num_layers']
    for i,name in enumerate(matrix_names(layers)):
        shape=base['network'][name].shape
        expected[f'linears.{i}.down']=(rank,shape[1]);expected[f'linears.{i}.up']=(shape[0],rank)
    for i,name in enumerate(norm_names(layers)):expected[f'norms.{i}.weight']=tuple(base['network'][name].shape)
    expected['critic.weight']=(1,base['config']['hidden_size'])
    warm=reference['parts'];policies={}
    for name in ['machine-DPO','chosen-CE']:
        raw=torch.load(SAVE/(name+'-policy.pt'),map_location='cpu',weights_only=True,mmap=True);policies[name]=raw
        if (raw['config']!=base['config'] or raw['settings']!=plan['own_actor_settings']
                or raw['actual_counts']['actual_unique']!=999706442 or raw['frozen_dense_sha256']!=actual_base_digest):
            raise ValueError('Every exact own policy configuration, source-base digest and count must reconstruct')
    rows=[warm,*[p['parts'] for p in policies.values()]]
    for parts in rows:
        if set(parts)!=set(expected) or any(tuple(v.shape)!=expected[k] or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()) for k,v in parts.items()):
            raise ValueError('Every actual actor/norm/critic tensor shape and finite value must reconstruct from the full dense architecture')
        if parts['critic.weight'].count_nonzero():raise ValueError('Every unused separate critic value must remain zero')
    if digest(warm.items())!=reference['binding']['frozen_actor_parts_sha256'] or 984399690+sum(v.numel() for v in warm.values())!=999706442:
        raise ValueError('Exactly one shared frozen base plus the full counted actor/critic must fit under1B')
    private=torch.load(SAVE/'machine-DPO-after1.pt',map_location='cpu',weights_only=True,mmap=True)
    core=private['core'];adam=core['optimizer'];selected=[n for n in expected if n!='critic.weight']
    if (private['own_reference_binding']!=reference['binding'] or core['completed']!=1 or core['planned']!=2
            or core['optimized_names']!=selected or len(adam['state'])!=len(selected) or len(selected)!=561
            or core['cursor']['CPU_mechanics_order']!=[396,27] or core['cursor']['full_source_epoch'] is not False):
        raise ValueError('Exact561leaf actor Adam, limited CPU source cursor and nonzero own reference must reconstruct')
    for index,name in enumerate(selected):
        state=adam['state'][index]
        if float(state['step'])!=1 or any(tuple(state[k].shape)!=expected[name] or state[k].dtype!=torch.float32 or not bool(torch.isfinite(state[k]).all()) for k in ['exp_avg','exp_avg_sq']):
            raise ValueError('Every actual durable first-step actor moment and clock must agree')
    updates=json.loads((SOURCE/'all-actual-update-records.json').read_text())
    if len(updates['machine_DPO'])!=2 or len(updates['chosen_CE'])!=2:raise ValueError('Both complete two-source arms must remain')
    for method in ['machine_DPO','chosen_CE']:
        for position,record in zip([396,27],updates[method]):
            source=data.rows[position];item=record['original_machine_pairs'][0]
            values=[]
            for field in ['chosen','rejected']:
                ids,labels=data.sequence(position,field);values.extend([ids,labels])
                if item[field+'_response_EOS_targets']!=sum(v!=-100 for v in labels[1:]):raise ValueError('Every original complete source/EOS denominator must reconstruct')
            if (record['original_source_positions']!=[position] or item['id']!=source['id']
                    or item['whole_native_pair_sha256']!=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
                    or record['human_preferences'] is not False or record['own_starting_actor_reference_queried']!=(method=='machine_DPO')):
                raise ValueError('Every actual native source, machine attribution and control/reference exposure must reconstruct')
    summaries={}
    for name,key in [('baseline','actual_baseline_source_measurements'),('machine_DPO','actual_machine_DPO_after'),('chosen_CE','actual_chosen_CE_after')]:
        observations=done[key];losses=[];margins=[];preferred=0;chosen_sum=targets=0
        for position,row in zip([396,27],observations):
            pc,pr,qc,qr=[torch.tensor(row[k],dtype=torch.float32) for k in ['policy_chosen_logp','policy_rejected_logp','reference_chosen_logp','reference_rejected_logp']]
            margin=float((pc-pr)-(qc-qr));loss=float(torch.nn.functional.softplus(-plan['training_options']['beta']*torch.tensor(margin)))
            count=sum(v!=-100 for v in data.sequence(position,'chosen')[1][1:])
            if row['position']!=position or row['own_reference_margin']!=margin or row['DPO_loss']!=loss or row['chosen_token_NLL']!=-float(pc)/count:
                raise ValueError('Every actual source preference margin/loss and target normalization must rederive')
            losses.append(loss);margins.append(margin);preferred+=float(pc)>float(pr);chosen_sum-=float(pc);targets+=count
        summaries[name]=dict(actual_pairs=2,mean_DPO_loss=math.fsum(losses)/2,mean_own_reference_margin=math.fsum(margins)/2,
            correct_program_higher_total_log_probability=preferred,chosen_target_mean_NLL=chosen_sum/targets)
    guard.check(full=True);OUT.mkdir()
    atomic_json(OUT/'complete.json',dict(at=datetime.now(timezone.utc).isoformat(),complete=True,
        actual_unique_student_values=999706442,actual_frozen_source_values=984399690,actual_actor_and_critic_tensors=len(expected),
        actual_durable_Adam_leaves=561,actual_changed_actor_tensors={name:sum(not torch.equal(raw['parts'][k],warm[k]) for k in selected) for name,raw in policies.items()},
        every_stored_actor_shape_finite_value_source_mask_and_scalar_measurement_rebuilt=True,
        actual_source_summaries=summaries,both_source_rankings_still_incorrect_under_total_log_probability=True,
        in_memory_frozen_base_integrity_and_neural_replay_cold_proofs_not_repeated=True,
        raw_frozen_base_fingerprint_agrees_with_actual_original_complete_main=True,
        no_new_forward_program_execution_or_optimizer_replay=True,no_generated_or_held_coding_gain_proved=True,
        source_files_sha256={str(p):file_hash(p) for p in [Path(__file__),plan_path,done_path,SAVE/'machine-DPO-after1.pt',
            SAVE/'machine-DPO-policy.pt',SAVE/'chosen-CE-policy.pt',SAVE/'private-own-starting-actor-reference.pt']},
        models_constructed=0,model_updates=0,candidate_promoted=False,goal_achieved=False))
    print(json.dumps(dict(report=str(OUT/'complete.json'),actual_source_summaries=summaries,held_gain_proved=False)),flush=True)


if __name__=='__main__':main()
