"""Independent complete policy-state, source-mask and DPO arithmetic readback.

No neural forward/update is replayed here. The original full-size probe owns
that evidence. The selected CPU ranking is explicitly a very short greeting.
"""
import json
import math
import os
from pathlib import Path

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_residual_checkpoint_v1 import _validate_adam,_validate_rng
from slm_lab.corpus import file_hash
from slm_lab.dense_human_preference_v1 import pair_information
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.own_dense_actor_v1 import FORMAT
from slm_lab.own_code_delta_v1 import matrix_names,norm_names
from slm_lab.own_dense_growth_v1 import DenseBackbone,OwnDenseAssistant
from slm_lab.whole_human_data_v2 import WholeHumanData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-human-actor-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Policy source/optimizer reader must explicitly hide GPUs')
    output=REPORTS/'independent-complete-readback.json'
    if output.exists():raise ValueError('Preserve original policy readback')
    torch.set_num_threads(2)
    plan=json.loads((REPORTS/'plan.json').read_text());proof=json.loads((REPORTS/'complete.json').read_text())
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    raw=torch.load(plan['initial_dense_settings']['initializer'],map_location='cpu',weights_only=True,mmap=True)
    if file_hash(plan['initial_dense_settings']['initializer'])!=plan['initial_dense_settings']['initializer_sha256']:
        raise ValueError('Exact original initial dense source required')
    with torch.device('meta'):
        base=OwnDenseAssistant(DenseBackbone(ModelConfig(**raw['config'])),**raw['settings'])
    # The policy factory supports real CPU/CUDA models, not meta devices. Read
    # every factor/norm/critic shape directly from durable original matrices;
    # this reader never needs CUDA or a second real neural model.
    rank=plan['actor_settings']['rank'];shapes={};dtypes={}
    for i,n in enumerate(matrix_names(base.config.num_layers)):
        weight=raw['network'][n];out_size,in_size=weight.shape
        for suffix,shape in [('down',(rank,in_size)),('up',(out_size,rank))]:
            name=f'linears.{i}.{suffix}';shapes[name]=shape;dtypes[name]=weight.dtype
    for i,n in enumerate(norm_names(base.config.num_layers)):
        name=f'norms.{i}.weight';shapes[name]=tuple(raw['network'][n].shape);dtypes[name]=raw['network'][n].dtype
    names=list(shapes);actor_values=sum(math.prod(shape) for shape in shapes.values())
    shapes['critic.weight']=(1,base.config.hidden_size);dtypes['critic.weight']=raw['network']['embedding.weight'].dtype
    total=base.counts()['total_unique']+actor_values+base.config.hidden_size
    counts=dict(frozen_dense_and_all_heads=base.counts()['total_unique'],actor=actor_values,
        separate_policy_critic=base.config.hidden_size,total_unique=total,actual_unique=total)
    if (counts!=proof['actual_counts'] or counts['actual_unique']!=999706442
            or actor_values!=15305216 or len(names)!=561 or counts['separate_policy_critic']!=1536):
        raise ValueError('Every actual dense/head/actor/separate critic value and all561actor tensors required')
    original=tensor_digest((n,raw[n.split('.',1)[0]][n.split('.',1)[1]]) for n,_ in base.named_parameters())
    if original!=proof['original_dense_sha256'] or original!=plan['original_dense_source_digest']:
        raise ValueError('Every original saved dense/auxiliary tensor must reconstruct its actual frozen reference digest')
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization')
    position=plan['original_human_ranking']['original_train_position'];wanted=pair_information(data,position)
    if wanted!=plan['original_human_ranking'] or wanted!=proof['complete_original_human_train_ranking']:
        raise ValueError('The actual complete original admitted human-ranking identity and masks must reconstruct')
    expected_sources=dict(full_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),
        initial_dense_sha256=plan['initial_dense_settings']['initializer_sha256'],
        human_admission_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json'))
    last=None
    for step in (1,2):
        path=REPORTS/f'resume-step{step}.pt'
        if file_hash(path)!=proof[f'retained_step{step}_sha256']:raise ValueError('Both original durable policy/Adam snapshots required')
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True);saved=state['model']
        if (state['completed']!=step or state['planned']!=2 or state['world_size']!=1 or state['groups']!=['actor']
                or state['optimized_names']!=names or state['metrics']!=proof['complete_source_DPO_metrics'][:step]
                or state['cursor']!=dict(original_human_ranking_position=position,completed_updates=step)
                or saved['format']!=FORMAT or saved['settings']!=plan['actor_settings'] or saved['config']!=raw['config']
                or saved['sources']!=expected_sources or saved['frozen_dense_sha256']!=original or saved['actual_counts']!=counts):
            raise ValueError('Every original policy model/source/feature inventory and complete source cursor must match')
        parts=saved['parts']
        if (parts.keys()!=shapes.keys() or any(tuple(t.shape)!=shapes[n] or t.dtype!=dtypes[n]
                or not bool(torch.isfinite(t).all()) for n,t in parts.items()) or parts['critic.weight'].count_nonzero()!=0):
            raise ValueError('Every policy/critic tensor must be finite with its exact shape/dtype, and unused critic remains zero')
        _validate_adam(state['optimizer'],[parts[n] for n in names],step);_validate_rng(state['rng'],'cpu')
        for record in state['metrics']:
            if (record['original_source_positions']!=[position] or record['DPO_beta']!=plan['DPO_beta']
                    or record['chosen_CE_weight']!=plan['chosen_CE_weight'] or len(record['original_human_pairs'])!=1
                    or any(record[k]!=wanted[k] for k in ('chosen_response_EOS_targets','rejected_response_EOS_targets','complete_context_tokens'))):
                raise ValueError('Every complete original human source/target count and fixed preference objective required')
            row=record['original_human_pairs'][0]
            if any(row[k]!=v for k,v in wanted.items()):raise ValueError('Every original human row/token fingerprint must match')
            for key in ('policy_chosen_logp','policy_rejected_logp','reference_chosen_logp','reference_rejected_logp'):
                if not math.isfinite(row[key]) or row[key]>0:raise ValueError('Finite original complete-response log probabilities required')
            margin=(row['policy_chosen_logp']-row['policy_rejected_logp'])-(row['reference_chosen_logp']-row['reference_rejected_logp'])
            x=-plan['DPO_beta']*margin;dpo=max(x,0)+math.log1p(math.exp(-abs(x)))
            if (not math.isclose(row['implicit_preference_margin'],margin,abs_tol=3e-6,rel_tol=3e-6)
                    or not math.isclose(row['pair_DPO'],dpo,abs_tol=3e-6,rel_tol=3e-6)
                    or record['pair_mean_DPO']!=row['pair_DPO']
                    or record['chosen_token_mean_NLL']!=-row['policy_chosen_logp']/wanted['chosen_response_EOS_targets']
                    or record['summed_objective']!=row['pair_DPO']-plan['chosen_CE_weight']*row['policy_chosen_logp']/wanted['chosen_response_EOS_targets']):
                raise ValueError('Actual complete chosen/rejected/reference DPO and chosen-token objective must reconstruct')
        last=saved
    checkpoint=REPORTS/'trained-disposable-policy.pt'
    if file_hash(checkpoint)!=proof['trained_policy_sha256']:raise ValueError('The actual cold-loaded disposable policy artifact required')
    final=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if (any(final[k]!=last[k] for k in ('format','settings','config','sources','actual_counts','frozen_dense_sha256'))
            or any(not torch.equal(t,last['parts'][n]) for n,t in final['parts'].items())):
        raise ValueError('Every exported trained policy/critic value must equal the complete final Adam snapshot')
    prefix=data.sequence(position,'chosen')[0][:data.rows[position]['native']['chosen']['target_start']]
    tokens=proof['actual_stochastic_rollout_tokens'][0];logps=proof['actual_stochastic_rollout_logps'][0]
    values=proof['actual_separate_critic_values_in_rollout'][0];mask=proof['actual_rollout_response_mask'][0]
    if (tokens[:len(prefix)]!=prefix or not 1<=len(tokens)-len(prefix)<=4 or len(tokens)-len(prefix)!=len(logps)
            or len(values)!=len(logps) or len(mask)!=len(logps) or not all(mask)
            or any(not math.isfinite(v) or v>0 for v in logps) or any(v!=0 for v in values)
            or any(type(v) is not int or not 0<=v<base.config.vocab_size for v in tokens)):
        raise ValueError('The actual source prompt, sampled token IDs/logps and separate zero critic deliveries must reconcile')
    guard.check(full=True)
    atomic_json(output,dict(reader_sha256=file_hash(__file__),original_complete_sha256=file_hash(REPORTS/'complete.json'),
        actual_counts=counts,actual_actor_Adam_states=561,actual_actor_values=15305216,
        all_original_saved_dense_and_auxiliary_tensors_reconstruct_frozen_feature_digest=True,
        both_policy_snapshots_all_Adam_CPU_RNG_sources_and_complete_human_masks_checked=True,
        all_saved_policy_tensors_exact_shape_dtype_finite_and_unused_critic_zero=True,
        every_original_pair_fingerprint_complete_logp_and_DPO_CE_arithmetic_reconstructed=True,
        all_exported_trained_policy_values_equal_complete_final_optimizer_snapshot=True,
        actual_sampled_source_prompt_token_ids_probabilities_and_separate_critic_deliveries_reconstructed=True,
        source_is_explicitly_short_greeting_Hi_period_vs_Hi_not_a_substantive_alignment_test=True,
        original_probe_replayed_next_actual_update_and_full_size_cold_reload=True,
        reader_did_not_replay_neural_forward_optimizer_or_sampling=True,
        whole_human_epoch_GPU_capacity_held_alignment_PPO_and_high_end_quality_unfinished=True,
        candidate_promoted=False,goal_achieved=False))
    print(output)


if __name__=='__main__':main()
