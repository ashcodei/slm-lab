"""Read both complete-stage policy snapshots and their actual original sources.

The tested stage validator reconstructs source/FP32 metric arithmetic. This
reader does not independently replay the neural optimizer or inference.
"""
import json
import math
import os
from pathlib import Path

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.code_residual_checkpoint_v1 import _validate_adam,_validate_rng
from slm_lab.corpus import file_hash
from slm_lab.dense_human_preference_stage_v1 import validate_metric
from slm_lab.dense_human_preference_v1 import pair_information
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.own_code_delta_v1 import matrix_names,norm_names
from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,DenseBackbone
from slm_lab.own_dense_actor_v1 import budget
from slm_lab.whole_human_data_v2 import WholeHumanData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/dense-human-preference-stage-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU source/optimizer readback must explicitly hide GPUs')
    output=REPORTS/'independent-complete-readback.json'
    if output.exists():raise ValueError('Preserve original readback')
    torch.set_num_threads(2)
    plan=json.loads((REPORTS/'plan.json').read_text());proof=json.loads((REPORTS/'complete.json').read_text())
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    if file_hash(plan['combined_base_checkpoint'])!=plan['combined_base_sha256']:raise ValueError('Exact combined source base required')
    base=torch.load(plan['combined_base_checkpoint'],map_location='cpu',weights_only=True,mmap=True)['model']
    with torch.device('meta'):
        meta=OwnDenseAssistant(DenseBackbone(ModelConfig(**base['config'])),**base['settings'])
    counts=budget(meta,plan['actor_settings']['rank']);counts['actual_unique']=counts['total_unique']
    if counts!=proof['actual_counts'] or counts['actual_unique']!=999706442:raise ValueError('Actual complete one-base policy/critic inventory required')
    frozen=tensor_digest((n,base[n.split('.',1)[0]][n.split('.',1)[1]]) for n,_ in meta.named_parameters())
    if frozen!=proof['frozen_combined_dense_sha256']:raise ValueError('Every saved combined dense/word/auxiliary value must reconstruct its frozen digest')
    rank=plan['actor_settings']['rank'];shapes={};dtypes={}
    for i,name in enumerate(matrix_names(meta.config.num_layers)):
        t=base['network'][name]
        for suffix,shape in [('down',(rank,t.shape[1])),('up',(t.shape[0],rank))]:
            n=f'linears.{i}.{suffix}';shapes[n]=shape;dtypes[n]=t.dtype
    for i,name in enumerate(norm_names(meta.config.num_layers)):
        n=f'norms.{i}.weight';shapes[n]=tuple(base['network'][name].shape);dtypes[n]=base['network'][name].dtype
    names=list(shapes);shapes['critic.weight']=(1,meta.config.hidden_size);dtypes['critic.weight']=torch.float32
    if len(names)!=561 or sum(math.prod(shapes[n]) for n in names)!=15305216:raise ValueError('Every actual actor matrix/norm and state required')
    data=WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences','train','optimization');setting=plan['training_options'];order=EpochOrder(len(data),setting['seed'])
    source_positions=[order.at(i) for i in range(2)]
    if [pair_information(data,p) for p in source_positions]!=proof['complete_selected_original_sources']:
        raise ValueError('The actual declared original shuffled sources must reconstruct without short-example selection')
    expected_sources=dict(full_stage_CPU_plan_sha256=file_hash(REPORTS/'plan.json'),combined_base_sha256=plan['combined_base_sha256'],
        human_admission_sha256=file_hash(ROOT/'data/whole-human-alignment-v2/admission.json'))
    last=None
    for step in (1,2):
        path=REPORTS/f'resume-step{step}.pt'
        if file_hash(path)!=proof[f'retained_step{step}_sha256']:raise ValueError('Both actual complete stage snapshots required')
        saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True);core=saved['core'];model=core['model'];parts=model['parts']
        cursor=dict(completed_updates=step,source_population=len(data),complete_passes=0,next_pass_sources=step,
            actual_pair_draws=step,repeated_rounding_pairs=0,seed=setting['seed'],world_size=1)
        if (saved['options']!=setting or saved['source_population']!=len(data) or saved['frozen_dense_sha256']!=frozen
                or core['completed']!=step or core['planned']!=setting['steps'] or core['cursor']!=cursor
                or core['optimized_names']!=names or core['groups']!=['actor'] or core['world_size']!=1
                or core['metrics']!=proof['actual_complete_stage_metrics'][:step] or model['sources']!=expected_sources
                or model['frozen_dense_sha256']!=frozen or model['actual_counts']!=counts or parts.keys()!=shapes.keys()):
            raise ValueError('Exact stage/options/source cursor/frozen representation and complete policy model required')
        for name,t in parts.items():
            if tuple(t.shape)!=shapes[name] or t.dtype!=dtypes[name] or not bool(torch.isfinite(t).all()):
                raise ValueError('Every saved actor/critic tensor must have its finite source-derived shape/dtype')
        if parts['critic.weight'].count_nonzero()!=0:raise ValueError('The separate unused DPO critic must stay zero')
        _validate_adam(core['optimizer'],[parts[n] for n in names],step);_validate_rng(core['rng'],'cpu')
        for i,record in enumerate(core['metrics']):validate_metric(data,[source_positions[i]],setting,record,i)
        last=model
    checkpoint=REPORTS/'trained-disposable-policy.pt'
    if file_hash(checkpoint)!=proof['trained_policy_sha256']:raise ValueError('Exact actual cold-loaded policy required')
    export=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if (any(export[k]!=last[k] for k in ('format','sources','settings','actual_counts','config','frozen_dense_sha256'))
            or any(not torch.equal(t,last['parts'][n]) for n,t in export['parts'].items())):
        raise ValueError('Every actual final policy export value must equal its complete optimizer snapshot')
    guard.check(full=True)
    atomic_json(output,dict(reader_sha256=file_hash(__file__),complete_sha256=file_hash(REPORTS/'complete.json'),actual_counts=counts,
        actual_actor_states=561,actual_actor_values=15305216,
        both_complete_stage_snapshots_actor_values_Adam_CPU_RNG_source_cursors_checked=True,
        every_original_fused_dense_word_auxiliary_value_reconstructs_frozen_reference_digest=True,
        complete_actual_original_shuffled_sources_reconstructed=proof['complete_selected_original_sources'],
        source_metric_arithmetic_reconstructed_with_tested_stage_validator=True,
        every_final_export_value_equal_complete_step2_snapshot=True,unused_separate_critic_stays_zero=True,
        original_probe_replayed_actual_next_update_and_full_size_cold_reload=True,
        reader_did_not_replay_neural_forward_optimizer_or_generation=True,
        CPU_accumulation1_does_not_prove_production4_GPU_capacity=True,
        all_probe_updates_disposable_not_complete_epoch_or_held_alignment_gain=True,candidate_promoted=False,goal_achieved=False))
    print(output)


if __name__=='__main__':main()
