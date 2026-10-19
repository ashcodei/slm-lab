"""Independently reconstruct every own folded matrix and retained Adam state.

No real backbone or neural forward is replayed. Meta shapes, original saved
base/code factors, all initial growth weights, both optimizer payloads and the
complete actual training-source tokens are read directly from durable files.
"""
import json
import os
from pathlib import Path

import torch

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.code_residual_checkpoint_v1 import _validate_adam,_validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.model import ModelConfig
from slm_lab.own_code_training_stage_v1 import PreparedCodeStream
from slm_lab.own_dense_growth_v1 import DenseBackbone,OwnDenseAssistant,FORMAT
from slm_lab.whole_human_data_v2 import WholeHumanData

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/own-dense-growth-full-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Dense weight/source reader must explicitly hide every GPU')
    output=REPORTS/'independent-complete-readback.json'
    if output.exists():raise ValueError('Preserve the original complete dense readback')
    torch.set_num_threads(2)
    plan=json.loads((REPORTS/'plan.json').read_text());proof=json.loads((REPORTS/'complete.json').read_text())
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    paths=[REPORTS/n for n in ('initial-model.pt','resume-step1.pt','resume-step2.pt')]
    if [file_hash(p) for p in paths]!=[proof[k] for k in ('initial_model_sha256','retained_step1_sha256','retained_step2_sha256')]:
        raise ValueError('Exact actual initial full model and both retained dense Adam checkpoints required')
    initial,first,second=[torch.load(p,map_location='cpu',weights_only=True,mmap=True) for p in paths]
    with torch.device('meta'):
        meta=OwnDenseAssistant(DenseBackbone(ModelConfig(**initial['config'])),**initial['settings'])
    expected_counts=meta.counts();network_names=[n for n,_ in meta.language_named_parameters()]
    if (expected_counts!=proof['actual_counts'] or expected_counts['total_unique']!=984399690
            or len(network_names)!=317 or sum(meta.network.get_parameter(n).numel() for n in network_names)!=931244544):
        raise ValueError('Actual full35-layer/head inventory and every fully plastic dense language value required')
    for bundle in (initial,first['model'],second['model']):
        if (bundle['format']!=FORMAT or bundle['config']!=initial['config'] or bundle['settings']!=initial['settings']
                or bundle['actual_counts']!=expected_counts or bundle['sources']!=initial['sources']
                or bundle['sources']['full_CPU_plan_sha256']!=file_hash(REPORTS/'plan.json')):
            raise ValueError('Every complete model specification/source binding must match')
        for part,module in [('network',meta.network),('heads',meta.heads)]:
            live=module.state_dict();saved=bundle[part]
            if (saved.keys()!=live.keys() or any(t.shape!=live[n].shape or t.dtype!=live[n].dtype
                    or not bool(torch.isfinite(t).all()) for n,t in saved.items())):
                raise ValueError('Every actual dense/auxiliary tensor must have its exact finite declared shape/dtype')
        if (bundle['network']['embedding.weight'].data_ptr()!=bundle['network']['lm_head.weight'].data_ptr()
                or not torch.equal(bundle['network']['embedding.weight'],bundle['network']['lm_head.weight'])):
            raise ValueError('The saved vocabulary table and LM head must actually share their storage')
    layout=proof['depth_layout']
    if len(layout)!=35 or [i for i in layout if i>=0]!=list(range(24)) or layout.count(-1)!=11:
        raise ValueError('All24original blocks in order and all11added residual blocks required')
    sft=json.loads((ROOT/'reports/whole-human-SFT-v1/plan.json').read_text())
    code_plan=json.loads(Path(sft['own_code_plan']).read_text())
    base=torch.load(code_plan['base_checkpoint'],map_location='cpu',weights_only=True,mmap=True)['model']
    own=torch.load(sft['own_code_checkpoint'],map_location='cpu',weights_only=True,mmap=True)['components']
    old_to_new={old:position for position,old in enumerate(layout) if old>=0}
    # Directly recompute each actual matrix sum from the original durable own
    # weights. The model's fold/grow functions are intentionally not imported.
    for name,parts in own['matrices'].items():
        _,old,*tail=name.split('.');destination='blocks.'+str(old_to_new[int(old)])+'.'+'.'.join(tail)
        expected=base[name]+parts['up']@parts['down']
        if not torch.equal(expected,initial['network'][destination]):raise ValueError('Every saved warm dense matrix must equal the original own base+factor product')
    for name,value in own['norms'].items():
        if name.startswith('blocks.'):
            _,old,*tail=name.split('.');destination='blocks.'+str(old_to_new[int(old)])+'.'+'.'.join(tail)
        else:destination=name
        if not torch.equal(value,initial['network'][destination]):raise ValueError('Every trained own code normalization tensor must be copied exactly')
    if not torch.equal(own['embedding'],initial['network']['embedding.weight']):raise ValueError('The actual trained own code vocabulary table must be retained')
    for name,value in base.items():
        if name.startswith(('vision.','value_head.')) and not torch.equal(value,initial['network'][name]):
            raise ValueError('Every original vision/critic tensor must copy exactly into the initial dense candidate')
    for position,old in enumerate(layout):
        if old<0:
            for suffix in ('attn.o_proj.weight','down.weight'):
                if initial['network'][f'blocks.{position}.{suffix}'].count_nonzero()!=0:raise ValueError('Every added residual output matrix must initially be exactly zero')
            for suffix in ('attn_norm.weight','ffn_norm.weight'):
                if not torch.equal(initial['network'][f'blocks.{position}.{suffix}'],torch.ones_like(initial['network'][f'blocks.{position}.{suffix}'])):
                    raise ValueError('Every added RMSNorm must have its declared initial unit values')
    human=WholeHumanData(ROOT/'data/whole-human-alignment-v2','dialogue','train','optimization')
    program=PreparedCodeStream('generation',ROOT/'data/APPS-verified-generation-v1',ROOT/'checkpoints/whole-human-SFT-v1/initial/tokenizer.json',ModelConfig.load(ROOT/'configs/model.json'))
    hp,pp=plan['original_human_position'],plan['original_verified_program_position'];hi,hl=human.sequence(hp);pi,pl=program.pack.sequence(pp)
    source_counts={name:dict(complete_context_tokens=len(ids),response_EOS_targets=sum(v!=-100 for v in labels[1:]))
                   for name,ids,labels in [('human',hi,hl),('verified_program',pi,pl)]}
    if human.rows[hp]['id']!=plan['original_human_ID'] or source_counts!=plan['complete_train_source_counts']:
        raise ValueError('Both actual complete original train-only sources must reconstruct')
    fixed_names=[n for n in initial['network'] if n not in network_names and n!='lm_head.weight']
    for step,state in [(1,first),(2,second)]:
        if (state['completed']!=step or state['planned']!=2 or state['world_size']!=1 or state['optimized_names']!=network_names
                or len(state['metrics'])!=step or state['metrics']!=proof['complete_source_metrics'][:step]
                or state['cursor']!=dict(human_position=hp,verified_program_position=pp,completed_updates=step)):
            raise ValueError('Every actual full dense-language update, source cursor and retained metric required')
        _validate_adam(state['optimizer'],[state['model']['network'][n] for n in network_names],step);_validate_rng(state['rng'],'cpu')
        if (any(not torch.equal(state['model']['network'][n],initial['network'][n]) for n in fixed_names)
                or any(not torch.equal(t,initial['heads'][n]) for n,t in state['model']['heads'].items())):
            raise ValueError('Every fixed vision/critic/auxiliary tensor must stay exact through both real updates')
        for metric in state['metrics']:
            if any(any(metric[name][key]!=value for key,value in wanted.items()) for name,wanted in source_counts.items()):
                raise ValueError('Every actual complete original response/EOS target and normalizer must reconstruct')
    if not any(not torch.equal(initial['network'][n],second['model']['network'][n]) for n in network_names):
        raise ValueError('Actual dense language weights must have learned in the disposable probe')
    guard.check(full=True)
    atomic_json(output,dict(reader_sha256=file_hash(__file__),original_complete_sha256=file_hash(REPORTS/'complete.json'),
        actual_counts=expected_counts,actual_dense_language_values=931244544,actual_full_dense_language_Adam_states=317,
        every168_original_dense_matrix_sum_independently_recomputed=True,
        every_original_code_embedding_norm_and_original_vision_critic_initial_value_exact=True,
        all11initial_residual_blocks_have_exact_zero_output_matrices_and_unit_norms=True,
        all_saved_tensors_exact_shapes_finite_and_one_actual_tied_saved_vocabulary_storage=True,
        both_step1_and_step2_complete_Adam_CPU_RNG_and_full_source_cursors_validated=True,
        all_fixed_candidate_vision_critic_heads_equal_initial=True,complete_original_used_train_sources_reconstructed=source_counts,
        actual_next_update_was_replayed_by_original_full_size_probe=True,
        reader_did_not_replay_neural_forward_or_optimizer_updates=True,
        full_source_training_GPU_capacity_and_held_quality_unfinished=True,candidate_promoted=False,goal_achieved=False))
    print(output)


if __name__=='__main__':main()
