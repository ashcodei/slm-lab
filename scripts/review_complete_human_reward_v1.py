"""Replay all learned reward heads on all original held frozen features.

Validate every original train draw, saved Adam state and unused extension.
Original GPU transformer features are source-read-back, not neurally replayed.
"""
from collections import Counter
import json
import math
import os
from pathlib import Path

import torch
from torch.nn import functional as F

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.code_residual_checkpoint_v1 import _validate_adam,_validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.whole_human_data_v2 import WholeHumanData
from slm_lab.whole_human_reward_v1 import RewardFeatures,FEATURE_ROUTE
from slm_lab.whole_human_reward_inference_v1 import expected_sources,FORMAT

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/whole-human-reward-v1'
FEATURES=ROOT/'data/whole-human-reward-features-v1'
OUTPUT=ROOT/'checkpoints/whole-human-reward-v1'


def summarize(rows):
    return dict(pairs=len(rows),correct=sum(r['correct'] for r in rows),tied=sum(r['tied'] for r in rows),
        accuracy=sum(r['correct'] for r in rows)/len(rows),pair_mean_nll=math.fsum(r['nll'] for r in rows)/len(rows))


@torch.no_grad()
def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Complete human reward reviewer must explicitly hide GPUs')
    destination=REPORTS/'independent-complete-reward-readback-v1.json'
    if destination.exists():raise ValueError('Preserve prior complete reward readback')
    torch.set_num_threads(2)
    plan=json.loads((REPORTS/'plan.json').read_text());complete=json.loads((REPORTS/'complete.json').read_text())
    guard=FileBindings(plan['source_files_sha256']);guard.check(full=True)
    admission=json.loads((FEATURES/'admission.json').read_text())
    for name,sha in admission['files_sha256'].items():
        path=Path(name) if Path(name).is_absolute() else FEATURES/name
        if file_hash(path)!=sha:raise ValueError('Every original complete feature shard must match its admitted bytes')
    bound=expected_sources(plan,file_hash(REPORTS/'plan.json'),file_hash(FEATURES/'admission.json'))
    feature_sources={k:v for k,v in bound.items() if k!='feature_admission_sha256'}
    data={role:WholeHumanData(ROOT/'data/whole-human-alignment-v2','preferences',role,purpose)
        for role,purpose in [('train','optimization'),('valid','evaluation')]}
    features={role:RewardFeatures(data[role],[torch.load(FEATURES/f'{role}-{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
        for rank in range(4)],feature_sources) for role in data}
    initial_path=ROOT/'checkpoints/whole-human-SFT-v1/initial/model.pt'
    initial=torch.load(initial_path,map_location='cpu',weights_only=True,mmap=True)['aligned_model']
    initial_head={n.removeprefix('reward.'):t for n,t in initial['extensions'].items() if n.startswith('reward.')}
    reward_names=['reward.'+n for n in initial_head];heads={'initial':initial_head};training={}
    original_proof=json.loads((ROOT/'reports/whole-human-reward-full-CPU-v2/complete.json').read_text())
    for arm in plan['arms']:
        folder=OUTPUT/arm;receipt=complete['trained_exports'][arm]
        if file_hash(folder/'model.pt')!=receipt['checkpoint_sha256'] or file_hash(folder/'resume.pt')!=receipt['resume_sha256']:
            raise ValueError('Exact actually completed head/export and full Adam snapshots required')
        saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=True,mmap=True)
        state=torch.load(folder/'resume.pt',map_location='cpu',weights_only=True,mmap=True);core=state['core'];parts=core['model']['extensions']
        setting={**plan['training_options'],'learning_rate':plan['arms'][arm]['learning_rate']};steps=setting['steps']
        if (saved['format']!=FORMAT or saved['arm']!=arm or saved['sources']!=bound
                or saved['feature_route']!=FEATURE_ROUTE or state['feature_route']!=FEATURE_ROUTE or saved['training_options']!=setting
                or saved['actual_human_reward_updates']!=steps or saved['source_exposure_pairs']!=len(data['train'])
                or saved['frozen_encoder_sha256']!=plan['frozen_encoder_sha256'] or saved['actual_counts']!=initial['actual_counts']
                or core['steps']!=steps or core['planned_steps']!=steps or core['world_size']!=1 or core['groups']!=['reward']
                or core['optimized_names']!=reward_names or state['options']!=setting or len(core['metrics'])!=steps
                or core['model']['sources']!=bound or parts.keys()!=initial['extensions'].keys()
                or any(not torch.equal(parts[n],t) for n,t in initial['extensions'].items() if not n.startswith('reward.'))
                or any(not torch.equal(parts['reward.'+n],t) for n,t in saved['reward'].items())):
            raise ValueError('Every completed head/source/options/feature basis and all unused extension values must match')
        _validate_adam(core['optimizer'],[parts[n] for n in reward_names],steps);_validate_rng(core['rng'],'cpu')
        group=core['optimizer']['param_groups'][0]
        if (group['lr']!=setting['learning_rate'] or tuple(group['betas'])!=tuple(setting['betas']) or group['weight_decay']!=setting['weight_decay']):
            raise ValueError('Exact final predeclared ordinary reward Adam settings required')
        cursor=dict(completed_updates=steps,completed_draws=len(data['train']),source_population=len(data['train']),seed=setting['seed'],final_batch_not_repeated=True)
        if core['sampler_state']!=cursor:raise ValueError('The complete once-only original human-pair cursor must reconstruct')
        order=EpochOrder(len(data['train']),setting['seed']);seen=Counter()
        for step,row in enumerate(core['metrics']):
            positions=[order.at(p) for p in range(step*setting['batch_size'],min((step+1)*setting['batch_size'],len(data['train'])))]
            if (row['step']!=step+1 or row['source_positions']!=positions or row['pairs']!=len(positions)
                    or row['learning_rate']!=setting['learning_rate']
                    or any(not math.isfinite(row[k]) or row[k]<0 for k in ('pair_mean_nll','gradient_norm_before_clip'))):
                raise ValueError('Every complete original training draw and finite actual update metric must reconstruct')
            seen.update(positions)
        if len(seen)!=len(data['train']) or set(seen.values())!={1}:raise ValueError('Each of all85786original human pairs exactly once required')
        heads[arm]=saved['reward'];training[arm]=dict(actual_updates=steps,actual_Adam_states=len(core['optimizer']['state']),
            every_original_train_pair_once=True,actual_last_batch_pairs=len(core['metrics'][-1]['source_positions']),
            all_unused_extensions_equal_original_initial_values=True,every_exported_head_tensor_equal_complete_resume=True)
    scores={};rows_by_arm={};eps=initial['config']['norm_eps']
    for arm,head in heads.items():
        path=REPORTS/(arm+'-human-rankings.json');report=json.loads(path.read_text());rows=report['results']
        if len(rows)!=len(data['valid']) or [r['source_position'] for r in rows]!=list(range(len(data['valid']))):
            raise ValueError('All original4642held human pairs exactly once required')
        # Independent direct tensor expression for the saved RMSNorm/MLP.
        # Same256-pair batch shape preserves the original CPU arithmetic.
        values=[]
        for start in range(0,len(data['valid']),256):
            x=features['valid'].values[start:start+256]
            normalized=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)).to(x.dtype)*head['0.weight'].to(x.dtype)
            result=F.linear(F.silu(F.linear(normalized,head['1.weight'])),head['3.weight']).squeeze(-1)
            values.extend(result.tolist())
        for p,(row,(chosen,rejected)) in enumerate(zip(rows,values)):
            original=data['valid'].rows[p];margin=chosen-rejected
            wanted=dict(id=original['id'],source=original['source'],chosen_score=chosen,rejected_score=rejected,margin=margin,
                nll=float(F.softplus(torch.tensor(-margin,dtype=torch.float64))),correct=margin>0,tied=margin==0,
                chosen_targets=original['native']['chosen']['length']-original['native']['chosen']['target_start'],
                rejected_targets=original['native']['rejected']['length']-original['native']['rejected']['target_start'])
            if any(row[k]!=v for k,v in wanted.items()):raise ValueError('Every original saved held score must equal the independently replayed actual learned neural head')
        controls={}
        for label,predicate in [('chosen_longer',lambda r:r['chosen_targets']>r['rejected_targets']),
            ('chosen_shorter',lambda r:r['chosen_targets']<r['rejected_targets']),('equal_length',lambda r:r['chosen_targets']==r['rejected_targets'])]:
            chosen=[r for r in rows if predicate(r)];controls[label]=summarize(chosen) if chosen else dict(pairs=0)
        by_source={n:summarize([r for r in rows if r['source']==n]) for n in sorted({r['source'] for r in rows})}
        if (report['overall']!=summarize(rows) or report['by_source']!=by_source or report['response_length_controls']!=controls
                or complete['initial_and_trained_held_human_ranking_metrics'][arm]!=report['overall']):
            raise ValueError('Every full-source and response-length held aggregate must reconstruct')
        scores[arm]=dict(overall=report['overall'],by_source=by_source,response_length_controls=controls,report_sha256=file_hash(path));rows_by_arm[arm]=rows
    best=min(plan['arms'],key=lambda a:(scores[a]['overall']['pair_mean_nll'],a))
    selected=best if scores[best]['overall']['pair_mean_nll']<scores['initial']['overall']['pair_mean_nll'] else 'initial'
    if selected!=complete['selected_arm']:raise ValueError('Only the declared full-held pair-NLL selection is permitted')
    a,b=rows_by_arm['initial'],rows_by_arm[selected]
    paired=dict(gained_ids=[y['id'] for x,y in zip(a,b) if not x['correct'] and y['correct']],
        lost_ids=[y['id'] for x,y in zip(a,b) if x['correct'] and not y['correct']])
    guard.check(full=True)
    atomic_json(destination,dict(reader_sha256=file_hash(__file__),complete_sha256=file_hash(REPORTS/'complete.json'),
        all_original_native_train_held_source_fingerprints_and_feature_shards_read_back=True,
        actual_complete_training=training,all_held_neural_head_scores_replayed_and_aggregates_reconstructed=scores,
        selected_arm=selected,paired_selected_changes=paired,
        initial_head_bound_to_original_initial_SFT_artifact_sha256=file_hash(initial_path),
        original_GPU_transformer_feature_neural_computation_not_replayed=True,
        optimizer_numerical_training_not_replayed=True,held_other_model_responses_not_fresh_own_output_human_judgments=True,
        frozen_general_encoder_required_not_interchangeable_with_new_dense_features=True,
        policy_updates=0,generated_policy_alignment_or_high_end_quality_not_proven=True,candidate_promoted=False,goal_achieved=False))
    print(destination)


if __name__=='__main__':main()
