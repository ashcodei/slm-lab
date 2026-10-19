"""Read a complete committed private prefix into a disposable CPU policy.

This is an inference-only reader for an explicitly incomplete source epoch.
It validates every private Adam/RNG/source journal before copying any weights;
it never resumes Adam, restores private RNG or certifies neural replay.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from .broad_anchored_execution_stage_v1 import FORMAT, semantic_digest, source_order
from .broad_anchored_policy_v3 import sources
from .code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from .own_dense_actor_checkpoint_v1 import FORMAT as CORE_FORMAT
from .own_dense_actor_v1 import OwnDenseActor
from .whole_human_sft_evidence_v1 import tensor_digest


def equal(a,b):
    if type(a) is not type(b):return False
    if isinstance(a,torch.Tensor):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def materialize(model,corpus,plan,plan_sha256,private_paths,journal_path,*,expected_completed,
                checked=lambda:None,allow_small_cpu_control=False):
    if (type(model) is not OwnDenseActor or any(p.device.type!='cpu' or p.dtype!=torch.float32 for p in model.parameters())
            or type(expected_completed) is not int or not 0<expected_completed<plan['planned_steps']
            or len(private_paths) not in (1,4)):
        raise ValueError('Exact disposable FP32 CPU policy and a complete explicitly interim private prefix required')
    control=bool(allow_small_cpu_control)
    if ((control and model.counts()['actual_unique']>=10**7) or
            (not control and (len(private_paths)!=4 or len(corpus)!=3721 or plan['planned_steps']!=3721
                             or model.counts()['actual_unique']!=999706442))):
        raise ValueError('Small CPU format controls cannot substitute for the full four-rank original experiment')
    checked();order=source_order(corpus,plan['training_options']['seed']);world=len(private_paths)
    if len(order)!=plan['planned_steps']:raise ValueError('Every original source must retain its complete epoch position')
    expected_sources=sources(plan,plan_sha256);named=model.named_selected(('actor',));parameters=[p for _,p in named]
    initial_parts=tensor_digest(model.parts.named_parameters());common=None;rng_hashes=[];private_hashes=[]
    for rank,path in enumerate(private_paths):
        checked();raw=torch.load(Path(path),map_location='cpu',weights_only=True,mmap=True);core=raw.get('core',{})
        settings=dict(method='anchor_CE',seed=plan['training_options']['seed'],world=world,rank=rank,
            local_group_size=1,KL_weight=plan['training_options']['KL_weight'],anchor_weight=plan['training_options']['anchor_weight'],
            cache_absolute_bound=plan['cache_calibration']['bound'],cache_calibration_sha256=plan['cache_calibration']['sha256'],
            CPU_control_positions=None,CPU_mechanics_horizon=plan.get('CPU_mechanics_horizon') if control else None,
            original_source_population=len(corpus),planned_updates=len(order),
            curriculum='documented_then_introductory_then_interview_then_competition_seeded_within_stratum',
            complete_original_reference_supervision_every_update=True,reference_is_frozen_own_starting_actor=True,
            precision='FP32_TF32_disabled',gradient_transport='Gloo_host_SUM_already_global_normalized')
        cursor=dict(completed_updates=expected_completed,planned_updates=len(order),source_order=order,
            source_population=len(corpus),outer_world_size=world,replica_rank=rank,seed=settings['seed'])
        if (raw.get('format')!=FORMAT or raw.get('settings')!=settings or raw.get('corpus_provenance')!=corpus.provenance
                or raw.get('reference_binding')!=plan['own_reference_binding']
                or raw.get('private_state_is_not_whole_distributed_checkpoint') is not True
                or raw.get('full_trajectory_records_require_external_durable_journal') is not True
                or raw.get('candidate_promoted') is not False or raw.get('goal_achieved') is not False
                or core.get('format')!=CORE_FORMAT or core.get('world_size')!=1 or core.get('groups')!=['actor']
                or core.get('optimized_names')!=[n for n,_ in named] or core.get('completed')!=expected_completed
                or core.get('planned')!=len(order) or core.get('cursor')!=cursor
                or not isinstance(core.get('metrics'),list) or len(core['metrics'])!=expected_completed
                or core.get('candidate_promoted') is not False or core.get('goal_achieved') is not False):
            raise ValueError('Every original private rank, epoch, source cursor and reference must match before mutation')
        model.validate_export(core['model'],expected_sources=expected_sources)
        if not torch.equal(core['model']['parts']['critic.weight'],model.parts['critic'].weight):
            raise ValueError('The unused critic cannot change in actor-only code training')
        _validate_adam(core.get('optimizer'),parameters,expected_completed)
        group=core['optimizer']['param_groups'][0];options=plan['training_options']
        if (group['lr']!=options['learning_rate'] or tuple(group['betas'])!=tuple(options['betas'])
                or group['weight_decay']!=options['weight_decay'] or group['eps']!=1e-8):
            raise ValueError('Original fixed actor Adam hyperparameters must match')
        rng=core.get('rng');_validate_rng(dict(rng,cuda=None),'cpu');cuda=rng['cuda']
        if not control and (not isinstance(cuda,torch.Tensor) or cuda.dtype!=torch.uint8 or cuda.ndim!=1 or cuda.numel()==0):
            raise ValueError('Every original opaque CUDA generator byte must be retained, without CPU replay')
        if control and cuda is not None:raise ValueError('Small CPU controls require actual CPU-only saved RNG')
        digest=hashlib.sha256(rng['torch'].numpy().tobytes()+(cuda.numpy().tobytes() if cuda is not None else b'')).hexdigest()
        rng_hashes.append(digest)
        if common is None:common=core
        elif any(not equal(core[k],common[k]) for k in ('model','optimizer','metrics')):
            raise ValueError('All original ranks must have identical common policy, Adam and complete metrics before mutation')
        with Path(path).open('rb') as stream:private_hashes.append(hashlib.file_digest(stream,'sha256').hexdigest())
    if world>1 and len(set(rng_hashes))!=world:raise ValueError('All original private generators must remain distinct')
    count=0;actions=0;targets=0
    with Path(journal_path).open('rb') as stream:
        for count,line in enumerate(stream,1):
            if count>expected_completed or not line.endswith(b'\n'):raise ValueError('Only the complete externally committed journal prefix is permitted')
            checked();record=json.loads(line);position=order[count-1];row=corpus.row(position);ids,labels=corpus.sequence(position)
            start=next(i for i,v in enumerate(labels) if v!=-100);trajectories=record.get('trajectories',[])
            if (record.get('step')!=count or record.get('position')!=position or record.get('source_id')!=row['id']
                    or record.get('source_epoch_order_position')!=count-1 or record.get('planned_source_updates')!=len(order)
                    or record.get('complete_prompt_ids')!=ids[:start] or record.get('source_reference_targets')!=row['response_EOS_targets']
                    or record.get('original_reference_used_regardless_of_rewards') is not True
                    or record.get('reference_binding')!=plan['own_reference_binding'] or record.get('method')!='anchor_CE'
                    or len(trajectories)!=world or record.get('world')!=world or record.get('local_group_size')!=1
                    or record.get('actual_counts')!=model.counts()):
                raise ValueError('Every complete original source and all original trajectories must reconstruct')
            compact={k:v for k,v in record.items() if k not in ('trajectories','complete_prompt_ids')}
            compact.update(global_trajectory_semantic_sha256=semantic_digest(trajectories),
                           original_source_context_positions=row['complete_context_positions'])
            if compact!=common['metrics'][count-1]:raise ValueError('Every full source record must match every private compact metric exactly')
            if record['global_rewards']!=[t['actual_execution_reward']['reward'] for t in trajectories]:
                raise ValueError('Every original execution reward must retain its actual source trajectory')
            for item in trajectories:
                response=item['response_ids'];cached=item['actual_cached_behavior_logps'];full=item['actual_full_behavior_logps']
                if (not response or len(response)>record['action_horizon'] or any(type(t) is not int or not 0<=t<model.config.vocab_size for t in response)
                        or 2 in response[:-1] or len(cached)!=len(response) or len(full)!=len(response)
                        or any(not math.isfinite(v) for v in cached+full)
                        or item['own_response']!=corpus.tokenizer.decode(response)
                        or item.get('temperature')!=1. or item.get('full_vocabulary_sampling') is not True):
                    raise ValueError('Actual full actions, source decoding and cached/full behavior probabilities must remain')
                actions+=len(response)
            targets+=row['response_EOS_targets']
    if count!=expected_completed:raise ValueError('Every consumed original update must be present in the committed journal')
    checked()
    model.import_export(common['model'],expected_sources=expected_sources);model.eval()
    return dict(complete_committed_interim_prefix_validated=True,completed_updates=expected_completed,
        original_planned_updates=len(order),original_sources_consumed=count,complete_reference_targets=targets,
        actual_sampled_actions=actions,all_private_policy_Adam_metrics_identical=True,private_checkpoint_sha256=private_hashes,
        opaque_private_RNG_sha256=rng_hashes,initial_actor_parts_sha256=initial_parts,
        interim_actor_parts_sha256=tensor_digest(model.parts.named_parameters()),actual_counts=model.counts(),
        optimizer_constructed_or_resumed=False,private_RNG_restored=False,CUDA_RNG_replayed=False,
        neural_training_or_behavior_likelihoods_replayed=False,small_CPU_format_control=control,
        original_source_epoch_is_incomplete=True,model_quality_proved=False,goal_achieved=False)
