"""CPU-only conditional RL trial state with exact episode/reference recovery.

This stage repeats one explicitly qualified original public training problem.
It is not a complete source epoch, a GPU admission, human RLHF or held evidence.
Full original program CE remains active in both matched CE and execution arms.
"""
import copy
import hashlib
import json
import math

import torch

from .admitted_public_prefix_episode_v1 import AdmittedPublicPrefixEpisode
from .anchored_prefix_execution_CPU_v1 import update
from .own_dense_actor_checkpoint_v1 import make_state,restore_state

FORMAT='own-qualified-prefix-CPU-conditional-private-training-trial-v1'


def trajectory_digest(trajectories):
    records=[]
    for item in trajectories:
        score=item['actual_execution_reward'];actual=score.get('actual_execution',score.get('raw_candidate_execution',{}))
        records.append(dict(actions=item['response_ids'],cached=item['actual_cached_behavior_logps'],
            full=item['actual_full_behavior_logps'],suffix=item['own_response'],program=item['complete_candidate_response'],
            supplied_prefix=item['supplied_public_reference_prefix_ids'],reward=score['reward'],status=score['status'],
            source_program_sha256=actual.get('original_program_sha256'),original_tests_sha256=actual.get('original_tests_sha256'),
            case_passes=actual.get('case_passes',actual.get('example_passes'))))
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class ConditionalPrefixStage:
    def __init__(self,model,optimizer,episode,reference,*,method,planned_updates,sources,
                 local_group_size=4,cache_absolute_bound=.0001,KL_weight=.01,anchor_weight=1.,
                 world=1,rank=0,control_group=None,gradient_group=None,CPU_mechanics_horizon=None):
        if (type(episode) is not AdmittedPublicPrefixEpisode or type(planned_updates) is not int or planned_updates<1
                or next(model.parameters()).device.type!='cpu' or method not in ['anchor_CE','anchor_execution']
                or not sources or any(not isinstance(k,str) or not isinstance(v,str) or len(v)!=64
                    or any(c not in '0123456789abcdef' for c in v) for k,v in sources.items())):
            raise ValueError('Explicit qualified CPU-only conditional trial and complete source bindings required')
        episode.check();reference.assert_bound(model)
        self.model,self.optimizer,self.episode,self.reference=model,optimizer,episode,reference
        self.sources=copy.deepcopy(sources);self.planned=planned_updates;self.completed=0;self.metrics=[]
        self.settings=dict(method=method,planned_updates=planned_updates,world=world,rank=rank,
            local_group_size=local_group_size,cache_absolute_bound=cache_absolute_bound,KL_weight=KL_weight,
            anchor_weight=anchor_weight,CPU_mechanics_horizon=CPU_mechanics_horizon,
            precision='CPU_FP32',torch_CPU_threads=torch.get_num_threads(),one_qualified_training_problem_not_complete_epoch=True,
            original_full_reference_targets=episode.binding()['full_original_reference_EOS_targets'])
        self.kwargs=dict(own_reference=reference,method=method,local_group_size=local_group_size,
            cache_absolute_bound=cache_absolute_bound,KL_weight=KL_weight,anchor_weight=anchor_weight,
            world=world,rank=rank,control_group=control_group,gradient_group=gradient_group,CPU_mechanics_horizon=CPU_mechanics_horizon)

    def cursor(self,completed):
        return dict(completed=completed,planned=self.planned,original_source_position=self.episode.position,
            source_id=self.episode.binding()['original_source_id'],retained_reference_quarters=self.episode.quarters,
            outer_world_size=self.settings['world'],replica_rank=self.settings['rank'])

    def update(self):
        if self.completed>=self.planned:raise ValueError('The declared conditional CPU trial is complete')
        record=update(self.model,self.optimizer,self.episode,**self.kwargs)
        record.update(step=self.completed+1,planned_trial_updates=self.planned,actual_counts=self.model.counts(),
            conditional_training_source_trial_not_a_source_epoch=True)
        compact={k:copy.deepcopy(v) for k,v in record.items() if k not in ['trajectories','complete_prompt_ids']}
        compact['global_actual_trajectory_semantic_sha256']=trajectory_digest(record['trajectories'])
        self.completed+=1;self.metrics.append(compact);return record

    def state(self):
        self.reference.assert_bound(self.model);self.episode.check()
        core=make_state(self.model,self.optimizer,groups=('actor',),sources=self.sources,
            completed=self.completed,planned=self.planned,cursor=self.cursor(self.completed),metrics=self.metrics,device='cpu')
        return dict(format=FORMAT,settings=copy.deepcopy(self.settings),episode_binding=self.episode.binding(),
            own_reference_binding=self.reference.binding(),core=core,
            no_claim_of_complete_source_epoch_or_GPU_admission=True,candidate_promoted=False,goal_achieved=False)

    def resume(self,payload,*,allow_completed=False):
        self.reference.assert_bound(self.model);self.episode.check();core=payload.get('core',{});completed=core.get('completed')
        if (payload.get('format')!=FORMAT or payload.get('settings')!=self.settings
                or payload.get('episode_binding')!=self.episode.binding() or payload.get('own_reference_binding')!=self.reference.binding()
                or payload.get('no_claim_of_complete_source_epoch_or_GPU_admission') is not True
                or payload.get('candidate_promoted') is not False or payload.get('goal_achieved') is not False
                or type(completed) is not int or not 0<completed<=self.planned
                or len(core.get('metrics',[]))!=completed):raise ValueError('Exact qualified source, public prefix, own reference and CPU trial required before mutation')
        for i,row in enumerate(core['metrics']):
            if (row.get('step')!=i+1 or row.get('planned_trial_updates')!=self.planned
                    or row.get('episode_binding')!=self.episode.binding() or row.get('reference_binding')!=self.reference.binding()
                    or row.get('method')!=self.settings['method'] or row.get('complete_original_anchor_preserved') is not True
                    or row.get('source_reference_targets')!=self.settings['original_full_reference_targets']
                    or len(row.get('global_rewards',[]))!=self.settings['world']*self.settings['local_group_size']
                    or any(not math.isfinite(v) or not 0<=v<=1 for v in row['global_rewards'])
                    or len(row.get('global_actual_trajectory_semantic_sha256',''))!=64):
                raise ValueError('Every original source, complete CE target and conditional actual trajectory binding must reconstruct')
        self.completed,self.metrics=restore_state(self.model,self.optimizer,core,groups=('actor',),expected_sources=self.sources,
            planned=self.planned,expected_cursor=self.cursor(completed),device='cpu',allow_completed=allow_completed)
        return self.completed
