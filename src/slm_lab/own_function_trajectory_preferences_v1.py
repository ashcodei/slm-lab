"""Preferences for original good code versus our actual failed trajectories.

A budget-exhausted rejection retains every actual sampled action and has no
invented EOS. These are censored finite-horizon machine preferences, not human
rankings or standard complete-answer DPO data. Source-train controls stay train
pool observations and are not relabelled independent held benchmarks.
"""
import hashlib
import json
from pathlib import Path

from .corpus import file_hash

FORMAT='own-public-function-actual-trajectory-machine-preferences-v1'


def control_group(group):return int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%4==0


def native_pair(row,prompt,chosen,actions,*,stopped,budget):
    if (row['split']!='train' or row['source_candidate']['role']!='train' or row['id']!=row['source_candidate']['id']
            or not prompt or prompt[0]!=1 or not chosen or chosen[-1]!=2 or 2 in chosen[:-1]
            or not actions or type(budget) is not int or budget<1 or len(actions)>budget
            or type(stopped) is not bool or stopped!=(actions[-1]==2) or 2 in actions[:-1]
            or not stopped and len(actions)!=budget
            or any(type(v) is not int or not 0<=v<32768 for v in [*prompt,*chosen,*actions])
            or max(len(prompt)+len(chosen),len(prompt)+len(actions))>4096):
        raise ValueError('Exact original source-train prompt/goldEOS and all actual bounded sampled actions are required')
    group=row['group_id']
    return dict(id=row['id'],original_source_group=group,original_source_role='train',
        optimization_role='unoptimized_source_train_control' if control_group(group) else 'optimization_source_train',
        original_native_prompt_ids=list(prompt),chosen_original_complete_response_EOS_ids=list(chosen),
        rejected_actual_sampled_action_ids=list(actions),rejected_actual_stopped_on_EOS=stopped,
        rejected_observation_budget=budget,rejected_is_budget_censored=not stopped,
        no_EOS_or_unobserved_rejected_token_inserted=True,original_public_example_answers_visible_in_model_input=True,
        human_preferences=False,held_benchmark=False)


def validate(pair):
    source=dict(id=pair['id'],split='train',group_id=pair['original_source_group'],source_candidate=dict(id=pair['id'],role=pair['original_source_role']))
    rebuilt=native_pair(source,pair['original_native_prompt_ids'],pair['chosen_original_complete_response_EOS_ids'],
        pair['rejected_actual_sampled_action_ids'],stopped=pair['rejected_actual_stopped_on_EOS'],budget=pair['rejected_observation_budget'])
    if pair!=rebuilt:raise ValueError('Every original source, observed trajectory, native mask and sealed optimization/control role must reconstruct')


def batch(pair,field,device='cpu'):
    import torch
    validate(pair)
    if field not in {'chosen','rejected'}:raise ValueError('Exact positive original code or negative observed trajectory field required')
    prompt=pair['original_native_prompt_ids']
    targets=pair['chosen_original_complete_response_EOS_ids'] if field=='chosen' else pair['rejected_actual_sampled_action_ids']
    ids=prompt+targets
    return dict(input_ids=torch.tensor([ids],dtype=torch.long,device=device),
        attention_mask=torch.ones((1,len(ids)),dtype=torch.long,device=device),
        labels=torch.tensor([[-100]*len(prompt)+targets],dtype=torch.long,device=device))


class OwnFunctionTrajectoryPreferences:
    def __init__(self,folder,purpose):
        self.folder=Path(folder).resolve();self.role='train';self.purpose=purpose
        if purpose not in {'optimization','audit'}:raise ValueError('Explicit source-train optimization or read-only audit purpose required')
        admission=json.loads((self.folder/'admission.json').read_text());manifest=json.loads((self.folder/'manifest.json').read_text())
        if (manifest.get('format')!=FORMAT or admission.get('actual_trajectory_machine_preference_training_admitted') is not True
                or admission.get('human_preferences') is not False or admission.get('source_valid_or_test_records')!=0
                or admission.get('manifest_sha256')!=file_hash(self.folder/'manifest.json')
                or manifest['pairs_sha256']!=file_hash(self.folder/'all-original-function-trajectory-pairs.jsonl')):
            raise ValueError('Complete original source/native/action/fresh-runtime trajectory preference admission required')
        proof_path=Path(admission['independent_readback'])
        if file_hash(proof_path)!=admission['independent_readback_sha256']:
            raise ValueError('Exact independent source/native/action/execution proof bytes required')
        proof=json.loads(proof_path.read_text())
        if (proof.get('complete') is not True or proof.get('all14_original_sources_native_actions_and_fresh_executions_rebuilt') is not True
                or proof.get('actual_original_source_train_trajectories')!=14):
            raise ValueError('Every original source-train trajectory must finish full independent admission')
        self.rows=list(map(json.loads,(self.folder/'all-original-function-trajectory-pairs.jsonl').open()))
        if len(self.rows)!=14 or len({r['id'] for r in self.rows})!=14:raise ValueError('Every predeclared original source-train trajectory must remain present')
        for row in self.rows:validate(row)
        self.optimization_positions=[i for i,r in enumerate(self.rows) if r['optimization_role']=='optimization_source_train']
        self.unoptimized_control_positions=[i for i,r in enumerate(self.rows) if r['optimization_role']=='unoptimized_source_train_control']
        if not self.optimization_positions or not self.unoptimized_control_positions:raise ValueError('Separate hash-sealed train-pool optimization and unoptimized-control groups are required')
        if (set(self.rows[i]['original_source_group'] for i in self.optimization_positions)
                & set(self.rows[i]['original_source_group'] for i in self.unoptimized_control_positions)):
            raise ValueError('Whole original source-file groups cannot cross the optimization/control split')
        self.source_files_sha256={str(self.folder/name):file_hash(self.folder/name) for name in ['manifest.json','admission.json','all-original-function-trajectory-pairs.jsonl']}
        self.source_files_sha256.update(manifest['source_files_sha256'])
        self.source_files_sha256[str(proof_path)]=admission['independent_readback_sha256']

    def batch(self,position,field,device='cpu',*,for_optimization=False):
        if type(position) is not int or not 0<=position<len(self.rows):raise ValueError('An exact original source-train trajectory position is required')
        if for_optimization and (self.purpose!='optimization' or position not in self.optimization_positions):
            raise ValueError('Unoptimized source-train controls cannot provide gradients')
        return batch(self.rows[position],field,device)
