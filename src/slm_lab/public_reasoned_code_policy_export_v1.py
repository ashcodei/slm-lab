"""Keep the complete trained coding policy and its shared human reward unfused.

The frozen backbone, nonzero inherited critic, reward head and every auxiliary
remain in one self-contained sub1B graph. This stage adds SFT, not fresh RLHF.
"""
import copy

import torch

from .complete_response_stopping_loss_v1 import PROFILES
from .corpus import file_hash
from .own_dense_actor_v1 import OwnDenseActor
from .own_policy_reward_standalone_v1 import export as export_shared
from .parallel_public_reasoned_code_SFT_stage_v1 import FORMAT as STATE
from .parallel_function_repair_stage_v1 import value_digest
from .public_reasoned_code_training_controller_v1 import TRAINING
from .public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from .whole_human_sft_evidence_v1 import tensor_digest

PROFILE = 'own-complete-public-reasoned-code-unfused-policy-reward-candidate-v1'


def export(model, reward, state, receipt, *, data, sources, tokenizer_sha256, allow_mechanics=False):
    if (type(model) is not OwnDenseActor or type(data) is not PublicReasonedCodeSFTData
            or data.role != 'train' or data.purpose != 'optimization' or data.views != 'both'
            or state.get('format') != STATE or receipt.get('format') != TRAINING or receipt.get('complete') is not True
            or state.get('mode') != receipt.get('mode') or state.get('profile') != receipt.get('profile')
            or state.get('profile') not in PROFILES or state.get('world_size') != receipt.get('world')
            or state.get('groups') != ['actor'] or type(state.get('completed')) is not int or state['completed'] < 1
            or state['completed'] != state.get('planned') or state['completed'] != receipt.get('actual_complete_updates')
            or len(state.get('metrics',[])) != state['completed']
            or receipt.get('every_original_complete_source_and_CE_target_retained') is not True
            or receipt.get('candidate_promoted') is not False or receipt.get('goal_achieved') is not False
            or receipt.get('actual_research_actor_parameters') != model.counts()['actual_unique']
            or receipt.get('actual_native_parameters') != model.base.counts()['total_unique']):
        raise ValueError('The exact complete own actor-only whole-code/reasoning state and actual controller receipt are required')
    mode = state['mode']
    entries = state['global_cursor']['all_consumed_global_source_batches']
    budget = data.budget(entries)
    if (len(entries) != state['completed'] or state['global_cursor']['completed'] != state['completed']
            or state['schedule_identity_sha256'] != data.identity(entries)
            or receipt['original_complete_training_budget'] != budget
            or sources.get('public_code_manifest_sha256') != data.manifest_sha256
            or sources.get('public_code_admission_sha256') != data.admission_sha256):
        raise ValueError('Every full source cursor, complete target budget and admitted package must reconstruct')
    if mode == 'train':
        if (state['world_size'] != 4 or model.counts()['actual_unique'] != 999706442
                or model.base.counts()['total_unique'] != 984399690
                or receipt.get('production_updates') != state['completed']
                or data.identity(entries) != data.identity(data.complete_schedule(epochs=1,width=16,seed=990231))
                or file_hash(data.folder/'tokenizer.json') != tokenizer_sha256):
            raise ValueError('Full export requires the complete original one-epoch16-wide own-tokenizer four-GPU production schedule')
    elif mode != 'mechanics' or allow_mechanics is not True or receipt.get('production_updates') != 0:
        raise ValueError('Only explicit small CPU mechanics may export without the complete production scope')
    model.validate_export(state['model'],expected_sources=sources)
    if (tensor_digest(model.parts.named_parameters()) != receipt['actual_final_actor_sha256']
            or tensor_digest(state['model']['parts'].items()) != receipt['actual_final_actor_sha256']
            or any(not torch.equal(value.detach().cpu(),state['model']['parts'][name]) for name,value in model.parts.state_dict().items())
            or value_digest(state['optimizer']) != receipt['actual_final_Adam_sha256']
            or tensor_digest(model.parts['critic'].named_parameters()) != sources.get('initial_policy_critic_sha256')):
        raise ValueError('Every completed actor/Adam value and the unchanged inherited policy critic must match')
    reward.check()
    provenance = dict(profile=PROFILE,mode=mode,training_profile=state['profile'],
        complete_training_budget=copy.deepcopy(budget),actual_completed_code_SFT_updates=state['completed'],
        whole_production_code_source_schedule_completed=mode=='train',
        completed_actor_sha256=receipt['actual_final_actor_sha256'],completed_Adam_sha256=receipt['actual_final_Adam_sha256'],
        initial_policy_critic_sha256=sources['initial_policy_critic_sha256'],
        inherited_policy_critic_and_shared_reward_preserved=True,policy_factors_not_folded_into_reward_features=True,
        new_supervision='public_precomputed_synthetic_code_and_whole_R1_weak_SFT',new_human_feedback_or_PPO_updates=0,
        complete_package_cold_GPU_and_original_full_assessments_still_required=True)
    payload = export_shared(model,reward,sources=sources,provenance=provenance,completed_updates=state['completed'],
        tokenizer_sha256=tokenizer_sha256,allow_mechanics=allow_mechanics)
    return payload,provenance
