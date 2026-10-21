"""One complete source epoch updating all own dense language parameters.

Shares the fixed admitted-data order with the adapter comparison. Vision,
critic and auxiliary tensors remain fixed; no human/PPO claim is implied.
Resource ownership, full-size capacity and final assessment are external.
"""
import math

from .complete_response_stopping_loss_v1 import metadata
from .serial_public_code_epoch_v1 import schedule
from .serial_dense_code_bf16_step_v1 import update


def reconcile(data,entries,records):
    if entries!=schedule(data) or len(records)!=len(entries):
        raise ValueError('Every complete original dense-language update and full source order is required')
    for i,(positions,row) in enumerate(zip(entries,records)):
        info=[data.information(p) for p in positions]
        if (row['actual_positions']!=positions or row['actual_Adam_step']!=i+1
                or row['source_metadata']!=metadata(info,'plain_tokens')
                or [m['position'] for m in row['actual_microbatches']]!=positions
                or [m['complete_context_tokens'] for m in row['actual_microbatches']]!=[r['complete_context_tokens'] for r in info]
                or row['fixed_vision_critic_and_auxiliary_parameters_unchanged'] is not True
                or row['private_RNG_unchanged'] is not True or row['complete_original_sources_and_EOS_preserved'] is not True
                or row['protocol']!='full_dense_language_serial_microbatch1_BF16_forward_FP32_CE_weights_Adam'):
            raise ValueError('Every original dense update source/target/objective/clock and fixed boundary must reconstruct')
    budget=data.budget(entries)
    for key in ['response_EOS_targets','complete_context_tokens']:
        if sum(r['source_metadata'][key] for r in records)!=budget[key]:raise ValueError('The complete original target/context population must reconcile')
    ce=math.fsum(m['original_CE_sum'] for r in records for m in r['actual_microbatches'])
    eos=math.fsum(m['EOS_CE_sum'] for r in records for m in r['actual_microbatches'])
    return dict(complete=True,profile='full_dense_plain_CE',schedule_sha256=data.identity(entries),
        original_complete_training_budget=budget,actual_final_batch_sources=len(entries[-1]),
        all_train_views_consumed_exactly_once=True,all_sources_responses_and_EOS_whole=True,
        training_token_mean_NLL=ce/budget['response_EOS_targets'],training_EOS_token_mean_NLL=eos/len(data),
        training_losses_are_not_held_quality=True,reserved_optimization_updates=0,
        fixed_vision_critic_and_auxiliary_parameters_unchanged=True,
        inherited_reward_feature_compatibility_after_language_updates_not_certified=True,
        synthetic_SFT_is_not_human_feedback_or_RL_oracle=True,candidate_promoted=False,goal_achieved=False)


def train(model,optimizer,data,*,checked,record,checkpoint,allow_mechanics=False):
    if optimizer.state:raise ValueError('Production arms must start with fresh full-language Adam')
    entries=schedule(data)
    if not allow_mechanics and (len(data)!=3675 or len(entries)!=230 or len(entries[-1])!=11):
        raise ValueError('Every original admitted source and the genuine eleven-source tail is required')
    rows=[]
    for i,positions in enumerate(entries):
        checked();row=update(model,optimizer,data,positions,'plain_tokens',checked=checked,allow_mechanics=allow_mechanics)
        rows.append(row);record(i+1,row)
        if i==0 or (i+1)%16==0 or i+1==len(entries):checkpoint(i+1,entries,rows)
    checked();data.audit_provenance();return reconcile(data,entries,rows)
