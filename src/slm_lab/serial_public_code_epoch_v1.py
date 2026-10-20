"""One unrepeated whole-source coding epoch, with durable-state callbacks.

Resource ownership is the caller's responsibility. This uses the capacity-
checked BF16 forward/FP32 CE and Adam protocol, not the older FP32 stage.
"""
import math

from .complete_response_stopping_loss_v1 import metadata
from .public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from .serial_public_code_bf16_step_v1 import update

WIDTH = 16
SEED = 990231
PROFILES = ('plain_tokens', 'eos_weight4')


def schedule(data):
    if type(data) is not PublicReasonedCodeSFTData or data.purpose != 'optimization' or data.views != 'both':
        raise ValueError('The complete admitted train role and both whole source views are required')
    entries = data.complete_schedule(epochs=1, width=WIDTH, seed=SEED)
    if sorted(p for entry in entries for p in entry) != list(range(len(data))):
        raise ValueError('Each original train view must occur exactly once without rounding repeats')
    return entries


def reconcile(data, entries, records, profile):
    if profile not in PROFILES or entries != schedule(data) or len(records) != len(entries):
        raise ValueError('One complete original epoch and every actual update record are required')
    for i, (positions, row) in enumerate(zip(entries, records)):
        information = [data.information(p) for p in positions]
        if (row['actual_positions'] != positions or row['actual_Adam_step'] != i+1
                or row['source_metadata'] != metadata(information, profile)
                or [m['position'] for m in row['actual_microbatches']] != positions
                or [m['complete_context_tokens'] for m in row['actual_microbatches']] != [s['complete_context_tokens'] for s in information]
                or row['complete_original_sources_and_EOS_preserved'] is not True
                or row['original_fixed_parameters_unchanged'] is not True
                or row['private_RNG_unchanged'] is not True
                or row['protocol'] != 'serial_microbatch1_BF16_forward_FP32_CE_weights_Adam'):
            raise ValueError('Every whole original source, objective, Adam clock and update boundary must agree')
    budget = data.budget(entries)
    for key in ['response_EOS_targets', 'complete_context_tokens']:
        if sum(row['source_metadata'][key] for row in records) != budget[key]:
            raise ValueError('The complete source/target population must reconcile')
    targets = budget['response_EOS_targets']
    original_ce = math.fsum(m['original_CE_sum'] for row in records for m in row['actual_microbatches'])
    eos_ce = math.fsum(m['EOS_CE_sum'] for row in records for m in row['actual_microbatches'])
    return dict(complete=True, profile=profile, schedule_sha256=data.identity(entries),
        original_complete_training_budget=budget, actual_final_batch_sources=len(entries[-1]),
        all_train_views_consumed_exactly_once=True, all_sources_responses_and_EOS_whole=True,
        training_token_mean_NLL=original_ce/targets, training_EOS_token_mean_NLL=eos_ce/len(data),
        training_losses_are_not_held_quality=True, reserved_optimization_updates=0,
        synthetic_SFT_is_not_human_feedback_or_RL_oracle=True, candidate_promoted=False, goal_achieved=False)


def train(model, optimizer, data, profile, *, checked, record, checkpoint, allow_mechanics=False):
    if profile not in PROFILES or optimizer.state:
        raise ValueError('A fresh matched objective and initially empty Adam state are required')
    entries = schedule(data)
    if not allow_mechanics and (len(data) != 3675 or len(entries) != 230 or len(entries[-1]) != 11):
        raise ValueError('The actual full admitted corpus and unrepeated final eleven-source tail are required')
    records = []
    for i, positions in enumerate(entries):
        checked()
        row = update(model, optimizer, data, positions, profile, checked=checked, allow_mechanics=allow_mechanics)
        records.append(row)
        record(i+1, row)
        if i == 0 or (i+1) % 16 == 0 or i+1 == len(entries):
            checkpoint(i+1, entries, records)
    checked()
    data.audit_provenance()
    return reconcile(data, entries, records, profile)
