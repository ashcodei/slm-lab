"""Keep all original capacity extremes with complete corrective accounting.

The first workload's actual source selection remains unchanged. Reconstruct
all checkpoint objectives using positive CE plus the separate original
negative forwards; their targets must never be subtracted from positive CE.
"""
from .complete_joint_capacity_plan_v1 import choose as original_choose
from .complete_joint_objective_bridge_v3 import complete_entries
from .complete_joint_source_schedule_v1 import digest
from .parallel_function_repair_stage_v1 import value_digest

FORMAT = 'own-complete-joint-original-source-capacity-workload-v3'


def choose(source, original_visual_capacity):
    result = original_choose(source, original_visual_capacity)
    original_sha256 = digest(result)
    entries = complete_entries(source)
    positive_contexts = sum(r['complete_context_tokens'] for part in source.streams.values() for r in part['entries'])
    positive_targets = sum(r['response_EOS_targets'] for part in source.streams.values() for r in part['entries'])
    # Native word CE and byte CTC are separate objectives. Only the native
    # objectives carry context/response counts, so no word is counted twice.
    complete_contexts = sum(o['complete_context_tokens'] for e in entries for o in e['objectives'])
    complete_targets = sum(o['response_EOS_targets'] for e in entries for o in e['objectives'])
    negative_targets = sum(r['complete_original_negative_response_EOS_targets'] for r in source.streams['visual']['entries'])
    if negative_targets != 80 or complete_targets != positive_targets + negative_targets or complete_contexts <= positive_contexts:
        raise ValueError('All original positive targets and all80additional corrective negative EOS targets must remain')
    for case in result['cases']:
        original = entries[case['original_joint_step'] - 1]
        case['original_checkpoint_entry_sha256'] = value_digest(original)
        case['checkpoint_entry'] = dict(original, step=case['capacity_step'])
    result.update(format=FORMAT, original_source_selection_workload_sha256=original_sha256,
        complete_checkpoint_entries_sha256=value_digest(entries),
        checkpoint_entry_digest_protocol='slm_lab.parallel_function_repair_stage_v1.value_digest',
        original_positive_native_context_tokens=positive_contexts, original_positive_response_EOS_targets=positive_targets,
        additional_original_corrective_negative_context_tokens=complete_contexts - positive_contexts,
        additional_original_corrective_negative_response_EOS_targets=negative_targets,
        complete_positive_and_negative_context_tokens=complete_contexts,
        complete_positive_and_negative_response_EOS_targets=complete_targets,
        every_actual_positive_CE_denominator_preserved=True)
    return result
