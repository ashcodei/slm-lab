"""Finalize closed full-quota source metadata without duplicate keyword fields."""
from copy import deepcopy


def validate_summary(summary,quotas):
    if (summary.get('optimization_admitted') is not False or summary.get('all_training_quotas_satisfied') is not True
            or set(summary.get('train_target_tokens_by_quota',{}))!=set(quotas)
            or any(summary['train_target_tokens_by_quota'][k]<v for k,v in quotas.items())
            or sum(quotas.values())!=10**9):
        raise ValueError('Every complete original billion-target source quota must finish without admission')
    for role in ['train','valid']:
        targets=summary[role+'_target_tokens_by_quota'];documents=sum(n[role] for n in summary['original_documents'].values())
        if (set(targets)!=set(quotas) or sum(targets.values())!=summary['actual_'+role+'_targets']
                or summary['native_context_token_counts'][role]!=sum(targets.values())+documents):
            raise ValueError('Every original role target/document/native-context denominator must reconcile')


def manifest(summary,quotas,tokenizer_sha256,artifacts):
    validate_summary(summary,quotas)
    result=deepcopy(summary)
    result.update(format='own-whole-public-billion-causal-candidate-v1',fixed_train_target_quotas=dict(quotas),
        tokenizer_sha256=tokenizer_sha256,native_token_dtype='little_endian_uint16',whole_original_documents_retained=True,
        source_files_sha256=dict(artifacts),independent_full_source_and_native_readback_required=True,
        human_preferences=False,execution_reward=False,goal_achieved=False)
    return result


def completion(summary,quotas,*,rows,source_files,dispositions,bindings):
    validate_summary(summary,quotas)
    if sum(dispositions.values())!=rows or dispositions['whole_original_document_retained_unadmitted']!=sum(
            sum(n.values()) for n in summary['original_documents'].values()):
        raise ValueError('Every observed source position and accepted whole document must reconcile')
    result=deepcopy(summary)
    result.update(complete=True,original_rows_observed=rows,actual_source_files=source_files,
        actual_prefix_dispositions=dict(dispositions),source_files_sha256=dict(bindings),
        whole_native_targets_collected_not_neurally_trained=True,
        independent_full_readback_and_training_capacity_still_required=True,
        neural_training_updates=0,candidate_promoted=False,goal_achieved=False)
    return result
