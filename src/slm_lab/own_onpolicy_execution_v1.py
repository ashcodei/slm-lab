"""Admit complete native own-policy wrong-value programs, never human labels."""
import hashlib

from .apps_execution_repair_v1 import counterexample
from .code_quality import instruction_code


def example(source, generation, positive, negative, tokenizer, *, policy_sha256):
    if source.get('split') != 'train' or source.get('publisher_split') != 'train':
        raise ValueError('Only original publisher-train sources may create on-policy training pairs')
    if (not isinstance(policy_sha256, str) or len(policy_sha256) != 64
            or any(c not in '0123456789abcdef' for c in policy_sha256)
            or generation.get('id') != source['id'] or generation.get('split') != 'train'
            or generation.get('publisher_split') != 'train' or generation.get('strength') != 0.
            or generation.get('prefix_ids') != source['prefix_ids']):
        raise ValueError('Actual original native greedy source identity/prefix and exact frozen policy required')
    ids = generation['response_ids']
    if (not ids or any(type(i) is not int or not 0 <= i < tokenizer.get_vocab_size() for i in ids)
            or tokenizer.decode(ids, skip_special_tokens=True) != generation['raw_response']
            or instruction_code(generation['raw_response']) != generation['code']):
        raise ValueError('Every unchanged actual own-policy response token and delivered extracted program must reconstruct')
    if not generation['stopped_on_all_EOS'] or ids[-1] != 2 or any(i in {0, 1, 2} for i in ids[:-1]):
        return None, 'own_response_not_complete_single_EOS_program'
    if (hashlib.sha256(generation['code'].encode()).hexdigest() != negative['original_program_sha256']
            or hashlib.sha256(source['original_human_program'].encode()).hexdigest() != positive['original_program_sha256']):
        raise ValueError('Execution must bind the exact original human and actually delivered own-policy program bytes')
    witness = counterexample(positive, negative, source['cases'])
    if witness is None:
        return None, 'no_complete_observed_wrong_value_counterexample'
    prefix = [1, *tokenizer.encode('User: ' + source['prompt'] + '\nAssistant:').ids]
    body = [*tokenizer.encode(source['original_human_program']).ids, 2]
    if prefix != source['prefix_ids'] or len(body) != source['target_tokens']:
        raise ValueError('Unchanged complete original native generation prompt/human response/EOS required')
    if len(prefix) + max(len(body), len(ids)) > 4096:
        return None, 'complete_original_pair_exceeds4096'
    chosen = [prefix + body, [-100] * len(prefix) + body]
    rejected = [prefix + ids, [-100] * len(prefix) + ids]
    if chosen == rejected:
        raise ValueError('An unchanged source reference cannot be both passing and observed wrong')
    return dict(id=source['id'] + ':own-onpolicy-execution-v1', split='train', publisher_split='train',
        feedback_source='machine_execution_of_original_publisher_tests', finite_execution_verified=True,
        human_preferences=False, genuine_human_RLHF=False, reference_policy_sha256=policy_sha256,
        source_problem_id=source['original_problem'], original_source_program_id=source['id'],
        chosen=chosen, rejected=rejected, witness=witness,
        original_human_target_bytes_unchanged=True, negative_is_actual_complete_own_policy_response=True,
        original_tests_sha256=source['cases']['original_tests_sha256'],
        original_human_program_sha256=positive['original_program_sha256'],
        actual_executed_negative_program_sha256=negative['original_program_sha256'],
        runtime='Original local APPS stdlib-prelude confinement and retained comparator; finite published train cases only',
        exact_held_source_screens_inherited_not_semantic_independence=True,
        external_model_teacher_used=False, algorithmic_or_efficiency_proof=False), None
