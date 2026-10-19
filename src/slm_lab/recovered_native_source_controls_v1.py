"""Reconstruct retained native control rows without inventing fresh executions."""
from .public_token_suffix_curriculum_v1 import LEVELS, view, descriptor
from .qualified_native_token_suffix_episode_v1 import PROOF
from .independent_native_suffix_controls_v1 import check as independent_check


def validate_retained(corpus, oracle, catalogue, record, index, original, independent, *, expected_proof_files):
    position, level = index // 4, LEVELS[index % 4]
    row = corpus.row(position)
    ids, labels = corpus.sequence(position)
    native = view(corpus.tokenizer, row['original'], ids, labels, level)
    desc = descriptor(native)
    previous = catalogue.rows[(position, level)]
    proof = record['native_profile_control_proof']
    legacy_eligible = previous['bounded_nontrivial_control_evidence']
    boundary = any(t in [0, 1, 3] for t in ids[1:-1])
    controls = proof['actual_controls']
    if (record.get('source_position') != position or record.get('corpus_id') != row['id'] or record.get('level') != level
            or record.get('all_original_unconditional_CE_targets_retained') is not True
            or record.get('source_training_admitted') is not False or record.get('production_RL_admitted') is not False
            or record.get('original_legacy_disposition') != previous
            or original['nominal_missing_non_EOS_tokens'] != level or original['native_view'] != desc or independent['nominal_missing_non_EOS_tokens'] != level
            or independent['disposition'] != previous['disposition']
            or proof.get('format') != PROOF or proof.get('scope') != 'complete-source-catalogue'
            or proof.get('source_position') != position or proof.get('nominal_missing_non_EOS_tokens') != level
            or proof.get('source_information') != oracle.source_information(position) or proof.get('oracle_binding') != oracle.binding()
            or proof.get('native_view') != desc or desc != previous['exact_native_descriptor']
            or proof.get('source_files_sha256') != expected_proof_files
            or proof.get('complete_legacy_catalogue_sha256') != catalogue.digest
            or proof.get('source_training_admitted') is not False or proof.get('production_RL_admitted') is not False
            or proof.get('model_generations') != 0 or proof.get('model_updates') != 0
            or not isinstance(controls, list) or len(controls) != (2 if legacy_eligible and not boundary else 0)):
        raise ValueError('Every retained original native source/view/role/whole target and control binding must reconstruct')
    legacy = [original['counterfactual_prefix_then_EOS_execution'], independent['fresh_prefix_execution']]
    for repeat, control in enumerate(controls):
        if (control.get('repeat') != repeat or control.get('legacy_prefix_then_EOS') != legacy[repeat]
                or control.get('retained_legacy_origin') != ('original_producer' if repeat == 0 else 'independent_fresh_reader')):
            raise ValueError('Both retained original legacy controls must keep their exact original execution provenance')
    qualified = bool(controls) and all(c['native_gold']['reward'] == 1.
        and c['native_prefix_then_EOS']['reward'] is not None and c['native_prefix_then_EOS']['reward'] < 1.
        for c in controls)
    if controls:
        _, independently_qualified = independent_check(corpus, oracle, position, level, desc, controls)
        if independently_qualified is not qualified:
            raise ValueError('Retained native outcomes and independently rebuilt whole program semantics disagree')
    disposition = ('native_nontrivial_controls_qualified' if qualified else
        'legacy_ineligible_CE_only' if not legacy_eligible else 'embedded_original_boundary_CE_only' if boundary else
        'native_control_failed_or_abstained_CE_only')
    if (proof.get('complete') is not qualified or record.get('eligible_for_later_native_profile') is not qualified
            or record.get('disposition') != disposition):
        raise ValueError('Retained failed/abstaining/trivial native views cannot be upgraded into eligibility')
    return dict(qualified=qualified, actual_retained_native_gold_control_invocations=len(controls),
        actual_retained_native_prefix_control_invocations=len(controls), new_executions=0,
        complete_original_context_tokens=len(ids), original_response_EOS_targets=sum(x != -100 for x in labels),
        exact_native_view=desc)
