"""Reconstruct closed independent-reader observations without rerunning them.

Retained observations keep their exact bytes and provenance. Only unfinished
sources get new executions; neither path grants model or training admission.
"""
from collections import Counter

from .independent_token_suffix_readback_v1 import rebuild, eligible

LEVELS = (4, 16, 64, 0)
SEMANTICS = ('reward', 'status', 'all_finite_original_cases_passed')


def prepare(corpus, position, original, metadata):
    row = corpus.row(position)
    ids, labels = corpus.sequence(position)
    if (original['source_position'] != position or original['corpus_id'] != row['id']
            or original['kind'] != row['kind'] or len(original['views']) != 4 or len(metadata) != 4
            or original['original_CE_context_tokens'] != len(ids)
            or original['original_CE_response_EOS_targets'] != len(ids) - row['prefix_positions']
            or original['source_role'] != 'train' or original['source_training_admitted'] is not False
            or original['production_RL_admitted'] is not False
            or original['counterfactuals_not_student_samples'] is not True
            or original['every_original_source_and_complete_CE_anchor_retained'] is not True):
        raise ValueError('Every original source, train role, complete CE mask and all four views must remain')
    reference = original['full_original_reference_execution']
    views = []
    for level, control, record in zip(LEVELS, original['views'], metadata):
        identity = {k: record[k] for k in ['source_position', 'corpus_id', 'kind']}
        descriptor = {k: v for k, v in record.items() if k not in identity}
        if identity != dict(source_position=position, corpus_id=row['id'], kind=row['kind']):
            raise ValueError('Each original native descriptor must retain its ordered source identity')
        native = rebuild(corpus.tokenizer, row['original'], ids, labels, level, descriptor)
        if (control['nominal_missing_non_EOS_tokens'] != level or control['native_view'] != descriptor
                or control['original_prefix_text'] != native['original_prefix_text']
                or control['source_training_admitted'] is not False
                or control['proposed_nontrivial_bounded_execution_view'] != eligible(descriptor, reference,
                    control['counterfactual_prefix_then_EOS_execution'])):
            raise ValueError('Every original prefix, descriptor and actual execution disposition must reconstruct')
        views.append(native)
    return row, views


def assemble(corpus, position, original, metadata, reference, prefixes):
    """Rebuild the unchanged V2 reader semantics from actual worker results."""
    row, views = prepare(corpus, position, original, metadata)
    if len(prefixes) != 4 or not isinstance(reference, dict) or any(not isinstance(p, dict) for p in prefixes):
        raise ValueError('One actual reference and all four actual prefix observations are required')
    old_reference = original['full_original_reference_execution']
    changed_reference = any(old_reference.get(k) != reference.get(k) for k in SEMANTICS)
    counts = Counter(fresh_full_reference_runs=1, fresh_reference_reward1=int(reference.get('reward') == 1.),
        complete_original_sources=1)
    discrepancies = []
    if changed_reference:
        discrepancies.append(dict(source_position=position, phase='full_reference',
            original_reward=old_reference.get('reward'), fresh_reward=reference.get('reward')))
    actual_views, catalogue = [], []
    totals = {}
    for level, native, control, actual in zip(LEVELS, views, original['views'], prefixes):
        info = native['metadata']
        old = control['counterfactual_prefix_then_EOS_execution']
        changed = changed_reference or any(old.get(k) != actual.get(k) for k in SEMANTICS)
        if changed:
            discrepancies.append(dict(source_position=position, level=level, phase='prefix',
                original_reward=old.get('reward'), fresh_reward=actual.get('reward')))
        both = eligible(info, old_reference, old) and eligible(info, reference, actual) and not changed
        disposition = ('nontrivial_in_both_complete_control_runs' if both else
            'fresh_execution_disagreement' if changed else 'trivial_prefix_rejected' if actual.get('reward') == 1 else
            'execution_abstention' if reference.get('reward') is None or actual.get('reward') is None else
            'full_reference_not_complete_credit' if reference.get('reward') != 1 else
            'embedded_original_boundary_token' if info['original_has_embedded_boundary_tokens'] else
            'no_actual_hint_at_this_nominal_level')
        catalogue.append(dict(source_position=position, corpus_id=row['id'], nominal_missing_non_EOS_tokens=level,
            exact_native_descriptor=info, disposition=disposition, bounded_nontrivial_control_evidence=both,
            all_original_unconditional_CE_targets_retained=True, source_training_admitted=False, production_RL_admitted=False))
        actual_views.append(dict(nominal_missing_non_EOS_tokens=level, fresh_prefix_execution=actual,
            execution_disagreed=changed, disposition=disposition))
        counts.update(fresh_prefix_control_runs=1, all_native_views_rebuilt=1,
            bounded_nontrivial_views_in_both_runs=int(both), fresh_prefix_reward1_rejected=int(actual.get('reward') == 1.))
        totals[str(level)] = dict(views=1, original_context_tokens=info['original_context_tokens'],
            original_response_EOS_targets=info['original_response_EOS_targets'])
    record = dict(source_position=position, corpus_id=row['id'], fresh_full_reference_execution=reference,
        full_reference_execution_disagreed=changed_reference, all_four_fresh_views=actual_views,
        no_student_generation_or_training=True)
    return dict(record=record, catalogue=catalogue, counts=counts, level_totals=totals, discrepancies=discrepancies)


def validate_retained(corpus, position, original, metadata, observed, catalogue):
    if len(observed['all_four_fresh_views']) != 4:
        raise ValueError('Every closed independent source needs all four original fresh execution observations')
    result = assemble(corpus, position, original, metadata, observed['fresh_full_reference_execution'],
        [v['fresh_prefix_execution'] for v in observed['all_four_fresh_views']])
    if observed != result['record'] or catalogue != result['catalogue']:
        raise ValueError('Every retained raw observation and independently reconstructed disposition must agree')
    return result


def fresh(corpus, position, original, metadata):
    row, views = prepare(corpus, position, original, metadata)

    def score(text):
        try:
            return corpus.score(position, text, stopped_on_eos=True)
        except Exception as exc:
            return dict(reward=None, status='preserved_execution_adapter_abstention', error=repr(exc))

    reference = score(row['original']['response'])
    prefixes = [score(v['original_prefix_text']) for v in views]
    return assemble(corpus, position, original, metadata, reference, prefixes)
