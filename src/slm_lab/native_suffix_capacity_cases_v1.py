"""Whole original native-code batches for later four-GPU capacity checks.

The geometry-only path uses every original human program without granting
execution eligibility. The production path requires the separate, complete
native-data admission and adds maxima from the actually qualified population.
Neither path constructs a model, launches a worker, or admits GPU training.
"""
from collections import Counter
import copy

from .native_suffix_train_data_v1 import NativeSuffixTrainData, LEVELS, schedule_source_views
from .parallel_function_repair_stage_v1 import value_digest
from .public_token_suffix_curriculum_v1 import view, descriptor

FORMAT = 'own-complete-native-suffix-four-replica-capacity-cases-v1'
GEOMETRY = 'own-original-human-native-suffix-source-geometry-v1'
KEYS = ('original_context_tokens', 'original_response_EOS_targets', 'conditional_prefix_tokens',
    'full_native_action_budget', 'prefix_plus_full_native_horizon')


def source_geometry(corpus):
    """Read all real source masks/IDs; no native execution eligibility is inferred."""
    from .expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    if (type(corpus) is not ExpandedExecutionCorpus or corpus.role != 'train'
            or corpus.purpose != 'execution_reward_and_complete_reference_CE' or len(corpus) != 3721):
        raise ValueError('The exact complete admitted base human train corpus is required')
    corpus.guard.check()
    rows = []
    for source in range(3721):
        original = corpus.row(source)
        ids, labels = corpus.sequence(source)
        for j, level in enumerate(LEVELS):
            prepared = view(corpus.tokenizer, original['original'], ids, labels, level)
            rows.append(dict(position=4 * source + j, original_source_position=source,
                original_source_id=original['id'], kind=original['kind'], level=level,
                original_context_tokens=len(ids), original_response_EOS_targets=sum(t != -100 for t in labels),
                original_complete_input_sha256=prepared['original_complete_input_sha256'],
                original_full_CE_labels_sha256=prepared['original_unconditional_labels_sha256'],
                conditional_prefix_tokens=prepared['conditional_rollout_prefix_tokens'],
                supplied_public_reference_tokens=prepared['retained_reference_tokens'],
                full_native_action_budget=prepared['full_native_action_budget'],
                native_execution_eligible=None, native_qualification_unknown=True, original_role='train',
                full_source_data_admitted=False, exact_native_descriptor=descriptor(prepared)))
    corpus.guard.check()
    if (len(rows) != 14884 or any(sum(r['original_context_tokens'] for r in rows if r['level'] == level) != 2287522
            or sum(r['original_response_EOS_targets'] for r in rows if r['level'] == level) != 584218 for level in LEVELS)):
        raise ValueError('Every complete original source/view/whole CEtarget must remain in the source-only geometry')
    return dict(format=GEOMETRY, records=rows, complete_original_human_sources=3721,
        all_original_native_views=14884, original_levels=list(LEVELS), source_geometry_sha256=value_digest(rows),
        native_execution_qualification_or_training_not_granted=True, models_constructed=0, neural_forwards=0,
        GPUcontexts=0, optimizer_updates=0, goal_achieved=False)


def _budget(rows, entries):
    flat = [rows[p] for e in entries for p in e]
    qualified = all(type(r['native_execution_eligible']) is bool for r in flat)
    return dict(updates=len(entries), source_presentations=len(flat),
        distinct_original_human_sources=len({r['original_source_position'] for r in flat}),
        complete_original_context_tokens=sum(r['original_context_tokens'] for r in flat),
        original_response_EOS_targets=sum(r['original_response_EOS_targets'] for r in flat),
        full_native_generation_horizon_sum=sum(r['full_native_action_budget'] for r in flat),
        execution_eligible_presentations=sum(r['native_execution_eligible'] for r in flat) if qualified else None,
        CE_only_presentations=sum(not r['native_execution_eligible'] for r in flat) if qualified else None,
        actual_single_source_three_empty_rank_steps=sum(len(e) == 1 for e in entries))


def _select(information, *, seed, epochs, admitted, identity, admission_sha256=None, manifest_sha256=None):
    if (type(seed) is not int or seed < 1 or seed >= 2**64 or epochs != 3 or type(epochs) is not int
            or len(information) != 14884 or [r['position'] for r in information] != list(range(14884))):
        raise ValueError('The full original three-epoch native-code experiment and exact source population are required')
    rows = copy.deepcopy(information)
    for r in rows:
        source, level_index = divmod(r['position'], 4)
        if (r['original_source_position'] != source or r['level'] != LEVELS[level_index] or r['original_role'] != 'train'
                or r['kind'] != ('documented' if source < 173 else 'apps') or r['full_source_data_admitted'] is not admitted
                or any(type(r[k]) is not int or r[k] < 1 for k in ['original_context_tokens', 'original_response_EOS_targets',
                    'conditional_prefix_tokens', 'full_native_action_budget'])
                or r['original_response_EOS_targets'] > r['original_context_tokens'] - 1
                or type(r['supplied_public_reference_tokens']) is not int or not 0 <= r['supplied_public_reference_tokens'] < r['original_response_EOS_targets']
                or r['conditional_prefix_tokens'] != r['original_context_tokens'] - r['original_response_EOS_targets'] + r['supplied_public_reference_tokens']
                or r['original_context_tokens'] - r['conditional_prefix_tokens'] > r['full_native_action_budget']
                or (r['supplied_public_reference_tokens'] != 0 if r['level'] == 0 else
                    r['supplied_public_reference_tokens'] > max(0, r['original_response_EOS_targets'] - 1 - r['level']))
                or r['conditional_prefix_tokens'] + r['full_native_action_budget'] > 4096
                or (type(r['native_execution_eligible']) is not bool if admitted else r['native_execution_eligible'] is not None)):
            raise ValueError('Every original exact native source/view/mask/horizon and actual eligibility disposition is required')
        r['prefix_plus_full_native_horizon'] = r['conditional_prefix_tokens'] + r['full_native_action_budget']
    for source in range(3721):
        views = rows[4*source:4*source+4]
        if any(len({r[k] for r in views}) != 1 for k in ['original_source_id', 'original_context_tokens',
                'original_response_EOS_targets', 'original_complete_input_sha256', 'original_full_CE_labels_sha256']):
            raise ValueError('Every hinted view must retain exactly the same original whole source/CE anchor')
    if len({r['original_source_id'] for r in rows}) != 3721:
        raise ValueError('All 3721 distinct original human source identities are required')
    if (Counter(r['kind'] for r in rows) != Counter(documented=173 * 4, apps=3548 * 4)
            or any(sum(r['original_context_tokens'] for r in rows if r['level'] == level) != 2287522
                or sum(r['original_response_EOS_targets'] for r in rows if r['level'] == level) != 584218 for level in LEVELS)):
        raise ValueError('All 3721 original human sources and complete context/CEtarget budgets must remain')
    entries = schedule_source_views(epochs=epochs, width=4, seed=seed)
    budget = _budget(rows, entries)
    if (len(entries) != 11172 or Counter(p for e in entries for p in e) != Counter({p: 3 for p in range(14884)})
            or budget['source_presentations'] != 44652 or budget['original_response_EOS_targets'] != 7010616
            or budget['actual_single_source_three_empty_rank_steps'] != 12):
        raise ValueError('The complete original source/view schedule and every real level-boundary tail must remain')
    first_full = {p:i for i,e in reversed(list(enumerate(entries))) if len(e) == 4 for p in e}
    chosen = {}
    def add(index, reason): chosen.setdefault(index, []).append(reason)
    def include_source(position, reason):
        if position not in first_full:
            raise ValueError('A selected source lacks a complete original batch for actual all-four-replica rotation')
        add(first_full[position], reason)
    for kind in ['documented', 'apps']:
        for level in LEVELS:
            population = [r for r in rows if r['kind'] == kind and r['level'] == level]
            # Every view contributes its full anchor, including CE-only views.
            for key in ['original_context_tokens', 'original_response_EOS_targets']:
                worst = max(population, key=lambda r:(r[key], -r['position']))
                include_source(worst['position'], dict(kind=kind, level=level, population='all_original_CE_views',
                    maximum=key, original_position=worst['position'], actual_value=worst[key]))
            rollout = [r for r in population if r['native_execution_eligible']] if admitted else population
            for key in ['conditional_prefix_tokens', 'full_native_action_budget', 'prefix_plus_full_native_horizon']:
                if not rollout: continue
                worst = max(rollout, key=lambda r:(r[key], -r['position']))
                include_source(worst['position'], dict(kind=kind, level=level,
                    population='actually_qualified_native_views' if admitted else 'unqualified_geometry_upper_bound_only',
                    maximum=key, original_position=worst['position'], actual_value=worst[key]))
    for key in ['original_context_tokens', 'original_response_EOS_targets']:
        index = max(range(len(entries)), key=lambda i:(sum(rows[p][key] for p in entries[i]), -i))
        add(index, dict(maximum='whole_actual_batch_' + key, actual_value=sum(rows[p][key] for p in entries[index])))
    shortest = min(rows, key=lambda r:(r['original_context_tokens'], r['position']))
    include_source(shortest['position'], dict(minimum='original_context_tokens', original_position=shortest['position'],
        actual_value=shortest['original_context_tokens']))
    tails = [i for i,e in enumerate(entries) if len(e) == 1]
    for index in tails:
        add(index, dict(actual_original_training_tail=True, original_position=entries[index][0]))
        include_source(entries[index][0], dict(physical_rotation_of_actual_tail_source=True, original_position=entries[index][0]))
    selected, cases, mapping = [], [], []
    rotated_sources = set()
    for step in sorted(chosen):
        original = entries[step]
        selected.append(dict(original_global_step=step + 1, intact_original_positions=list(original),
            original_complete_sources=[rows[p] for p in original], reasons=copy.deepcopy(chosen[step])))
        for rotation in range(4 if len(original) == 4 else 1):
            actual = original[rotation:] + original[:rotation]
            cases.append(actual)
            mapping.append(dict(capacity_step=len(cases), original_global_step=step + 1,
                original_intact_batch=list(original), actual_original_positions=list(actual), whole_batch_rank_rotation=rotation,
                actual_rank_source_counts=[int(r < len(actual)) for r in range(4)],
                actual_original_training_tail=len(original) == 1, original_training_schedule_untouched=True))
        if len(original) == 4: rotated_sources.update(original)
    if not all(p in rotated_sources for e in cases for p in e):
        raise ValueError('Every selected source, including each real tail source, must reach every physical replica in a full batch')
    return dict(format=FORMAT, geometry_only=not admitted, full_source_data_admitted=admitted,
        complete_source_selection=True, mechanics_only=False, source_selection_only=True,
        seed=seed, epochs=epochs, world=4, complete_training_source_identity=identity,
        native_data_admission_sha256=admission_sha256, native_data_manifest_sha256=manifest_sha256,
        complete_training_schedule_sha256=value_digest(entries), complete_training_budget=budget,
        original_human_sources=3721, original_source_views=14884, original_levels=list(LEVELS),
        actual_source_strata=[dict(kind=k, level=l) for k in ['documented', 'apps'] for l in LEVELS],
        selected_intact_original_batches=selected, capacity_entries=cases, capacity_case_mapping=mapping,
        capacity_budget=_budget(rows, cases), all_selected_sources_physically_rotated_to_each_replica=True,
        all_12_real_single_source_three_empty_rank_training_tails_retained=True,
        original_tail_global_steps=[i+1 for i in tails],
        full_original_action_budgets=sorted({r['full_native_action_budget'] for r in rows}),
        maximum_original_context=max(r['original_context_tokens'] for r in rows),
        maximum_original_CE_targets=max(r['original_response_EOS_targets'] for r in rows),
        maximum_original_conditional_prefix=max(r['conditional_prefix_tokens'] for r in rows),
        maximum_original_prefix_plus_full_native_horizon=max(r['prefix_plus_full_native_horizon'] for r in rows),
        actual_execution_eligible_views=sum(bool(r['native_execution_eligible']) for r in rows) if admitted else None,
        all_original_anchors_and_failed_CE_only_dispositions_retained=True,
        models_constructed=0, neural_forwards=0, GPUcontexts=0, optimizer_updates=0,
        training_admitted=False, full_size_capacity_passed=False, goal_achieved=False)


def choose_geometry(geometry, *, seed=989711, epochs=3):
    if (geometry.get('format') != GEOMETRY or geometry.get('native_execution_qualification_or_training_not_granted') is not True
            or value_digest(geometry['records']) != geometry['source_geometry_sha256']):
        raise ValueError('Only complete actual original-source geometry is allowed, without inferred execution eligibility')
    return _select(geometry['records'], seed=seed, epochs=epochs, admitted=False, identity=geometry['source_geometry_sha256'])


def choose_admitted(data, *, seed=989711, epochs=3):
    if (type(data) is not NativeSuffixTrainData or data.role != 'train' or data.purpose not in ['optimization', 'capacity']
            or not data.full_source_data_admitted or data.admission_sha256 is None or len(data) != 14884):
        raise ValueError('Separate complete original native-source data admission is required for production capacity selection')
    data.guard.check()
    entries = data.complete_schedule(epochs=epochs, width=4, seed=seed)
    plan = _select([data.information(p) for p in data.slots], seed=seed, epochs=epochs, admitted=True,
        identity=data.identity(entries), admission_sha256=data.admission_sha256, manifest_sha256=data.manifest_sha256)
    if plan['complete_training_budget'] != data.budget(entries) or plan['capacity_budget'] != data.budget(plan['capacity_entries']):
        raise ValueError('The actual downstream native-data schedule/budget must agree with every selected original source')
    data.guard.check()
    return plan
