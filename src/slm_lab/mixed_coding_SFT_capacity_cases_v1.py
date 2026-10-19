"""Actual intact worst source batches for the later four-GPU coding stage."""
from collections import Counter
import copy

from .mixed_coding_SFT_data_v1 import MixedCodingSFTData, FAMILIES, EXPECTED
from .parallel_function_repair_stage_v1 import value_digest

FORMAT = 'own-complete-mixed-coding-SFT-capacity-source-selection-v1'


def choose(data, *, epochs=3, seed=989923):
    if type(data) is not MixedCodingSFTData or data.role != 'train' or data.purpose != 'optimization':
        raise ValueError('Only the exact complete train-only mixed curriculum may define capacity cases')
    data.guard.check()
    entries = data.complete_schedule(epochs=epochs, width=16, seed=seed)
    budget = data.budget(entries)
    if (Counter(p for e in entries for p in e) != Counter({p: epochs for p in data.slots})
            or any(budget[k] != epochs * v for k, v in EXPECTED.items())
            or any(len(e) != (12 if (i + 1) % 1752 == 0 else 16) for i, e in enumerate(entries))):
        raise ValueError('Every complete source epoch, full target and genuine12-source tail must remain')
    info = [data.information(p) for p in data.slots]
    membership = {p: i for i, e in reversed(list(enumerate(entries))) for p in e}
    selected = {}
    def add(i, reason): selected.setdefault(i, []).append(reason)
    for family in FAMILIES:
        rows = [r for r in info if r['family'] == family]
        for key in ['original_prompt_positions', 'complete_context_tokens', 'response_EOS_targets']:
            r = max(rows, key=lambda r: (r[key], -r['position']))
            add(membership[r['position']], dict(family=family, maximum=key, original_position=r['position'], actual_value=r[key]))
    for key in ['complete_context_tokens', 'response_EOS_targets']:
        i = max(range(len(entries)), key=lambda i: (sum(info[p][key] for p in entries[i]), -i))
        add(i, dict(maximum='whole_actual_batch_' + key, actual_value=sum(info[p][key] for p in entries[i])))
        # Individual rank sum can dominate activation accumulation despite the
        # same complete global batch. Include the actual worst observed shard.
        i, rank = max(((i, rank) for i in range(len(entries)) for rank in range(4)),
            key=lambda ir: (sum(info[p][key] for p in entries[ir[0]][ir[1]::4]), -ir[0], -ir[1]))
        add(i, dict(maximum='actual_rank_shard_' + key, original_rank=rank,
            actual_value=sum(info[p][key] for p in entries[i][rank::4])))
    for i, e in enumerate(entries):
        if len(e) == 12: add(i, dict(actual_complete_training_tail=True, actual_rank_source_counts=[3, 3, 3, 3]))
    chosen, cases, mapping = [], [], []
    for i in sorted(selected):
        original = entries[i]
        chosen.append(dict(original_global_step=i + 1, intact_original_positions=list(original),
            original_complete_sources=[info[p] for p in original], reasons=copy.deepcopy(selected[i])))
        for rotation in range(4):
            actual = original[rotation:] + original[:rotation]
            cases.append(actual)
            mapping.append(dict(capacity_step=len(cases), original_global_step=i + 1,
                original_intact_batch=list(original), actual_original_positions=list(actual), whole_batch_rank_rotation=rotation,
                actual_rank_source_counts=[len(actual[r::4]) for r in range(4)], original_training_schedule_untouched=True))
    worst = max(info, key=lambda r: (r['complete_context_tokens'], -r['position']))
    cases.append([worst['position']])
    mapping.append(dict(capacity_step=len(cases), actual_original_positions=[worst['position']],
        actual_rank_source_counts=[1, 0, 0, 0], separate_single_source_zero_gradient_control=True,
        actual_original_training_tail=False, intact_original_source=worst))
    return dict(format=FORMAT, complete_source_selection=True, mechanics_only=False, world=4,
        epochs=epochs, seed=seed, width=16, complete_mixed_source_identity_sha256=data.identity_sha256,
        complete_training_schedule_identity=data.identity(entries), complete_training_schedule_sha256=value_digest(entries),
        complete_training_budget=budget, selected_intact_original_batches=chosen, capacity_entries=cases,
        capacity_case_mapping=mapping, capacity_budget=data.budget(cases),
        all_four_replicas_receive_each_selected_source=True, actual_original_training_has_empty_ranks=False,
        actual_complete_training_tail_sources_per_rank=[3, 3, 3, 3],
        original_families=list(FAMILIES), source_selection_only=True, models_constructed=0,
        neural_forwards=0, optimizer_updates=0, GPUcontexts=0, training_admitted=False,
        full_size_capacity_passed=False, goal_achieved=False)
