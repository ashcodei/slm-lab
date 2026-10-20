"""Select whole admitted train batches for the actual four-GPU capacity gate."""
from collections import Counter
import copy

from .public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from .parallel_function_repair_stage_v1 import value_digest

FORMAT = 'own-public-reasoned-code-complete-capacity-source-selection-v1'


def choose(data, *, epochs=1, seed=990231):
    if (type(data) is not PublicReasonedCodeSFTData or data.role != 'train'
            or data.purpose not in ('optimization','capacity') or data.views != 'both'
            or data.use_as_RL_reward_oracle_forbidden is not True):
        raise ValueError('Exact independently admitted train-only whole code/reasoning selection required')
    data.guard.check()
    entries = data.complete_schedule(epochs=epochs, width=16, seed=seed)
    budget = data.budget(entries)
    if Counter(p for entry in entries for p in entry) != Counter({p:epochs for p in data.slots}):
        raise ValueError('Every original admitted train view must occur once per declared epoch')
    for key, source in [('source_presentations','examples'),('complete_context_tokens','context_tokens'),('response_EOS_targets','target_tokens')]:
        if budget[key] != epochs*data.pack.manifest['counts']['train'][source]:
            raise ValueError('Complete admitted original train contexts and response/EOS budgets required')
    count = len(data)
    updates_per_epoch = (count+15)//16
    tail = count%16 or 16
    if any(len(entry) != (tail if (i+1)%updates_per_epoch == 0 else 16) for i,entry in enumerate(entries)):
        raise ValueError('Original epoch tails cannot be cropped, rounded or replayed')
    info = [data.information(p) for p in data.slots]
    membership = {p:i for i,entry in reversed(list(enumerate(entries))) for p in entry}
    selected = {}
    def add(index, reason): selected.setdefault(index,[]).append(reason)
    views = sorted({row['view'] for row in info})
    for view in views:
        rows = [row for row in info if row['view'] == view]
        for key in ['original_prompt_positions','complete_context_tokens','response_EOS_targets']:
            row = max(rows, key=lambda row:(row[key],-row['position']))
            add(membership[row['position']], dict(view=view, maximum=key, original_position=row['position'], actual_value=row[key]))
    for key in ['complete_context_tokens','response_EOS_targets']:
        index = max(range(len(entries)), key=lambda i:(sum(info[p][key] for p in entries[i]),-i))
        add(index, dict(maximum='whole_actual_batch_'+key, actual_value=sum(info[p][key] for p in entries[index])))
        index, rank = max(((i,r) for i in range(len(entries)) for r in range(4)),
            key=lambda ir:(sum(info[p][key] for p in entries[ir[0]][ir[1]::4]),-ir[0],-ir[1]))
        add(index, dict(maximum='actual_rank_shard_'+key, original_rank=rank,
            actual_value=sum(info[p][key] for p in entries[index][rank::4])))
    for i,entry in enumerate(entries):
        if (i+1)%updates_per_epoch == 0:
            add(i, dict(actual_original_epoch_end=True, genuine_tail_sources=tail,
                actual_rank_source_counts=[len(entry[r::4]) for r in range(4)]))
    chosen, cases, mapping = [], [], []
    for i in sorted(selected):
        original = entries[i]
        chosen.append(dict(original_global_step=i+1, intact_original_positions=list(original),
            original_complete_sources=[info[p] for p in original], reasons=copy.deepcopy(selected[i])))
        for rotation in range(4 if len(original)%4 == 0 else len(original)):
            actual = original[rotation:]+original[:rotation]
            cases.append(actual)
            mapping.append(dict(capacity_step=len(cases), original_global_step=i+1,
                original_intact_batch=list(original), actual_original_positions=list(actual), whole_batch_rank_rotation=rotation,
                actual_rank_source_counts=[len(actual[r::4]) for r in range(4)], original_training_schedule_untouched=True))
    selected_sources = {p for batch in chosen for p in batch['intact_original_positions']}
    def seen_ranks(position):
        return {i%4 for entry in cases for i,p in enumerate(entry) if p == position}
    # A short original tail cannot assign its sources to all four ranks by
    # cyclic rotations alone. Add explicitly separate capacity-only groups of
    # four distinct whole admitted sources. Never insert these into training.
    for position in sorted(selected_sources):
        if seen_ranks(position) == {0,1,2,3} or len(data) < 4:
            continue
        others = sorted((p for p in data.slots if p != position),
            key=lambda p:(-info[p]['complete_context_tokens'],p))[:3]
        original = [position,*others]
        for rotation in range(4):
            actual = original[rotation:]+original[:rotation]
            cases.append(actual)
            mapping.append(dict(capacity_step=len(cases), actual_original_positions=list(actual),
                actual_rank_source_counts=[1,1,1,1], separate_whole_source_rank_coverage_control=True,
                original_source_requiring_all_rank_coverage=position, whole_batch_rank_rotation=rotation,
                actual_original_training_batch=False, original_training_schedule_untouched=True))
    complete_rank_coverage = all(seen_ranks(position) == {0,1,2,3} for position in selected_sources)
    worst = max(info,key=lambda row:(row['complete_context_tokens'],-row['position']))
    cases.append([worst['position']])
    mapping.append(dict(capacity_step=len(cases), actual_original_positions=[worst['position']],
        actual_rank_source_counts=[1,0,0,0], separate_single_source_zero_gradient_control=True,
        actual_original_training_tail=False, intact_original_source=worst))
    return dict(format=FORMAT, complete_source_selection=True, mechanics_only=False, world=4,
        epochs=epochs, seed=seed, width=16, dataset_manifest_sha256=data.manifest_sha256,
        native_admission_sha256=data.admission_sha256, historical_provenance_identity=data.provenance_identity,
        complete_training_schedule_identity=data.identity(entries), complete_training_schedule_sha256=value_digest(entries),
        complete_training_budget=budget, selected_intact_original_batches=chosen, capacity_entries=cases,
        capacity_case_mapping=mapping, capacity_budget=data.budget(cases),
        all_four_replicas_receive_each_selected_source_in_declared_plan=complete_rank_coverage,
        actual_original_training_has_empty_ranks=tail<4,
        actual_complete_training_tail_sources_per_rank=[len(entries[-1][r::4]) for r in range(4)],
        original_views=views, source_selection_only=True, models_constructed=0,
        neural_forwards=0, optimizer_updates=0, GPUcontexts=0, training_admitted=False,
        full_size_capacity_passed=False, public_synthetic_weak_SFT_only=True, goal_achieved=False)
