"""Complete-source worst cases for later owned four-GPU RL capacity.

Choose intact actual training batches, then explicitly rotate each whole batch
so every physical replica sees every selected worst source. A separate genuine
one-source control exercises zero-gradient ranks; the full2568-source epoch is
divisible by four and has no actual empty training tail. This is no admission.
"""
from collections import Counter
import copy

from .parallel_function_repair_stage_v1 import value_digest
from .public_reasoning_train_data_v1 import PublicReasoningTrainData

FORMAT = 'own-complete-public-reasoning-four-replica-capacity-cases-v1'


def choose(data, *, seed, epochs):
    if type(data) is not PublicReasoningTrainData or data.role != 'train' or data.purpose != 'optimization':
        raise ValueError('Only the exact independently admitted complete train partition is allowed')
    data.guard.check()
    entries = data.complete_schedule(epochs=epochs, width=4, seed=seed)
    budget = data.budget(entries)
    information = [data.information(p) for p in data.slots]
    tasks = sorted({r['task'] for r in information})
    if (data.slots != list(range(2568)) or len(information) != 2568 or len(tasks) != 12
            or Counter(p for e in entries for p in e) != Counter({p: epochs for p in data.slots})
            or budget['source_presentations'] != 2568 * epochs or budget['response_EOS_targets'] != 19080 * epochs
            or any(len(e) != 4 for e in entries)):
        raise ValueError('Every original source/target in every complete declared epoch is required')
    first_membership = {p: i for i, e in reversed(list(enumerate(entries))) for p in e}
    chosen = {}
    def add(index, reason): chosen.setdefault(index, []).append(reason)
    for task in tasks:
        subset = [r for r in information if r['task'] == task]
        for key in ['original_prompt_positions', 'complete_context_tokens', 'response_EOS_targets']:
            worst = max(subset, key=lambda r: (r[key], -r['position']))
            add(first_membership[worst['position']], dict(task=task, maximum=key,
                original_position=worst['position'], actual_value=worst[key]))
    for key in ['complete_context_tokens', 'response_EOS_targets']:
        number = max(range(len(entries)), key=lambda i: (sum(information[p][key] for p in entries[i]), -i))
        add(number, dict(maximum='whole_actual_batch_' + key,
            actual_value=sum(information[p][key] for p in entries[number])))
    shortest = min(information, key=lambda r: (r['complete_context_tokens'], r['position']))
    add(first_membership[shortest['position']], dict(minimum='complete_context_tokens',
        original_position=shortest['position'], actual_value=shortest['complete_context_tokens']))
    selected, cases, mapping = [], [], []
    for original_step in sorted(chosen):
        original = entries[original_step]
        selected.append(dict(original_global_step=original_step + 1, intact_original_positions=list(original),
            original_complete_sources=[information[p] for p in original], reasons=copy.deepcopy(chosen[original_step])))
        for rotation in range(4):
            actual = original[rotation:] + original[:rotation]
            cases.append(list(actual))
            mapping.append(dict(capacity_step=len(cases), original_global_step=original_step + 1,
                original_intact_batch=list(original), actual_original_positions=list(actual),
                whole_batch_rank_rotation=rotation, capacity_permutation_only=rotation != 0,
                actual_rank_source_counts=[1, 1, 1, 1], original_training_schedule_untouched=True))
    worst = max(information, key=lambda r: (r['original_prompt_positions'] + r['full_training_action_budget'], -r['position']))
    cases.append([worst['position']])
    mapping.append(dict(capacity_step=len(cases), actual_original_positions=[worst['position']],
        actual_rank_source_counts=[1, 0, 0, 0], separate_single_source_zero_gradient_control=True,
        actual_original_training_tail=False, intact_original_source=worst))
    return dict(format=FORMAT, complete_source_selection=True, mechanics_only=False, seed=seed, epochs=epochs, world=4,
        full_train_source_manifest_sha256=data.manifest_sha256, complete_training_schedule_identity=data.identity(entries),
        complete_training_schedule_sha256=value_digest(entries), complete_training_budget=budget,
        actual_train_task_families=tasks, actual_original_train_source_population=2568,
        every_selected_original_batch_and_positive_target_preserved=True,
        selected_intact_original_batches=selected, capacity_entries=cases, capacity_case_mapping=mapping,
        capacity_budget=data.budget(cases), all_four_replicas_receive_each_selected_source=True,
        actual_original_training_has_empty_tail=False,
        maximum_original_prompt_positions=max(r['original_prompt_positions'] for r in information),
        maximum_original_complete_context=max(r['complete_context_tokens'] for r in information),
        maximum_original_response_EOS_targets=max(r['response_EOS_targets'] for r in information),
        maximum_original_prompt_plus_full_horizon=max(r['original_prompt_positions'] + r['full_training_action_budget'] for r in information),
        full_original_action_budgets=sorted({r['full_training_action_budget'] for r in information}),
        all_source_horizons_and_unconditional_complete_CE_anchors_retained=True,
        source_selection_only=True, models_constructed=0, GPU_contexts=0, optimizer_updates=0,
        training_admitted=False, full_size_capacity_passed=False, goal_achieved=False)
