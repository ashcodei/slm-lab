"""Whole reserved reasoning population with actual native greedy GPU actions.

Only this curriculum's declared reserved role is established; no historical
exposure or semantic independence is asserted. Explicit small CPU mechanics
shares the transport/scorer but never grants full-size GPU admission.
"""
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch

from .corpus import file_hash
from .diverse_images import atomic_json
from .inference_rotary_cache_v1 import inference_cache
from .native_greedy_observer_v1 import observe, signature
from .public_reasoning_reserved_evaluation_v1 import PublicReasoningReserved

FORMAT = 'own-full-source-reserved-reasoning-native-shard-v1'


@torch.no_grad()
def evaluate(model, population, path, *, rank, world, checkpoint_sha256, checked,
             record, allow_CPU_mechanics=False):
    if (type(population) is not PublicReasoningReserved or population.purpose != 'evaluation'
            or type(world) is not int or world != 4 or type(rank) is not int or not 0 <= rank < world
            or model.training or not callable(checked) or not callable(record) or Path(path).exists()
            or len(checkpoint_sha256) != 64 or any(p.grad is not None for p in model.parameters())):
        raise ValueError('Exact entire evaluation-only population, four original shards and fresh actual native deliveries required')
    device = next(model.parameters()).device
    count = sum(p.numel() for p in model.parameters())
    if (device.type == 'cuda' and count != 984399690
            or device.type != 'cuda' and not (device.type == 'cpu' and allow_CPU_mechanics is True
                and count < 10**6 and not torch.cuda.is_initialized())):
        raise ValueError('Actual full-size GPU native or explicit own-small CPU mechanics required')
    population.guard.check(full=True)
    before, rows = signature(model), []
    selected = list(range(rank, len(population.cases), world))
    with torch.autocast(device.type, dtype=torch.bfloat16), inference_cache(), observe(model, record) as events:
        for position in selected:
            checked()
            case = population.cases[position]
            prefix = case['original_prompt_ids']
            tensor = torch.tensor([prefix], dtype=torch.long, device=device)
            generated = model.generate(tensor, max_new_tokens=case['native_generation_budget'],
                temperature=0., eos_token_id=2, use_cache=True)
            actions = generated[0, len(prefix):].tolist()
            event = events[-1]
            if (event['original_input_token_ids'] != prefix or event['actual_generated_token_ids'] != actions
                    or event['requested_native_action_budget'] != case['native_generation_budget']):
                raise ValueError('Each actual whole original prompt/action/EOS/budget must match its observed delivery')
            row = dict(position=position, source_id=case['source_id'], source_group=case['source_group'], task=case['task'],
                original_prompt_ids=list(prefix), actual_generated_token_ids=actions,
                original_native_generation_budget=case['native_generation_budget'],
                score=population.grade(position, actions), actual_native_event_position=event['case_position'])
            rows.append(row)
    if signature(model) != before or len(rows) != len(selected) or any(p.grad is not None for p in model.parameters()):
        raise ValueError('All original assigned cases and unchanged actual model values are required')
    population.guard.check(full=True)
    result = dict(format=FORMAT, complete=True, rank=rank, world=world, results=rows,
        selected_original_positions=selected, checkpoint_sha256=checkpoint_sha256, actual_student_values=count,
        inference_device_type=device.type, inference_precision='bf16_autocast',
        CPU_mechanics_only=device.type == 'cpu', source_role='validation', optimization_forbidden=True,
        source_reserved_only_historical_exposure_not_certified=True, actual_native_actions_unchanged=True,
        original_complete_population=289, original_full_native_budgets=[64, 256], model_updates=0, goal_achieved=False)
    atomic_json(path, result)
    return result


def reconcile(population, paths, output):
    import json
    if type(population) is not PublicReasoningReserved or len(paths) != 4 or Path(output).exists():
        raise ValueError('Every full reserved shard and fresh complete output are required')
    reports = [json.loads(Path(path).read_text()) for path in paths]
    rows = []
    fields = ['checkpoint_sha256', 'actual_student_values', 'inference_device_type', 'inference_precision', 'CPU_mechanics_only']
    for rank, report in enumerate(reports):
        selected = list(range(rank, 289, 4))
        if (report['format'] != FORMAT or not report['complete'] or report['rank'] != rank or report['world'] != 4
                or report['selected_original_positions'] != selected or report['source_role'] != 'validation'
                or report['optimization_forbidden'] is not True or [r['position'] for r in report['results']] != selected
                or any(report[k] != reports[0][k] for k in fields)):
            raise ValueError('Every original source shard and identical actual inference protocol must remain')
        for row in report['results']:
            case = population.cases[row['position']]
            if (row['source_id'] != case['source_id'] or row['source_group'] != case['source_group'] or row['task'] != case['task']
                    or row['original_prompt_ids'] != case['original_prompt_ids']
                    or row['original_native_generation_budget'] != case['native_generation_budget']
                    or row['score'] != population.grade(row['position'], row['actual_generated_token_ids'])):
                raise ValueError('Every actual original prompt, native action and strict reserved verdict must reconstruct')
        rows.extend(report['results'])
    rows.sort(key=lambda r: r['position'])
    if [r['position'] for r in rows] != list(range(289)) or len({r['task'] for r in rows}) != 12:
        raise ValueError('All289reserved sources and all12tasks are required')
    result = dict(complete=True, examples=289, correct=sum(r['score']['correct'] for r in rows), results=rows,
        per_task={task: dict(examples=sum(r['task'] == task for r in rows),
            correct=sum(r['task'] == task and r['score']['correct'] for r in rows)) for task in sorted({r['task'] for r in rows})},
        **{k: reports[0][k] for k in fields}, all_original_whole_prompts_and_full64or256budgets_retained=True,
        model_inputs_exclude_reference_answer_tokens=True, source_role='validation', optimization_forbidden=True,
        historical_exposure_or_semantic_independence_not_certified=True,
        source_files_sha256={str(Path(p).resolve()): file_hash(p) for p in paths}, model_updates=0,
        candidate_promoted=False, goal_achieved=False)
    atomic_json(output, result)
    return result
