"""Reconstruct complete source draws and reference-bound own trained weights."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .apps_native_metrics_v1 import summarize
from .code_quality import execute, instruction_code
from .corpus import file_hash
from .diverse_images import atomic_json
from .own_code_delta_v1 import matrix_names, norm_names
from .own_code_training_stage_v1 import EpochOrder, LanguageCycleOrder, STREAMS


def exposure_budget(streams, options):
    exposures = options['steps'] * options['global_examples_per_stream']
    result = {}
    for i, (name, stream) in enumerate(streams.items()):
        order = (LanguageCycleOrder(stream.language_positions, options['seed'] + i * 1009) if name == 'variety'
                 else EpochOrder(len(stream), options['seed'] + i * 1009))
        positions = [order.at(p) for p in range(exposures)]
        indices = [stream.pack.index[stream.pack.positions[p]] for p in positions]
        targets = sum(int(row['length']) - int(row['target_start']) for row in indices)
        result[name] = dict(original_train_population=len(stream), actual_exposures=exposures,
            unique_original_positions=len(set(positions)), repeated_original_positions=exposures - len(set(positions)),
            complete_context_tokens=sum(int(row['length']) for row in indices), response_EOS_targets=targets,
            absolute_draw_positions_sha256=hashlib.sha256(np.asarray(positions, dtype='<i8').tobytes()).hexdigest())
        if name == 'variety':
            reverse = {p: language for language, values in stream.language_positions.items() for p in values}
            result[name]['sampled_original_languages'] = dict(Counter(reverse[p] for p in positions))
            result[name]['full_original_corpus_pass'] = len(set(positions)) == len(stream)
        else:
            result[name].update(complete_passes=exposures // len(stream), next_pass_remainder=exposures % len(stream))
    return result


def reconcile(root, reports, output, streams):
    root, reports, output = map(Path, [root, reports, output])
    plan = json.loads((reports / 'plan.json').read_text()); options = plan['training_options']
    receipt = json.loads((reports / 'training-complete.json').read_text())
    if (file_hash(output / 'resume.pt') != receipt['resumable_checkpoint_sha256']
            or file_hash(output / 'model.pt') != receipt['trained_components_sha256']):
        raise ValueError('Actual complete saved training and inference artifacts required')
    state = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
    export = torch.load(output / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    if (state['steps'] != options['steps'] or state['planned_steps'] != options['steps'] or state['world_size'] != 4
            or state['options'] != options or state['sources']['run_plan_sha256'] != file_hash(reports / 'plan.json')
            or len(state['worker_rng_states']) != 4 or state['actual_counts'] != plan['actual_model_counts']
            or export['own_code_training']['actual_steps'] != options['steps']
            or export['own_code_training']['run_plan_sha256'] != file_hash(reports / 'plan.json')):
        raise ValueError('Actual complete own training/source/budget/model provenance required')
    expected = {}
    for i, name in enumerate(matrix_names(state['config']['num_layers'])):
        for part in ['down', 'up']:
            expected[f'linears.{i}.{part}'] = export['components']['matrices'][name][part]
    expected['embedding.weight'] = export['components']['embedding']
    for i, name in enumerate(norm_names(state['config']['num_layers'])):
        expected[f'norms.{i}.weight'] = export['components']['norms'][name]
    if expected.keys() != state['parts'].keys() or sum(t.numel() for t in expected.values()) != plan['actual_model_counts']['trainable_code']:
        raise ValueError('Every unique trained own-code component required')
    for name, value in expected.items():
        if not torch.equal(value, state['parts'][name]) or not bool(torch.isfinite(value).all()):
            raise ValueError('Actual finite exported own-code components must equal complete resume values')
    optimizer = state['optimizer']; ordered = optimizer['param_groups'][0]['params']
    if len(optimizer['param_groups']) != 1 or len(ordered) != len(expected) or set(optimizer['state']) != set(ordered):
        raise ValueError('Every unique own-code component needs actual complete Adam state')
    for identity, value in zip(ordered, state['parts'].values()):
        moments = optimizer['state'][identity]
        if (float(moments['step']) != options['steps'] or any(moments[k].shape != value.shape
                or not bool(torch.isfinite(moments[k]).all()) for k in ['exp_avg', 'exp_avg_sq'])):
            raise ValueError('Complete finite exact-step own-code Adam moments required')
    if (file_hash(plan['base_checkpoint']) != plan['base_checkpoint_sha256']
            or export['base_checkpoint_sha256'] != plan['base_checkpoint_sha256']
            or state.get('own_base_is_immutable_reference') is not True
            or receipt['frozen_base_digest_before'] != receipt['frozen_base_digest_after']):
        raise ValueError('The entire immutable original general/vision/critic reference must remain identical')
    budget = exposure_budget(streams, options)
    if budget != plan['predeclared_exposure_budget']:
        raise ValueError('All original generation/repair/language-balanced replay source draws must reconstruct')
    if len(state['metrics']) != options['steps']:
        raise ValueError('Every actual update metric and original source draw required')
    for name in STREAMS:
        if sum(r['objectives'][name]['global_target_tokens'] for r in state['metrics']) != budget[name]['response_EOS_targets']:
            raise ValueError('Every separately normalized target-token count must match original complete packs')
    orders = {name: (LanguageCycleOrder(stream.language_positions, options['seed'] + i * 1009) if name == 'variety'
                    else EpochOrder(len(stream), options['seed'] + i * 1009))
              for i, (name, stream) in enumerate(streams.items())}
    exposures = options['steps'] * options['global_examples_per_stream']
    expected_cursors = {name: (orders[name].cursor(exposures) if name == 'variety' else
        dict(population=len(stream), seed=options['seed'] + i * 1009, actual_exposures=exposures,
             complete_passes=exposures // len(stream), next_pass_remainder=exposures % len(stream)))
        for i, (name, stream) in enumerate(streams.items())}
    if state['sampler_state'] != expected_cursors:
        raise ValueError('Every final original stream/language cursor must reconstruct')
    # Reconstruct each actual per-rank/micro draw, including repeated sources.
    for step, record in enumerate(state['metrics']):
        if record['step'] != step + 1 or [d['rank'] for d in record['worker_draws']] != list(range(4)):
            raise ValueError('Every exact original update and worker order required')
        for i, name in enumerate(STREAMS):
            stream = streams[name]
            order = orders[name]
            tokens = 0
            for rank, actual in enumerate(record['worker_draws']):
                if actual['positions'][name] != order.rank_groups(step, rank, 4, options['microbatch_per_rank'], options['accumulation']):
                    raise ValueError('Actual rank/micro source sequence differs from the sealed schedule')
                for group in actual['positions'][name]:
                    for p in group:
                        index = stream.pack.index[stream.pack.positions[p]]
                        tokens += int(index['length']) - int(index['target_start'])
            metric = record['objectives'][name]
            if (metric['global_target_tokens'] != tokens or metric['coefficient'] != options['coefficients'][name]
                    or not math.isfinite(metric['token_mean_nll']) or not math.isclose(metric['weighted_objective'],
                        metric['coefficient'] * metric['token_mean_nll'], rel_tol=1e-6, abs_tol=1e-7)):
                raise ValueError('Every separately normalized per-update objective/target count must reconstruct')
    comparisons = {}
    for name in ['code-completion.json', 'code-instruction.json', 'C++.json', 'knowledge.json', 'ocr.json', 'objects.json', 'vision.json']:
        before_path, after_path = reports / 'baseline' / name, reports / 'candidate' / name
        before, after = [json.loads(p.read_text()) for p in [before_path, after_path]]
        if (before['checkpoint_sha256'] != plan['initial_component_checkpoint_sha256']
                or after['checkpoint_sha256'] != receipt['trained_components_sha256']):
            raise ValueError('Exact original and trained conditional-code artifacts required in all native reports')
        for key in ['inference_device_type', 'inference_precision', 'tokenizer_sha256', 'evaluation_provenance']:
            if before[key] != after[key]:
                raise ValueError('Actual before/after evaluation protocol differs: ' + key)
        comparisons[name] = dict(baseline=summarize(before, name), candidate=summarize(after, name),
                                reports_sha256={str(p): file_hash(p) for p in [before_path, after_path]})
    source = {r['id']: r for r in map(json.loads, (root / 'data/public-code-assessment-v1/questions.jsonl').read_text().splitlines())}
    phases = {}
    protocols = []
    for phase in ['baseline', 'candidate']:
        native = []
        for shard in range(4):
            p = (root / 'reports/own-code-delta-public-v1' / f'candidate-{shard}.json' if phase == 'baseline'
                 else reports / 'candidate' / f'public-{shard}.json')
            report = json.loads(p.read_text())
            data_path = root / 'reports/own-code-delta-public-v1/inputs' / f'shard-{shard}' / 'questions.jsonl'
            if (report['generation_budget'] != 512 or report['inference_precision'] != 'bf16'
                    or report['inference_device_type'] != 'cuda' or report['data_sha256'] != file_hash(data_path)
                    or report['checkpoint_sha256'] != (plan['initial_component_checkpoint_sha256'] if phase == 'baseline'
                                                      else receipt['trained_components_sha256'])):
                raise ValueError('Exact original public CUDA/BF16/512token protocol required')
            protocols.append(report['evaluation_provenance'])
            native.extend(report['results'])
        if len(native) != 175 or {r['id'] for r in native} != source.keys():
            raise ValueError('Every original175supported public problem required')
        phases[phase] = {r['id']: r for r in native}
    if any(p != protocols[0] for p in protocols):
        raise ValueError('Every original public generation/execution runtime source must match')
    reexecuted = []
    for identity, row in phases['candidate'].items():
        if row['code'] != instruction_code(row['raw_response']):
            raise ValueError('Original delivered code extraction differs')
        if row['execution'].get('functional_passed'):
            task = source[identity]
            again = execute(dict(language='Python', code=row['code'], function=task['function'], cases=task['cases'], stress=[]))
            if not again.get('functional_passed'):
                raise ValueError('Every candidate public case pass must reexecute on all original cases')
            reexecuted.append(dict(id=identity, execution=again))
    public_counts = {phase: dict(tasks=175, syntax=sum(r['execution'].get('syntax_valid', False) for r in rows.values()),
        functional=sum(r['execution'].get('functional_passed', False) for r in rows.values())) for phase, rows in phases.items()}
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), actual_steps=options['steps'],
        actual_unique_parameters=plan['actual_model_counts'], actual_component_tensors=len(expected),
        actual_complete_Adam_states=len(optimizer['state']), all_exported_components_equal_complete_resume=True,
        all_original_source_draws_and_complete_target_counts_reconciled=True, exposure_budget=budget,
        original_general_vision_critic_digest_unchanged=True, four_worker_RNG_payloads_retained=True,
        native_comparisons=comparisons, public_subset=public_counts, every_new_public_pass_reexecuted=reexecuted,
        public_subset_not_full_standard_MBPP=True, finite_cases_and_empirical_stress_not_algorithmic_proof=True,
        genuine_final_policy_RLHF_still_pending=True, candidate_promoted=False, goal_achieved=False)
    atomic_json(reports / 'independent-comparison.json', result)
    return result
