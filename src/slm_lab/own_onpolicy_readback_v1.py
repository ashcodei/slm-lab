"""Complete own arm/component/Adam/source/DPO and native-result readback."""
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .apps_native_metrics_v1 import summarize
from .codenet_stage_v1 import EpochOrder
from .code_quality import execute, instruction_code
from .corpus import file_hash
from .diverse_images import atomic_json
from .own_code_delta_v1 import matrix_names, norm_names
from .own_code_training_stage_v1 import LanguageCycleOrder
from .own_onpolicy_inference_v1 import FORMAT as EXPORT_FORMAT, options, provenance
from .own_onpolicy_stage_v1 import FORMAT, validate_reference


def near(a, b):
    if not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('Every retained numerical source/objective statistic must reconstruct')


def public_reports(folder, checkpoint_sha, root, tokenizer_sha):
    source = list(map(json.loads, (root / 'data/public-code-assessment-v1/questions.jsonl').read_text().splitlines()))
    mapped = {}; first = None
    for shard in range(4):
        path = folder / f'public-{shard}.json'; report = json.loads(path.read_text())
        expected = source[shard::4]
        original = root / 'reports/own-code-delta-public-v1/inputs' / f'shard-{shard}' / 'questions.jsonl'
        if (report['checkpoint_sha256'] != checkpoint_sha or report['generation_budget'] != 512
                or report['inference_device_type'] != 'cuda' or report['inference_precision'] != 'bf16'
                or report['tokenizer_sha256'] != tokenizer_sha or report['data_sha256'] != file_hash(original)
                or [r['id'] for r in report['results']] != [r['id'] for r in expected]):
            raise ValueError('Every original supported problem, native prompt/runtime/tokenizer/budget/precision must match')
        summarize(report, 'public-Python.json')
        if first is not None and report['evaluation_provenance'] != first:
            raise ValueError('All original native evaluator/runtime sources must match')
        first = report['evaluation_provenance']
        for row, original_row in zip(report['results'], expected):
            if row['id'] in mapped or row['code'] != instruction_code(row['raw_response']) or row['task_id'] != original_row['task_id']:
                raise ValueError('Every unchanged delivered program and original problem exactly once required')
            mapped[row['id']] = row
    if len(source) != 175 or len(mapped) != 175:
        raise ValueError('All175original supported public-development problems required')
    counts = dict(tasks=175, syntax=sum(r['execution'].get('syntax_valid', False) for r in mapped.values()),
        functional=sum(r['execution'].get('functional_passed', False) for r in mapped.values()))
    return mapped, counts, first


def reconcile(root, reports, output, data, pairs):
    root, reports, output = map(Path, [root, reports, output])
    if torch.cuda.is_initialized():
        raise ValueError('Independent final readback must remain CPU-only')
    plan = json.loads((reports / 'plan.json').read_text()); plan_sha = file_hash(reports / 'plan.json')
    capacity = json.loads((reports / 'capacity.json').read_text())
    reference_sha = file_hash(reports / 'reference.json')
    if reference_sha != capacity['reference_scores_sha256']:
        raise ValueError('Exact four-GPU measured frozen own reference required')
    reference = json.loads((reports / 'reference.json').read_text())
    validate_reference(pairs, reference, plan['initial_manifest_sha256'])
    baseline_folder = root / 'reports/own-code-generation-repair-v1/candidate'
    initial, baseline_public, native = public_reports(baseline_folder, plan['initial_code_checkpoint_sha256'], root, plan['tokenizer_sha256'])
    source = {r['id']: r for r in map(json.loads, (root / 'data/public-code-assessment-v1/questions.jsonl').read_text().splitlines())}
    all_arms = {}
    for arm in plan['arms']:
        opts = options(plan, arm); steps = opts['steps']
        receipt = json.loads((reports / (arm + '-training-complete.json')).read_text())
        if (receipt['actual_updates'] != steps or receipt['reference_scores_sha256'] != reference_sha
                or file_hash(output / arm / 'model.pt') != receipt['trained_components_sha256']
                or file_hash(output / arm / 'resume.pt') != receipt['resume_sha256']
                or receipt['frozen_states_before'] != receipt['frozen_states_after']
                or not receipt['all_original_general_vision_critic_head_bytes_versions_and_native_route_probes_exact']):
            raise ValueError('Actual completed arm/export/resume/reference and all frozen native paths required')
        saved = torch.load(output / arm / 'resume.pt', map_location='cpu', weights_only=True, mmap=True)
        export = torch.load(output / arm / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
        expected_sources = dict(run_plan_sha256=plan_sha, initial_policy_manifest_sha256=plan['initial_manifest_sha256'],
            reference_scores_sha256=reference_sha, data_admissions_sha256=plan['data_admissions_sha256'],
            initial_code_checkpoint_sha256=plan['initial_code_checkpoint_sha256'], base_checkpoint_sha256=plan['base_checkpoint_sha256'])
        if (saved['format'] != FORMAT or saved['sources'] != expected_sources or saved['options'] != opts
                or saved['steps'] != steps or saved['planned_steps'] != steps or saved['world_size'] != 1
                or saved['actual_counts'] != plan['actual_model_counts'] or len(saved['worker_rng_states']) != 1
                or not saved['own_base_is_immutable_reference'] or not saved['own_reference_scores_are_source_bound']
                or export['format'] != EXPORT_FORMAT or export['own_onpolicy_training'] != provenance(plan, plan_sha, arm, reference_sha)):
            raise ValueError('Complete exact own source/budget/reference/arm metadata required')
        parts = {}
        for index, name in enumerate(matrix_names(saved['config']['num_layers'])):
            for direction in ['down', 'up']:
                parts[f'linears.{index}.{direction}'] = export['components']['matrices'][name][direction]
        parts['embedding.weight'] = export['components']['embedding']
        for index, name in enumerate(norm_names(saved['config']['num_layers'])):
            parts[f'norms.{index}.weight'] = export['components']['norms'][name]
        if parts.keys() != saved['parts'].keys() or sum(t.numel() for t in parts.values()) != plan['actual_model_counts']['trainable_code']:
            raise ValueError('Every unique actual own-code parameter must be accounted for')
        for name, tensor in parts.items():
            if not bool(torch.isfinite(tensor).all()) or not torch.equal(tensor, saved['parts'][name]):
                raise ValueError('Every actual finite exported code tensor must equal its resumable counterpart')
        optimizer = saved['optimizer']; groups = optimizer['param_groups']; order = groups[0]['params']
        if len(groups) != 1 or len(order) != len(parts) or set(optimizer['state']) != set(order):
            raise ValueError('Every actual own-code parameter requires complete Adam state')
        for identity, tensor in zip(order, saved['parts'].values()):
            moments = optimizer['state'][identity]
            if (float(moments['step']) != steps or any(moments[k].shape != tensor.shape or moments[k].dtype != tensor.dtype
                    or not bool(torch.isfinite(moments[k]).all()) for k in ['exp_avg', 'exp_avg_sq'])):
                raise ValueError('Every actual Adam moment and exact step must be finite and correctly shaped')
        rng = saved['worker_rng_states'][0]; random.Random().setstate(rng['python']); torch.Generator(device='cpu').set_state(rng['torch'])
        n = rng['numpy']; np.random.RandomState().set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
        if rng['cuda'].dtype != torch.uint8 or rng['cuda'].ndim != 1 or not rng['cuda'].numel():
            raise ValueError('Complete original CUDA RNG bytes must be retained without CPU-reader CUDA initialization')
        orders = dict(generation=EpochOrder(len(data['generation']), opts['seed']),
            variety=LanguageCycleOrder(data['variety'].language_positions, opts['seed']+1009), pairs=EpochOrder(len(pairs), opts['seed']+2018))
        expected_cursors = {}
        for index, name in enumerate(['generation', 'variety', 'pairs']):
            population = len(pairs) if name == 'pairs' else len(data[name])
            expected_cursors[name] = (orders[name].cursor(steps) if name == 'variety' else dict(population=population,
                seed=opts['seed']+index*1009, actual_exposures=steps, complete_passes=steps//population, next_pass_remainder=steps%population))
        if saved['sampler_state'] != expected_cursors:
            raise ValueError('Every final original source/language/pair cursor must reconstruct')
        metrics = list(map(json.loads, (reports / (arm+'-metrics.jsonl')).read_text().splitlines()))
        if metrics != saved['metrics'] or len(metrics) != steps:
            raise ValueError('Every actual retained original optimizer update must match its snapshot')
        totals = Counter()
        for step, record in enumerate(metrics):
            expected = {name: [[order.at(step)]] for name, order in orders.items()}
            if record['step'] != step+1 or record['worker_draws'] != [dict(rank=0, positions=expected)]:
                raise ValueError('Every actual complete original source draw must reconstruct')
            for name, stream in data.items():
                row = stream.pack.index[stream.pack.positions[expected[name][0][0]]]
                count = int(row['length']) - int(row['target_start']); totals[name] += count
                objective = record['CE']['objectives'][name]
                if objective['global_target_tokens'] != count or objective['coefficient'] != opts['coefficients'][name]:
                    raise ValueError('Every independent global stream target normalizer/coefficient must match')
            pair = pairs[expected['pairs'][0][0]]; pref = record['DPO']; r = pref['records'][0]; ref = reference['records'][pair['id']]
            if record['actual_pair_id'] != pair['id'] or r['id'] != pair['id'] or pref['alpha'] != opts['DPO_alpha'] or pref['beta'] != opts['DPO_beta']:
                raise ValueError('Every actual original own-policy pair and fixed treatment coefficient required')
            chosen = sum(v != -100 for v in pair['chosen'][1]); rejected = sum(v != -100 for v in pair['rejected'][1])
            if r['chosen_targets'] != chosen or r['rejected_targets'] != rejected:
                raise ValueError('Every complete original chosen/rejected response and EOS must count')
            totals['chosen'] += chosen; totals['rejected'] += rejected
            margin = (r['chosen_sum_logp']-r['rejected_sum_logp'])-(ref['chosen_logp']-ref['rejected_logp'])
            near(r['reference_anchored_margin'], margin); x = opts['DPO_beta']*margin
            loss = max(-x, 0.)+math.log1p(math.exp(-abs(x)))
            near(pref['unscaled_mean_DPO'], loss); near(pref['weighted_DPO'], opts['DPO_alpha']*loss)
            sigmoid = math.exp(-x)/(1+math.exp(-x)) if x >= 0 else 1/(1+math.exp(x))
            near(r['gradient_coefficient'], opts['DPO_alpha']*opts['DPO_beta']*sigmoid)
            near(record['summed_objective'], record['CE']['summed_objective']+pref['weighted_DPO'])
        for name in data:
            if totals[name] != plan['predeclared_exposure_budget'][name]['response_EOS_targets']:
                raise ValueError('All original source target counts must reconcile to sealed budget')
        if (totals['chosen'] != plan['predeclared_exposure_budget']['pairs']['chosen_human_response_EOS_targets']
                or totals['rejected'] != plan['predeclared_exposure_budget']['pairs']['rejected_actual_own_response_EOS_targets']):
            raise ValueError('All original actual own-policy pair targets must reconcile')
        reload = json.loads((reports / arm / 'trained-shared-reload.json').read_text())
        if not reload['all_three_actual_trained_GPU_routes_exact_after_reload'] or reload['actual_counts']['total_unique'] != 986450944:
            raise ValueError('Actual complete trained one-backbone shared reload/route proof required')
        current, public_counts, current_native = public_reports(reports / arm, receipt['trained_components_sha256'], root, plan['tokenizer_sha256'])
        if current_native != native:
            raise ValueError('Matched original public evaluation/runtime sources must stay unchanged')
        passing = []
        for identity, row in current.items():
            if row['execution'].get('functional_passed'):
                original = source[identity]
                actual = execute(dict(language='Python', code=row['code'], function=original['function'], cases=original['cases'], stress=[]))
                if not actual.get('functional_passed') or actual['passed_cases'] != len(original['cases']):
                    raise ValueError('Every candidate public pass must reexecute all original published cases')
                passing.append(dict(id=identity, actual=actual))
        local = {}
        for name in ['code-completion.json', 'code-instruction.json', 'C++.json']:
            baseline = json.loads((baseline_folder / name).read_text()); current_report = json.loads((reports / arm / name).read_text())
            if (current_report['checkpoint_sha256'] != receipt['trained_components_sha256']
                    or current_report['generation_budget'] != baseline['generation_budget']
                    or current_report['evaluation_provenance'] != baseline['evaluation_provenance']
                    or current_report['tokenizer_sha256'] != baseline['tokenizer_sha256']
                    or current_report['inference_device_type'] != baseline['inference_device_type']
                    or current_report['inference_precision'] != baseline['inference_precision']):
                raise ValueError('Matched original local generated-code runtime/prompt/budget/precision required')
            local[name] = dict(initial=summarize(baseline, name), candidate=summarize(current_report, name))
        all_arms[arm] = dict(actual_updates=steps, actual_code_component_tensors=len(parts), complete_Adam_states=len(order),
            all_exported_components_equal_resume=True, all_source_draws_masks_targets_cursors_DPO_scores_and_RNG_reconciled=True,
            actual_trained_shared_reload=reload, baseline_public=baseline_public, candidate_public=public_counts,
            gained_public_ids=sorted(i for i in current if current[i]['execution'].get('functional_passed') and not initial[i]['execution'].get('functional_passed')),
            lost_public_ids=sorted(i for i in current if initial[i]['execution'].get('functional_passed') and not current[i]['execution'].get('functional_passed')),
            every_candidate_public_pass_reexecuted=passing, native_local_comparisons=local,
            actual_source_targets=dict(totals), checkpoint_sha256=receipt['trained_components_sha256'])
        del saved, export
    atomic_json(reports / 'independent-comparison.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        arms=all_arms, all_original_general_vision_critic_grounding_bytes_versions_and_route_probes_retained=True,
        actual_shared_unique_parameters=986450944, actual_complete_fixed_native_public_problems=175,
        small_28_pair_machine_feedback_pilot_not_final_production=True,
        public_development_not_standard_full_or_certified_independent_benchmark=True,
        finite_cases_empirical_stress_not_algorithmic_proof=True, genuine_final_policy_human_RLHF_incomplete=True,
        candidate_promoted=False, goal_achieved=False))
