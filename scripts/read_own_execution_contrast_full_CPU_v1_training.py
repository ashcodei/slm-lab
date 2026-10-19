"""Reconcile both actual40-update private states and every fixed pair/objective.

This verifies retained training evidence, not held code correctness. No fresh
neural forward/backward or cached-KL neural recomputation is performed.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from slm_lab.code_residual_checkpoint_v1 import _validate_adam, _validate_rng
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_execution_contrast_CPU_v1 import ExecutionContrastPairs
from slm_lab.own_execution_contrast_policy_CPU_v1 import sources
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/own-execution-contrast-full-CPU-v1'
CHECKPOINTS = ROOT / 'checkpoints/own-execution-contrast-full-CPU-v1'
OUT = ROOT / 'reports/own-execution-contrast-full-CPU-v1-training-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden complete40-update training readback required')
    torch.set_num_threads(1)
    plan_path = RUN / 'plan.json'; plan = json.loads(plan_path.read_text()); digest = file_hash(plan_path)
    data = ExecutionContrastPairs(ROOT / 'data/own-execution-negative-pairs-CPU-v1')
    initial_path = CHECKPOINTS / 'initial-actor.pt'; initial = torch.load(initial_path, map_location='cpu', weights_only=True, mmap=True)['policy']
    inputs = plan['base_inputs']; source_path = Path(inputs['checkpoint'])
    if file_hash(source_path) != inputs['checkpoint_sha256']: raise ValueError('Exact immutable own partial foundation required')
    base = torch.load(source_path, map_location='cpu', weights_only=True, mmap=True)['model']
    base_items = [('network.' + n, t) for n, t in base['network'].items() if n != 'lm_head.weight']
    base_items += [('heads.' + n, t) for n, t in base['heads'].items()]
    base_digest = tensor_digest(base_items)
    if base_digest != initial['frozen_dense_sha256']: raise ValueError('Every actual frozen base/head source value must independently hash correctly')
    cache_path = CHECKPOINTS / 'own-initial-full-context-logps.pt'
    if file_hash(cache_path) != plan['own_initial_reference_cache_sha256']: raise ValueError('Exact original own initial reference distributions required')
    cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
    if cache['initial_parts_sha256'] != plan['initial_parts_sha256'] or cache['own_base_sha256'] != base_digest:
        raise ValueError('Actual own reference cache must bind the same initial student')
    for key, index in data.by_source.items():
        row = data.rows[index]; values = cache['logps'][key]
        if (cache['source_to_complete_native_input_ids'][key] != row['complete_chosen_ids']
                or values.shape != (1, len(row['complete_chosen_ids']) - 1, 32768)
                or not bool(torch.isfinite(values).all()) or float(values.logsumexp(-1).abs().max()) > .0001):
            raise ValueError('All complete normalized own initial distributions must address the original full source')
    summaries = {}; final_parts = {}; files = [Path(__file__), plan_path, initial_path, source_path, cache_path, data.folder / 'manifest.json']
    for arm in ['anchor_CE', 'anchor_contrast']:
        private_path = CHECKPOINTS / (arm + '-final-state.pt'); native_path = CHECKPOINTS / (arm + '-actor.pt')
        state = torch.load(private_path, map_location='cpu', weights_only=True, mmap=True)
        native = torch.load(native_path, map_location='cpu', weights_only=True, mmap=True); parts = state['model']['parts']
        names = state['optimized_names']; expected_sources = sources(plan, digest)
        if (state['completed'] != 40 or state['planned'] != 40 or len(state['metrics']) != 40 or len(names) != 561
                or state['cursor']['method'] != arm or state['cursor']['next_update'] != 40
                or state['model']['frozen_dense_sha256'] != base_digest or state['model']['sources'] != expected_sources
                or native['arm'] != arm or native['completed_trial_updates'] != 40
                or native['policy']['sources'] != expected_sources or native['policy']['parts'].keys() != parts.keys()
                or any(not torch.equal(t, native['policy']['parts'][n]) for n, t in parts.items())
                or not torch.equal(parts['critic.weight'], initial['parts']['critic.weight'])):
            raise ValueError('Every complete actual40-update native/private tensor, source and fixed critic must agree')
        _validate_adam(state['optimizer'], [parts[n] for n in names], 40); _validate_rng(state['rng'], 'cpu')
        targets = negative_actions = 0
        for index, record in enumerate(state['metrics']):
            row = data.rows[plan['fixed_pair_order'][index]]
            expected_hash = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            x = .5 - 2. * (record['mean_chosen_logp'] - record['mean_rejected_logp'])
            preference = max(x, 0.) + math.log1p(math.exp(-abs(x)))
            objective = record['original_reference_CE'] + .01 * record['own_initial_full_context_forward_KL']
            if arm == 'anchor_contrast': objective += .2 * record['length_normalized_preference_loss']
            if (record['step'] != index + 1 or record['method'] != arm or record['pair_index'] != row['pair_index']
                    or record['native_pair_sha256'] != expected_hash or record['pair_id'] != row['pair_id']
                    or record['source_id'] != row['source_id'] or not record['all_original_reference_EOS_targets_preserved']
                    or record['original_reference_EOS_targets'] != row['complete_original_reference_EOS_targets']
                    or record['genuine_negative_action_targets'] != len(row['rejected_actual_action_ids'])
                    or record['rejected_actual_EOS'] != row['rejected_generated_EOS']
                    or record['censored_negative_boundary_not_completed'] != row['rejected_is_censored_at_complete512_action_budget']
                    or not math.isclose(record['length_normalized_preference_loss'], preference, abs_tol=2e-6, rel_tol=1e-5)
                    or not math.isclose(record['actual_objective'], objective, abs_tol=2e-6, rel_tol=1e-5)
                    or record['original_reference_CE'] != -record['mean_chosen_logp']):
                raise ValueError('Every full pair, target/EOS/censoring boundary and recorded CE/preference/KL arithmetic must rederive')
            path = RUN / (arm + '-' + ('first' if index == 0 else 'replayed-second' if index == 1 else 'step-' + str(index + 1)) + '.json')
            if json.loads(path.read_text()) != record: raise ValueError('Every external actual update must equal its private-state metric')
            files.append(path); targets += record['original_reference_EOS_targets']; negative_actions += record['genuine_negative_action_targets']
        changed = sum(not torch.equal(t, initial['parts'][name]) for name, t in parts.items() if name != 'critic.weight')
        if targets != 5700 or negative_actions != 16626 or not changed: raise ValueError('All complete source exposures and actually changed actor values required')
        summaries[arm] = dict(actual_updates=40, actual_selected_Adam_states=561, all_actual_Adam_clocks=40,
            changed_actor_tensors=changed, complete_original_reference_EOS_targets=targets,
            actual_negative_actions_observed=negative_actions, native_private_values_all_bitexact=True,
            original_four_sources_each_positive_seen_ten_times=True, actual_private_parts_sha256=tensor_digest(parts.items()))
        final_parts[arm] = parts; files += [private_path, native_path, RUN / (arm + '-training-complete.json')]
    differences = sum(not torch.equal(t, final_parts['anchor_contrast'][name]) for name, t in final_parts['anchor_CE'].items() if name != 'critic.weight')
    if not differences: raise ValueError('Actual contrastive objective must yield a distinct trained actor')
    OUT.mkdir(); atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256={str(p): file_hash(p) for p in files}, actual_full_size_student_values_each_arm=999706442,
        arms=summaries, final_differing_actor_tensors_between_objectives=differences,
        frozen_actual_dense_source_hash_independently_rederived=True,
        all_original20_pairs_each_used_twice_per_arm=True, all_original_pair_masks_EOS_censoring_and_objective_arithmetic_rederived=True,
        all_actual_finite_Adam_moments_and_private_CPU_RNG_valid=True, own_reference_distributions_are_cached_outputs_not_an_extra_neural_model=True,
        recorded_neural_logps_and_KL_not_freshly_recomputed_by_reader=True, new_neural_updates=0,
        native_development_quality_assessments_still_required=True, candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_updates_each_arm=40, actual_original_targets_each_arm=5700,
        differing_actor_tensors=differences)), flush=True)


if __name__ == '__main__': main()
