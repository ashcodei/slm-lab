"""Read both actual full-size three-update states and replay original programs.

No new neural forward/backward is performed. Endpoint evaluations may still
be running; this verifies training-state and sampled-training-record evidence.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch

from slm_lab.admitted_public_prefix_episode_v1 import AdmittedPublicPrefixEpisode
from slm_lab.conditional_prefix_stage_CPU_v1 import trajectory_digest
from slm_lab.conditional_trial_policy_CPU_v1 import sources
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.execution_group_controlled_v2 import advantages
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/conditional-prefix-full-CPU-v1'
OUTPUT = ROOT / 'checkpoints/conditional-prefix-full-CPU-v1'
OUT = ROOT / 'reports/conditional-prefix-full-CPU-v1-training-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden actual training-state readback required')
    torch.set_num_threads(1)
    plan_path = RUN / 'plan.json'; plan = json.loads(plan_path.read_text()); digest = file_hash(plan_path)
    if plan['planned_updates_each_arm'] != 3 or plan['actual_exported_student_values_each_arm'] != 999706442:
        raise ValueError('Exact actual full-size three-update comparison required')
    for name, expected in plan['source_files_sha256'].items():
        if file_hash(name) != expected: raise ValueError('Bound original CPU trial source changed')
    corpus = ExpandedExecutionCorpus(ROOT); position = corpus.by_id[plan['selected_original_train_source']]
    episode = AdmittedPublicPrefixEpisode(corpus, position, 3, root=ROOT); prepared = episode.rollout_view()
    binding = episode.binding(); expected_sources = sources(plan, digest)
    initial_path = OUTPUT / 'initial-actor.pt'; initial = torch.load(initial_path, map_location='cpu', weights_only=True, mmap=True)['policy']
    if tensor_digest(initial['parts'].items()) != plan['initial_parts_sha256']: raise ValueError('Every actual own initial private value required')
    files = [Path(__file__), plan_path, initial_path]; states = {}; first_states = {}; actual_replays = []; summaries = {}
    for arm in ['anchor_CE', 'anchor_execution']:
        first_path = OUTPUT / (arm + '-after-first.pt'); final_path = OUTPUT / (arm + '-final-state.pt'); native_path = OUTPUT / (arm + '-actor.pt')
        first = torch.load(first_path, map_location='cpu', weights_only=True, mmap=True)
        final = torch.load(final_path, map_location='cpu', weights_only=True, mmap=True)
        native = torch.load(native_path, map_location='cpu', weights_only=True, mmap=True)
        if (final['settings']['method'] != arm or final['settings']['CPU_mechanics_horizon'] is not None
                or final['settings']['local_group_size'] != 4 or final['episode_binding'] != binding
                or first['episode_binding'] != binding or final['own_reference_binding'] != first['own_reference_binding']):
            raise ValueError('Exact full native source, own reference and unshortened action horizon required')
        for state, completed in [(first, 1), (final, 3)]:
            core = state['core']; model = core['model']; parts = model['parts']; adam = core['optimizer']
            if (core['completed'] != completed or core['planned'] != 3 or core['groups'] != ['actor']
                    or len(core['metrics']) != completed or len(core['optimized_names']) != 561
                    or len(adam['state']) != 561 or model['sources'] != expected_sources
                    or model['actual_counts']['actual_unique'] != 999706442
                    or model['frozen_dense_sha256'] != initial['frozen_dense_sha256']
                    or not torch.equal(parts['critic.weight'], initial['parts']['critic.weight'])):
                raise ValueError('Every actual selected state/count/source/frozen base and unused critic required')
            for key, name in zip(adam['param_groups'][0]['params'], core['optimized_names']):
                moment = adam['state'][key]
                if (float(moment['step']) != completed or moment['exp_avg'].shape != parts[name].shape
                        or moment['exp_avg_sq'].shape != parts[name].shape
                        or any(not bool(torch.isfinite(t).all()) for t in moment.values())
                        or bool((moment['exp_avg_sq'] < 0).any()) or not bool(torch.isfinite(parts[name]).all())):
                    raise ValueError('Every actual selected finite Adam moment, matching shape and exact clock required')
            rng = core['rng']; random.Random().setstate(rng['python']); torch.Generator(device='cpu').set_state(rng['torch'])
            n = rng['numpy']; np.random.RandomState().set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'], n['has_gauss'], n['cached_gaussian']))
            if rng['cuda'] is not None: raise ValueError('Actual CPU trial must not contain a CUDA generator')
        core = final['core']; parts = core['model']['parts']
        if (native['completed_trial_updates'] != 3 or native['arm'] != arm or native['complete_human_alignment_or_source_epoch']
                or native['policy']['parts'].keys() != parts.keys()
                or any(not torch.equal(t, native['policy']['parts'][name]) for name, t in parts.items())):
            raise ValueError('Every actual final native private tensor must equal the private training state')
        logical = []
        for step, label in [(1, 'first'), (2, 'replayed-second'), (3, 'third-final')]:
            path = RUN / (arm + '-' + label + '-actual-native-step.json'); record = json.loads(path.read_text()); files.append(path)
            compact = {k: v for k, v in record.items() if k not in ['trajectories', 'complete_prompt_ids']}
            compact['global_actual_trajectory_semantic_sha256'] = trajectory_digest(record['trajectories'])
            if compact != core['metrics'][step - 1] or record['complete_prompt_ids'] != prepared['native_rollout_prefix_ids']:
                raise ValueError('Every complete original journal record must match the actual private-state metric')
            rewards = []; action_count = 0
            for sample_index, trajectory in enumerate(record['trajectories']):
                actions = trajectory['response_ids']; stopped = actions[-1] == 2
                if (trajectory['supplied_public_reference_prefix_ids'] != prepared['public_reference_prefix_ids']
                        or not actions or 2 in actions[:-1] or (not stopped and len(actions) != 512)
                        or trajectory['complete_candidate_response'] != corpus.tokenizer.decode(prepared['public_reference_prefix_ids'] + actions)
                        or trajectory['own_response'] != corpus.tokenizer.decode(actions)):
                    raise ValueError('Exact genuine sampled suffix and supplied human prefix must independently reconstruct')
                fresh = corpus.score(position, trajectory['complete_candidate_response'], stopped_on_eos=stopped)
                old = trajectory['actual_execution_reward']
                if fresh['reward'] != old['reward'] or fresh['status'] != old['status']:
                    raise ValueError('Actual original public execution reward/status differs on fresh replay')
                a = trajectory['actual_cached_behavior_logps']; b = trajectory['actual_full_behavior_logps']
                if (len(a) != len(actions) or len(b) != len(actions) or any(not math.isfinite(v) or v > 0 for v in a + b)
                        or max(abs(x - y) for x, y in zip(a, b)) > .0001):
                    raise ValueError('Every actual action probability and complete cached/full guard required')
                rewards.append(fresh['reward']); action_count += len(actions)
                actual_replays.append(dict(arm=arm, step=step, sample=sample_index,
                    genuine_EOS=stopped, genuine_action_tokens=len(actions), fresh_original_execution=fresh,
                    supplied_prefix_not_attributed_to_neural_policy=True))
            relative = advantages(rewards)
            expected_surrogate = math.fsum(-relative[i] * math.fsum(t['actual_full_behavior_logps']) / action_count
                for i, t in enumerate(record['trajectories']))
            if (rewards != record['global_rewards'] or relative != record['global_advantages']
                    or action_count != record['global_sampled_action_tokens'] or record['source_reference_targets'] != 166
                    or not record['original_reference_used_regardless_of_rewards'] or not record['complete_original_anchor_preserved']
                    or not math.isclose(expected_surrogate, record['policy_surrogate'], abs_tol=5e-5, rel_tol=1e-5)
                    or not math.isclose(record['original_reference_CE_sum'] / 166, record['original_reference_token_mean_CE'], abs_tol=1e-12)):
                raise ValueError('Actual full CE targets, reward/advantage/action denominators and policy surrogate must rederive')
            logical.append(record)
        expected_path = RUN / (arm + '-expected-second-actual-native-step.json'); expected = json.loads(expected_path.read_text()); files.append(expected_path)
        if trajectory_digest(expected['trajectories']) != trajectory_digest(logical[1]['trajectories']):
            raise ValueError('Actual expected/recovered second trajectories must agree exactly')
        first_states[arm] = first['core']; states[arm] = core
        changed = sum(not torch.equal(t, initial['parts'][name]) for name, t in parts.items() if name != 'critic.weight')
        if changed == 0: raise ValueError('Actual selected parameters must change')
        summaries[arm] = dict(actual_logical_updates=3, actual_optimized_tensors=561, all_actual_Adam_clocks=3,
            changed_actor_tensors=changed, private_parts_sha256=tensor_digest(parts.items()),
            actual_reward_groups=[r['global_rewards'] for r in logical], full_original_reference_targets=498,
            actual_private_rng_states_valid=True, native_private_state_all_tensors_bitexact=True)
        files += [first_path, final_path, native_path]
    left = first_states['anchor_CE']; right = first_states['anchor_execution']
    if any(not torch.equal(t, right['model']['parts'][name]) for name, t in left['model']['parts'].items()):
        raise ValueError('Actual first flat-reward update must match across both controlled arms')
    difference = sum(not torch.equal(t, states['anchor_execution']['model']['parts'][name])
        for name, t in states['anchor_CE']['model']['parts'].items() if name != 'critic.weight')
    if difference == 0: raise ValueError('Actual RL objective must produce a distinct final trained actor')
    OUT.mkdir()
    with (OUT / 'fresh-original-training-program-executions.jsonl').open('x') as stream:
        for row in actual_replays: stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    files.append(OUT / 'fresh-original-training-program-executions.jsonl')
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256={str(p): file_hash(p) for p in files},
        actual_full_size_student_values_each_arm=999706442, actual_logical_updates_each_arm=3, arms=summaries,
        actual_original_training_candidates_freshly_reexecuted=24, all_rewards_advantages_targets_and_policy_surrogates_rederived=True,
        actual_first_flat_reward_update_parts_identical_across_arms=True, actual_final_differing_actor_tensors=difference,
        frozen_dense_base_and_unused_critic_identical=True, no_new_neural_forward_or_training_by_reader=True,
        no_independent_KL_recomputation_without_full_reference_logits=True,
        source_training_and_partial_prefix_not_unassisted_held_quality=True, endpoint_quality_comparison_not_established_by_training_state=True,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_fresh_training_replays=24, differing_actor_tensors=difference)), flush=True)


if __name__ == '__main__': main()
