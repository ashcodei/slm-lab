"""Read every expanded native training sequence and exercise actual reward APIs."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/expanded-execution-corpus-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or REPORTS.exists():
        raise ValueError('Fresh CPU-only whole-source integration proof required')
    import torch
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    torch.set_num_threads(1)
    corpus = ExpandedExecutionCorpus(ROOT)
    REPORTS.mkdir()
    totals = Counter(); extremes = {}; targets_by_rank = Counter()
    for position in range(len(corpus)):
        row = corpus.row(position); ids, labels = corpus.sequence(position)
        start = row['prefix_positions']; targets = row['response_EOS_targets']
        if (ids[0] != 1 or ids[-1] != 2 or labels != [-100] * start + ids[start:]
                or len(ids) != row['complete_context_positions'] or len(ids) - start != targets
                or not targets <= row['full_training_action_budget'] <= 4096 - start):
            raise ValueError('Every actual complete native sequence/target/EOS/budget required')
        text = 'User: ' + row['original']['prompt'] + '\nAssistant:'
        if [1] + corpus.tokenizer.encode(text).ids != ids[:start]:
            raise ValueError('Original native prompt must contain no appended expected-case values')
        if corpus.tokenizer.encode(row['original']['response']).ids + [2] != ids[start:]:
            raise ValueError('Every unchanged original reference-code byte sequence must encode the stored target')
        totals[row['kind']] += 1; totals['complete_context_positions'] += len(ids)
        totals['response_EOS_targets'] += targets; targets_by_rank[position % 4] += targets
        for key in ['complete_context_positions', 'prefix_positions', 'response_EOS_targets', 'full_training_action_budget']:
            name = row['kind'] + ':' + key
            if name not in extremes or row[key] > corpus.rows[extremes[name]][key]:
                extremes[name] = position
    observations = []
    for position in sorted(set(extremes.values())):
        row = corpus.row(position); batch = corpus.batch(position, 'cpu')
        if (int(batch['attention_mask'].sum()) != row['complete_context_positions']
                or int((batch['labels'][:, 1:] != -100).sum()) != row['response_EOS_targets']):
            raise ValueError('Real native extreme batch lost complete source targets')
        ref = corpus.score(position, row['original']['response'], stopped_on_eos=True)
        if row['kind'] == 'apps':
            from slm_lab.apps_published_cases_v2 import contract
            from slm_lab.apps_execution_reward_v1 import noop
            cases = contract(Path(ROOT / 'data/public-apps-original-inventory-v1/original-train' /
                                  row['original']['source_problem_id'] / 'input_output.json').read_bytes())
            empty = corpus.score(position, noop(cases), stopped_on_eos=True)
            if (ref['informative'] and ref['reward'] != 1.) or empty['reward'] != 0.:
                raise ValueError('Real integrated public-reference/no-op rewards differ from controls')
        else:
            empty = corpus.score(position, 'pass', stopped_on_eos=True)
            if empty['reward'] != 0.:
                raise ValueError('Non-function no-op cannot earn documented-function reward')
        observations.append(dict(position=position, source_id=row['id'], kind=row['kind'],
            actual_native_context_positions=row['complete_context_positions'],
            actual_response_EOS_targets=row['response_EOS_targets'],
            actual_training_action_budget=row['full_training_action_budget'],
            actual_original_reference_reward=ref, actual_noop_reward=empty))
    if totals['documented'] != 173 or totals['apps'] != 3548 or sum(targets_by_rank.values()) != totals['response_EOS_targets']:
        raise ValueError('All original sources and four-way source partition must reconcile')
    files = [Path(__file__), ROOT / 'src/slm_lab/expanded_execution_corpus_v1.py',
        ROOT / 'src/slm_lab/anchored_code_objective_v1.py', ROOT / 'tests/test_anchored_code_objective_v1.py',
        ROOT / 'reports/anchored-code-objective-v1-tests.log', ROOT / 'data/apps-execution-rewards-v1/admission.json',
        ROOT / 'reports/apps-execution-rewards-v1/independent-reader.json',
        ROOT / 'reports/bounded-code-probes-v1/admission.json']
    for pack in corpus.packs.values():
        files.extend(pack.folder / name for name in ['manifest.json', 'admission.json'])
    atomic_json(REPORTS / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(),
        all_original_sources=3721, counts=dict(totals), four_rank_source_partition_targets=dict(targets_by_rank),
        all_complete_native_contexts_reference_targets_and_EOS_rederived=True,
        all_original_prompts_unchanged_no_new_expected_cases_added=True,
        extrema=extremes, actual_extreme_native_batches_and_rewards=observations,
        provenance=corpus.provenance, source_files_sha256={str(p): file_hash(p) for p in files},
        exact_source_identity_screens_do_not_establish_semantic_independence=True,
        all_3721_policy_or_optimizer_updates_not_started=True, full_new_profile_GPU_capacity_still_required=True,
        reference_anchor_loss_tested_not_integrated_in_a_live_production_learner=True,
        model_capability_gain_tested=False, human_preferences=False, genuine_human_RLHF=False,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(sources=3721, counts=dict(totals), actual_extreme_batches=len(observations),
        neural_updates=0, capability_gain_tested=False)), flush=True)


if __name__ == '__main__':
    main()
