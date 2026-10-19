"""Reconstruct and freshly score all30 complete matched endpoint deliveries."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.public_reference_prefix_curriculum_v1 import view

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/conditional-prefix-full-CPU-v1'
OUT = ROOT / 'reports/conditional-prefix-full-CPU-v1-endpoint-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh complete CPU endpoint readback required')
    completed = json.loads((RUN / 'complete.json').read_text()); plan = json.loads((RUN / 'plan.json').read_text())
    if not completed['complete'] or completed['exported_student_values_each_arm'] != 999706442: raise ValueError('Both completed actual full-size endpoints required')
    corpus = ExpandedExecutionCorpus(ROOT); position = corpus.by_id[plan['selected_original_train_source']]
    source = corpus.row(position); native_ids, labels = corpus.sequence(position)
    stages = ['initial', 'anchor_CE-final', 'anchor_execution-final']; all_rows = {}; summaries = {}; executions = []
    files = [Path(__file__), RUN / 'complete.json', RUN / 'plan.json', ROOT / 'reports/conditional-prefix-full-CPU-v1-training-readback/complete.json']
    for stage in stages:
        path = RUN / (stage + '-actual-evaluation-prefix.json'); records = json.loads(path.read_text()); files.append(path)
        if len(records) != 10 or [(r['quarters'], r['repetition']) for r in records] != [(q, r) for q in [0, 3] for r in range(5)]:
            raise ValueError('Every original full endpoint condition and private-seed repetition required')
        for record in records:
            quarter, repetition = record['quarters'], record['repetition']; actual = record['actual_rollout']
            prepared = view(corpus.tokenizer, source['original'], native_ids, labels, quarter)
            actions = actual['actual_generated_action_ids']; stopped = actual['stopped_on_generated_EOS']
            if (actual['original_native_prefix_ids'] != prepared['native_rollout_prefix_ids']
                    or actual['supplied_public_reference_prefix_ids'] != prepared['public_reference_prefix_ids']
                    or not actions or stopped != (actions[-1] == 2) or 2 in actions[:-1]
                    or (not stopped and len(actions) != 512) or actual['horizon'] != 512
                    or actual['seed'] != 965617 + quarter * 1009 + repetition * 9176
                    or actual['sampler'] != ('greedy' if repetition == 0 else 'full_vocabulary_temperature1')
                    or actual['complete_candidate_response'] != corpus.tokenizer.decode(prepared['public_reference_prefix_ids'] + actions)):
                raise ValueError('Whole native source, exact supplied hint and genuine action/EOS/budget/seed must reconstruct')
            cached = actual['actual_cached_logps']; full = actual['actual_full_forward_logps']
            if (len(cached) != len(actions) or len(full) != len(actions)
                    or any(not math.isfinite(v) or v > 0 for v in cached + full)
                    or max(abs(a - b) for a, b in zip(cached, full)) > .0001):
                raise ValueError('Complete genuine cached/full probability observations required')
            fresh = corpus.score(position, actual['complete_candidate_response'], stopped_on_eos=stopped)
            if fresh['reward'] != record['actual_execution_reward']['reward'] or fresh['status'] != record['actual_execution_reward']['status']:
                raise ValueError('Fresh complete original execution outcome differs')
            executions.append(dict(stage=stage, quarters=quarter, repetition=repetition, actual_actions=len(actions),
                genuine_EOS=stopped, fresh_original_execution=fresh, no_neural_hint_credit=True))
        all_rows[stage] = records
        summaries[stage] = {str(q): dict(episodes=5, positive_rewards=sum(r['actual_execution_reward']['reward'] > 0 for r in records if r['quarters'] == q),
            full_finite_case_passes=sum(r['actual_execution_reward']['reward'] == 1. for r in records if r['quarters'] == q)) for q in [0, 3]}
    pairwise = {}
    for left, right in [('initial', 'anchor_CE-final'), ('initial', 'anchor_execution-final'), ('anchor_CE-final', 'anchor_execution-final')]:
        pairwise[left + ':' + right] = dict(exactly_identical_generated_action_sequences=sum(
            a['actual_rollout']['actual_generated_action_ids'] == b['actual_rollout']['actual_generated_action_ids'] for a, b in zip(all_rows[left], all_rows[right])),
            paired_deliveries=10)
    OUT.mkdir()
    path = OUT / 'fresh-original-endpoint-executions.jsonl'
    with path.open('x') as stream:
        for row in executions: stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    files.append(path)
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256={str(p): file_hash(p) for p in files},
        independently_reexecuted_actual_endpoint_programs=30, all_original_sources_hints_tokens_budgets_seeds_and_rewards_rederived=True,
        actual_endpoint_summaries=summaries, actual_paired_sequence_comparisons=pairwise,
        no_new_neural_forward_or_update=True, source_is_public_train_not_held=True,
        no_positive_reward_or_quality_improvement_observed_in_this_three_update_trial=all(x['positive_rewards'] == 0 for s in summaries.values() for x in s.values()),
        small_fixed_trial_not_a_general_rejection_of_RL_or_longer_training=True,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), actual_fresh_endpoint_replays=30, outcomes=summaries)), flush=True)


if __name__ == '__main__': main()
