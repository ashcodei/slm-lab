"""Independently reconstruct/re-execute all four early contrastive source deliveries."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.own_execution_contrast_CPU_v1 import ExecutionContrastPairs

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/own-execution-contrast-early-contrast-source-greedy-CPU-v1'
OUT = ROOT / 'reports/own-execution-contrast-early-contrast-source-greedy-CPU-v1-readback'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden complete source-only outcome readback required')
    complete = json.loads((RUN / 'complete.json').read_text()); plan = json.loads((RUN / 'plan.json').read_text())
    for path, digest in complete['source_files_sha256'].items():
        if file_hash(path) != digest: raise ValueError('Completed original early neural diagnostic changed')
    data = ExecutionContrastPairs(ROOT / 'data/own-execution-negative-pairs-CPU-v1'); corpus = ExpandedExecutionCorpus(ROOT)
    if plan['original_source_ids'] != list(data.by_source) or complete['actual_neural_source_episodes'] != 4:
        raise ValueError('Every predeclared original source and both matched endpoints required')
    executions = []; summaries = {}; files = [Path(__file__), RUN / 'complete.json', RUN / 'plan.json']
    for endpoint in ['anchor_contrast']:
        path = RUN / (endpoint + '-actual-source-greedy-prefix.json'); rows = json.loads(path.read_text()); files.append(path)
        if [r['source_id'] for r in rows] != plan['original_source_ids']: raise ValueError('Every original source exactly once required')
        full = exact = 0
        for row in rows:
            position = corpus.by_id[row['source_id']]; source = corpus.row(position); ids, labels = corpus.sequence(position)
            prefix = ids[:source['prefix_positions']]; gold_actions = ids[source['prefix_positions']:]
            actual = row['actual_unassisted_greedy']; actions = actual['actual_generated_action_ids']; stopped = actual['stopped_on_generated_EOS']
            if (actual['original_native_prefix_ids'] != prefix or actual['supplied_public_reference_prefix_ids']
                    or actual['sampler'] != 'greedy' or actual['horizon'] != 512 or actual['seed'] != 971517
                    or not actions or stopped != (actions[-1] == 2) or 2 in actions[:-1]
                    or (not stopped and len(actions) != 512) or actual['complete_candidate_response'] != corpus.tokenizer.decode(actions)
                    or row['full_original_EOS_targets'] != len(gold_actions)
                    or row['actual_greedy_actions_exactly_match_original_human_reference_and_EOS'] != (actions == gold_actions)):
                raise ValueError('Whole original prompt, actual unassisted source actions/EOS and original human target must reconstruct')
            cached = actual['actual_cached_logps']; forward = actual['actual_full_forward_logps']
            if (len(cached) != len(actions) or len(forward) != len(actions) or any(not math.isfinite(v) or v > 0 for v in cached + forward)
                    or max(abs(a - b) for a, b in zip(cached, forward)) > .0001):
                raise ValueError('Every actual cached/full behavior probability must remain within the declared numerical observation bound')
            fresh = corpus.score(position, actual['complete_candidate_response'], stopped_on_eos=stopped)
            if fresh['reward'] != row['fresh_original_execution']['reward'] or fresh['status'] != row['fresh_original_execution']['status']:
                raise ValueError('Every original source execution reward/status must reproduce')
            full += int(fresh['reward'] == 1.); exact += int(actions == gold_actions)
            executions.append(dict(endpoint=endpoint, source_id=row['source_id'], genuine_EOS=stopped,
                exactly_original_human_actions=actions == gold_actions, fresh_original_execution=fresh,
                actual_neural_program=actual['complete_candidate_response'], public_training_source_not_held=True))
        summaries[endpoint] = dict(original_train_sources=4, full_finite_case_passes=full, exact_original_human_programs_and_EOS=exact)
        if full != complete['actual_outcomes'][endpoint]['full_finite_case_passes'] or exact != complete['actual_outcomes'][endpoint]['exact_original_human_programs_with_EOS']:
            raise ValueError('Every completed source-only aggregate must independently rederive')
    comparison_path = RUN / 'matched-CE40-versus-contrast40-training-source-comparison.json'
    comparison = json.loads(comparison_path.read_text())
    old_path = ROOT / 'reports/own-execution-contrast-early-source-greedy-CPU-v1/anchor_CE-actual-source-greedy-prefix.json'
    old_rows = json.loads(old_path.read_text()); trained_rows = json.loads((RUN / 'anchor_contrast-actual-source-greedy-prefix.json').read_text())
    independent_comparison = []
    for before,after in zip(old_rows,trained_rows):
        if before['source_id'] != after['source_id']: raise ValueError('Every paired original source required')
        first,second=before['actual_unassisted_greedy'],after['actual_unassisted_greedy']
        if any(first[k] != second[k] for k in ['original_native_prefix_ids','supplied_public_reference_prefix_ids','horizon','sampler','seed']):
            raise ValueError('Matched full unassisted original prompts and generation protocol required')
        position=corpus.by_id[before['source_id']]
        ce_fresh=corpus.score(position,first['complete_candidate_response'],stopped_on_eos=first['stopped_on_generated_EOS'])
        if ce_fresh['reward']!=before['fresh_original_execution']['reward']: raise ValueError('Every earlier actual CE program must freshly execute to the same original finite-case credit')
        second_fresh=next(r['fresh_original_execution'] for r in executions if r['source_id']==after['source_id'])
        independent_comparison.append(dict(source_id=before['source_id'],CE40_reward=ce_fresh['reward'],contrast40_reward=second_fresh['reward'],
            same_actual_generated_actions=first['actual_generated_action_ids']==second['actual_generated_action_ids']))
    if (len(independent_comparison)!=4 or comparison['matched_original_sources']!=independent_comparison
            or comparison['CE40_finite_passes']!=sum(r['CE40_reward']==1. for r in independent_comparison)
            or comparison['contrast40_finite_passes']!=sum(r['contrast40_reward']==1. for r in independent_comparison)):
        raise ValueError('Every paired training-source advantage or regression must independently rederive')
    files += [comparison_path,old_path]
    OUT.mkdir(); output = OUT / 'fresh-original-source-executions.jsonl'
    with output.open('x') as stream:
        for row in executions: stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    files.append(output)
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256={str(p): file_hash(p) for p in files},
        independently_reexecuted_actual_neural_programs=4, earlier_CE40_actual_programs_also_freshly_reexecuted=4,
        independent_matched_CE40_vs_contrast40_comparison=independent_comparison, actual_outcomes=summaries,
        all_source_prompts_generated_tokens_original_human_targets_EOS_budgets_and_rewards_rederived=True,
        actual_model_generated_code_from_complete_original_prompt_without_hint_or_reference_at_inference=True,
        new_neural_forward_or_update_by_reader=False, training_source_fit_not_new_problem_generalization=True,
        not_a_general_algorithmic_efficiency_or_frontier_model_score=True, native_full_development_code_suite_still_pending=True,
        candidate_promoted=False, goal_achieved=False))
    print(json.dumps(dict(complete=str(OUT / 'complete.json'), outcomes=summaries)), flush=True)


if __name__ == '__main__': main()
