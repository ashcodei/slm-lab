"""Early full-size contrastive source decoding, matched to existing CE40 data.

Every four original training problems receives a fresh unassisted512-budget
native generation. Previously retained CE40 deliveries supply the comparator;
no CE neural generation is repeated and no held/generalization claim is made.
"""
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path

import torch

from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.native_prefix_rollout_CPU_v1 import sample
from slm_lab.own_execution_contrast_CPU_v1 import ExecutionContrastPairs, batch
from slm_lab.own_execution_contrast_policy_CPU_v1 import load

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'reports/own-execution-contrast-full-CPU-v1'
CHECKPOINTS = ROOT / 'checkpoints/own-execution-contrast-full-CPU-v1'
OUT = ROOT / 'reports/own-execution-contrast-early-contrast-source-greedy-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden source-only early check required')
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    receipt_path = RUN / 'anchor_contrast-training-complete.json'; receipt = json.loads(receipt_path.read_text())
    if receipt['actual_logical_updates'] != 40 or receipt['all561_Adam_clocks'] != 40: raise ValueError('Exact completed40-update contrastive arm required')
    plan_path = RUN / 'plan.json'; plan = json.loads(plan_path.read_text()); digest = file_hash(plan_path)
    data = ExecutionContrastPairs(ROOT / 'data/own-execution-negative-pairs-CPU-v1'); corpus = ExpandedExecutionCorpus(ROOT)
    files = [Path(__file__), receipt_path, plan_path, data.folder / 'manifest.json']
    OUT.mkdir(); owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    atomic_json(OUT / 'plan.json', dict(at=datetime.now(timezone.utc).isoformat(), owner=owner,
        original_source_ids=list(data.by_source), source_selection='Every four original train source in the predeclared20-pair comparison',
        endpoints=['anchor_contrast'], full_greedy_action_budget=512, CPU_threads=2, precision='CPU_FP32',
        public_training_diagnostic_not_held_capability=True, no_training_or_promotion=True, goal_achieved=False))
    outcomes = {}
    for endpoint in ['anchor_contrast']:
        checkpoint = CHECKPOINTS / (endpoint + '-actor.pt'); files.append(checkpoint)
        model, _ = load(checkpoint, plan_path, endpoint, expected_checkpoint_sha256=file_hash(checkpoint), expected_plan_sha256=digest)
        records = []
        for source_id, position in data.by_source.items():
            row = data.rows[position]; preferred = batch(row, 'chosen'); target = preferred['labels'][:, 1:]; mask = target != -100
            with torch.no_grad():
                logits = model(preferred['input_ids'])['logits'][:, :-1].float()
                predicted = logits.argmax(-1)[mask].tolist(); gold = preferred['input_ids'][:, 1:][mask].tolist()
                per_token = torch.nn.functional.cross_entropy(logits[mask], target[mask], reduction='none')
            actual = sample(model.policy, corpus.tokenizer, row['original_native_prefix_ids'], [], horizon=512, sampler='greedy', seed=971517)
            score = corpus.score(row['source_position'], actual['complete_candidate_response'], stopped_on_eos=actual['stopped_on_generated_EOS'])
            exact_human_actions = actual['actual_generated_action_ids'] == gold
            if predicted == gold and not exact_human_actions:
                raise ValueError('All teacher-forced greedy targets correct but actual unassisted greedy actions differ; inspect causal/cache behavior')
            record = dict(endpoint=endpoint, source_id=source_id, full_original_EOS_targets=len(gold),
                actual_teacher_forced_target_argmax_correct=sum(a == b for a, b in zip(predicted, gold)),
                actual_teacher_forced_reference_NLL_sum=float(per_token.sum()), all_teacher_forced_argmax_targets_correct=predicted == gold,
                actual_unassisted_greedy=actual, fresh_original_execution=score,
                actual_greedy_actions_exactly_match_original_human_reference_and_EOS=exact_human_actions,
                public_train_source_not_held_benchmark=True)
            records.append(record); atomic_json(OUT / (endpoint + '-actual-source-greedy-prefix.json'), records)
            print(json.dumps(dict(endpoint=endpoint, source_id=source_id, reward=score['reward'],
                actual_EOS=actual['stopped_on_generated_EOS'], exact_human_actions=exact_human_actions)), flush=True)
        outcomes[endpoint] = dict(source_problems=4, full_finite_case_passes=sum(r['fresh_original_execution']['reward'] == 1. for r in records),
            exact_original_human_programs_with_EOS=sum(r['actual_greedy_actions_exactly_match_original_human_reference_and_EOS'] for r in records),
            all_original_sources_evaluated=True)
        del model; gc.collect()
    previous = ROOT / 'reports/own-execution-contrast-early-source-greedy-CPU-v1'
    previous_readback = ROOT / 'reports/own-execution-contrast-early-source-greedy-CPU-v1-readback/complete.json'
    old = json.loads(previous_readback.read_text())
    if old['actual_outcomes']['anchor_CE']['full_finite_case_passes'] != 3: raise ValueError('Exact independently re-executed CE40 comparator required')
    ce_path = previous / 'anchor_CE-actual-source-greedy-prefix.json'; ce = json.loads(ce_path.read_text())
    compare = []; fresh = records
    for a,b in zip(ce,fresh):
        x,y = a['actual_unassisted_greedy'],b['actual_unassisted_greedy']
        if (a['source_id'] != b['source_id'] or a['full_original_EOS_targets'] != b['full_original_EOS_targets']
                or any(x[k] != y[k] for k in ['original_native_prefix_ids','supplied_public_reference_prefix_ids','horizon','sampler','seed'])):
            raise ValueError('Every original source, unassisted prompt and native generation budget must match')
        compare.append(dict(source_id=a['source_id'], CE40_reward=a['fresh_original_execution']['reward'],
            contrast40_reward=b['fresh_original_execution']['reward'],
            same_actual_generated_actions=x['actual_generated_action_ids']==y['actual_generated_action_ids']))
    if len(compare)!=4: raise ValueError('Every original source must be compared')
    comparison_path = OUT / 'matched-CE40-versus-contrast40-training-source-comparison.json'
    atomic_json(comparison_path,dict(actual_original_train_sources=4,actual_new_contrastive_neural_deliveries=4,
        earlier_CE40_deliveries_not_new_neural_generation=True, matched_original_sources=compare,
        CE40_finite_passes=sum(r['CE40_reward']==1. for r in compare),
        contrast40_finite_passes=sum(r['contrast40_reward']==1. for r in compare),
        no_held_generalization_or_algorithmic_efficiency_or_human_RLHF_claim=True,automatic_promotion=False,goal_achieved=False))
    files += [previous_readback,ce_path,comparison_path]
    if torch.cuda.is_initialized(): raise ValueError('CPU diagnostic initialized CUDA')
    files += [OUT / 'plan.json', *OUT.glob('*-actual-source-greedy-prefix.json')]
    atomic_json(OUT / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256={str(p): file_hash(p) for p in files},
        actual_outcomes=outcomes, actual_neural_source_episodes=4, actual_student_values_each_endpoint=999706442,
        fresh_unassisted_generation_no_reference_prefix_or_teacher=True, training_updates=0,
        training_source_fit_not_held_coding_knowledge_or_efficiency=True,
        production_full48_case_native_development_assessments_remain_required=True, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__': main()
