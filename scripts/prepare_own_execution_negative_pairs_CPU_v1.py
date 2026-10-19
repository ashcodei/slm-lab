"""Prepare every unassisted failure from the completed own diagnostic.

Public human programs are positives; execution-rejected own neural deliveries
are negatives. Censored512-action failures keep their exact non-EOS boundary.
These machine labels are not human rankings or independent held assessment.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from slm_lab.active_execution_input_bindings_v1 import input_bindings, merge_bindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
from slm_lab.local_source_closure_v2 import python_sources

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/prefix-reward-density-full-CPU-v1'
OUT = ROOT / 'data/own-execution-negative-pairs-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or OUT.exists(): raise ValueError('Fresh CPU-hidden pair preparation required')
    readback_path = ROOT / 'reports/prefix-reward-density-full-CPU-v2-readback/complete.json'
    reader = json.loads(readback_path.read_text())
    if not reader['all40_actual_candidate_programs_independently_rechecked_with_same_original_outcomes']:
        raise ValueError('All original actual neural actions need independent runtime readback first')
    for name, digest in reader['source_files_sha256'].items():
        if file_hash(name) != digest: raise ValueError('Completed own diagnostic readback changed')
    diagnostic = json.loads((SOURCE / 'complete.json').read_text())
    records = [json.loads(line) for line in (SOURCE / 'actual-neural-episodes.jsonl').read_text().splitlines()]
    records = [r for r in records if r['task']['retained_reference_quarters'] == 0]
    if len(records) != 20 or diagnostic['checkpoint_actual_updates'] != 2048: raise ValueError('All20 actual unassisted own step2048 deliveries required')
    corpus = ExpandedExecutionCorpus(ROOT); pairs = []; positives = {}; raw_executions = []
    for index, record in enumerate(records):
        task = record['task']; original = corpus.row(task['position']); actual = record['actual_rollout']
        ids, labels = corpus.sequence(task['position']); start = original['prefix_positions']; prefix = ids[:start]
        actions = actual['actual_generated_action_ids']; stopped = actual['stopped_on_generated_EOS']
        target_hash = hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest()
        if (original['id'] != task['source_id'] or original['original']['split'] != 'train'
                or original['original']['publisher_split'] != 'train' or actual['original_native_prefix_ids'] != prefix
                or actual['supplied_public_reference_prefix_ids'] or record['public_reference_prefix_tokens'] != 0
                or target_hash != record['original_source_native_input_sha256']
                or not actions or len(prefix) + len(actions) > 4096 or actual['horizon'] != 512
                or stopped != (actions[-1] == 2) or 2 in actions[:-1] or (not stopped and len(actions) != 512)
                or corpus.tokenizer.decode(actions) != actual['complete_candidate_response']):
            raise ValueError('Whole original train prompt and exact actual unassisted neural delivery required')
        if original['id'] not in positives:
            response = original['original']['response']
            if corpus.tokenizer.decode(ids[start:-1]) != response or ids[-1] != 2:
                raise ValueError('Every unchanged full original human program and EOS must reconstruct')
            score = corpus.score(task['position'], response, stopped_on_eos=True)
            if score['reward'] != 1.: raise ValueError('Original human positive must pass all admitted unchanged finite cases')
            positives[original['id']] = dict(source_id=original['id'], source_position=task['position'],
                chosen_response=response, complete_chosen_ids=ids, complete_chosen_labels=labels,
                complete_original_reference_EOS_targets=len(ids) - start, fresh_original_reference_execution=score)
        bad = corpus.score(task['position'], actual['complete_candidate_response'], stopped_on_eos=stopped)
        if bad['reward'] != 0. or record['actual_execution_reward']['reward'] != 0.:
            raise ValueError('Every selected actual unassisted failure must independently receive zero reward')
        raw_executions.append(dict(pair_index=index, source_id=original['id'], fresh_original_protocol_execution=bad))
        pair = dict(pair_index=index, pair_id=original['id'] + ':own-unassisted-episode:' + str(task['index']),
            **{k: v for k, v in positives[original['id']].items() if k != 'fresh_original_reference_execution'},
            rejected_actual_action_ids=actions, complete_rejected_ids=prefix + actions,
            complete_rejected_labels=[-100] * len(prefix) + actions,
            rejected_actual_response=actual['complete_candidate_response'], original_native_prefix_ids=prefix,
            source_diagnostic_task=task, rejected_generated_EOS=stopped,
            rejected_is_censored_at_complete512_action_budget=not stopped,
            rejected_full_EOS_was_not_invented=True, original_split='train', publisher_split='train',
            positive_finite_case_credit=1., negative_finite_case_credit=0.,
            machine_execution_preference_not_human_ranked=True, external_model_teacher=False)
        pairs.append(pair)
    if len(positives) != 4 or len({r['pair_id'] for r in pairs}) != 20: raise ValueError('Exactly20 pairs across all four original train sources required')
    bindings = input_bindings(ROOT, corpus)
    files = [Path(__file__), readback_path, SOURCE / 'complete.json', SOURCE / 'plan.json', SOURCE / 'actual-neural-episodes.jsonl',
             *python_sources(ROOT, [Path(__file__)])]
    merge_bindings(bindings, {str(p): file_hash(p) for p in files})
    for name, digest in bindings.items():
        if file_hash(name) != digest: raise ValueError('Pair preparation source changed')
    OUT.mkdir()
    for name, rows in [('pairs.jsonl', pairs), ('fresh-negative-executions.jsonl', raw_executions), ('fresh-positive-executions.jsonl', list(positives.values()))]:
        with (OUT / name).open('x') as stream:
            for row in rows: stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    atomic_json(OUT / 'manifest.json', dict(format='own-original-human-code-versus-execution-rejected-unassisted-CPU-pairs-v1',
        at=datetime.now(timezone.utc).isoformat(), pairs=20, distinct_original_train_sources=4,
        genuine_EOS_negatives=sum(r['rejected_generated_EOS'] for r in pairs),
        censored_full512_action_negatives=sum(r['rejected_is_censored_at_complete512_action_budget'] for r in pairs),
        positive_original_EOS_targets_with_pair_repetitions=sum(r['complete_original_reference_EOS_targets'] for r in pairs),
        negative_actual_action_targets=sum(len(r['rejected_actual_action_ids']) for r in pairs),
        all20_original_unassisted_diagnostic_deliveries_included=True, source_selection_not_based_on_final_held_scores=True,
        all_original_reference_and_actual_wrong_deliveries_freshly_reexecuted=True,
        source_files_sha256=bindings, files_sha256={p.name: file_hash(p) for p in OUT.glob('*.jsonl')},
        original_own_tokenizer_sha256=corpus.provenance['tokenizer_sha256'], generation_own_checkpoint_updates=2048,
        own_generation_parameter_values=984399690, human_rankings=False, human_RLHF=False, held_assessment=False,
        external_teacher=False, model_updates=0, candidate_promoted=False, goal_achieved=False,
        limitations=['Four public train problems only; no final independent coding or source-epoch claim.',
            'Fourteen negative sequences end at the actual512-action budget without EOS; neither EOS nor missing text is fabricated.',
            'Length-normalized preference training on censored negatives is an explicit experimental extension, not an exact SimPO replication.',
            'Publisher/bounded execution outcomes do not prove general algorithmic correctness or efficiency.']))
    print(json.dumps(dict(prepared=str(OUT / 'manifest.json'), pairs=20, sources=4,
        negative_EOS=sum(r['rejected_generated_EOS'] for r in pairs), model_updates=0)), flush=True)


if __name__ == '__main__': main()
