"""CPU source/per-position/matrix readback of the completed conditioning study."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re

import numpy as np
import torch
from tokenizers import Tokenizer

import assess_joint_java_comparison_v1 as resources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.own_code_conditioning_v1 import sequences, summarize

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'reports/own-code-conditioning-v1'
REPORTS = ROOT / 'reports/own-code-conditioning-v2-readback'


def main():
    if REPORTS.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or torch.cuda.is_initialized():
        raise ValueError('Fresh GPU-hidden source/score readback required')
    plan = json.loads((ORIGINAL / 'plan.json').read_text())
    if resources.same(plan['owner']) or any(resources.same(a) for a in json.loads((ORIGINAL / 'children.json').read_text())):
        raise ValueError('Every actual original diagnostic actor must end before final readback')
    if not json.loads((ORIGINAL / 'restored.json').read_text())['all_registered_point_actors_terminal']:
        raise ValueError('Original diagnostic must have restored its waiter')
    files = dict(plan['source_files_sha256'])
    for path in [Path(__file__).resolve(), *ORIGINAL.glob('*.json'), *ORIGINAL.glob('*.jsonl')]:
        files[str(path)] = file_hash(path)
    guard = FileBindings(files); REPORTS.mkdir()
    atomic_json(REPORTS / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=files, model_updates=0, full_GPU_neural_scores_not_replayed=True, goal_achieved=False))
    rows = list(map(json.loads, (ORIGINAL / 'calibration.jsonl').read_text().splitlines()))
    tokenizer = Tokenizer.from_file(plan['tokenizer']); total = 0
    for row in rows:
        if row['split'] != 'train' or row['publisher_split'] != 'train':
            raise ValueError('All original source roles must be train')
        folder = ROOT / 'data/public-apps-original-inventory-v1/original-train' / row['original_problem']
        question = (folder / 'question.txt').read_bytes().decode()
        starter_path = folder / 'starter_code.py'; starter = starter_path.read_bytes().decode() if starter_path.exists() else None
        interface = ('Use the original required function or class interface.' if starter or row['cases']['function']
            else 'Read standard input and write standard output as specified.')
        prompt = 'Write a Python solution to the following problem. ' + interface + ' Return only the complete code.\n\n' + question
        if starter:
            prompt += '\n\nOriginal starter code:\n' + starter
        prefix = [1, *tokenizer.encode('User: ' + prompt + '\nAssistant:').ids]
        body = [*tokenizer.encode(row['original_human_program']).ids, 2]
        index = int(re.search(r':solution:(\d+):verified-generation-v1$', row['id']).group(1))
        original_program = json.loads((folder / 'solutions.json').read_bytes())[index]
        if (original_program != row['original_human_program'] or prompt != row['prompt'] or prefix != row['prefix_ids']
                or row['target_tokens'] != len(body)
                or hashlib.sha256(np.asarray(prefix + body, dtype='<u2').tobytes()).hexdigest() != row['complete_input_ids_sha256']):
            raise ValueError('Complete actual original question, source solution index, human program and all own-token boundaries must reconstruct')
        total += len(body)
    expected = sequences(rows, tokenizer); mapping = {s['sequence_sha256']: s for s in expected}
    records = []; digests = []
    for shard in range(4):
        complete = json.loads((ORIGINAL / f'complete-{shard}.json').read_text())
        if not complete['all_weights_and_versions_unchanged'] or complete['component_digests_before'] != complete['component_digests_after']:
            raise ValueError('Complete original/code/head weight and version retention required')
        path = ORIGINAL / f'scores-{shard}.jsonl'
        if file_hash(path) != complete['scores_sha256']:
            raise ValueError('All actual retained score bytes required')
        part = list(map(json.loads, path.read_text().splitlines()))
        assigned = {r['id'] for i, r in enumerate(rows) if i % 4 == shard}
        wanted = [(s['sequence_sha256'], route) for s in expected if s['target_id'] in assigned for route in plan['routes']]
        if [(r['sequence_sha256'], r['route']) for r in part] != wanted:
            raise ValueError('All actual disjoint complete same-interface matrix cells and routes required')
        for record in part:
            source = mapping[record['sequence_sha256']]
            for field in ['target_id', 'question_id', 'target_mode', 'condition', 'prefix_tokens', 'target_tokens']:
                if record[field] != source[field]:
                    raise ValueError('Actual score must bind its entire original source/mask and condition')
            targets = source['input_ids'][source['prefix_tokens']:]; values = record['per_target_NLL']
            top = record['teacher_argmax_ids']; n = len(targets)
            if (targets != record['target_ids'] or len(values) != n or len(top) != n
                    or any(not math.isfinite(v) or v < 0 for v in values)
                    or not math.isclose(math.fsum(values), record['whole_response_EOS_sum_NLL'], rel_tol=1e-6, abs_tol=1e-4)
                    or not math.isclose(math.fsum(values) / n, record['whole_response_EOS_mean_NLL'], rel_tol=1e-6, abs_tol=1e-6)
                    or record['teacher_argmax_correct'] != sum(a == b for a, b in zip(top, targets))
                    or record['first_target_correct'] != (top[0] == targets[0])
                    or record['first_target_NLL'] != values[0]):
                raise ValueError('Every actual target/EOS, per-position loss and argmax statistic must reconstruct')
        records.extend(part); digests.append(complete['component_digests_after'])
    if any(d != digests[0] for d in digests):
        raise ValueError('All four GPU workers must use the same original/code/head weight bytes')
    summary = summarize(records, rows, routes=plan['routes'])
    comparison = json.loads((ORIGINAL / 'comparison.json').read_text())
    if json.loads(json.dumps(summary)) != comparison['routes'] or len(records) != 288 or total != 2057:
        raise ValueError('All288score cells,2057original target/EOS positions and route summaries must reconstruct')
    guard.check(full=True)
    result = dict(completed_at=datetime.now(timezone.utc).isoformat(), original_plan_sha256=file_hash(ORIGINAL / 'plan.json'),
        original_comparison_sha256=file_hash(ORIGINAL / 'comparison.json'), actual_original_train_programs=len(rows),
        actual_original_response_EOS_positions_per_proper_route=total, complete_question_response_route_scores=len(records),
        all_original_human_programs_source_indices_questions_masks_and_tokens_reconstructed=True,
        every_retained_per_position_loss_and_argmax_recounted=True, all_four_original_code_head_states_equal_and_unchanged=True,
        routes=summary, GPU_neural_forwards_not_independently_replayed=True,
        proper_code_matching_question_best=16, proper_code_greedy_case_passes_in_previous_study=1,
        interpretation='The code view distinguishes the matching question on these16training programs. Teacher forcing under the original human prefix does not establish correct free generation. Do not proceed on the hypothesis that this study shows complete question ignorance.',
        controls_have_different_lengths_and_uncertified_semantic_relationship=True,
        not_held_or_execution_accuracy=True, model_updates=0, CUDA_initialized=torch.cuda.is_initialized(),
        candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'complete.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
