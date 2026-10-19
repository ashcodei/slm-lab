"""Parallel complete held human pairs with unchanged masks and source readback."""
from contextlib import nullcontext
import json
import math
from pathlib import Path

import torch

from .corpus import file_hash
from .dense_human_preference_evaluation_v1 import information, scored, summary, readback
from .dense_human_preference_v1 import sequence_logp
from .diverse_images import atomic_json
from .own_dense_actor_v1 import OwnDenseActor
from .whole_human_data_v2 import WholeHumanData

FORMAT = 'own-original-held-human-preference-shards-v1'


def positions(data, rank, world):
    if (type(data) is not WholeHumanData or (data.family, data.role, data.purpose) != ('preferences', 'valid', 'evaluation')
            or type(world) is not int or not 1 <= world <= len(data)
            or type(rank) is not int or not 0 <= rank < world):
        raise ValueError('Original evaluation-only held data and nonempty exact shard required')
    return list(range(rank, len(data), world))


def profile(provenance, beta):
    keys = {'checkpoint_sha256', 'run_plan_sha256', 'human_manifest_sha256',
            'frozen_dense_sha256', 'inference_device_type', 'inference_precision'}
    if (set(provenance) != keys or not math.isfinite(beta) or not 0 < beta <= 1
            or any(not isinstance(provenance[k], str) or len(provenance[k]) != 64
                   or any(c not in '0123456789abcdef' for c in provenance[k])
                   for k in keys if k.endswith('sha256'))
            or provenance['inference_device_type'] not in ('cpu', 'cuda')
            or provenance['inference_precision'] not in ('float32', 'bf16')):
        raise ValueError('Complete exact held-policy/source/runtime profile required')
    return dict(provenance, DPO_beta=beta)


@torch.no_grad()
def evaluate_shard(model, data, path, *, rank, world, provenance, beta=.1, amp=None, checked=lambda:None):
    selected = positions(data, rank, world)
    expected_profile = profile(provenance, beta)
    if type(model) is not OwnDenseActor or any(p.grad is not None for p in model.parameters()):
        raise ValueError('Exact counted own policy and clean evaluation-only gradients required')
    path = Path(path)
    journal = path.with_suffix('.jsonl')
    if path.exists() or journal.exists():
        raise ValueError('Preserve all earlier full or partial held shard results')
    rows = []
    model.eval()
    device = next(model.parameters()).device
    if device.type != provenance['inference_device_type']:
        raise ValueError('Declared and actual inference devices must match')
    amp = amp or nullcontext
    with journal.open('x') as stream:
        for p in selected:
            checked()
            row = information(data, p)
            for field in ('chosen', 'rejected'):
                batch = data.batch([p], field, device)
                inputs = {k:v for k,v in batch.items() if k != 'labels'}
                with amp():
                    policy = model(**inputs)['logits']
                    reference = model.base(**inputs)['logits']
                for name, logits in [('policy', policy), ('reference', reference)]:
                    value, count = sequence_logp(logits, batch['labels'])
                    if (not math.isfinite(float(value)) or value > 0
                            or count != row[field+'_response_EOS_targets']):
                        raise ValueError('Finite complete original held response/EOS scores required')
                    row[name+'_'+field+'_logp'] = float(value)
            row.update(scored(row, beta))
            rows.append(row)
            stream.write(json.dumps(row)+'\n')
            stream.flush()
    result = dict(format=FORMAT, rank=rank, world=world, profile=expected_profile,
        original_held_positions=selected, raw_journal=str(journal.resolve()), raw_journal_sha256=file_hash(journal),
        results=rows, overall=summary(rows), held_gradients=False,
        implementation_sha256=file_hash(__file__), goal_achieved=False)
    atomic_json(path, result)
    return result


def reconcile(data, paths, output):
    paths = [Path(p) for p in paths]
    output = Path(output)
    if output.exists() or output.with_suffix('.jsonl').exists() or not paths:
        raise ValueError('Fresh complete held preference destination required')
    world = len(paths)
    seen = set()
    rows = {}
    shared = None
    bindings = {}
    for path in paths:
        shard = json.loads(path.read_text())
        rank = shard.get('rank')
        selected = positions(data, rank, world)
        journal = Path(shard['raw_journal'])
        if (shard.get('format') != FORMAT or shard.get('world') != world or rank in seen
                or shard.get('original_held_positions') != selected
                or shard.get('implementation_sha256') != file_hash(__file__)
                or shard.get('held_gradients') is not False
                or file_hash(journal) != shard['raw_journal_sha256']
                or shard['results'] != [json.loads(line) for line in journal.open()]):
            raise ValueError('Every unchanged raw held shard, role and exact source allocation required')
        current = dict(shard['profile'])
        beta = current.pop('DPO_beta')
        validated = profile(current, beta)
        if shared is None:
            shared = validated
        if shared != validated or [r['original_held_position'] for r in shard['results']] != selected:
            raise ValueError('Every held policy, base, mask, budget and source identity must match')
        for row in shard['results']:
            expected = information(data, row['original_held_position'])
            if (any(row.get(k) != v for k,v in expected.items())
                    or any(not math.isfinite(row[n+'_'+f+'_logp']) or row[n+'_'+f+'_logp'] > 0
                        for n in ('policy', 'reference') for f in ('chosen', 'rejected'))
                    or any(row.get(k) != v for k,v in scored(row, beta).items())):
                raise ValueError('Every complete original held fingerprint and preference score must reconstruct')
            rows[row['original_held_position']] = row
        if shard['overall'] != summary(shard['results']):
            raise ValueError('Every shard aggregate must reconstruct')
        seen.add(rank)
        bindings[str(path.resolve())] = file_hash(path)
        bindings[str(journal)] = file_hash(journal)
    if seen != set(range(world)) or set(rows) != set(range(len(data))):
        raise ValueError('Every original held human pair must appear exactly once; no omitted failures')
    ordered = [rows[p] for p in range(len(data))]
    report = dict(DPO_beta=shared['DPO_beta'], overall=summary(ordered),
        by_source={n:summary([r for r in ordered if r['source_blob'] == n]) for n in sorted({r['source_blob'] for r in ordered})},
        results=ordered, every_original_held_pair_once=True, held_gradients=False,
        teacher_other_model_responses_not_own_output_human_judgments=True,
        ties_receive_half_credit=True, profile=shared, source_files_sha256=bindings,
        raw_reports_preserved=True, shards=world, candidate_promoted=False, goal_achieved=False)
    with output.with_suffix('.jsonl').open('x') as stream:
        for row in ordered:
            stream.write(json.dumps(row)+'\n')
    atomic_json(output, report)
    readback(data, output)
    return report
