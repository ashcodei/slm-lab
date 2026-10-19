"""Join whole declared source streams without repeating a shorter stream.

Each component retains its own source identities, target denominators and
rank assignment. Shared update references store component records once.
This metadata primitive does not admit a model, GPU job or training run.
"""
from collections import Counter
import hashlib
import json

FORMAT = 'own-complete-code-visual-human-source-schedule-v1'
STREAMS = ('code', 'visual', 'human')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def combine(streams):
    if type(streams) is not dict or set(streams) != set(STREAMS):
        raise ValueError('All complete code, visual and human streams are required')
    budgets = {}; identities = {}; sizes = {}
    for name in STREAMS:
        stream = streams[name]
        if (type(stream) is not dict or stream.get('source_role') != 'train'
                or type(stream.get('source_identity_sha256')) is not str or len(stream['source_identity_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in stream['source_identity_sha256'])
                or type(stream.get('entries')) is not list or not stream['entries']):
            raise ValueError('A whole train-only stream with its exact original source identity is required')
        rows = stream['entries']; seen = set()
        for index, row in enumerate(rows, 1):
            if (type(row) is not dict or type(row.get('source_step')) is not int or row['source_step'] != index
                    or type(row.get('epoch')) is not int or row['epoch'] < 0
                    or type(row.get('family')) is not str or not row['family']
                    or type(row.get('original_positions')) is not list or not row['original_positions']
                    or any(type(p) is not int or p < 0 for p in row['original_positions'])
                    or len(set(row['original_positions'])) != len(row['original_positions'])
                    or row.get('rank_source_counts') != [len(row['original_positions'][r::4]) for r in range(4)]
                    or any(type(row.get(k)) is not int or row[k] < 0 for k in ['complete_context_tokens', 'response_EOS_targets'])
                    or row['response_EOS_targets'] > row['complete_context_tokens']):
                raise ValueError('Every original source step, epoch, full budget and actual rank assignment must remain explicit')
            for position in row['original_positions']:
                key = (row['epoch'], row['family'], position)
                if key in seen: raise ValueError('A declared source occurrence cannot repeat within its original epoch')
                seen.add(key)
        budgets[name] = dict(updates=len(rows), source_presentations=sum(len(r['original_positions']) for r in rows),
            complete_context_tokens=sum(r['complete_context_tokens'] for r in rows),
            response_EOS_targets=sum(r['response_EOS_targets'] for r in rows),
            task_empty_rank_calls=sum(sum(n == 0 for n in r['rank_source_counts']) for r in rows))
        identities[name] = digest(stream); sizes[name] = len(rows)
    entries = []
    combinations = Counter()
    for index in range(max(sizes.values())):
        parts = []
        for name in STREAMS:
            if index < sizes[name]:
                parts.append(dict(stream=name, source_step=index + 1, original_entry_sha256=digest(streams[name]['entries'][index])))
        entries.append(dict(step=index + 1, components=parts))
        combinations['+'.join(p['stream'] for p in parts)] += 1
    result = dict(format=FORMAT, world_size=4, source_role='train', component_streams_sha256=identities,
        original_stream_budgets=budgets, actual_joint_updates=len(entries), task_combinations=dict(combinations), entries=entries,
        every_original_component_step_consumed_once=True, shorter_streams_not_repeated=True,
        independent_component_denominators_preserved=True, actual_partial_task_tails_preserved=True,
        production_training_admitted=False, models_constructed=0, optimizer_updates=0, goal_achieved=False)
    return result


def require_original_scope(joint):
    budgets = joint['original_stream_budgets']
    expected = dict(code=dict(updates=5256, source_presentations=84084, complete_context_tokens=42271575, response_EOS_targets=21316734),
        visual=dict(updates=4733, source_presentations=302627, complete_context_tokens=17681556, response_EOS_targets=4669243),
        human=dict(updates=5362, source_presentations=85786, complete_context_tokens=31147346, response_EOS_targets=8514104))
    if (joint.get('format') != FORMAT or joint.get('actual_joint_updates') != 5362
            or joint.get('task_combinations') != {'code+visual+human': 4733, 'code+human': 523, 'human': 106}
            or set(budgets) != set(expected) or any(any(budgets[n].get(k) != v for k, v in values.items()) for n, values in expected.items())
            or joint.get('production_training_admitted') is not False):
        raise ValueError('The entire original code/visual/human populations and full5362step schedule are required')
