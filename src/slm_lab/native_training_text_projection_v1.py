"""Exact input-token projection of all current full-source training families.

Image-bearing examples retain their real text prompt/target. This projection
does not claim to inspect unannotated text inside the image pixels.
"""
import numpy as np

from .expanded_joint_sources_v1 import ExpandedJointSources


class NativeTrainingTextProjection:
    def __init__(self, data):
        if type(data) is not ExpandedJointSources or (data.role, data.purpose) != ('train','optimization'):
            raise ValueError('Exact current original train-only source graph required')
        self.data = data

    def sequence(self, name, position):
        data = self.data
        if name not in data.populations or type(position) is not int or not 0 <= position < data.populations[name]:
            raise ValueError('Exact original training family/position required')
        if name in data.expanded_owners:
            kind, family = data.expanded_owners[name]
            if kind == 'packed':
                ids, labels = data.packed_encoder.sequence(data.packed.row(family, position), None)
            elif kind == 'views':
                ids, labels = data.views.sequence(data.view_positions[family][position])
            else:
                source = data.later if kind == 'later' else data.fact
                ids, labels = data.expanded_encoder.sequence(source.row(family, position), 'response')
        elif name in data.base.populations:
            stream = data.base.streams[name]
            if name == 'education':
                row = stream.window(position); length = sum(row['attention_mask'])
                ids, labels = row['input_ids'][:length], row['labels'][:length]
            elif name == 'human': ids, labels = stream.sequence(position, 'response')
            else: ids, labels = stream.pack.sequence(position)
        elif name == 'documented': ids, labels = data.documented.sequence(position)
        elif name == 'words':
            row = data.words.rows[position]; item = row['native']
            ids = data.words.tokens[item['offset']:item['offset']+item['length']].astype(np.int64).tolist()
            labels = [-100]*item['target_start'] + ids[item['target_start']:]
        else:
            group, family = name.split(':', 1)
            source = data.replay if group == 'replay' else data.images
            ids, labels = data.replay_encoder.sequence(source.row(family, position), 'response')
        if (len(ids) != data.lengths[name][position]
                or sum(value != -100 for value in labels[1:]) != data.targets[name][position]):
            raise ValueError('Every native input and original response/next-token/EOS mask must match actual training')
        return ids, labels

    def family(self, name):
        if name not in self.data.populations: raise ValueError('Unknown training family')
        for position in range(self.data.populations[name]):
            yield position, *self.sequence(name, position)
