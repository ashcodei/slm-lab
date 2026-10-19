"""Lossless native-record factoring before collectives and resident history.

The full immutable own-reference binding stays once in the stage/checkpoint.
All dynamic source/action/probability/verdict/timing fields remain ordinary
copied values. Expanding a record is exact; no neural values or rewards change.
"""
import copy
import json

from .parallel_function_repair_stage_v1 import value_digest

PROTOCOL = 'own-native-records-one-exact-reference-before-collective-history-v1'
MARKER = 'own-native-record-reference-sha256-v1'


def canonical(value): return (json.dumps(value,ensure_ascii=True,allow_nan=False)+'\n').encode()


class NativeReferenceRecords:
    def __init__(self,reference):
        if (not isinstance(reference,dict) or reference.get('format')!='frozen-own-starting-actor-shared-dense-reference-v1'
                or not reference.get('sources')):
            raise ValueError('The entire actual starting-own-policy reference binding is required')
        self.reference=copy.deepcopy(reference)
        self.identity=value_digest(reference)
        self.original_bytes=canonical(reference)

    @property
    def marker(self):
        return dict(format=MARKER,sha256=self.identity)

    def check(self,reference):
        if (value_digest(reference)!=self.identity or value_digest(self.reference)!=self.identity
                or canonical(reference)!=self.original_bytes or canonical(self.reference)!=self.original_bytes):
            raise ValueError('Every original immutable reference field and value type must remain')

    def pack_observation(self,observation):
        if (not isinstance(observation,dict) or set(observation)!={'rank','actual_original_positions','kernel'}
                or type(observation['rank']) is not int or not isinstance(observation['actual_original_positions'],list)):
            raise ValueError('The exact native rank observation and original source positions are required')
        kernel=observation['kernel']
        if kernel is None:
            if observation['actual_original_positions']:raise ValueError('Only a genuine empty rank may omit its source kernel')
            packed=None
        else:
            if (not isinstance(kernel,dict) or 'own_reference_binding' not in kernel
                    or len(observation['actual_original_positions'])!=1
                    or value_digest(kernel['own_reference_binding'])!=self.identity
                    or canonical(kernel['own_reference_binding'])!=self.original_bytes):
                raise ValueError('Each actual whole-source kernel must retain the exact original own reference')
            packed={k:dict(self.marker) if k=='own_reference_binding' else copy.deepcopy(v) for k,v in kernel.items()}
        return {k:packed if k=='kernel' else copy.deepcopy(v) for k,v in observation.items()}

    def expand_observation(self,observation,*,share_reference=False):
        if (not isinstance(observation,dict) or set(observation)!={'rank','actual_original_positions','kernel'}
                or type(observation['rank']) is not int or not isinstance(observation['actual_original_positions'],list)):
            raise ValueError('The exact complete compact rank observation is required')
        kernel=observation['kernel']
        if kernel is None:
            if observation['actual_original_positions']:raise ValueError('A nonempty original source cannot disappear from compact history')
            expanded=None
        else:
            if (not isinstance(kernel,dict) or kernel.get('own_reference_binding')!=self.marker
                    or len(observation['actual_original_positions'])!=1):
                raise ValueError('A complete kernel must bind exactly this unchanged reference, never another source')
            expanded={k:(self.reference if share_reference else copy.deepcopy(self.reference))
                if k=='own_reference_binding' else copy.deepcopy(v) for k,v in kernel.items()}
        return {k:expanded if k=='kernel' else copy.deepcopy(v) for k,v in observation.items()}

    def pack(self,row):
        if (not isinstance(row,dict) or not isinstance(row.get('actual_rank_observations'),list)
                or not 1<=len(row['actual_rank_observations'])<=4):
            raise ValueError('Every actual original source-rank record is required')
        return {k:[self.pack_observation(r) for r in v] if k=='actual_rank_observations' else copy.deepcopy(v) for k,v in row.items()}

    def expand(self,row,*,share_reference=False):
        if (not isinstance(row,dict) or not isinstance(row.get('actual_rank_observations'),list)
                or not 1<=len(row['actual_rank_observations'])<=4):
            raise ValueError('Every compact original source-rank record is required')
        return {k:[self.expand_observation(r,share_reference=share_reference) for r in v]
            if k=='actual_rank_observations' else copy.deepcopy(v) for k,v in row.items()}
