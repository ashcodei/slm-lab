"""Complete original reviewed human conversations for code-stage rehearsal."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .codenet_replay_v1 import INDEX_DTYPE
from .corpus import file_hash

FAMILY='reviewed_human_dialogue'
SOURCE='OpenAssistant/oasst2 original human conversation export'


def dialogue_role(row):
    quality=row.get('human_quality') or {}
    if (row.get('split')!='train' or row.get('source_split','train')!='train'
            or row.get('source')!=SOURCE or row.get('synthetic') is not False or row.get('image')
            or row.get('source_message_id') is None or row.get('source_message_tree_id') is None
            or quality.get('count',0)<1 or quality.get('value',0)<.5
            or row.get('group_id')!='oasst2:'+row['source_message_tree_id']
            or row.get('id')!='oasst2-'+row['source_message_id']
            or not isinstance(row.get('prompt'),str) or not row['prompt'].endswith('\nAssistant:')
            or not isinstance(row.get('response'),str) or not row['response']
            or hashlib.sha256(row['response'].encode()).hexdigest()!=row.get('source_text_sha256')):
        raise ValueError('Original training-only reviewed human dialogue required')


class DialogueReplayCorpus:
    def __init__(self,folder):
        self.folder=Path(folder);self.manifest=json.loads((self.folder/'manifest.json').read_text())
        m=self.manifest
        if m.get('format')!='slm-complete-reviewed-dialogue-replay-v1' or m['family_names']!=[FAMILY]:
            raise ValueError('Explicit reviewed-dialogue replay format/family required')
        if set(m['artifact_files_sha256'])!={'index.npy','records.jsonl','source-admission.json'}:
            raise ValueError('Complete original-dialogue artifact inventory required')
        for name,digest in m['artifact_files_sha256'].items():
            if file_hash(self.folder/name)!=digest:raise ValueError('Dialogue replay artifact changed')
        source_manifest=Path(m['original_source_manifest'])
        if file_hash(source_manifest)!=m['original_source_manifest_sha256']:
            raise ValueError('Original reviewed-dialogue source manifest changed')
        original=json.loads(source_manifest.read_text());admission=json.loads((self.folder/'source-admission.json').read_text())
        if (admission.get('passed') is not True or admission.get('validation_rows_not_returned_for_training') is not True
                or admission.get('manifest_sha256')!=m['original_source_manifest_sha256']
                or admission.get('original_training_examples')!=m['complete_fit_rows']
                or original['splits']['train']['examples']!=m['complete_fit_rows']
                or original['files_sha256']['train.jsonl']!=m['original_training_source_sha256']):
            raise ValueError('Complete original train-only dialogue source binding required')
        self.source=Path(m['original_training_source'])
        if file_hash(self.source)!=m['original_training_source_sha256']:
            raise ValueError('Original reviewed-dialogue bytes changed')
        self.source_state=self._state(self.source)
        self.index=np.load(self.folder/'index.npy',mmap_mode='r',allow_pickle=False)
        if self.index.dtype!=INDEX_DTYPE or self.index.ndim!=1 or len(self.index)!=m['complete_fit_rows']:
            raise ValueError('Original dialogue index schema/population differs')
        expected=np.concatenate((np.zeros(1,dtype=np.uint64),np.cumsum(self.index['bytes'].astype(np.uint64))[:-1]))
        if (not np.array_equal(self.index['offset'],expected) or np.any(self.index['family']!=0)
                or np.any(self.index['bytes']==0) or int(self.index['bytes'].astype(np.uint64).sum())!=self.source.stat().st_size
                or np.any(self.index['context_tokens']<3) or np.any(self.index['context_tokens']>4096)
                or np.any(self.index['target_tokens']<1) or np.any(self.index['target_tokens']>=self.index['context_tokens'])):
            raise ValueError('Every complete original dialogue row and target span required')
        self.families=[FAMILY]

    @staticmethod
    def _state(path):
        s=path.stat();return s.st_ino,s.st_size,s.st_mtime_ns

    def size(self,name):
        if name!=FAMILY:raise KeyError(name)
        return len(self.index)

    def row(self,name,position):
        if name!=FAMILY:raise KeyError(name)
        if type(position) is not int or not 0<=position<len(self.index):raise IndexError('Dialogue source position outside original population')
        if self._state(self.source)!=self.source_state:raise ValueError('Original dialogue source changed during replay')
        item=self.index[position]
        with self.source.open('rb') as source:
            source.seek(int(item['offset']));raw=source.read(int(item['bytes']))
        if hashlib.sha256(raw).digest()!=item['row_sha256'].tobytes():raise ValueError('Original dialogue row bytes differ')
        row=json.loads(raw);dialogue_role(row)
        return {**row,'_replay_input_sha256':item['input_sha256'].tobytes().hex(),
            '_replay_context_tokens':int(item['context_tokens']),'_replay_target_tokens':int(item['target_tokens'])}
