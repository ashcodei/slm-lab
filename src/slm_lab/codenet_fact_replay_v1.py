"""Complete original human fact-QA rehearsal, separate from library-code families."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .codenet_replay_v1 import INDEX_DTYPE
from .corpus import file_hash


FAMILY='human_fact_questions'


def fact_role(row):
    if (row.get('split')!='train' or row.get('source_split')!='train'
            or row.get('task')!='nq_open_question' or row.get('image')
            or not row.get('reference_answers') or row['response']!=row['reference_answers'][0]):
        raise ValueError('Original training-only human fact question, answer and aliases required')


class FactReplayCorpus:
    def __init__(self,folder):
        self.folder=Path(folder);self.manifest=json.loads((self.folder/'manifest.json').read_text())
        if self.manifest.get('format')!='slm-complete-human-fact-replay-v1' or self.manifest['family_names']!=[FAMILY]:
            raise ValueError('Explicit original human fact-replay format/family required')
        if set(self.manifest['artifact_files_sha256'])!={'index.npy','records.jsonl','source-admission.json'}:
            raise ValueError('Complete fixed fact-replay artifact inventory required')
        for name,digest in self.manifest['artifact_files_sha256'].items():
            if file_hash(self.folder/name)!=digest:raise ValueError('Fact-replay artifact bytes differ')
        admission=json.loads((self.folder/'source-admission.json').read_text())
        if (admission.get('passed') is not True or admission.get('validation_rows_not_returned_for_training') is not True
                or admission.get('original_training_questions')!=self.manifest['complete_fit_rows']
                or admission.get('manifest_sha256')!=self.manifest['original_human_fact_manifest_sha256']):
            raise ValueError('Completed original human fact training-only admission required')
        self.source=Path(self.manifest['original_training_source'])
        if file_hash(self.source)!=self.manifest['original_training_source_sha256']:
            raise ValueError('Original human fact source changed')
        self.source_state=self._state(self.source)
        self.index=np.load(self.folder/'index.npy',mmap_mode='r',allow_pickle=False)
        if self.index.dtype!=INDEX_DTYPE or self.index.ndim!=1 or len(self.index)!=self.manifest['complete_fit_rows']:
            raise ValueError('Original fact-replay index/schema/population differs')
        expected=np.concatenate((np.zeros(1,dtype=np.uint64),np.cumsum(self.index['bytes'].astype(np.uint64))[:-1]))
        if (not np.array_equal(self.index['offset'],expected) or np.any(self.index['family']!=0)
                or np.any(self.index['bytes']==0) or int(self.index['bytes'].astype(np.uint64).sum())!=self.source.stat().st_size
                or np.any(self.index['context_tokens']<3) or np.any(self.index['context_tokens']>4096)
                or np.any(self.index['target_tokens']<1) or np.any(self.index['target_tokens']>=self.index['context_tokens'])):
            raise ValueError('Every complete original fact row and target span required')
        self.families=[FAMILY]

    @staticmethod
    def _state(path):
        s=path.stat();return s.st_ino,s.st_size,s.st_mtime_ns

    def size(self,name):
        if name!=FAMILY:raise KeyError(name)
        return len(self.index)

    def row(self,name,position):
        if name!=FAMILY:raise KeyError(name)
        if type(position) is not int or not 0<=position<len(self.index):raise IndexError('Fact replay position outside original source')
        if self._state(self.source)!=self.source_state:raise ValueError('Original human fact source changed during replay')
        item=self.index[position]
        with self.source.open('rb') as f:
            f.seek(int(item['offset']));raw=f.read(int(item['bytes']))
        if hashlib.sha256(raw).digest()!=item['row_sha256'].tobytes():
            raise ValueError('Original fact row bytes differ')
        row=json.loads(raw);fact_role(row)
        return {**row,'_replay_input_sha256':item['input_sha256'].tobytes().hex(),
            '_replay_context_tokens':int(item['context_tokens']),'_replay_target_tokens':int(item['target_tokens'])}


class CombinedReplay:
    """Duck-typed training_update interface; final runner binding stays separate."""
    def __init__(self,*components):
        if not components:raise ValueError('Explicit complete replay components required')
        self.components=components;self.families=[];self.owners={}
        for component in components:
            for name in component.families:
                if name in self.owners:raise ValueError('Replay components must have distinct source families')
                self.owners[name]=component;self.families.append(name)

    def size(self,name):return self.owners[name].size(name)

    def row(self,name,position):return self.owners[name].row(name,position)
