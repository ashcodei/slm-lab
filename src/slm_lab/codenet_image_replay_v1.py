"""Original later-image sources with complete masks and explicit source-family roles."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .codenet_replay_v1 import INDEX_DTYPE,training_role
from .corpus import file_hash

FAMILIES=('additional_photo_captions','additional_real_words','additional_commons',
          'original_visual_reasoning','human_photo_reading','own_context_photo_reading')
TASKS=dict(zip(FAMILIES,('caption','word_ocr','caption','visual_reasoning','text_reading_question','own_reader_question')))
ARTIFACTS={'index.npy','records.jsonl','review.jsonl','image-files.jsonl','source-admission.json','own-context-original-rows.jsonl'}


def image_role(row,name):
    training_role(row)
    if (name not in TASKS or row.get('task')!=TASKS[name] or not row.get('image')
            or not row.get('id') or not row.get('group_id') or not isinstance(row.get('prompt'),str)
            or not isinstance(row.get('response'),str) or not row['response']
            or row.get('source_subset') not in {None,'train'}
            or any(not isinstance(row.get(k),str) or len(row[k])!=64 for k in ['stored_image_sha256','model_pixel_sha256_448'])):
        raise ValueError('Original complete training-image source role/task/pixel identity required')
    if name=='additional_photo_captions' and (row.get('visual_source')!='additional_everyday_photos' or row.get('split')!='train' or row.get('source_subset')!='train'):
        raise ValueError('Only original additional training captions required')
    if name=='additional_real_words' and (row.get('split')!='train' or row.get('source_subset')!='train'):
        raise ValueError('Only original additional training words required')
    if name=='additional_commons' and row.get('visual_source')!='commons':
        raise ValueError('Only original additional Commons training rows required')
    if name=='original_visual_reasoning' and row.get('source_subset')!='train':
        raise ValueError('Original training-only visual reasoning required')
    if name=='human_photo_reading' and row.get('split')!='train':
        raise ValueError('Original training-only human reading required')
    if name=='own_context_photo_reading' and (row.get('split')!='train' or row.get('own_predicted_context_supplied') is not True or row.get('feedback_source')!='human'):
        raise ValueError('Only admitted training own-context questions required')


def canonical_image(row,source,root):
    path=Path(row['image'])
    path=(path if path.is_absolute() else Path(source).parent/path).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve()):raise ValueError('Replay image escapes its original project')
    return path


class ImageReplayCorpus:
    def __init__(self,folder):
        self.folder=Path(folder);self.manifest=json.loads((self.folder/'manifest.json').read_text());m=self.manifest
        if m.get('format')!='slm-complete-later-image-replay-v1' or m['family_names']!=list(FAMILIES):
            raise ValueError('Explicit complete later-image replay families required')
        if set(m['artifact_files_sha256'])!=ARTIFACTS:raise ValueError('Complete later-image artifact inventory required')
        for name,digest in m['artifact_files_sha256'].items():
            if file_hash(self.folder/name)!=digest:raise ValueError('Later-image replay artifact changed')
        receipt=json.loads((self.folder/'source-admission.json').read_text())
        if receipt.get('passed') is not True or receipt.get('held_roles_excluded') is not True:
            raise ValueError('Completed original-source image admission checks required')
        self.families=list(FAMILIES);self.sources={};self.source_states={}
        if [s['family'] for s in m['sources']]!=self.families:raise ValueError('Exact ordered image source inventory required')
        for source in m['sources']:
            path=Path(source['path'])
            if file_hash(path)!=source['sha256'] or file_hash(source['original_manifest'])!=source['original_manifest_sha256']:
                raise ValueError('Original later-image source or source manifest changed')
            original=json.loads(Path(source['original_manifest']).read_text())
            if original['files_sha256'][source['original_file_key']]!=source['original_file_sha256']:
                raise ValueError('Original train-source manifest authorization differs')
            if file_hash(source['original_file_path'])!=source['original_file_sha256']:
                raise ValueError('Original prepared image source bytes changed')
            if receipt['source_manifest_sha256'].get(source['family'])!=source['original_manifest_sha256']:
                raise ValueError('Original source-admission manifest differs')
            self.sources[source['family']]=source;self.source_states[source['family']]=self._state(path)
        self.index=np.load(self.folder/'index.npy',mmap_mode='r',allow_pickle=False)
        if (self.index.dtype!=INDEX_DTYPE or self.index.ndim!=1 or len(self.index)!=m['complete_fit_rows']
                or np.any(self.index['family']>=len(self.families)) or np.any(self.index['bytes']==0)
                or np.any(self.index['context_tokens']<3) or np.any(self.index['context_tokens']>4096)
                or np.any(self.index['target_tokens']<1) or np.any(self.index['target_tokens']>=self.index['context_tokens'])):
            raise ValueError('Complete original image index/schema/target masks required')
        self.positions={name:np.flatnonzero(self.index['family']==i) for i,name in enumerate(self.families)}
        for name in self.families:
            items=self.index[self.positions[name]];source=self.sources[name]
            if (len(items)!=m['family_counts'][name] or len(items)==0
                    or len(items)+source['review_rows']!=source['selected_rows']
                    or np.any(items['offset'][1:]<=items['offset'][:-1])
                    or np.any(items['offset']+items['bytes']>Path(source['path']).stat().st_size)):
                raise ValueError('Original image family/source coverage differs')
        self.images={}
        for line in (self.folder/'image-files.jsonl').open():
            record=json.loads(line);path=Path(record['path'])
            if str(path) in self.images or not path.is_relative_to(Path(m['project_root'])) or file_hash(path)!=record['sha256']:
                raise ValueError('Original image inventory/bytes differ')
            self.images[str(path)]={**record,'state':self._state(path)}
        if len(self.images)!=m['unique_original_image_files']:raise ValueError('Complete original image population required')

    @staticmethod
    def _state(path):
        s=Path(path).stat();return s.st_ino,s.st_size,s.st_mtime_ns

    def size(self,name):return len(self.positions[name])

    def row(self,name,position):
        if name not in self.sources:raise KeyError(name)
        if type(position) is not int or not 0<=position<self.size(name):raise IndexError('Image replay position outside original source')
        source=self.sources[name];path=Path(source['path'])
        if self._state(path)!=self.source_states[name]:raise ValueError('Original image source changed during replay')
        item=self.index[self.positions[name][position]]
        with path.open('rb') as stream:
            stream.seek(int(item['offset']));raw=stream.read(int(item['bytes']))
        if hashlib.sha256(raw).digest()!=item['row_sha256'].tobytes():raise ValueError('Original image row bytes differ')
        row=json.loads(raw);image_role(row,name)
        image=canonical_image(row,path,self.manifest['project_root']);expected=self.images.get(str(image))
        if (expected is None or self._state(image)!=expected['state'] or file_hash(image)!=row['stored_image_sha256']
                or expected['sha256']!=row['stored_image_sha256'] or expected['model_pixel_sha256_448']!=row['model_pixel_sha256_448']):
            raise ValueError('Actual original replay image bytes/pixel identity changed')
        return {**row,'image':str(image),'_replay_input_sha256':item['input_sha256'].tobytes().hex(),
            '_replay_context_tokens':int(item['context_tokens']),'_replay_target_tokens':int(item['target_tokens'])}
