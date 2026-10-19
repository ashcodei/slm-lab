"""Complete source-index backed visual/native/word batch delivery.

Audit mode is usable before full pixel/source training admission. Capacity or
optimization require an additional source-admission receipt; none is inferred
from a prepared index or from admission of the four small corrections alone.
No neural model is instantiated by this data adapter.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageOps
from tokenizers import Tokenizer
import torch

from .codenet_replay_v1 import FullReplayEncoder,training_role
from .multimodal_source_index_format_v1 import FORMAT,DTYPE
from .shared_word_data_v1 import image_arrays,array_digest,native_bytes


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):value.update(block)
    return value.hexdigest()


def state(path):
    value = Path(path).stat();return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)


class CorrectedMultimodalBatches:
    role = 'train'

    def __init__(self,root,config,*,purpose):
        if purpose not in {'audit','capacity','optimization'}:raise ValueError('Explicit train data audit/capacity/optimization purpose required')
        self.root = Path(root).resolve();self.purpose = purpose;self.config = config
        self.folder = self.root/'reports/full-multimodal-corrected-source-index-CPU-v1'
        reader_path = self.root/'reports/full-multimodal-corrected-source-index-v1-readback/complete.json'
        self.proof = read(self.folder/'complete.json');reader = read(reader_path)
        if (not self.proof.get('complete') or not reader.get('complete') or self.proof['format'] != FORMAT
                or self.proof['training_admitted'] or reader['training_admitted']
                or self.proof['all_original_train_views'] != 302627 or reader['actual_all_original_entries'] != 302627
                or reader['actual_current_positive_native_entries_changed'] != 4):
            raise ValueError('The full original corrected index and exact complete fresh readback required')
        self.training_admitted = False
        if purpose != 'audit':
            admission_path = self.folder/'source-training-admission.json'
            if not admission_path.is_file():raise ValueError('Full physical-pixel and batch-source admission is still required before capacity or optimization')
            admission = read(admission_path)
            if (admission.get('training_admitted') is not True or admission.get('admitted_role') != 'train'
                    or admission.get('index_sha256') != self.proof['index_sha256']
                    or admission.get('index_readback_sha256') != sha(reader_path)
                    or not admission.get('source_files_sha256')):
                raise ValueError('Exact separate complete-source training admission required')
            for name,expected in admission['source_files_sha256'].items():
                if sha(name) != expected:raise ValueError('Source-admission evidence changed')
            for key in ['full_pixel_report','full_pixel_readback','batch_source_readback']:
                evidence = read(admission[key])
                if not evidence.get('complete') or evidence.get('passed') is not True:
                    raise ValueError('All full source pixel and batch readback evidence must pass')
            self.training_admitted = True
        self.files = {**self.proof['source_files_sha256'],**reader['source_files_sha256']}
        self.files[str(self.folder/'complete.json')] = sha(self.folder/'complete.json')
        self.files[str(reader_path)] = sha(reader_path)
        self.index_path = self.folder/'complete-native-source-index.npy'
        if sha(self.index_path) != self.proof['index_sha256'] or any(sha(p) != expected for p,expected in self.files.items()):
            raise ValueError('Every original/current source and complete index must retain its exact bytes')
        self.states = {p:state(p) for p in self.files}
        self.index = np.load(self.index_path,mmap_mode='r',allow_pickle=False)
        if self.index.dtype != DTYPE or self.index.shape != (302627,):raise ValueError('Exact full compact native source schema required')
        self.names = self.proof['family_names'];self.positions = {};self.lengths = {};self.targets = {}
        for number,name in enumerate(self.names):
            positions = np.flatnonzero(self.index['family'] == number)
            if (len(positions) != self.proof['actual_family_budgets'][name]['original_views']
                    or not np.array_equal(self.index['original_local_position'][positions],np.arange(len(positions)))):
                raise ValueError('Each complete original family must retain every ordered local occurrence')
            self.positions[name] = positions
            self.lengths[name] = self.index['context_tokens'][positions].astype(int).tolist()
            self.targets[name] = self.index['response_EOS_targets'][positions].astype(int).tolist()
        self.metadata_path = self.root/'reports/full-multimodal-source-epoch-CPU-v1/all302627-original-native-views.jsonl'
        self.correction_folder = self.root/'data/public-document-visible-corrections-admitted-v1'
        admission = read(self.correction_folder/'admission.json')
        correction_file = self.correction_folder/'all4-complete-corrections-and-preferences.jsonl'
        if not admission['training_admitted'] or sha(correction_file) != admission['data_file_sha256']:
            raise ValueError('Exactly the independently admitted corrective data required')
        self.corrections = [json.loads(line) for line in correction_file.open()]
        if len(self.corrections) != 4:raise ValueError('All four qualified current response replacements required')
        self.inventory = {}
        for name in ['codenet-image-replay-v1','codenet-replay-v1']:
            for row in map(json.loads,(self.root/'data'/name/'image-files.jsonl').open()):
                key = str(Path(row['path']).resolve());entry = self.inventory.setdefault(key,{})
                for field,value in [('stored',row['sha256']),('native_RGB448',row.get('model_pixel_sha256_448'))]:
                    if value is None:continue
                    if field in entry and entry[field] != value:raise ValueError('Conflicting original physical pixel protocols')
                    entry[field] = value
        self.tokenizer = Tokenizer.from_file(str(self.root/'data/tokenizer-corpus/tokenizer.json'))
        if self.tokenizer.truncation is not None or self.tokenizer.padding is not None:
            raise ValueError('No hidden native clipping/padding is allowed')
        self.encoder = FullReplayEncoder(self.tokenizer,config,self.root,448)
        self.word_tokens_path = self.root/'data/shared-byte-words-v1/train.tokens.u16'
        self.word_tokens = np.memmap(self.word_tokens_path,dtype='<u2',mode='r')

    def guard(self,path):
        key = str(Path(path))
        if key not in self.states or state(key) != self.states[key]:raise ValueError('An exact bound source changed or an unbound source was requested')

    def row(self,name,position):
        if name not in self.positions or type(position) is not int or not 0 <= position < len(self.positions[name]):
            raise ValueError('Exact original train family/local source position required')
        self.guard(self.index_path);self.guard(self.metadata_path)
        entry = self.index[int(self.positions[name][position])]
        with self.metadata_path.open('rb') as stream:stream.seek(int(entry['metadata_offset']));raw = stream.read(int(entry['metadata_bytes']))
        if hashlib.sha256(raw).digest() != entry['metadata_sha256'].tobytes():raise ValueError('Whole original metadata bytes differ')
        metadata = json.loads(raw);reference = metadata['original_source_reference'];source = Path(reference['path']);self.guard(source)
        with source.open('rb') as stream:stream.seek(int(entry['source_offset']));raw = stream.read(int(entry['source_bytes']))
        if hashlib.sha256(raw).digest() != entry['source_sha256'].tobytes():raise ValueError('Whole original source bytes differ')
        original = json.loads(raw);training_role(original)
        if (metadata['family'] != name or metadata['original_local_position'] != position or metadata['source_role'] != 'train'
                or original.get('id') != metadata['source_id']):raise ValueError('Original source identity/family/role must remain exact')
        original = {**original,'image':metadata['image_path']};correction = int(entry['correction_index'])
        current = original
        if correction >= 0:
            if not 0 <= correction < len(self.corrections):raise ValueError('Unknown current correction branch')
            item = self.corrections[correction]
            if (not item['training_admitted'] or item['source_role'] != 'train' or item['original_prepared_native_view'] != metadata
                    or item['positive_native']['input_ids_sha256'] != entry['input_ids_sha256'].tobytes().hex()):
                raise ValueError('Exact qualified positive branch required; original negative must not become current CE')
            current = item['corrected_row'];training_role(current)
        return dict(metadata=metadata,entry=entry,original_row=original,current_row=current,correction_index=correction)

    def negative(self,name,position):
        value = self.row(name,position)
        if value['correction_index'] < 0:return None
        item = self.corrections[value['correction_index']]
        return dict(source_id=item['source_id'],native=item['complete_original_negative_native'],response=item['original_response'],
            supervision='machine_source_derived_preference_negative_not_human_or_student_sample')

    def batch(self,name,positions,*,device='cpu'):
        destination = torch.device(device)
        if self.purpose == 'audit' and destination.type != 'cpu':raise ValueError('Audit batches remain CPU-only before separate full-source training admission')
        if name not in self.positions or not isinstance(positions,list) or len(positions) > 64 or len(set(positions)) != len(positions):
            raise ValueError('One complete homogeneous bounded batch of distinct original source occurrences required')
        if not positions:return None
        native_ids = [];native_labels = [];native_pixels = [];word_pixels = [];texts = [];byte_targets = [];matching_ids = [];keys = [];observations = []
        for position in positions:
            value = self.row(name,position);metadata,entry,row = value['metadata'],value['entry'],value['current_row']
            path = Path(metadata['image_path']).resolve(strict=True)
            if not path.is_relative_to(self.root):raise ValueError('Physical source images must remain inside this project')
            before = state(path)
            if before[2] > 64*1024**2:raise ValueError('Bounded original physical image bytes required')
            raw = path.read_bytes();stored = hashlib.sha256(raw).hexdigest();inventory = self.inventory.get(str(path),{})
            expected_stored = metadata['original_stored_image_sha256'] or inventory.get('stored')
            expected_RGB = metadata['original_RGB448_metadata_sha256'] or inventory.get('native_RGB448')
            if not expected_stored or not expected_RGB or stored != expected_stored:raise ValueError('Exact original image-file and separately named native RGB448 hashes required')
            with Image.open(io.BytesIO(raw)) as opened:image = opened.convert('RGB')
            if name == 'words':
                values,required = native_bytes(row['response']);packed = row['native'];self.guard(self.word_tokens_path)
                seq = self.word_tokens[packed['offset']:packed['offset']+packed['length']]
                if (row['split'] != 'train' or row['native_byte_targets'] != values or row['minimum_CTC_steps'] != required
                        or len(seq) != int(entry['context_tokens']) or packed['target_start'] != int(entry['target_start'])):
                    raise ValueError('Every original whole word/native byte/alignment target required')
                ids = seq.astype(int).tolist();labels = [-100]*int(entry['target_start'])+ids[int(entry['target_start']):]
                word,array = image_arrays(image)
                if (array_digest(word) != row['word_input_pixels_sha256'] or array_digest(array) != row['native_input_pixels_sha256']
                        or array_digest(word) != metadata['word_pixel_float_CHW_metadata_sha256']
                        or array_digest(array) != metadata['native_word_pixel_float_CHW_metadata_sha256']
                        or list(image.size) != row['source_dimensions'] or hashlib.sha256(image.tobytes()).hexdigest() != row['source_RGB_pixels_sha256']):
                    raise ValueError('Whole original word/source/native float pixel protocols must independently match')
                rgb = np.rint((array.transpose(1,2,0)+1)*127.5).clip(0,255).astype(np.uint8)
                word_pixels.append(torch.from_numpy(word));texts.append(row['response']);byte_targets.append(values)
            else:
                encoded = {**row,'image':str(path),'_replay_input_sha256':entry['input_ids_sha256'].tobytes().hex(),
                    '_replay_context_tokens':int(entry['context_tokens']),'_replay_target_tokens':int(entry['response_EOS_targets'])}
                ids,labels = self.encoder.sequence(encoded,'response')
                rgb = np.asarray(ImageOps.pad(image,(448,448),method=Image.Resampling.BICUBIC,color=(128,128,128)),dtype=np.uint8)
                array = np.ascontiguousarray((rgb.astype(np.float32)/127.5-1).transpose(2,0,1))
            if (hashlib.sha256(rgb.tobytes()).hexdigest() != expected_RGB or state(path) != before
                    or hashlib.sha256(np.asarray(ids,dtype='<u2').tobytes()).digest() != entry['input_ids_sha256'].tobytes()
                    or len(ids) != int(entry['context_tokens']) or ids[0] != 1 or ids[-1] != 2
                    or any(not 4 <= t < self.config.vocab_size for t in ids[1:-1])
                    or labels != [-100]*int(entry['target_start'])+ids[int(entry['target_start']):]
                    or int(entry['response_EOS_targets']) != len(ids)-int(entry['target_start'])):
                raise ValueError('Every actual image/native input, complete response and genuine EOS mask must match the corrected index')
            native_ids.append(ids);native_labels.append(labels);native_pixels.append(torch.from_numpy(array))
            if name == 'images:additional_photo_captions':
                caption = [1]+self.tokenizer.encode(row['response'],add_special_tokens=False).ids+[2]
                if any(t < 4 for t in caption[1:-1]):raise ValueError('Matching captions retain only their complete original source content')
                matching_ids.append(caption);texts.append(row['response']);keys.append(expected_RGB)
            observations.append(dict(family=name,original_local_position=position,source_id=metadata['source_id'],correction_index=value['correction_index'],
                input_ids_sha256=entry['input_ids_sha256'].tobytes().hex(),complete_context_tokens=len(ids),response_EOS_targets=int(entry['response_EOS_targets']),
                original_image_sha256=stored,native_RGB448_sha256=expected_RGB))
        maximum = max(map(len,native_ids));ids = torch.zeros(len(positions),maximum,dtype=torch.long);labels = torch.full_like(ids,-100);mask = torch.zeros_like(ids,dtype=torch.bool)
        for i,(seq,target) in enumerate(zip(native_ids,native_labels)):
            ids[i,:len(seq)] = torch.tensor(seq);labels[i,:len(seq)] = torch.tensor(target);mask[i,:len(seq)] = True
        native = dict(input_ids=ids,labels=labels,attention_mask=mask,pixel_values=torch.stack(native_pixels))
        word = dict(word_pixels=torch.stack(word_pixels),texts=texts,native_byte_targets=byte_targets,whole_UTF8_bytes=sum(map(len,byte_targets))) if name == 'words' else None
        matching = None
        if matching_ids:
            width = max(map(len,matching_ids));caption_ids = torch.zeros(len(positions),width,dtype=torch.long);caption_mask = torch.zeros_like(caption_ids,dtype=torch.bool)
            for i,seq in enumerate(matching_ids):caption_ids[i,:len(seq)] = torch.tensor(seq);caption_mask[i,:len(seq)] = True
            matching = dict(text_ids=caption_ids,text_mask=caption_mask,pixel_keys=keys,captions=texts)
        if destination.type != 'cpu':
            native = {k:v.to(destination) for k,v in native.items()}
            if word is not None:word = {**word,'word_pixels':word['word_pixels'].to(destination)}
            if matching is not None:matching = {**matching,'text_ids':matching['text_ids'].to(destination),'text_mask':matching['text_mask'].to(destination)}
        if (int(native['attention_mask'].sum()) != sum(self.lengths[name][p] for p in positions)
                or int((native['labels'][:,1:] != -100).sum()) != sum(self.targets[name][p] for p in positions)):
            raise ValueError('Every full original/current context and response/EOS target must reach batch delivery')
        return dict(native=native,word=word,matching=matching,source_observations=observations,
            source_role='train',purpose=self.purpose,training_admitted=self.training_admitted,
            complete_context_tokens=int(mask.sum()),response_EOS_targets=int((labels[:,1:] != -100).sum()))
