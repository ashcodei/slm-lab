"""Whole original word crops for the shared vision/byte and native LM heads.

The native byte alphabet is0..255 with CTC blank256. It differs from the old
standalone reader's byte+1 alphabet. Pixel geometry is fixed independently of
transcript length. A word crop is not a full-photo text detector.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from .corpus import file_hash

FORMAT = 'own-shared-vision-whole-byte-word-crops-v1'
WORD_HEIGHT, WORD_WIDTH, NATIVE_SIZE, CTC_FRAMES = 64, 448, 448, 140


def native_bytes(text):
    raw = list(text.encode('utf-8'))
    required = len(raw)+sum(a == b for a,b in zip(raw, raw[1:]))
    if not raw or len(raw) > 64 or required > CTC_FRAMES:
        raise ValueError('Complete original UTF-8 transcript including repeated-byte CTC alignment must fit')
    return raw, required


def image_arrays(image):
    image = image.convert('RGB')
    word = np.asarray(image.resize((WORD_WIDTH, WORD_HEIGHT), Image.Resampling.BICUBIC), dtype=np.float32)/127.5-1
    native = np.asarray(ImageOps.pad(image, (NATIVE_SIZE, NATIVE_SIZE), method=Image.Resampling.BICUBIC,
                                    color=(128, 128, 128)), dtype=np.float32)/127.5-1
    return np.ascontiguousarray(word.transpose(2,0,1)), np.ascontiguousarray(native.transpose(2,0,1))


def array_digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def original_crop(parent, metadata, annotation, family):
    """Independently reconstruct the original fixed-margin real-word crop."""
    if (str(annotation['image_id']) != str(metadata['id']) or len(annotation['bbox']) != 4
            or min(metadata['width'], metadata['height']) <= 0):
        raise ValueError('Exact original word/photo identity and geometry required')
    width, height = metadata['width'], metadata['height']
    if abs(parent.width/parent.height-width/height) > .04*width/height:
        raise ValueError('Original source photograph and annotation aspect ratios differ')
    x,y,w,h = map(float, annotation['bbox'])
    if not all(math.isfinite(v) for v in (x,y,w,h)) or min(w,h) < 4:
        raise ValueError('Original complete nondegenerate human word box required')
    bounds = (max(0, math.floor((x-.05*w)*parent.width/width)), max(0, math.floor((y-.1*h)*parent.height/height)),
              min(parent.width, math.ceil((x+1.05*w)*parent.width/width)), min(parent.height, math.ceil((y+1.1*h)*parent.height/height)))
    if min(bounds[2]-bounds[0], bounds[3]-bounds[1]) < 4:
        raise ValueError('Original human word crop must contain pixels')
    maximum = 448 if family == 'additional_real_word' else 768
    return ImageOps.contain(parent.crop(bounds).convert('RGB'), (maximum, maximum), Image.Resampling.LANCZOS)


class SharedWordData:
    def __init__(self, root, role, purpose):
        if role not in ('train', 'valid') or purpose not in ('audit', 'optimization', 'evaluation', 'capacity'):
            raise ValueError('Explicit whole-word source role and purpose required')
        if (purpose in ('optimization', 'capacity') and role != 'train') or (purpose == 'evaluation' and role != 'valid'):
            raise ValueError('Held original word crops never supply training or capacity gradients')
        self.root = Path(root).resolve(); self.role = role; self.purpose = purpose
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        if (self.manifest['format'] != FORMAT or self.manifest['input_geometry'] != dict(word=[3,64,448], native=[3,448,448])
                or self.manifest['ctc_blank'] != 256 or self.manifest['ctc_frames'] != CTC_FRAMES):
            raise ValueError('Exact shared word pixels, full native bytes and blank256 contract required')
        if purpose in ('optimization', 'capacity'):
            admission = json.loads((self.root/'admission.json').read_text())
            if (admission.get('training_admitted') is not True or admission['manifest_sha256'] != file_hash(self.root/'manifest.json')
                    or admission['independent_readback_sha256'] != file_hash(self.root/'independent-source-readback.json')):
                raise ValueError('Complete source/pixel/token readback and train-only admission required')
        for suffix in ('.jsonl', '.tokens.u16'):
            path = self.root/(role+suffix)
            if file_hash(path) != self.manifest['files_sha256'][path.name]:
                raise ValueError('Every complete original word index/native token byte must match')
        self.rows = [json.loads(line) for line in (self.root/(role+'.jsonl')).open()]
        self.tokens = np.memmap(self.root/(role+'.tokens.u16'), dtype='<u2', mode='r')
        cursor = 0
        for row in self.rows:
            target, required = native_bytes(row['response']); item = row['native']
            if (row['split'] != role or row['native_byte_targets'] != target or row['minimum_CTC_steps'] != required
                    or item['offset'] != cursor or not 1 <= item['target_start'] < item['length'] <= 4096):
                raise ValueError('Complete original transcript, role, byte and native response offsets required')
            tokens = self.tokens[cursor:cursor+item['length']]
            if len(tokens) != item['length'] or int(tokens[0]) != 1 or int(tokens[-1]) != 2:
                raise ValueError('Complete original own BOS/response/EOS sequence required')
            cursor += item['length']
        if cursor != len(self.tokens) or len(self.rows) != self.manifest['splits'][role]['examples']:
            raise ValueError('Every original packed word and native token must be covered')

    def __len__(self):
        return len(self.rows)

    def batch(self, positions, device='cpu'):
        if not positions: raise ValueError('Whole original word batch must be nonempty')
        items = []; word_pixels = []; native_pixels = []; texts = []
        for position in positions:
            if type(position) is not int or not 0 <= position < len(self): raise ValueError('Exact original word position required')
            row = self.rows[position]; path = Path(row['image'])
            if file_hash(path) != row['stored_image_sha256']: raise ValueError('Original word image bytes changed')
            with Image.open(path) as image: word, native = image_arrays(image)
            if (array_digest(word) != row['word_input_pixels_sha256'] or array_digest(native) != row['native_input_pixels_sha256']):
                raise ValueError('Full source-derived fixed-geometry word/native pixels must match')
            item = row['native']; seq = self.tokens[item['offset']:item['offset']+item['length']].astype(np.int64).tolist()
            items.append((seq, item['target_start'])); texts.append(row['response'])
            word_pixels.append(torch.from_numpy(word)); native_pixels.append(torch.from_numpy(native))
        maximum = max(len(seq) for seq,_ in items)
        ids = torch.zeros(len(items), maximum, dtype=torch.long); labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i,(seq,start) in enumerate(items):
            ids[i,:len(seq)] = torch.tensor(seq); labels[i,start:len(seq)] = torch.tensor(seq[start:]); mask[i,:len(seq)] = True
        return dict(input_ids=ids.to(device), labels=labels.to(device), attention_mask=mask.to(device),
                    pixel_values=torch.stack(native_pixels).to(device), word_pixels=torch.stack(word_pixels).to(device), texts=texts)
