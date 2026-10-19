"""Load all previous words plus new native-photo words with bounded frame reuse."""
from collections import OrderedDict, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset, Sampler

from .byte_word_reader import labels, load_rows, pixels
from .corpus import file_hash
from .evaluation_provenance import capture, pixels_sha256
from .more_real_words import ROOT, crop_tensor, word_geometry
from .rendered_text import text_group


def load(data):
    data = Path(data); manifest = json.loads((data / 'manifest.json').read_text())
    provenance = manifest['source_provenance']
    if capture([ROOT / name for name in provenance['files_sha256']]) != provenance: raise ValueError('Prepared native-word implementation/runtime changed')
    previous_data = ROOT / 'data/byte-word-reader-v1'
    if file_hash(previous_data / 'manifest.json') != manifest['previous_reader_manifest_sha256']: raise ValueError('Original word training/validation manifest changed')
    previous, _ = load_rows(previous_data); existing = {(row['group_id'], str(row.get('original_word_id', row['id']))) for row in previous['train']}
    protected = {row['group_id'] for row in previous['valid']}
    protected_words = {text_group(row['response']) for row in previous['valid'] if row['byte_family'] == 'rendered'}
    for path, digest in {**manifest['photo_inventories_sha256'], **manifest['native_source_photos_sha256'],
                         str(ROOT / 'data/image-source-cache-v1/TextOCR_0.1_train.json'): manifest['original_annotation_source']['sha256']}.items():
        if file_hash(path) != digest: raise ValueError('Prepared native source/photo inventory changed')
    path = data / 'additional.train.jsonl.gz'
    if file_hash(path) != manifest['files_sha256'][path.name]: raise ValueError('Compressed additional word rows changed')
    digest = hashlib.sha256(); additional = []; seen = set()
    with gzip.open(path, 'rb') as stream:
        for line in stream:
            digest.update(line); row = json.loads(line); identity = (row['group_id'], row['original_word_id'])
            if identity in existing or identity in seen or row['group_id'] in protected or text_group(row['response']) in protected_words:
                raise ValueError('Additional native word repeats old annotation or enters protected validation')
            labels(row['response'])
            text, bounds = word_geometry({'utf8_string': row['response'], 'bbox': row['original_bbox']}, row['original_annotation_dimensions'], row['native_dimensions'])
            if text != row['response'] or bounds != row['native_crop_bounds'] or row['source_image_sha256'] != manifest['native_source_photos_sha256'][row['image']]:
                raise ValueError('Additional word geometry/source binding differs')
            if row['split'] != 'train' or row['byte_family'] != 'additional_native_real_word': raise ValueError('Wrong native-word training family/split')
            seen.add(identity); additional.append({**row, 'refinement_backend': 'native'})
    if digest.hexdigest() != manifest['decoded_JSONL_sha256'] or len(additional) != manifest['additional_statistics']['new_original_words']:
        raise ValueError('Decoded additional word bytes/count differ')
    train = [{**row, 'refinement_backend': 'previous'} for row in previous['train']] + additional
    if len({row['id'] for row in train}) != len(train): raise ValueError('Duplicate composite word identity')
    return train, previous['valid'], manifest


class NativeFrames:
    """Each decoded frame is pinned to read-and-hashed bytes, at most192MiB."""
    def __init__(self, maximum_bytes=192 * 1024**2):
        self.maximum_bytes = maximum_bytes; self.frames = OrderedDict(); self.bytes = 0
    def image(self, row):
        key = (row['image'], row['source_image_sha256'])
        if key in self.frames:
            self.frames.move_to_end(key); return self.frames[key]
        path = Path(row['image']).resolve()
        if not path.is_relative_to(ROOT): raise ValueError('Native frame leaves project')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['source_image_sha256']: raise ValueError('Actual native frame bytes changed before admission')
        with Image.open(io.BytesIO(raw)) as opened:
            image = opened.convert('RGB')
        if list(image.size) != row['native_dimensions']: raise ValueError('Actual native frame dimensions changed')
        size = image.width * image.height * 3
        while self.frames and self.bytes + size > self.maximum_bytes:
            _, old = self.frames.popitem(last=False); self.bytes -= old.width * old.height * 3
        if size <= self.maximum_bytes:
            self.frames[key] = image; self.bytes += size
        return image
    def pixels(self, row):
        value = crop_tensor(self.image(row), row['native_crop_bounds'])
        if pixels_sha256(value) != row['reader_input_pixels_sha256']: raise ValueError('Cached native crop pixels differ from prepared input')
        return value


class Words(Dataset):
    def __init__(self, rows): self.rows = rows; self.native = NativeFrames()
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        value = self.native.pixels(row) if row['refinement_backend'] == 'native' else pixels(row)
        return value, row['response']


class GroupChunks(Sampler):
    """Every row exactly once per pass; shuffle photo/text chunks of at most8."""
    def __init__(self, rows, seed=42, chunk_size=8):
        self.rows = rows; self.seed = seed; self.chunk_size = chunk_size; self.epoch = 0
        if not isinstance(chunk_size, int) or chunk_size <= 0: raise ValueError('Positive chunk size required')
    def __len__(self): return len(self.rows)
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch); groups = defaultdict(list)
        for index, row in enumerate(self.rows): groups[row['group_id']].append(index)
        chunks = []
        for group in sorted(groups):
            indices = groups[group]; rng.shuffle(indices)
            chunks.extend(indices[start:start + self.chunk_size] for start in range(0, len(indices), self.chunk_size))
        rng.shuffle(chunks)
        return iter(index for chunk in chunks for index in chunk)
