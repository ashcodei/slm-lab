"""Own compact center/size word detector on full photos, initialized from scratch.

The center-point idea follows Objects as Points (arXiv:1904.07850), but this
small CNN, three-level assignment, loss and protocol are our own implementation.
This is a standalone research component, not a pretrained OCR system.
"""
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F

from .corpus import file_hash
from .text_localization_metrics import box, iou

ROOT = Path(__file__).resolve().parents[2]
SIZE = 768
STRIDES = (4, 8, 16)
CONFIDENCE = .3
NMS_IOU = .5
MAX_PREDICTIONS = 1000


def assignments(targets, size=SIZE):
    """Assign every legible box once; fall back to another level on a collision.

    The actual prepared corpus has no unrepresented boxes. Overflow is an
    error, never a silently dropped training or evaluation word.
    """
    if size % 16 or size <= 0: raise ValueError('Input size must be positive and divisible by16')
    used = set(); result = []; counts = Counter()
    words = [(index, target, box(target['normalized_padded_box'])) for index, target in enumerate(targets) if target['legible']]
    words.sort(key=lambda item: ((item[2][2] - item[2][0]) * (item[2][3] - item[2][1]), str(item[1]['original_annotation_id'])))
    for index, target, rectangle in words:
        x0, y0, x1, y1 = rectangle
        cx, cy = (x0 + x1) * size / 2, (y0 + y1) * size / 2
        width, height = (x1 - x0) * size, (y1 - y0) * size
        preferred = 4 if height <= 24 else 8 if height <= 64 else 16
        order = sorted(STRIDES, key=lambda stride: (abs(math.log2(stride / preferred)), stride))
        for stride in order:
            x, y = int(cx / stride), int(cy / stride)
            key = (stride, y, x)
            if key in used: continue
            used.add(key); counts['level_' + str(stride)] += 1
            if stride != preferred: counts['fallback_level'] += 1
            result.append({'target_index': index, 'stride': stride, 'x': x, 'y': y,
                           'offset': [cx / stride - x, cy / stride - y],
                           'log_size': [math.log(width / stride), math.log(height / stride)],
                           'radius': min(3, int(min(width, height) / stride / 6))})
            break
        else: raise ValueError('Unrepresented original word after all center-grid collision fallbacks: ' + str(target['original_annotation_id']))
    counts['assigned_legible_words'] = len(result)
    return result, dict(counts)


def encode_targets(targets, size=SIZE):
    encoded = {}
    for stride in STRIDES:
        cells = size // stride
        encoded[stride] = {'heat': torch.zeros(1, cells, cells), 'valid': torch.ones(1, cells, cells, dtype=torch.bool),
                           'positive': torch.zeros(1, cells, cells, dtype=torch.bool), 'geometry': torch.zeros(4, cells, cells)}
        # Original unreadable regions do not teach a negative text/background
        # decision. Legible positive centers later override the ignore mask.
        for target in targets:
            rectangle = box(target['normalized_padded_box'])
            if target['legible']: continue
            x0, y0, x1, y1 = rectangle
            left, top = max(0, math.ceil(x0 * cells - .5)), max(0, math.ceil(y0 * cells - .5))
            right, bottom = min(cells, math.ceil(x1 * cells - .5)), min(cells, math.ceil(y1 * cells - .5))
            encoded[stride]['valid'][:, top:bottom, left:right] = False
    assigned, _ = assignments(targets, size)
    for item in assigned:
        level = encoded[item['stride']]; x, y, radius = item['x'], item['y'], item['radius']
        cells = level['heat'].shape[-1]
        left, right = max(0, x - radius), min(cells, x + radius + 1)
        top, bottom = max(0, y - radius), min(cells, y + radius + 1)
        xx = torch.arange(left, right).float() - x; yy = torch.arange(top, bottom).float() - y
        gaussian = torch.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / (2 * max(.5, (2 * radius + 1) / 6) ** 2))
        patch = level['heat'][0, top:bottom, left:right]
        patch.copy_(torch.maximum(patch, gaussian))
        level['positive'][0, y, x] = True; level['valid'][0, y, x] = True
        level['geometry'][:, y, x] = torch.tensor(item['offset'] + item['log_size'])
    return encoded


class Conv(nn.Sequential):
    def __init__(self, incoming, outgoing, stride=1):
        super().__init__(nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(8, outgoing), nn.GELU())


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__(); self.body = nn.Sequential(Conv(channels, channels), nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.GroupNorm(8, channels))
    def forward(self, value): return F.gelu(value + self.body(value))


class WordLocalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Conv(3, 32, 2)
        self.stages = nn.ModuleList([nn.Sequential(Conv(incoming, outgoing, 2), Residual(outgoing))
                                    for incoming, outgoing in [(32, 48), (48, 80), (80, 128), (128, 192)]])
        self.lateral = nn.ModuleList([nn.Conv2d(channels, 48, 1) for channels in [48, 80, 128, 192]])
        self.smooth = nn.ModuleList([Conv(48, 48) for _ in STRIDES])
        self.heads = nn.ModuleList([nn.Sequential(Conv(48, 48), nn.Conv2d(48, 5, 1)) for _ in STRIDES])
        for head in self.heads:
            nn.init.zeros_(head[-1].bias)
            with torch.no_grad(): head[-1].bias[0] = math.log(.01 / .99)

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[-2] != images.shape[-1] or images.shape[-1] % 32:
            raise ValueError('Detector expects square RGB input divisible by32')
        features = []; value = self.stem(images)
        for stage in self.stages:
            value = stage(value); features.append(value)
        pyramids = [None] * 4
        for index in range(3, -1, -1):
            value = self.lateral[index](features[index])
            if index < 3: value = value + F.interpolate(pyramids[index + 1], size=value.shape[-2:], mode='nearest')
            pyramids[index] = value
        return {stride: head(smooth(pyramids[index])) for index, (stride, smooth, head) in enumerate(zip(STRIDES, self.smooth, self.heads))}


def loss(outputs, targets):
    heat_loss = outputs[4].float().sum() * 0; geometry_loss = heat_loss
    count = sum(level['positive'].sum() for level in targets.values()).clamp_min(1)
    for stride in STRIDES:
        output = outputs[stride].float(); level = targets[stride]
        heat = level['heat']; positive = level['positive']; valid = level['valid']
        if output.shape[0] != heat.shape[0] or output.shape[-2:] != heat.shape[-2:]: raise ValueError('Detector target/output geometry differs')
        probability = output[:, :1].sigmoid().clamp(1e-6, 1 - 1e-6)
        positive_loss = -(1 - probability).square() * probability.log()
        negative_loss = -probability.square() * torch.log1p(-probability) * (1 - heat).pow(4)
        heat_loss = heat_loss + (positive_loss * positive + negative_loss * (~positive & valid)).sum()
        # Offset is in a cell; sizes are log(width/stride), log(height/stride).
        geometry = torch.cat([output[:, 1:3].sigmoid(), output[:, 3:5]], dim=1)
        geometry_loss = geometry_loss + (F.smooth_l1_loss(geometry, level['geometry'], reduction='none') * positive).sum()
    return (heat_loss + geometry_loss) / count, {'center_focal': heat_loss.detach() / count, 'box_geometry': geometry_loss.detach() / count}


def decode(outputs, size=SIZE, confidence=CONFIDENCE, nms_iou=NMS_IOU, maximum=MAX_PREDICTIONS):
    """Decode pixels-only predictions, with fixed thresholds and no target input."""
    if not 0 < confidence < 1 or not 0 < nms_iou <= 1 or maximum <= 0: raise ValueError('Invalid fixed detector decode protocol')
    candidates = []
    for stride in STRIDES:
        output = outputs[stride]
        if output.ndim != 3 or output.shape != (5, size // stride, size // stride): raise ValueError('Unexpected single-photo detector output dimensions')
        output = output.detach().float().cpu()
        if not torch.isfinite(output).all(): raise ValueError('Nonfinite learned word detections')
        probability = output[0].sigmoid()
        peaks = (probability == F.max_pool2d(probability[None, None], 3, stride=1, padding=1)[0, 0]) & (probability >= confidence)
        locations = peaks.nonzero()
        # A bounded per-level shortlist is based only on predicted scores.
        if len(locations) > maximum:
            values = probability[locations[:, 0], locations[:, 1]]
            locations = locations[torch.argsort(values, descending=True, stable=True)[:maximum]]
        for y, x in locations.cpu().tolist():
            offset = output[1:3, y, x].sigmoid().cpu().tolist()
            width, height = output[3:5, y, x].clamp(-12, 12).exp().mul(stride / size).cpu().tolist()
            cx, cy = (x + offset[0]) * stride / size, (y + offset[1]) * stride / size
            rectangle = [max(0., cx - width / 2), max(0., cy - height / 2), min(1., cx + width / 2), min(1., cy + height / 2)]
            if rectangle[2] <= rectangle[0] or rectangle[3] <= rectangle[1]: continue
            candidates.append({'box': rectangle, 'score': float(probability[y, x]), 'stride': stride})
    candidates.sort(key=lambda item: (-item['score'], item['stride'], item['box']))
    selected = []
    for candidate in candidates:
        if any(iou(candidate['box'], previous['box']) > nms_iou for previous in selected): continue
        selected.append(candidate)
        if len(selected) == maximum: break
    return selected


def load_rows(data):
    data = Path(data); manifest = json.loads((data / 'manifest.json').read_text()); rows = {}
    if manifest['input_size'] != SIZE: raise ValueError('Localization data/input resolution differs')
    preparation = ROOT / 'src/slm_lab/text_localization_data.py'
    if file_hash(preparation) != manifest['implementation_sha256']: raise ValueError('Prepared localization implementation changed')
    for path, digest in manifest['source_files_sha256'].items():
        if file_hash(path) != digest: raise ValueError('Original localization annotation/photo inventory changed')
    for split in ['train', 'valid']:
        path = data / (split + '.jsonl.gz')
        if file_hash(path) != manifest['files_sha256'][path.name]: raise ValueError('Compressed localization targets changed')
        decoded = hashlib.sha256(); records = []; seen = set(); groups = set(); statistics = Counter()
        with gzip.open(path, 'rb') as stream:
            for line in stream:
                decoded.update(line); row = json.loads(line)
                if row['split'] != split or row['id'] in seen or row['group_id'] in groups or row['transcript_supplied_to_detector'] is not False:
                    raise ValueError('Repeated/wrong-split or transcript-bearing full-photo detector input')
                seen.add(row['id']); groups.add(row['group_id'])
                if len({target['original_annotation_id'] for target in row['targets']}) != len(row['targets']): raise ValueError('Repeated original word annotation')
                for target in row['targets']: box(target['normalized_padded_box'])
                assigned, _ = assignments(row['targets'])
                statistics['full_photos'] += 1; statistics['word_regions'] += len(row['targets'])
                statistics['legible_regions'] += len(assigned); statistics['unreadable_ignore_regions'] += len(row['targets']) - len(assigned)
                records.append(row)
        if decoded.hexdigest() != manifest['decoded_files_sha256'][split + '.jsonl']: raise ValueError('Decoded localization JSONL bytes changed')
        if dict(statistics) != manifest['splits'][split]: raise ValueError('Actual localization word/photo counts differ from prepared manifest')
        rows[split] = records
    if {row['group_id'] for row in rows['train']} & {row['group_id'] for row in rows['valid']}: raise ValueError('Detector image parents overlap splits')
    return rows, manifest


def pixels(row):
    path = Path(row['image']).resolve()
    if not path.is_relative_to(ROOT): raise ValueError('Detector full-photo input path leaves project')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['stored_image_sha256']: raise ValueError('Actual detector full-photo bytes changed')
    with Image.open(io.BytesIO(raw)) as original:
        if list(original.size) != row['stored_dimensions']: raise ValueError('Detector original full-photo dimensions changed')
        image = ImageOps.pad(original.convert('RGB'), (SIZE, SIZE), method=Image.Resampling.BICUBIC, color=(128, 128, 128))
        image_bytes = image.tobytes()
    if hashlib.sha256(image_bytes).hexdigest() != row['model_pixel_sha256_768']: raise ValueError('Actual full-photo detector pixels differ from prepared geometry')
    return torch.from_numpy(np.frombuffer(image_bytes, dtype=np.uint8).copy().reshape(SIZE, SIZE, 3)).permute(2, 0, 1).float().div(127.5).sub(1)
