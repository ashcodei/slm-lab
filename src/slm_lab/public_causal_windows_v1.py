"""Lossless target coverage over bounded causal windows of whole native documents.

Original documents remain intact in the source corpus. Long documents have a
4096-token context limit; windows overlap by one context token, so each text/EOS
target occurs exactly once. Within-role document concatenation permits causal
attention across document boundaries. BOS is always masked, including internal
BOS tokens. This module grants no source admission itself.
"""
from bisect import bisect_left, bisect_right
import json
from pathlib import Path

import numpy as np

from .corpus import file_hash

WINDOW_DTYPE = np.dtype([('offset', '<u8'), ('length', '<u2'), ('targets', '<u2')])


def windows(total_tokens, document_starts, context=4096):
    starts = [int(n) for n in document_starts]
    if (type(total_tokens) is not int or total_tokens < 3 or type(context) is not int
            or not 3 <= context <= 4096 or not starts or starts[0] != 0
            or any(b-a < 3 for a, b in zip(starts, starts[1:]))
            or total_tokens-starts[-1] < 3):
        raise ValueError('Complete ordered native document spans and bounded causal context required')
    result = []
    for offset in range(0, total_tokens-1, context-1):
        length = min(context, total_tokens-offset)
        internal_bos = bisect_left(starts, offset+length)-bisect_right(starts, offset)
        targets = length-1-internal_bos
        if targets < 1:
            raise ValueError('Every emitted causal window needs a genuine text/EOS target')
        result.append((offset, length, targets))
    result = np.asarray(result, dtype=WINDOW_DTYPE)
    if int(result['targets'].astype(np.uint64).sum()) != total_tokens-len(starts):
        raise ValueError('Every whole original text and EOS target must occur exactly once')
    return result


class CausalPublicPack:
    """Read independently admitted native values; never train on the held role."""
    def __init__(self, folder, role, purpose='audit'):
        self.folder = Path(folder).resolve()
        if role not in {'train', 'valid'} or purpose not in {'audit', 'optimization'}:
            raise ValueError('Explicit sealed source role and purpose required')
        if purpose == 'optimization' and role != 'train':
            raise ValueError('Held public documents cannot be used for optimization')
        manifest_path = self.folder/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('format') != 'own-public-causal-windows-v1':
            raise ValueError('Independently rebuilt complete public causal windows required')
        if purpose == 'optimization':
            receipt = json.loads((self.folder/'admission.json').read_text())
            if (receipt.get('admitted_to_training') is not True
                    or receipt.get('manifest_sha256') != file_hash(manifest_path)
                    or receipt.get('full_source_reader_complete') is not True):
                raise ValueError('Actual complete independent source admission required')
            reader_path = Path(receipt['independent_readback'])
            reader = json.loads(reader_path.read_text())
            if (file_hash(reader_path) != receipt['independent_readback_sha256']
                    or reader.get('complete') is not True
                    or reader.get('all_original_source_rows_and_native_targets_rebuilt') is not True):
                raise ValueError('Exact complete independent original-source reader required')
        for name, digest in manifest['source_files_sha256'].items():
            if file_hash(Path(name)) != digest:
                raise ValueError('Admitted original native/index/source bytes changed')
        self.manifest = manifest
        self.role = role
        self.index = np.load(self.folder/(role+'.windows.npy'), allow_pickle=False, mmap_mode='r')
        self.starts = np.load(self.folder/(role+'.document-starts.npy'), allow_pickle=False, mmap_mode='r')
        path = Path(manifest['native_files'][role])
        if self.index.dtype != WINDOW_DTYPE or self.index.ndim != 1 or self.starts.dtype != np.dtype('<u8'):
            raise ValueError('Exact independent window index and original BOS positions required')
        if path.stat().st_size % 2:
            raise ValueError('Every native little-endian u16 token must remain whole')
        self.ids = np.memmap(path, dtype='<u2', mode='r')
        expected = windows(len(self.ids), self.starts, manifest['context_limit'])
        if (not np.array_equal(self.index, expected)
                or int(self.index['targets'].astype(np.uint64).sum()) != manifest['roles'][role]['target_tokens']):
            raise ValueError('Exact complete-source causal target partition required')

    def __len__(self):
        return len(self.index)

    def sequence(self, position):
        if type(position) is not int or not 0 <= position < len(self):
            raise IndexError('Original sealed-role causal window position required')
        row = self.index[position]
        ids = self.ids[int(row['offset']):int(row['offset'])+int(row['length'])].astype(np.int64)
        if np.any((ids < 1) | (ids >= 32768) | (ids == 3)):
            raise ValueError('Original native framing and own vocabulary required')
        labels = ids.copy()
        labels[0] = -100
        labels[ids == 1] = -100
        if int((labels[1:] != -100).sum()) != int(row['targets']):
            raise ValueError('Every original BOS is masked and every text/EOS target retained')
        return ids.tolist(), labels.tolist()

    def batch(self, positions, device='cpu'):
        if not positions:
            return None
        import torch
        rows = [self.sequence(p) for p in positions]
        maximum = max(len(ids) for ids, _ in rows)
        ids = torch.zeros(len(rows), maximum, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros(len(rows), maximum, dtype=torch.bool)
        for p, (tokens, targets) in enumerate(rows):
            n = len(tokens)
            ids[p, :n] = torch.tensor(tokens)
            labels[p, :n] = torch.tensor(targets)
            attention[p, :n] = True
        result = dict(input_ids=ids, labels=labels, attention_mask=attention)
        if torch.device(device).type != 'cpu':
            result = {k:v.pin_memory().to(device, non_blocking=True) for k,v in result.items()}
        return result
