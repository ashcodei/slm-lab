"""Reconstruct one exact manifest-ordered raw token record from its bounds.

The record can cross any number of shards. Boundaries and full length are
checked before a caller decodes or normalizes text; no substring scanner is
used here and missing/oversized records fail closed.
"""
import hashlib
from pathlib import Path

import numpy as np


def reconstruct(folder, shards, witness, *, maximum_tokens=2_000_000):
    if type(maximum_tokens) is not int or maximum_tokens < 1 or not shards: raise ValueError('Explicit positive whole-record bound and complete manifest required')
    names = [r['path'] for r in shards]
    if len(set(names)) != len(names): raise ValueError('Distinct ordered source shards required')
    start, end = witness['start'], witness['end']
    if start['shard'] not in names or type(start['token_offset']) is not int: raise ValueError('Original start shard/offset required')
    first = names.index(start['shard']); terminal = end.get('end_of_manifest_ordered_train_stream') is True
    if terminal:
        last = len(shards) - 1; stop = shards[-1]['tokens']
    else:
        if end.get('shard') not in names or type(end.get('token_offset')) is not int: raise ValueError('Original closing shard/offset required')
        last = names.index(end['shard']); stop = end['token_offset']
    if first > last or type(witness['token_count']) is not int or not 1 <= witness['token_count'] <= maximum_tokens:
        raise ValueError('Ordered nonempty whole source record within its resource limit required')
    def mapped(index):
        item = shards[index]; path = Path(folder) / item['path']
        if type(item['tokens']) is not int or item['tokens'] < 1 or path.stat().st_size != 2 * item['tokens']:
            raise ValueError('Every full original shard byte must match its declared native uint16 token count')
        return np.memmap(path, dtype='<u2', mode='r')
    chunks = []; count = 0
    for index in range(first, last + 1):
        values = mapped(index); left = start['token_offset'] if index == first else 0; right = stop if index == last else len(values)
        if not 0 <= left <= right <= len(values): raise ValueError('Exact in-range whole-record slice required')
        if index == first:
            if left:
                previous = int(values[left - 1])
                if previous not in [0, 1, 2]: raise ValueError('Witness begins inside a source record')
            elif first:
                previous_shard = mapped(first - 1)
                if int(previous_shard[-1]) not in [0, 1, 2]: raise ValueError('Witness dropped the preceding shard prefix of its record')
        if index == last and not terminal:
            if right >= len(values) or int(values[right]) not in [0, 1, 2]: raise ValueError('Witness closes inside a source record')
        chunk = np.asarray(values[left:right]).copy(); count += len(chunk)
        if count > maximum_tokens or (len(chunk) and (chunk.min() < 3 or chunk.max() >= 32768)):
            raise ValueError('Whole source record may contain no unrelated record-boundary or invalid vocabulary token')
        chunks.append(chunk)
    if count != witness['token_count']: raise ValueError('Every declared original record token must reconstruct')
    result = np.concatenate(chunks)
    return result.astype(np.int64).tolist(), hashlib.sha256(result.tobytes()).hexdigest()
