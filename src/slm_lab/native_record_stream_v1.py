"""Reconstruct complete token records across shard boundaries before decoding.

Own BOS/EOS/PAD IDs are hard boundaries. Unfinished records are retained across
shards, so neither UTF-8 decoding nor Unicode normalization happens at an
arbitrary byte/token chunk boundary. A resource limit fails closed rather than
silently dropping a long record.
"""
import numpy as np


class NativeRecordStream:
    def __init__(self, maximum_record_tokens=2_000_000):
        if type(maximum_record_tokens) is not int or maximum_record_tokens < 1:
            raise ValueError('Positive explicit complete-record resource bound required')
        self.maximum_record_tokens = maximum_record_tokens
        self.pending = []; self.pending_tokens = 0; self.start = None
        self.input_tokens = 0; self.boundary_tokens = 0; self.output_tokens = 0
        self.records = 0

    def _append(self, values, source, offset):
        if not len(values): return
        if self.pending_tokens + len(values) > self.maximum_record_tokens:
            raise ValueError('Complete original token record exceeds audit resource bound; no clean result may be issued')
        if self.start is None: self.start = dict(source, token_offset=offset)
        self.pending.append(np.asarray(values, dtype='<u2').copy())
        self.pending_tokens += len(values)

    def _emit(self, end):
        if not self.pending_tokens: return None
        values = np.concatenate(self.pending).astype(np.int64).tolist()
        row = dict(record=self.records, start=self.start, end=end, token_count=len(values))
        self.records += 1; self.output_tokens += len(values)
        self.pending = []; self.pending_tokens = 0; self.start = None
        return values, row

    def feed(self, values, source):
        values = np.asarray(values)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or (len(values) and (values.min() < 0 or values.max() >= 32768)):
            raise ValueError('One-dimensional own-vocabulary native token stream required')
        self.input_tokens += len(values)
        boundaries = np.flatnonzero((values == 0) | (values == 1) | (values == 2))
        self.boundary_tokens += len(boundaries)
        previous = 0
        for boundary in boundaries:
            boundary = int(boundary)
            self._append(values[previous:boundary], source, previous)
            row = self._emit(dict(source, token_offset=boundary))
            if row is not None: yield row
            previous = boundary + 1
        self._append(values[previous:], source, previous)

    def finish(self):
        row = self._emit(dict(end_of_manifest_ordered_train_stream=True))
        if row is not None: yield row
        if self.pending_tokens or self.input_tokens != self.boundary_tokens + self.output_tokens:
            raise ValueError('Every original input token must be reconstructed or an explicit boundary')
