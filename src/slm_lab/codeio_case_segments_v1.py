"""Atomic, source-bound segments for whole CodeI/O execution observations.

Closed segments form a SHA256 chain. Recovery checks every original frame;
an interrupted temporary file is preserved but never counted as complete.
This is data qualification storage, not source admission or model training.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

FORMAT = 'whole-CodeIO-source-observation-segments-v1'
ZERO = '0'*64
STATUSES = {
    'original_complete_schema_qualification',
    'reference_runtime_or_resource_population_qualification',
    'complete_local_context_cached_value_mismatch',
    'no_OP_runtime_or_trivial_population_qualification',
    'all_original_cached_values_match_two_local_seed_contexts',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_record(row, position):
    if (not isinstance(row, dict) or row.get('original_position') != position
            or not isinstance(row.get('original_row_sha256'), str) or len(row['original_row_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in row['original_row_sha256'])
            or row.get('status') not in STATUSES
            or row.get('source_only_no_training_admission') is not True
            or row.get('independent_semantic_correctness_certified') is not False
            or row.get('RL_reward_admitted') is not False
            or row.get('fixed_role', {}).get('source_role') not in ['train', 'reserved']):
        raise ValueError('Each original ordered fixed-role source needs a complete qualification disposition')
    if ((row['fixed_role']['source_role'] == 'reserved' or row.get('whole_component_quarantined') is True)
            and row.get('source_optimization_forbidden') is not True):
        raise ValueError('Original reserved roles must remain optimization-forbidden')
    positive = row['status'] == 'all_original_cached_values_match_two_local_seed_contexts'
    if row.get('all_original_cases_and_seeds_match') is not positive:
        raise ValueError('A partial or qualified source cannot become a cached-case positive')
    if positive:
        observations = row.get('reference_observations', [])
        control = row.get('no_OP_control', {})
        total = row.get('total_original_cases')
        if (type(total) is not int or total < 1 or len(observations) != 2
                or [o.get('seed') for o in observations] != [0, 1]
                or any(o.get('total_original_cases') != total or len(o.get('records', [])) != total
                    or o.get('all_original_cases_observed') is not True
                    or o.get('all_original_cached_values_match') is not True
                    or o.get('case_passes') != [True]*total
                    or o.get('expected_outputs_not_sent_to_children') is not True
                    or o.get('original_program_sha256') != row.get('original_program_sha256')
                    or o.get('original_cases_sha256') != row.get('original_cases_sha256')
                    or [r.get('case_position') for r in o.get('records', [])] != list(range(total))
                    or o.get('training_or_RL_admitted') is not False for o in observations)
                or control.get('all_original_cases_observed') is not True
                or control.get('total_original_cases') != total or len(control.get('records', [])) != total
                or control.get('all_original_cached_values_match') is not False):
            raise ValueError('A positive requires both complete seed populations and the complete no-op negative control')


class Segments:
    def __init__(self, folder, *, bindings, total, width=16, minimum_free_bytes=8*1024**3,
                 maximum_total_bytes=512*1024**2, maximum_segment_bytes=64*1024**2):
        if (not isinstance(bindings, dict) or not bindings or type(total) is not int or total < 1
                or type(width) is not int or not 1 <= width <= 64
                or any(type(v) is not int or v < 0 for v in [minimum_free_bytes, maximum_total_bytes, maximum_segment_bytes])
                or maximum_segment_bytes < 1 or maximum_total_bytes < maximum_segment_bytes):
            raise ValueError('Explicit complete source/binding and bounded storage budgets required')
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest = dict(format=FORMAT, bindings=bindings, total=total, width=width,
            maximum_total_bytes=maximum_total_bytes, maximum_segment_bytes=maximum_segment_bytes,
            minimum_free_bytes=minimum_free_bytes, whole_sources_only=True, training_admitted=False)
        path = self.folder/'manifest.json'
        if path.exists():
            if json.loads(path.read_text()) != self.manifest:
                raise ValueError('Recovery must retain the exact original source, population and storage contract')
        else:
            if any(self.folder.iterdir()):
                raise ValueError('An unbound existing output cannot be adopted')
            self._atomic_json(path, self.manifest)
        self.manifest_sha256 = digest(self.manifest)
        self.next_position, self.tip, self.counts, self.committed_bytes = 0, ZERO, {}, 0
        self.segment_paths = []
        self.prefixes = {0: dict(tip=ZERO, counts={}, allocated=0)}
        for path in sorted(self.folder.glob('segment-*.json.gz')):
            item, file_sha = self._inspect(path)
            self._adopt(path, item, file_sha)
        # A rename can be durable before the mutable cursor is published.
        # Reconstructing the complete chain safely adopts that closed segment.
        cursor = self.folder/'cursor.json'
        if cursor.exists():
            old = json.loads(cursor.read_text())
            if (old.get('manifest_sha256') != self.manifest_sha256
                    or type(old.get('next_original_position')) is not int
                    or not 0 <= old['next_original_position'] <= self.next_position):
                raise ValueError('An existing cursor cannot exceed or contradict its closed segment chain')
            prefix = self.prefixes.get(old['next_original_position'])
            if (prefix is None or old.get('last_closed_segment_sha256') != prefix['tip']
                    or old.get('complete_sources_by_disposition') != prefix['counts']
                    or old.get('committed_allocated_bytes') != prefix['allocated']):
                raise ValueError('The original published cursor must still reconstruct its exact closed source prefix')
        self._write_cursor()

    def _atomic_json(self, path, value):
        temporary = self.folder/(path.name+'.'+uuid.uuid4().hex+'.pending')
        with temporary.open('xb') as stream:
            stream.write(canonical(value)+b'\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(self.folder)

    def _inspect(self, path):
        if path.stat().st_size > self.manifest['maximum_segment_bytes']:
            raise ValueError('A retained segment exceeds its original compressed bound')
        with gzip.open(path, 'rb') as stream:
            raw = stream.read(128*1024**2+1)
        if len(raw) > 128*1024**2:
            raise ValueError('An entire retained source segment exceeds its declared decode bound')
        item = json.loads(raw)
        if (item.get('format') != FORMAT or item.get('manifest_sha256') != self.manifest_sha256
                or item.get('previous_segment_sha256') != self.tip
                or item.get('start') != self.next_position
                or item.get('stop') != min(self.next_position+self.manifest['width'], self.manifest['total'])
                or not isinstance(item.get('records'), list)
                or len(item['records']) != item['stop']-item['start']):
            raise ValueError('Every whole original source segment and previous hash must reconstruct')
        expected_name = f'segment-{item["start"]:09d}-{item["stop"]-1:09d}.json.gz'
        if path.name != expected_name:
            raise ValueError('Original segment filename and complete ordered source range must agree')
        for position, row in enumerate(item['records'], item['start']):
            validate_record(row, position)
        return item, sha(path)

    def _adopt(self, path, item, file_sha):
        self.next_position = item['stop']
        self.tip = file_sha
        self.committed_bytes += path.stat().st_blocks*512
        if self.committed_bytes > self.manifest['maximum_total_bytes']:
            raise ValueError('The entire original durable source budget has been exceeded')
        self.segment_paths.append(path)
        for row in item['records']:
            self.counts[row['status']] = self.counts.get(row['status'], 0)+1
        self.prefixes[self.next_position] = dict(tip=self.tip, counts=dict(self.counts), allocated=self.committed_bytes)

    def _write_cursor(self):
        self._atomic_json(self.folder/'cursor.json', dict(manifest_sha256=self.manifest_sha256,
            next_original_position=self.next_position, total_original_sources=self.manifest['total'],
            last_closed_segment_sha256=self.tip, complete_sources_by_disposition=self.counts,
            committed_allocated_bytes=self.committed_bytes,
            interrupted_temporary_files_preserved=[p.name for p in sorted(self.folder.glob('*.pending'))],
            complete=self.next_position == self.manifest['total'], training_admitted=False))

    def append(self, rows):
        start = self.next_position
        stop = min(start+self.manifest['width'], self.manifest['total'])
        if start == stop or not isinstance(rows, list) or len(rows) != stop-start:
            raise ValueError('Each commit must retain the entire next original source block, including its real tail')
        for position, row in enumerate(rows, start):
            validate_record(row, position)
        item = dict(format=FORMAT, manifest_sha256=self.manifest_sha256, previous_segment_sha256=self.tip,
            start=start, stop=stop, records=rows)
        raw = canonical(item)
        if len(raw) > 128*1024**2:
            raise ValueError('The complete original source block exceeds its decode bound; never crop it')
        packed = gzip.compress(raw, compresslevel=6, mtime=0)
        allocated = ((len(packed)+4095)//4096)*4096
        if (len(packed) > self.manifest['maximum_segment_bytes']
                or self.committed_bytes+allocated > self.manifest['maximum_total_bytes']
                or shutil.disk_usage(self.folder).free-len(packed) < self.manifest['minimum_free_bytes']):
            raise ValueError('The complete source block must fit the original durable budget and other writers reserve')
        path = self.folder/f'segment-{start:09d}-{stop-1:09d}.json.gz'
        if path.exists():
            raise ValueError('A complete original source segment cannot be overwritten')
        temporary = self.folder/(path.name+'.'+uuid.uuid4().hex+'.pending')
        with temporary.open('xb') as stream:
            stream.write(packed);stream.flush();os.fsync(stream.fileno())
        # Link publishes a complete fsynced inode without ever replacing an
        # existing segment, including when a second writer races this call.
        os.link(temporary, path)
        sync_directory(self.folder)
        temporary.unlink()
        closed, file_sha = self._inspect(path)
        self._adopt(path, closed, file_sha)
        self._write_cursor()

    def rows(self):
        for path in self.segment_paths:
            with gzip.open(path, 'rt') as stream:
                yield from json.load(stream)['records']
