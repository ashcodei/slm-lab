"""Constant-space accounting for actual indexed-history block allocations.

Configure on a fresh, exclusively owned Writer before any sampling. Count
sealed record/receipt inodes once and inspect only the current write paths.
This augments the unchanged source guard and never grants model admission.
"""
from pathlib import Path

from .corpus import file_hash
from .indexed_native_history_v1 import RECEIPT_LIMIT
from .indexed_native_history_v2 import Writer

PROTOCOL = 'own-native-actual-allocated-block-prefix-guard-v1'


def inode(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('The complete original owned regular history inode is required')
    stat = path.stat()
    return (stat.st_dev, stat.st_ino), stat.st_blocks * 512


class AllocatedHistoryGuard:
    def __init__(self, writer, *, maximum_allocated_bytes, source_sha256):
        if (type(writer) is not Writer or writer.records or writer.ticket is not None or writer.failure_slot is not None
                or writer.failed or type(maximum_allocated_bytes) is not int
                or not 0 < maximum_allocated_bytes <= writer.budgets['maximum_physical_bytes']
                or source_sha256 != file_hash(Path(__file__).resolve())):
            raise ValueError('A fresh actual Writer, independently funded block quota and exact guard source are required')
        self.writer, self.root = writer, writer.root
        self.maximum = maximum_allocated_bytes
        self.source_sha256 = source_sha256
        self.source_guard = writer.guard
        self.sealed_count = self.sealed_allocated_bytes = 0
        self.maximum_observed_allocated_bytes = 0
        self.maximum_observed_funded_peak_bytes = 0
        self.calls = 0
        self.header_inode = inode(self.root / 'reference.json.gz')[0]
        root_stat = self.root.stat()
        self.directory_inode = root_stat.st_dev, root_stat.st_ino

    def __call__(self):
        self.source_guard()
        if file_hash(Path(__file__).resolve()) != self.source_sha256 or self.writer.root != self.root:
            raise ValueError('The original allocation-guard source and exclusively owned root must remain')
        writer = self.writer
        root_stat = self.root.stat()
        if (writer.records < self.sealed_count or writer.records > self.sealed_count + 1
                or (root_stat.st_dev, root_stat.st_ino) != self.directory_inode
                or inode(self.root / 'reference.json.gz')[0] != self.header_inode):
            raise ValueError('Every actual completed write must preserve the exact owned block-accounting prefix')
        if writer.records == self.sealed_count + 1:
            name = 'record-' + str(writer.records).zfill(12)
            record, receipt = inode(self.root / (name + '.json.gz')), inode(self.root / (name + '.receipt.json'))
            if record[0] == receipt[0]: raise ValueError('Record data and external receipt require independent inodes')
            self.sealed_allocated_bytes += record[1] + receipt[1]
            self.sealed_count += 1
        # Already sealed immutable rows are owned only by this writer. The
        # independent full-directory readback checks that ownership contract.
        # The current reserve/publication names can alias one physical inode.
        prefix = 'record-' + str(self.sealed_count + 1).zfill(12)
        names = ['reference.json.gz', 'independent-failure.reserved', prefix + '.reserved',
            prefix + '.json.gz', prefix + '.receipt.json']
        seen, current = set(), self.sealed_allocated_bytes + root_stat.st_blocks * 512
        for name in names:
            path = self.root / name
            if path.exists() or path.is_symlink():
                identity, blocks = inode(path)
                if identity not in seen: seen.add(identity); current += blocks
        primary = 2 * writer.budgets['maximum_compact_record_bytes'] + 2 * RECEIPT_LIMIT + 64 * 1024
        failure = 2 * writer.budgets['maximum_expanded_record_bytes'] + 2 * RECEIPT_LIMIT + 64 * 1024
        additional = 0
        if (writer.ticket is None and not writer.pending_model_commit and not writer.failed
                and writer.starting_completed + writer.records < writer.budgets['maximum_records']):
            additional = primary + (failure if writer.failure_slot is None else 0) + 128 * 1024
        peak = current + additional
        if current > self.maximum or peak > self.maximum:
            raise ValueError('All actual history blocks and the complete next success/failure allocation must fit before sampling')
        self.maximum_observed_allocated_bytes = max(self.maximum_observed_allocated_bytes, current)
        self.maximum_observed_funded_peak_bytes = max(self.maximum_observed_funded_peak_bytes, peak)
        self.calls += 1

    def report(self):
        return dict(protocol=PROTOCOL, exclusive_owned_directory=str(self.root),
            maximum_allocated_bytes=self.maximum, actual_sealed_record_count=self.sealed_count,
            actual_sealed_record_and_receipt_allocated_bytes=self.sealed_allocated_bytes,
            maximum_observed_actual_allocated_bytes=self.maximum_observed_allocated_bytes,
            maximum_observed_funded_peak_bytes=self.maximum_observed_funded_peak_bytes,
            actual_guard_calls=self.calls, no_original_source_guard_removed=True,
            no_observation_history_retained=True, independent_full_directory_readback_still_required=True,
            production_training_admitted=False)


def attach(stage):
    error = None
    if stage.rank == 0:
        try:
            writer = stage.history_writer
            path = Path(__file__).resolve()
            wrapper = AllocatedHistoryGuard(writer,
                maximum_allocated_bytes=writer.budgets['maximum_physical_bytes'], source_sha256=stage.sources.get(str(path)))
            writer.guard = wrapper
        except Exception as exc: error = repr(exc)
    stage.coordinated(error, 'actual allocated block accounting before capacity sampling')
