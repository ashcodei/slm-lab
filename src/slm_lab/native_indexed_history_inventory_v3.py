"""Inventory healthy reservations and exact consumed failure evidence.

A terminal failed attempt has consumed its emergency allocation. Its complete
typed payload and original hardlink pair replace the unused-space claim; it
can never be reused for another update. Healthy attempts must still retain
the entire real preallocated failure slot. No neural observations are removed.
"""
import copy
import json
import re

from .indexed_native_history_v1 import RECEIPT_LIMIT, read_regular
from .indexed_native_history_v2 import read_failure
from .corpus import file_hash

PROTOCOL = 'own-native-indexed-healthy-or-consumed-failure-inventory-v3'


def history_inventory(stage):
    directories = {(r['directory'], r['header_sha256']) for r in stage.metrics}
    if stage.history_reader is not None:
        directories.add((str(stage.history_reader.root), stage.history_reader.header_sha256))
    result = []
    for directory, header in sorted(directories):
        reader = stage.history_file_reader(directory, header)
        slot = reader.root / 'independent-failure.reserved'
        if slot.is_symlink() or not slot.is_file(): raise ValueError('The original independent failure-slot name must survive')
        state = slot.stat()
        allocation = 2 * reader.budgets['maximum_expanded_record_bytes'] + 2 * RECEIPT_LIMIT + 64 * 1024
        receipts = list(reader.root.glob('failure-step-*.receipt.json'))
        payloads = list(reader.root.glob('failure-step-*.pt'))
        consumed = None
        if receipts or payloads:
            if len(receipts) != 1 or len(payloads) != 1:
                raise ValueError('One terminal attempt can consume exactly one original independent failure allocation')
            receipt = json.loads(read_regular(receipts[0], RECEIPT_LIMIT))
            step = receipt.get('attempted_step')
            if (type(step) is not int or not reader.header['starting_completed'] < step <= reader.budgets['maximum_records']
                    or receipt.get('expected_reference_header_sha256') != header):
                raise ValueError('The consumed failure must retain its actual original attempt/header/source boundary')
            payload = read_failure(reader.root, receipt, maximum_bytes=reader.budgets['maximum_expanded_record_bytes'])
            actual = (reader.root / receipt['file']).stat()
            if (receipts[0].name != 'failure-step-' + str(step).zfill(12) + '.receipt.json'
                    or payloads[0].name != receipt['file'] or state.st_nlink != 2 or actual.st_nlink != 2
                    or (state.st_dev, state.st_ino) != (actual.st_dev, actual.st_ino)
                    or state.st_size != receipt['actual_bytes'] or state.st_blocks * 512 < state.st_size):
                raise ValueError('Both original failure names must retain the same complete physically stored immutable inode')
            consumed = dict(actual_attempted_step=step, actual_receipt_sha256=file_hash(receipts[0]),
                actual_payload_sha256=receipt['file_sha256'], actual_original_typed_observation_sha256=receipt['original_typed_observation_sha256'],
                original_complete_failure_readback_passed=True,
                successful_optimizer_commit_certified=payload['successful_optimizer_commit_certified'],
                allocation_consumed_and_never_available_for_new_updates=True)
        elif state.st_size != allocation or state.st_blocks * 512 < allocation or state.st_nlink != 1:
            raise ValueError('Every healthy attempt must retain its entire physically funded unused failure allocation')
        seen, allocated, files = set(), reader.root.stat().st_blocks * 512, 0
        for path in reader.root.iterdir():
            if path.is_symlink() or not path.is_file(): raise ValueError('Only complete original regular history artifacts are eligible')
            if (path.name not in ['reference.json.gz', 'independent-failure.reserved']
                    and re.fullmatch(r'record-\d{12}\.(json\.gz|receipt\.json|reserved)', path.name) is None
                    and re.fullmatch(r'failure-step-\d{12}\.(pt|receipt\.json)', path.name) is None):
                raise ValueError('An unknown writer cannot change an exclusively owned history inventory')
            stat = path.stat(); files += 1; identity = stat.st_dev, stat.st_ino
            if identity not in seen:
                seen.add(identity); allocated += stat.st_blocks * 512
        if allocated > reader.budgets['maximum_physical_bytes']:
            raise ValueError('Every retained actual allocated block must fit the original complete physical quota')
        result.append(dict(directory=directory, header_sha256=header,
            starting_completed=reader.header['starting_completed'], actual_regular_files=files,
            actual_unique_inode_allocated_bytes=allocated,
            actual_independent_failure_slot_allocated_bytes=state.st_blocks * 512,
            original_indexed_history_budgets=copy.deepcopy(reader.budgets),
            inventory_protocol=PROTOCOL, actual_independent_failure_slot_disposition='consumed-complete-terminal-evidence' if consumed else 'unused-fully-funded',
            actual_consumed_failure=consumed))
    return result
