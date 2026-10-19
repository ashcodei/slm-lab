"""Retire redundant completed optimizer histories while retaining verified weights."""
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
from .corpus import file_hash
from .diverse_images import atomic_json


def retire(directory, audit):
    directory, audit = Path(directory), Path(audit)
    state = directory / 'training-state.pt'; model = directory / 'model.pt'
    report = json.loads((directory / 'training.json').read_text())
    if 'completed' not in report.get('status', ''):
        raise ValueError('Optimizer state is not from a completed run')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try: arguments = (process / 'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, PermissionError, ProcessLookupError): continue
        for raw in arguments:
            if not 0 < len(raw) < 4096: continue
            try: resolved = Path(raw.decode()).resolve()
            except (OSError, UnicodeError, ValueError): continue
            if resolved in {state.resolve(), directory.resolve()}:
                raise ValueError('Completed state/output directory referenced by a live process')
    source = torch.load(state, map_location='cpu', weights_only=True, mmap=True)
    retained = torch.load(model, map_location='cpu', weights_only=True, mmap=True)
    if source['steps'] != source['planned_steps'] or 'optimizer' not in source:
        raise ValueError('Incomplete or non-optimizer checkpoint')
    for key in ['steps', 'stage', 'data_sha256', 'tokenizer_sha256']:
        if source[key] != retained[key] or source[key] != report[key]:
            raise ValueError('Completed state/model/report identity differs: ' + key)
    if (source['config'] != retained['config'] or set(source['model']) != set(retained['model'])
            or any(not torch.equal(value, retained['model'][key]) for key, value in source['model'].items())):
        raise ValueError('Retained final learned tensors differ')
    record = {'created_at': datetime.now(timezone.utc).isoformat(), 'state': str(state.resolve()),
              'state_sha256': file_hash(state), 'bytes': state.stat().st_size,
              'retained_model_sha256': file_hash(model), 'all_learned_tensors_verified_equal': True,
              'audit_completed_before_removal': True, 'removed': False,
              'limitation': 'Exact completed optimizer history cannot resume; learned weights/reports remain.'}
    atomic_json(audit, record)
    del source, retained
    state.unlink(); record['removed'] = True
    atomic_json(audit, record)
    return record
