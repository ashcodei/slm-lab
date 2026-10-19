"""Recover terminal source observers using the unchanged full recipe builders.

Only their new metadata destination/publisher and indexed admission path are
rebound in this process. Every original numerical driver, source population,
predecessor, horizon, training budget and final assessment remains required.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / 'scripts' / name
    spec = importlib.util.spec_from_file_location('_recovered_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main(phase):
    if (phase not in ['original', 'indexed'] or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Only a fresh one-thread CUDA-hidden idle source-recovery process is eligible')
    from slm_lab.metadata_observer_recovery_v1 import ORIGINAL_OUT, INDEXED_OUT, capture, read, sha
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab import reserved_atomic_metadata_v1 as metadata
    recoveries = [capture(ROOT / 'reports' / name, ROOT / 'reports' / (name + '.log'), kind='source') for name in
        ['native-code-suffix-aligned-preparation-CPU-v2', 'native-code-suffix-aligned-preparation-CPU-v4']]
    output = ORIGINAL_OUT if phase == 'original' else INDEXED_OUT
    if output.exists():
        raise ValueError('Preserve every existing recovery owner/output;there is no automatic retry')
    legacy = load_script('prepare_native_source_recipe_recovery_CPU_v1.py' if phase == 'original' else 'prepare_native_code_suffix_aligned_CPU_v4.py')
    legacy.OUT = output
    entries = [Path(__file__), ROOT / 'scripts/train_native_code_suffix_aligned_GPU_v5.py',
        ROOT / 'src/slm_lab/native_suffix_owned_runtime_contract_v5.py',
        ROOT / 'scripts/prepare_native_source_recipe_recovery_CPU_v1.py']
    runtime = {str(p): sha(p) for p in python_sources(ROOT, entries)}
    files = dict(runtime)
    for recovery in recoveries:
        for path, digest in recovery['source_files_sha256'].items():
            if path in files and files[path] != digest:
                raise ValueError('All retired and replacement runtime sources must agree')
            files[path] = digest
    original_write = metadata.write_json

    def routed(path, value, *, root, minimum_free_bytes, maximum_metadata_bytes, guard, **kwargs):
        path = Path(path)
        if path.parent != output or minimum_free_bytes != 40 * 1024**3:
            raise ValueError('Only this new observer metadata may use its actual scratch filesystem with the unchanged reserve')
        if path.name == 'plan.json':
            value['source_files_sha256'].update(files)
            if 'added_runtime_sources' in value:
                value['added_runtime_sources'].update(runtime)
            value['versioned_metadata_recovery'] = dict(at=datetime.now(timezone.utc).isoformat(), phase=phase,
                exact_retired_observers=recoveries, original_GPU_funding_root=str(ROOT), actual_metadata_filesystem=str(output),
                all_original_scopes_preserved=True, GPU_work_started=False, neural_retries=0)
        def checked():
            guard()
            if 'torch' in sys.modules or any(sha(p) != h for p, h in files.items()):
                raise ValueError('The new source observer must remain non-neural and preserve every captured original/new source byte')
        return original_write(path, value, root=output, minimum_free_bytes=minimum_free_bytes,
            maximum_metadata_bytes=maximum_metadata_bytes, guard=checked, **kwargs)

    metadata.write_json = routed
    previous_preparation = None
    try:
        if phase == 'original':
            old = read(ROOT / 'reports/native-code-suffix-aligned-preparation-CPU-v2/plan.json')
            legacy.main(old['exact_original_source_parents']['full_native_capacity_sources'])
        else:
            from slm_lab import native_suffix_preparation_v4 as preparation
            from slm_lab.native_suffix_owned_runtime_contract_v5 import PROTOCOL, require_complete_preparation
            legacy.ORIGINAL = ORIGINAL_OUT
            previous_preparation = (preparation, preparation.PROTOCOL, preparation.require_complete_preparation)
            preparation.PROTOCOL = PROTOCOL
            preparation.require_complete_preparation = require_complete_preparation
            legacy.main()
    finally:
        metadata.write_json = original_write
        if previous_preparation is not None:
            preparation, protocol, require_complete_preparation = previous_preparation
            preparation.PROTOCOL = protocol
            preparation.require_complete_preparation = require_complete_preparation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase', choices=['original', 'indexed'], required=True)
    main(parser.parse_args().phase)
