"""Restore a V4 opaque checkpoint using its explicitly retained raw anchor."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.nice(max(0, 19-os.nice(0)))
    if importlib.util.find_spec('zstandard') is None:
        package = Path('/opt/conda/lib/python3.11/site-packages/zstandard')
        spec = importlib.util.spec_from_file_location('zstandard', package/'__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec); sys.modules['zstandard'] = module; spec.loader.exec_module(module)
    from slm_lab.exact_checkpoint_raw_anchor_v4 import restore
    print(json.dumps(restore(args.archive, args.output), indent=2))
