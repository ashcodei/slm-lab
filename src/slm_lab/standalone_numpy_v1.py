"""Expose only the installed NumPy package to the constrained Python runner."""
import importlib.metadata
from pathlib import Path

import numpy as np

from .standalone_program_v1 import PYTHON_BOOTSTRAP, prepare_python


def numpy_runtime_paths():
    package = Path(np.__file__).resolve().parent
    distribution = Path(importlib.metadata.distribution('numpy')._path).resolve()
    paths = [package, distribution]
    libraries = package.parent / 'numpy.libs'
    if libraries.is_dir():
        paths.append(libraries.resolve())
    return paths


def prepare_numpy(code, work):
    constraints = prepare_python(code, work)
    package = Path(np.__file__).resolve().parent
    constraints['paths'].append((str(package.parent), 1 << 3))
    for path in numpy_runtime_paths():
        constraints['paths'].append((str(path), (1 << 2) | (1 << 3)))
    constraints['command'][5] = ('import sys;sys.path.insert(0,' + repr(str(package.parent)) + ');' + PYTHON_BOOTSTRAP)
    constraints['environment'].update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return constraints
