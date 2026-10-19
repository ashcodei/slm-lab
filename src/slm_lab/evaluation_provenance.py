"""Bind local evaluation reports to the implementation/runtime used for them."""
from importlib.metadata import PackageNotFoundError, version
import hashlib
from pathlib import Path
import sys

from .corpus import file_hash

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = [
    'model', 'training', 'corpus', 'data', 'diverse_images', 'code_tasks',
    'code_quality', 'evaluation', 'cpp_tasks', 'cpp_evaluation', 'native_code',
    'math_corpus', 'math_evaluation', 'knowledge_evaluation', 'ocr_evaluation',
    'object_evaluation', 'vision_evaluation', 'public_code_benchmark',
    'evaluation_provenance',
]


def capture(files=None):
    paths = ([ROOT/'src/slm_lab'/(name+'.py') for name in SOURCE_NAMES]
             + [ROOT/'scripts/code_worker.py', ROOT/'scripts/native_sandbox.py']) if files is None else list(files)
    packages = {}
    for name in ['torch', 'tokenizers', 'pillow', 'quickjs']:
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    return {'version': 1, 'python': sys.version, 'packages': packages,
            'files_sha256': {str(Path(path).resolve().relative_to(ROOT)) if Path(path).resolve().is_relative_to(ROOT)
                             else str(Path(path).resolve()): file_hash(path) for path in paths}}


def pixels_sha256(pixels):
    return hashlib.sha256(pixels.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def seal(report, provenance, files=None, data=None, digest=None, device=None, tokenizer_sha256=None):
    if capture(files) != provenance:
        raise ValueError('Evaluation implementation/runtime changed during this run')
    if data is not None and (digest is None or file_hash(data) != digest):
        raise ValueError('Evaluation data changed during this run')
    if device is not None:
        report['inference_device_type'] = 'cuda' if str(device).startswith('cuda') else 'cpu'
        report['inference_precision'] = 'bf16' if str(device).startswith('cuda') else 'fp32'
    if tokenizer_sha256 is not None:report['tokenizer_sha256']=tokenizer_sha256
    report['evaluation_provenance'] = provenance
    return report
