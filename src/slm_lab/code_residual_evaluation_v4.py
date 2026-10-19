"""Use completed own residual exports with unchanged checkpoint evaluators.

Each view hard-links the exact inference/base file and its own tokenizer. No
weights are copied or merged. A process-local loader replacement lasts for one
evaluation; prompts, generation methods and scoring functions are untouched.
This bridge does not certify a completed benchmark suite or promote a model.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import os
from pathlib import Path
import threading

import torch

from .code_residual_checkpoint_v1 import _digest
from .code_residual_collective_checkpoint_v2 import fingerprint
from .code_residual_export_v2 import load_inference
from .codenet_checkpoint_v1 import FileBindings
from .corpus import file_hash
from .diverse_images import atomic_json
from .evaluation_provenance import capture
from .training import load_checkpoint as load_own_base


FORMAT = 'own-complete-code-evaluation-subject-v4'
VIEW_FORMAT = 'own-complete-code-evaluation-view-v4'
PATHS = ('inference_checkpoint', 'base_checkpoint', 'tokenizer', 'run_plan',
    'training_checkpoint', 'source_manifest')
_LOADER_LEASE = threading.Lock()
METADATA_FIELDS = ('stage', 'steps', 'planned_steps', 'tokenizer_sha256', 'data_sha256',
    'own_base_checkpoint_sha256', 'run_plan_sha256', 'training_checkpoint_sha256',
    'model_counts', 'world_size')


def _plain(value):
    """Keep provenance JSON-safe without accepting opaque executable objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise ValueError('Evaluation arguments must be explicit JSON values or paths')


def validate_subject(subject):
    if (set(subject) != {'format', 'expected_model_counts'} | set(PATHS) | {k + '_sha256' for k in PATHS}
            or subject['format'] != FORMAT
            or any(not isinstance(subject[k], str) or not Path(subject[k]).is_absolute() for k in PATHS)
            or any(not _digest(subject[k + '_sha256']) for k in PATHS)):
        raise ValueError('Exact completed own inference/base/tokenizer/plan/training/source subject required')
    counts = subject['expected_model_counts']
    if (not isinstance(counts, dict) or set(counts) != {'total_unique', 'frozen_base', 'trainable_code'}
            or any(type(v) is not int or v < 1 for v in counts.values())
            or counts['frozen_base'] + counts['trainable_code'] != counts['total_unique']
            or counts['total_unique'] >= 10**9):
        raise ValueError('All unique own base/vision/critic/code parameters must fit below1B')
    files = {}
    for key in PATHS:
        name, digest = subject[key], subject[key + '_sha256']
        if name in files and files[name] != digest:
            raise ValueError('Conflicting evaluation subject source identity')
        files[name] = digest
    return files


def prepare_view(subject, directory, route):
    """Create one new same-filesystem, byte-identical evaluator input directory."""
    files = validate_subject(subject)
    if route not in {'general', 'code'}:
        raise ValueError('An explicit general or code evaluation route is required')
    directory = Path(directory).absolute()
    if directory.exists() or any(Path(p).resolve().is_relative_to(directory.resolve()) for p in files):
        raise ValueError('New distinct evaluation view required; preserve existing inputs/results')
    FileBindings(files).check(full=True)
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(subject['inference_checkpoint' if route == 'code' else 'base_checkpoint'])
    # Real files preserve the evaluator's sibling-tokenizer lookup even when it
    # resolves checkpoint paths. A symlink to model.pt would lose that sibling.
    os.link(source, directory / 'model.pt')
    os.link(subject['tokenizer'], directory / 'tokenizer.json')
    receipt = dict(format=VIEW_FORMAT, subject_fingerprint=fingerprint(subject), route=route,
        checkpoint_sha256=file_hash(source), tokenizer_sha256=subject['tokenizer_sha256'],
        model_source=str(source), tokenizer_source=subject['tokenizer'],
        model_bytes_copied=0, weights_merged=False, goal_achieved=False)
    atomic_json(directory / 'view.json', receipt)
    # Adding hard links changes inode ctime. Capture the new immutable file
    # states after linking, while independently checking all original digests.
    FileBindings({**files, str(directory / 'model.pt'): receipt['checkpoint_sha256'],
        str(directory / 'tokenizer.json'): subject['tokenizer_sha256']}).check(full=True)
    return directory / 'model.pt'


class BoundLoader:
    """Exact inference-only replacement for an evaluator's imported loader."""
    def __init__(self, subject, directory, device):
        files = validate_subject(subject)
        self.subject = dict(subject)
        self.directory = Path(directory).absolute()
        self.checkpoint = self.directory / 'model.pt'
        self.device = str(torch.device(device))
        if self.device != 'cpu' and (not self.device.startswith('cuda:') or not self.device[5:].isdigit()):
            raise ValueError('Explicit CPU or indexed CUDA evaluation device required')
        import json
        receipt = json.loads((self.directory / 'view.json').read_text())
        if (receipt.get('format') != VIEW_FORMAT or receipt.get('subject_fingerprint') != fingerprint(subject)
                or receipt.get('route') not in {'general', 'code'}):
            raise ValueError('Exact owned route/view/subject receipt required')
        self.route = receipt['route']
        source_key = 'inference_checkpoint' if self.route == 'code' else 'base_checkpoint'
        if (receipt.get('checkpoint_sha256') != subject[source_key + '_sha256']
                or receipt.get('tokenizer_sha256') != subject['tokenizer_sha256']
                or not os.path.samefile(self.checkpoint, subject[source_key])
                or not os.path.samefile(self.directory / 'tokenizer.json', subject['tokenizer'])):
            raise ValueError('Actual exact-source hard links required, not substituted evaluator files')
        self.files = {**files, str(self.checkpoint): subject[source_key + '_sha256'],
            str(self.directory / 'tokenizer.json'): subject['tokenizer_sha256'],
            str(self.directory / 'view.json'): file_hash(self.directory / 'view.json')}
        self.guard = FileBindings(self.files)
        self.calls = 0
        self.loaded_metadata = []

    def __call__(self, path, device='cpu', training_state=True):
        if (Path(path).absolute() != self.checkpoint or training_state is not False
                or str(torch.device(device)) != self.device):
            raise ValueError('Only the exact bound view/device and inference-only load are allowed')
        self.guard.check(full=True)
        s = self.subject
        if self.route == 'code':
            model, metadata = load_inference(self.checkpoint, s['base_checkpoint'], self.directory / 'tokenizer.json',
                expected_checkpoint_sha256=s['inference_checkpoint_sha256'],
                expected_base_sha256=s['base_checkpoint_sha256'], expected_plan_sha256=s['run_plan_sha256'],
                expected_training_checkpoint_sha256=s['training_checkpoint_sha256'],
                expected_data_sha256=s['source_manifest_sha256'], device=self.device)
            if metadata['model_counts'] != s['expected_model_counts']:
                raise ValueError('Loaded complete residual parameter inventory differs from the bound subject')
        else:
            model, metadata = load_own_base(self.checkpoint, self.device, training_state=False)
        expected = s['expected_model_counts']['total_unique' if self.route == 'code' else 'frozen_base']
        if sum(p.numel() for p in model.parameters()) != expected or metadata['tokenizer_sha256'] != s['tokenizer_sha256']:
            raise ValueError('Exact original own parameter count and tokenizer required')
        model.eval().requires_grad_(False)
        self.guard.check(full=True)
        self.calls += 1
        self.loaded_metadata.append({**{k: metadata[k] for k in METADATA_FIELDS if k in metadata}, 'route': self.route})
        return model, metadata


@contextmanager
def evaluator_loader(module, loader):
    """Restore the original imported function even if generation/scoring fails."""
    if not isinstance(loader, BoundLoader) or not callable(getattr(module, 'load_checkpoint', None)):
        raise ValueError('A bound own loader and existing checkpoint evaluator are required')
    if not _LOADER_LEASE.acquire(blocking=False):
        raise ValueError('Only one process-local evaluator loader replacement may be active')
    original = module.load_checkpoint
    try:
        module.load_checkpoint = loader
        yield
    finally:
        module.load_checkpoint = original
        _LOADER_LEASE.release()


def evaluate_bound(subject, directory, module, output, receipt, *, device='cpu',
                   output_argument='output', arguments=None):
    """Run an existing evaluator and write separate exact loader provenance.

    This preserves the evaluator's own report bytes/protocol. A suite controller
    must still require every original baseline/guard and compare all case scores.
    """
    import json
    arguments = {} if arguments is None else _plain(dict(arguments))
    json.dumps(arguments, allow_nan=False)
    if output_argument not in {'output', 'reports'} or {'checkpoint', 'device', 'output', 'reports'} & arguments.keys():
        raise ValueError('Checkpoint/device/output are owned by the evaluation bridge')
    output, receipt = Path(output).absolute(), Path(receipt).absolute()
    if output.exists() or receipt.exists() or output == receipt or receipt.is_relative_to(output):
        raise ValueError('New distinct evaluation output and external receipt required')
    loader = BoundLoader(subject, directory, device)
    if any(Path(p).resolve().is_relative_to(output.resolve()) or Path(p).resolve() == receipt.resolve() for p in loader.files):
        raise ValueError('Evaluation outputs must preserve every source and view input')
    files = [Path(__file__), Path(module.__file__)] + [Path(__file__).with_name(name) for name in (
        'code_residual_export_v2.py', 'code_residual_branch_v1.py', 'code_residual_checkpoint_v1.py',
        'code_residual_collective_checkpoint_v2.py', 'codenet_checkpoint_v1.py')]
    runtime = capture(files)
    guard = FileBindings({str(Path(p).resolve()): file_hash(p) for p in files})
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with evaluator_loader(module, loader):
        result = module.evaluate(checkpoint=loader.checkpoint, device=loader.device,
            **{output_argument: output}, **arguments)
    loader.guard.check(full=True)
    guard.check(full=True)
    if capture(files) != runtime or loader.calls < 1 or not output.exists():
        raise ValueError('Actual unchanged evaluator inference and completed output required')
    reports = list(output.rglob('*')) if output.is_dir() else [output]
    atomic_json(receipt, dict(completed_at=datetime.now(timezone.utc).isoformat(),
        subject_fingerprint=fingerprint(subject), route=loader.route, subject=subject,
        evaluator_module=module.__name__, evaluator_source=str(Path(module.__file__).resolve()),
        evaluator_arguments=arguments, inference_device=loader.device, loader_calls=loader.calls,
        loaded_metadata=loader.loaded_metadata, runtime_provenance=runtime,
        input_files_sha256=loader.files,
        report_files_sha256={str(p): file_hash(p) for p in reports if p.is_file()},
        checkpoint_optimizer_loaded_on_GPU=False, prompt_generation_and_scoring_functions_unchanged=True,
        complete_capability_suite_certified=False, candidate_promoted=False, goal_achieved=False))
    return result
