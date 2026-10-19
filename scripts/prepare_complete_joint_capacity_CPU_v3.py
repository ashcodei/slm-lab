"""Read all actual joint sources and retain complete original capacity cases.

An explicit separate scratch output is allowed for metadata preparation; this
does not fund persistent full-size training or acquire any GPU resources.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 40 * 1024**3
MAX_OUTPUT = 64 * 1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main(output):
    output = Path(output).absolute()
    if (output.exists() or output.resolve() != output or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.nice(0) < 19 or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(output.parent).free < RESERVE + MAX_OUTPUT):
        raise ValueError('Fresh CUDA-hidden idle CPU preparation needs its complete metadata budget above40GiBon its actual output filesystem')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.complete_joint_capacity_plan_v3 import choose
    from slm_lab.complete_joint_training_data_v1 import CompleteJointTrainingDataV1
    from slm_lab.model import ModelConfig
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    files = {str(p): sha(p) for p in python_sources(ROOT, [Path(__file__)])}
    visual = ROOT / 'reports/joint-visual-actual-capacity-cases-CPU-v1/actual-complete-source-capacity-plan.json'
    files[str(visual)] = sha(visual)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    output.mkdir()

    def write(name, value):
        raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if (sum(p.stat().st_size for p in output.iterdir() if p.is_file()) + len(raw) > MAX_OUTPUT
                or shutil.disk_usage(output).free - len(raw) < RESERVE):
            raise ValueError('Preserve complete bounded source metadata and the unchanged40GiBoutput writer reserve')
        with (output / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())

    config = ModelConfig(num_layers=35)
    write('plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        configuration=asdict(config), word_expansion=5, minimum_free_bytes=RESERVE, maximum_output_bytes=MAX_OUTPUT,
        actual_output_filesystem=str(output), scratch_metadata_is_not_persistent_training_funding=True,
        models_constructed=0, GPU_contexts=0, production_training_admitted=False, goal_achieved=False))
    try:
        source = CompleteJointTrainingDataV1(ROOT, config, word_expansion=5)
        for path, digest in source.files.items():
            if path in files and files[path] != digest:
                raise ValueError('Original source and actual runtime closure must not disagree')
            files[path] = digest
        guard = FileBindings(files); guard.check(full=True)
        chosen = choose(source, json.loads(visual.read_text()))
        if torch.cuda.is_initialized():
            raise ValueError('Source-only preparation cannot initialize a GPU context')
        guard.check(full=True)
        write('actual-complete-joint-capacity-plan.json', chosen)
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, source_identity_sha256=source.identity_sha256,
            capacity_plan_sha256=sha(output / 'actual-complete-joint-capacity-plan.json'),
            actual_original_joint_steps=5362, actual_complete_capacity_cases=chosen['actual_complete_capacity_cases'],
            original_stream_budgets=chosen['original_stream_budgets'], original_full_shape_limits=chosen['original_full_shape_limits'],
            all_original_visual_capacity_cases=10, additional_original_negative_context_tokens=chosen['additional_original_corrective_negative_context_tokens'],
            additional_original_negative_EOS_targets=chosen['additional_original_corrective_negative_response_EOS_targets'],
            complete_positive_and_negative_context_tokens=chosen['complete_positive_and_negative_context_tokens'],
            complete_positive_and_negative_response_EOS_targets=chosen['complete_positive_and_negative_response_EOS_targets'], models_constructed=0, neural_forwards=0, optimizer_updates=0,
            GPU_contexts=0, actual_full_size_GPU_capacity_passed=False, production_training_admitted=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, output=str(output), actual_complete_capacity_cases=chosen['actual_complete_capacity_cases'])), flush=True)
    except BaseException as exc:
        write('failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/complete-joint-actual-capacity-cases-CPU-v3')
    main(parser.parse_args().output)
