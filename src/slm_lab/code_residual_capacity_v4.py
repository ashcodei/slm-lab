"""Disposable, final-input-bound four-GPU capacity check for the coding stage.

No checkpoint or candidate is exported. The final production plan must bind the
result's exact digest; this program cannot replace a missing source or baseline.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import distributed as dist

from .code_residual_branch_v1 import RoutedCodeResidual
from .code_residual_checkpoint_v1 import _digest
from .code_residual_collective_checkpoint_v2 import _validated, fingerprint
from .code_residual_lifecycle_v4 import rank_zero, report_json
from .code_residual_production_v4 import (ControllerBoundGuard, DONE, MODEL_COUNTS,
    PACKAGE, ROOT, RUNTIME_FILES, _merge, _process, _read_bound)
from .code_residual_source_contract_v4 import (FIELDS, SourceContractResidualKernel,
    open_final_source_replay, source_contract_update)
from .code_residual_stage_v2 import validate_options
from .code_residual_stage_v3 import ExpandedReplayEncoder, _EncodedBatches
from .codenet_checkpoint_v1 import FileBindings
from .codenet_fact_replay_v1 import CombinedReplay
from .codenet_runner_v1 import average_gradients
from .codenet_stage_v1 import Objective, exposure_budget, objective_gradients
from .codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from .corpus import file_hash
from .evaluation_provenance import capture
from .model import ModelConfig
from .recurrent_ablation import state_digest
from .training import load_checkpoint


FORMAT = 'own-complete-code-residual-capacity-plan-v4'
PROOF_FORMAT = 'own-complete-code-residual-capacity-proof-v4'


def validate_plan(plan):
    validate_options(plan['training_options'])
    if (plan.get('format') != FORMAT or plan.get('world_size') != 4
            or plan.get('residual_width') != 2048 or plan.get('expected_model_counts') != MODEL_COUNTS
            or plan['training_options']['image_size'] != 448
            or plan.get('candidate_export') is not False
            or not _digest(plan.get('replay_contract_fingerprint'))):
        raise ValueError('Exact own857M/four-worker/448px disposable capacity plan required')
    paths = ('base_checkpoint', 'tokenizer', 'code_pack', 'source_bundle', 'final_base_binding', 'output')
    hashes = ('base_checkpoint', 'tokenizer', 'code_manifest', 'code_admission', 'source_bundle_manifest', 'final_base_binding')
    if (any(not isinstance(plan.get(k), str) or not Path(plan[k]).is_absolute() for k in paths)
            or any(not _digest(plan.get(k + '_sha256')) for k in hashes)):
        raise ValueError('Absolute exact final capacity inputs required')
    sources = plan.get('source_files_sha256', {})
    required = {str(PACKAGE / name) for name in RUNTIME_FILES} | {str(Path(__file__))}
    if (not isinstance(sources, dict) or not required <= sources.keys()
            or any(not Path(p).is_absolute() or not _digest(h) for p, h in sources.items())):
        raise ValueError('Complete production and capacity runtime source bindings required')
    actors = plan.get('preceding_processes', [])
    if (len(actors) != 4 or {a.get('role') for a in actors} != {'main', 'joint', 'feedback', 'resumer'}
            or any(type(a.get('pid')) is not int or a['pid'] < 1
                or not isinstance(a.get('start_ticks'), str) or not a['start_ticks'].isdigit() for a in actors)):
        raise ValueError('Exact original predecessor process births required')
    storage = plan.get('storage', {})
    if (set(storage) != {'maximum_report_bytes', 'minimum_free_bytes'}
            or any(type(v) is not int or v < 1 for v in storage.values())
            or storage['maximum_report_bytes'] < 1024**2 or storage['minimum_free_bytes'] < 26 * 1024**3):
        raise ValueError('Funded bounded reports above the original26GiB reserve required')


def bind_inputs(plan):
    """Resolve final source/base evidence before any GPU inspection or context."""
    validate_plan(plan)
    path = Path(plan['source_bundle']) / 'manifest.json'
    bundle = _read_bound(path, plan['source_bundle_manifest_sha256'])
    if (bundle.get('format') != 'own-complete-code-final-source-bundle-v4'
            or any(bundle.get(k) is not True for k in ('all30_actual_prepared_families_opened',
                'full_original_paired_admission_completed', 'every_prepared_logical_record_reached'))
            or any(bundle.get(k) != plan[k] for k in ('tokenizer_sha256', 'code_manifest_sha256',
                'code_admission_sha256', 'replay_contract_fingerprint'))):
        raise ValueError('Actual complete final-source bundle and full replay coverage required')
    files = {}
    for added in (plan['source_files_sha256'], bundle['source_files_sha256'],
            {str(path): plan['source_bundle_manifest_sha256']}):
        _merge(files, added)
    for name, digest in bundle['artifact_files_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Owned source artifact basenames required')
        _merge(files, {str(path.parent / name): digest})
    for key in ('base_checkpoint', 'tokenizer', 'final_base_binding'):
        _merge(files, {plan[key]: plan[key + '_sha256']})
    for name in ('manifest', 'admission'):
        _merge(files, {str(Path(plan['code_pack']) / (name + '.json')): plan['code_' + name + '_sha256']})
    binding = _read_bound(plan['final_base_binding'], plan['final_base_binding_sha256'])
    if (binding.get('format') != 'own-complete-code-final-base-binding-v4'
            or any(binding.get(k) != plan[k] for k in ('base_checkpoint_sha256', 'tokenizer_sha256',
                'source_bundle_manifest_sha256', 'replay_contract_fingerprint'))
            or any(binding.get(k) is not True for k in ('final_checkpoint_replay_match_verified', 'all_later_queued_sources_covered'))
            or set(binding.get('completion_status_files', {})) != set(DONE)):
        raise ValueError('Completed final own policy and all-later-source correspondence required')
    evidence = binding['files_sha256']
    _merge(files, evidence)
    for role, report in binding['completion_status_files'].items():
        if report not in evidence or _read_bound(report, evidence[report]).get('status') != DONE[role]:
            raise ValueError('Original predecessor training/evaluation is incomplete: ' + role)
    selection, coverage = binding['selection_file'], binding['coverage_report']
    if selection not in evidence or coverage not in evidence:
        raise ValueError('Actual final selection and independent coverage evidence required')
    if _read_bound(selection, evidence[selection]).get('selected_checkpoint_sha256') != plan['base_checkpoint_sha256']:
        raise ValueError('Final guarded selection differs from the capacity base')
    cov = _read_bound(coverage, evidence[coverage])
    if (cov.get('matched_checkpoint_sha256') != plan['base_checkpoint_sha256']
            or cov.get('source_bundle_manifest_sha256') != plan['source_bundle_manifest_sha256']
            or cov.get('final_checkpoint_replay_match_verified') is not True
            or cov.get('source_rows_dropped_or_duplicated') != 0 or cov.get('encoder_truncated_sequences') != 0):
        raise ValueError('Exact complete unclipped final-policy/source correspondence required')
    guard = FileBindings(files)
    guard.check(full=True)
    return bundle, guard, files


def capacity_extremes(code, replay, encoder, bundle, contract):
    """Choose admitted original examples; never synthesize long inputs."""
    position = int(np.argmax(code.index['length'][code.positions]))
    ids, labels = code.sequence(position)
    if len(ids) != 4096:
        raise ValueError('An actual complete4096-token training program is required')
    candidates = []
    for family in bundle['selected_actual_full_encoder_checks']:
        for check in family['checks']:
            if check['actual448px_golden_pixels_checked']:
                candidates.append((check['context_tokens'], family['name'], check['position']))
    if not candidates:
        raise ValueError('An actual admitted448px image source is required')
    _, family, image_position = max(candidates)
    row = replay.row(family, image_position)
    field = FIELDS[next(f['objective'] for f in contract['families'] if f['name'] == family)]
    batch = encoder.batch([row], field, 'cpu')
    if not row.get('image') or tuple(batch['pixel_values'].shape[-2:]) != (448, 448):
        raise ValueError('The capacity image must pass actual original448px encoding')
    return position, row, field, dict(code_position=position, actual_code_context_tokens=len(ids),
        code_target_tokens=sum(y != -100 for y in labels[1:]), image_family=family,
        image_position=image_position, image_context_tokens=batch['input_ids'].shape[1], image_size=448)


def exercise_capacity(kernel, code_position, image_row, image_field):
    """Actual kernel update, every source's full layout, then combined extremes.

    Probe updates use actual schedule positions but are deliberately noncontiguous.
    Their disposable weights/history are never offered as a resumable candidate.
    """
    base_before = state_digest(kernel.model.base.state_dict())
    residual_before = state_digest(kernel.model.residuals.state_dict())
    first = kernel.update()
    seen = {first['replay_family']}
    probe_metrics = []
    for step in range(kernel.steps):
        family, occurrence = kernel.schedule.at(step)
        if family in seen:
            continue
        kernel._guard()
        metric = source_contract_update(kernel.wrapped, kernel.optimizer, kernel.code, kernel.code_encoder,
            kernel.replay, kernel.replay_encoder, kernel.objectives, step, kernel.options,
            kernel.device, kernel.amp, kernel.rank, kernel.world)
        kernel.optimizer.zero_grad(set_to_none=True)
        probe_metrics.append(metric)
        seen.add(family)
        if len(seen) == len(kernel.replay.families):
            break
    if seen != set(kernel.replay.families):
        raise ValueError('Every actual declared replay family must receive a full-layout capacity update')
    options = kernel.options
    code_groups = [[dict(packed_code_index=code_position) for _ in range(options['batch_size'])]
        for _ in range(options['gradient_accumulation'])]
    image_groups = [[image_row for _ in range(options['replay_batch_size'])]
        for _ in range(options['replay_accumulation'])]
    def prepare():
        return (_EncodedBatches(kernel.code_encoder, code_groups, 'response', kernel.device),
            _EncodedBatches(kernel.replay_encoder, image_groups, image_field, kernel.device))
    encoded_code, encoded_image = _validated(prepare)
    actual_context = max(batch['input_ids'].shape[1] for batch in encoded_code.batches)
    image_sizes = [list(batch['pixel_values'].shape[-2:]) for batch in encoded_image.batches]
    kernel.optimizer.zero_grad(set_to_none=True)
    extreme_metric = objective_gradients(kernel.wrapped,
        [Objective('maximum_complete_code', encoded_code, code_groups, 'response', 1.),
         Objective('maximum_original_image', encoded_image, image_groups, image_field, options['replay_coefficient'])],
        kernel.device, kernel.amp, world=kernel.world)
    average_gradients(kernel.wrapped, kernel.device, kernel.world)
    norm = torch.nn.utils.clip_grad_norm_(kernel.wrapped.parameters(), options['gradient_clip'], error_if_nonfinite=True)
    if (any(p.grad is not None for p in kernel.model.base.parameters())
            or state_digest(kernel.model.base.state_dict()) != base_before
            or state_digest(kernel.model.residuals.state_dict()) == residual_before
            or kernel.code_encoder.truncated or kernel.replay_encoder.truncated):
        raise ValueError('Actual finite residual learning and unchanged unclipped original backbone required')
    kernel.optimizer.zero_grad(set_to_none=True)
    return dict(actual_full_kernel_optimizer_updates=1, first_kernel_metric=first,
        additional_noncontiguous_disposable_probe_updates=len(probe_metrics),
        probe_metrics=probe_metrics, all_declared_source_families_exercised=sorted(seen),
        actual_maximum_context_backward=actual_context, actual_image_batch_sizes=image_sizes,
        combined_extreme_backward=extreme_metric, extreme_gradient_norm=float(norm),
        original_backbone_unchanged=True, residual_parameters_changed=True,
        candidate_weights_saved=False, capability_result=False)


def controller(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    bundle, guard, files = bind_inputs(plan)
    output, storage = Path(plan['output']).resolve(), plan['storage']
    if (not output.is_relative_to(ROOT / 'reports') or output == ROOT / 'reports'
            or any(Path(p).resolve().is_relative_to(output) for p in files)
            or output.exists()):
        raise ValueError('A new distinct owned report directory is required')
    def available():
        for actor in plan['preceding_processes']:
            if _process(actor['pid'], actor['start_ticks']):
                raise ValueError('Original GPU predecessor remains live: ' + actor['role'])
        if output.exists():
            raise ValueError('Capacity output became occupied; inspect rather than overwrite')
        if shutil.disk_usage(ROOT).free < storage['minimum_free_bytes'] + storage['maximum_report_bytes']:
            raise ValueError('Capacity reports and original free-space reserve are unfunded')
    available()
    lock = ROOT / 'reports/code-residual-production-v4.gpu.lock'
    with lock.open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        available()
        memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            check=True, capture_output=True, text=True).stdout.splitlines()
        if len(memory) != 4 or any(int(v.strip()) > 128 for v in memory):
            raise ValueError('All four GPUs must be idle before capacity workers launch')
        guard.check(full=True)
        output.mkdir()
        pid = os.getpid()
        owner = dict(pid=pid, start_ticks=(Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()[19],
            plan_sha256=file_hash(plan_path), parent_cmdline_sha256=file_hash(Path('/proc') / str(pid) / 'cmdline'))
        report_json(output / 'owner.json', owner, storage['maximum_report_bytes'])
        argv = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4',
            '-m', 'slm_lab.code_residual_capacity_v4', '--worker', '--plan', str(plan_path)]
        try:
            subprocess.run(argv, cwd=ROOT, check=True)
            proof = json.loads((output / 'proof.json').read_text())
            if proof.get('passed') is not True:
                raise ValueError('Capacity workers ended without an actual passing proof')
        except Exception as error:
            report_json(output / 'failure.json', dict(error=str(error), automatic_restart=False,
                passed=False, goal_achieved=False), storage['maximum_report_bytes'])
            raise


def worker(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output, storage = Path(plan['output']), plan['storage']
    owner_path = output / 'owner.json'
    owner = json.loads(owner_path.read_text())
    if owner['plan_sha256'] != file_hash(plan_path) or not _process(owner['pid'], owner['start_ticks']):
        raise ValueError('Exact live checked capacity parent required')
    bundle, initial, files = bind_inputs(plan)
    _merge(files, {str(plan_path): file_hash(plan_path), str(owner_path): file_hash(owner_path)})
    guard = ControllerBoundGuard(FileBindings(files), owner)
    guard.check(full=True)
    rank, world, local = [int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK')]
    if world != 4 or not 0 <= rank < 4 or not 0 <= local < 4 or torch.cuda.device_count() != 4:
        raise ValueError('Exactly four visible actual CUDA workers required')
    code = CompleteCodePack(plan['code_pack'], 'train', purpose='training')
    replay, contract = open_final_source_replay(bundle['replay_folders'], plan['tokenizer_sha256'])
    options = plan['training_options']
    if (fingerprint(contract) != plan['replay_contract_fingerprint']
            or exposure_budget(len(code), options['passes'], 4, options['batch_size'], options['gradient_accumulation']) != bundle['prospective_code_budget']
            or options['seed'] != bundle['prospective_seed']
            or 4 * options['replay_batch_size'] * options['replay_accumulation'] != bundle['prospective_global_replay_width']):
        raise ValueError('Capacity must exercise the exact final production source/layout contract')
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_name(device) != 'NVIDIA L40':
        raise ValueError('Actual BF16 L40 capacity evidence required')
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=timedelta(minutes=15))
    try:
        base, metadata = load_checkpoint(plan['base_checkpoint'], device, training_state=False)
        if metadata['tokenizer_sha256'] != plan['tokenizer_sha256'] or base.config != ModelConfig.load(code.folder / 'model-config.json'):
            raise ValueError('Exact final own architecture/tokenizer required')
        torch.manual_seed(options['seed'])
        model = RoutedCodeResidual(base, width=2048)
        if model.parameter_counts() != MODEL_COUNTS:
            raise ValueError('Exact complete857M parameter inventory required')
        optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=options['learning_rate'],
            weight_decay=options['weight_decay'], foreach=False)
        joint = replay
        while isinstance(joint, CombinedReplay):
            joint = joint.owners['human_instruction']
        token = Tokenizer.from_file(plan['tokenizer'])
        encoder = ExpandedReplayEncoder(token, model.config, joint.source.parent, 448)
        code_position, image_row, field, extremes = capacity_extremes(code, replay, encoder, bundle, contract)
        sources = dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
            run_plan_sha256=file_hash(plan_path))
        kernel = SourceContractResidualKernel(model, optimizer, code, packed_encoder(code, token, model.config),
            replay, encoder, options, sources, device,
            lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16),
            replay_contract=contract, expected_contract_fingerprint=fingerprint(contract), guard=guard)
        random.seed(options['seed'] + rank)
        np.random.seed((options['seed'] + rank) % 2**32)
        torch.manual_seed(options['seed'] + rank)
        torch.cuda.reset_peak_memory_stats(device)
        proof = exercise_capacity(kernel, code_position, image_row, field)
        torch.cuda.synchronize(device)
        proof.update(rank=rank, device_name=torch.cuda.get_device_name(device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device), selected_original_extremes=extremes)
        if proof['actual_maximum_context_backward'] != 4096 or any(size != [448, 448] for size in proof['actual_image_batch_sizes']):
            raise ValueError('Actual4096-token and448px backward evidence required')
        guard.check(full=True)
        rank_reports = [None] * 4
        dist.all_gather_object(rank_reports, proof)
        def finish():
            guard.check(full=True)
            for item in rank_reports:
                report_json(output / ('rank-' + str(item['rank']) + '.json'), item, storage['maximum_report_bytes'])
            report_json(output / 'proof.json', dict(format=PROOF_FORMAT, passed=True,
                completed_at=datetime.now(timezone.utc).isoformat(), base_checkpoint_sha256=plan['base_checkpoint_sha256'],
                replay_contract_fingerprint=plan['replay_contract_fingerprint'],
                kernel_source_sha256=files[str(PACKAGE / 'code_residual_source_contract_v4.py')],
                training_options_fingerprint=fingerprint(options), model_counts=model.parameter_counts(),
                actual_world_size=4, actual_full_kernel_optimizer_updates=1,
                actual_maximum_context_backward=4096, actual448px_image_backward=True,
                runtime_provenance=capture([PACKAGE / name for name in sorted(RUNTIME_FILES)]),
                files_sha256={**files, **{str(output / ('rank-' + str(i) + '.json')):
                    file_hash(output / ('rank-' + str(i) + '.json')) for i in range(4)}},
                all_source_families_full_layout_probed=True, candidate_weights_saved=False,
                noncontiguous_probe_weights_discarded=True, goal_achieved=False), storage['maximum_report_bytes'])
        rank_zero(finish)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.plan)
    else:
        controller(args.plan)


if __name__ == '__main__':
    main()
