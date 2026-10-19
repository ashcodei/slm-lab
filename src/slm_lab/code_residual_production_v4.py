"""Explicit-plan own857M coding stage; no implicit launch, resume or promotion."""
import argparse
from datetime import datetime, timezone
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
from .code_residual_collective_checkpoint_v2 import fingerprint, _validated
from .code_residual_export_v2 import export_inference, load_inference
from .code_residual_lifecycle_v4 import checkpoint_layout, rank_zero, report_json, run_updates
from .code_residual_source_contract_v4 import SourceContractResidualKernel, open_final_source_replay
from .code_residual_stage_v2 import validate_options
from .code_residual_stage_v3 import ExpandedReplayEncoder
from .codenet_checkpoint_v1 import FileBindings
from .codenet_runner_v1 import bind_evaluations
from .codenet_stage_v1 import exposure_budget
from .codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from .corpus import file_hash
from .diverse_images import atomic_json
from .model import ModelConfig
from .evaluation_provenance import capture
from .training import load_checkpoint


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
FORMAT = 'own-complete-code-residual-production-plan-v4'
MODEL_COUNTS = dict(total_unique=857007616, frozen_base=706012672, trainable_code=150994944)
DONE = dict(main='extended language and multimodal training/evaluation completed; broader capability unverified',
    joint='human dialogue retention training/evaluation completed; broader capability unverified',
    feedback='human-feedback pilot training/evaluation completed; broader goal unverified')
RUNTIME_FILES = {
    'code_residual_production_v4.py', 'code_residual_lifecycle_v4.py', 'code_residual_source_contract_v4.py',
    'code_residual_stage_v3.py', 'code_residual_stage_v2.py', 'code_residual_branch_v1.py',
    'code_residual_collective_checkpoint_v2.py', 'code_residual_checkpoint_v1.py', 'code_residual_export_v2.py',
    'codenet_checkpoint_v1.py', 'codenet_runner_v1.py', 'codenet_stage_v1.py', 'codenet_token_pack_v1.py',
    'codenet_replay_schedule_v1.py', 'codenet_replay_v1.py', 'codenet_fact_replay_v1.py',
    'codenet_dialogue_replay_v1.py', 'codenet_image_replay_v1.py', 'codenet_later_instruction_replay_v2.py',
    'codenet_packed_replay_v1.py', 'codenet_paired_photo_replay_v1.py', 'model.py', 'training.py', 'corpus.py',
    'diverse_images.py', 'recurrent_ablation.py', 'evaluation_provenance.py', 'vqa_pair_evaluation.py'}
PATHS = ('base_checkpoint', 'tokenizer', 'code_pack', 'source_bundle', 'final_base_binding',
    'capacity_proof', 'evaluation_binding', 'output')
HASHES = ('base_checkpoint', 'tokenizer', 'code_manifest', 'code_admission', 'source_bundle_manifest',
    'final_base_binding', 'capacity_proof', 'evaluation_binding')


def validate_production_plan(plan):
    validate_options(plan['training_options'])
    options = plan['training_options']
    if (plan.get('format') != FORMAT or plan.get('world_size') != 4 or plan.get('residual_width') != 2048
            or plan.get('expected_model_counts') != MODEL_COUNTS or options['image_size'] != 448
            or plan.get('training_ready') is not True or plan.get('automatic_candidate_promotion') is not False
            or plan.get('gradient_reduction') != 'fixed_parameter_order_worker_average_v1'
            or plan.get('replay_selection') != 'full_pass_floor_then_uniform_extra_family_updates_v1'
            or not _digest(plan.get('replay_contract_fingerprint'))):
        raise ValueError('Complete exact own857M/four-worker/full-source ready production plan required')
    if any(not isinstance(plan.get(key), str) or not Path(plan[key]).is_absolute() for key in PATHS):
        raise ValueError('Absolute immutable production source/output paths required')
    if any(not _digest(plan.get(key + '_sha256')) for key in HASHES):
        raise ValueError('Exact production source, final-base, capacity and baseline hashes required')
    sources = plan.get('source_files_sha256')
    if (not isinstance(sources, dict) or not {str(PACKAGE / name) for name in RUNTIME_FILES} <= set(sources)
            or any(not Path(name).is_absolute() or not _digest(digest) for name, digest in sources.items())):
        raise ValueError('Complete exact neural/objective/data/runtime source binding required')
    storage = plan['storage']
    if (set(storage) != {'maximum_checkpoint_bytes', 'maximum_inference_bytes', 'maximum_report_bytes', 'minimum_free_bytes'}
            or any(type(value) is not int or value < 1 for value in storage.values())
            or storage['maximum_checkpoint_bytes'] < 2 * 1024**3
            or storage['maximum_inference_bytes'] < 768 * 1024**2
            or storage['maximum_report_bytes'] < 512 * 1024**2
            or storage['minimum_free_bytes'] < 26 * 1024**3):
        raise ValueError('Three complete2GiB slots,768MiB inference,512MiB reports and26GiB reserve required')
    actors = plan.get('preceding_processes')
    if (not isinstance(actors, list) or {item.get('role') for item in actors} != {'main', 'joint', 'feedback', 'resumer'}
            or len(actors) != 4 or any(type(item.get('pid')) is not int or item['pid'] < 1
                or not isinstance(item.get('start_ticks'), str) or not item['start_ticks'].isdigit() for item in actors)):
        raise ValueError('Exact original main/joint/feedback/resumer process birth identities required')
    return options


def _read_bound(path, digest):
    path = Path(path)
    if file_hash(path) != digest:
        raise ValueError('Bound completed production evidence changed: ' + str(path))
    return json.loads(path.read_text())


def _merge(bindings, added):
    for name, digest in added.items():
        key = str(Path(name))
        if not Path(key).is_absolute() or not _digest(digest) or (key in bindings and bindings[key] != digest):
            raise ValueError('Conflicting or invalid exact production evidence binding: ' + key)
        bindings[key] = digest


def bind_completed_inputs(plan):
    """Only return after physical source/coverage/baseline evidence is bound."""
    bundle_path = Path(plan['source_bundle']) / 'manifest.json'
    bundle = _read_bound(bundle_path, plan['source_bundle_manifest_sha256'])
    if (bundle.get('format') != 'own-complete-code-final-source-bundle-v4'
            or bundle.get('all30_actual_prepared_families_opened') is not True
            or bundle.get('full_original_paired_admission_completed') is not True
            or bundle.get('every_prepared_logical_record_reached') is not True
            or bundle.get('replay_contract_fingerprint') != plan['replay_contract_fingerprint']
            or bundle.get('tokenizer_sha256') != plan['tokenizer_sha256']
            or bundle.get('code_manifest_sha256') != plan['code_manifest_sha256']
            or bundle.get('code_admission_sha256') != plan['code_admission_sha256']):
        raise ValueError('Completed actual30-family source bundle and full draw coverage required')
    bindings = {}
    _merge(bindings, plan['source_files_sha256'])
    _merge(bindings, bundle['source_files_sha256'])
    _merge(bindings, {str(bundle_path): plan['source_bundle_manifest_sha256']})
    for name, digest in bundle['artifact_files_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Owned source artifact basenames required')
        _merge(bindings, {str(Path(plan['source_bundle']) / name): digest})
    for key in ('base_checkpoint', 'tokenizer', 'final_base_binding', 'capacity_proof', 'evaluation_binding'):
        _merge(bindings, {plan[key]: plan[key + '_sha256']})
    _merge(bindings, {str(Path(plan['code_pack']) / 'manifest.json'): plan['code_manifest_sha256'],
        str(Path(plan['code_pack']) / 'admission.json'): plan['code_admission_sha256']})
    binding = _read_bound(plan['final_base_binding'], plan['final_base_binding_sha256'])
    if (binding.get('format') != 'own-complete-code-final-base-binding-v4'
            or binding.get('base_checkpoint_sha256') != plan['base_checkpoint_sha256']
            or binding.get('tokenizer_sha256') != plan['tokenizer_sha256']
            or binding.get('source_bundle_manifest_sha256') != plan['source_bundle_manifest_sha256']
            or binding.get('replay_contract_fingerprint') != plan['replay_contract_fingerprint']
            or binding.get('final_checkpoint_replay_match_verified') is not True
            or binding.get('all_later_queued_sources_covered') is not True
            or set(binding.get('completion_status_files', {})) != set(DONE)
            or not binding.get('files_sha256')):
        raise ValueError('Exact final own base/all-later source correspondence evidence required')
    for role, path in binding['completion_status_files'].items():
        if path not in binding['files_sha256'] or json.loads(Path(path).read_text()).get('status') != DONE[role]:
            raise ValueError('Original training/evaluation/feedback predecessor is incomplete: ' + role)
    selection_path = binding['selection_file']
    if selection_path not in binding['files_sha256']:
        raise ValueError('Exact actual guarded feedback selection report required')
    selection = json.loads(Path(selection_path).read_text())
    if selection.get('selected_checkpoint_sha256') != plan['base_checkpoint_sha256']:
        raise ValueError('Starting base must match actual final guarded selection')
    coverage_path = binding['coverage_report']
    if coverage_path not in binding['files_sha256']:
        raise ValueError('Exact independent final-policy replay coverage report required')
    coverage = json.loads(Path(coverage_path).read_text())
    if (coverage.get('matched_checkpoint_sha256') != plan['base_checkpoint_sha256']
            or coverage.get('source_bundle_manifest_sha256') != plan['source_bundle_manifest_sha256']
            or coverage.get('final_checkpoint_replay_match_verified') is not True
            or coverage.get('source_rows_dropped_or_duplicated') != 0 or coverage.get('encoder_truncated_sequences') != 0):
        raise ValueError('Complete final-policy original-source correspondence and unclipped replay required')
    _merge(bindings, binding['files_sha256'])
    _merge(bindings, bind_evaluations(plan))
    capacity = _read_bound(plan['capacity_proof'], plan['capacity_proof_sha256'])
    if (capacity.get('format') != 'own-complete-code-residual-capacity-proof-v4' or capacity.get('passed') is not True
            or capacity.get('base_checkpoint_sha256') != plan['base_checkpoint_sha256']
            or capacity.get('replay_contract_fingerprint') != plan['replay_contract_fingerprint']
            or capacity.get('kernel_source_sha256') != plan['source_files_sha256'][str(PACKAGE / 'code_residual_source_contract_v4.py')]
            or capacity.get('training_options_fingerprint') != fingerprint(plan['training_options'])
            or capacity.get('model_counts') != MODEL_COUNTS or capacity.get('actual_world_size') != 4
            or capacity.get('actual_full_kernel_optimizer_updates', 0) < 1
            or capacity.get('actual_maximum_context_backward') != 4096
            or capacity.get('actual448px_image_backward') is not True or not capacity.get('files_sha256')
            or capacity.get('runtime_provenance') != capture([PACKAGE / name for name in sorted(RUNTIME_FILES)])):
        raise ValueError('Actual bound four-L40/full857M kernel and maximum-context/image capacity proof required')
    _merge(bindings, capacity['files_sha256'])
    for name, digest in bindings.items():
        if not Path(name).is_absolute() or not _digest(digest):
            raise ValueError('Absolute exact transitive production evidence bindings required')
    return bundle, FileBindings(bindings), bindings


def _process(pid, ticks):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[0] != 'Z' and fields[19] == ticks


class ControllerBoundGuard:
    def __init__(self, files, owner):
        self.files, self.owner = files, owner

    def check(self, full=False):
        self.files.check(full=full)
        if (not _process(self.owner['pid'], self.owner['start_ticks'])
                or file_hash(Path('/proc') / str(self.owner['pid']) / 'cmdline') != self.owner['parent_cmdline_sha256']):
            raise ValueError('Exact checked parent controller ended or changed')


def finish_inference(output, plan, plan_path, result):
    """Finish/recheck a completed export without overwriting unexplained artifacts."""
    output, plan_path = Path(output), Path(plan_path)
    complete, inference = output / 'resume.pt', output / 'inference.pt'
    latest = result['latest_checkpoint']
    if (result.get('training_complete') is not True or result.get('completed_steps') != result.get('planned_steps')
            or latest.get('steps') != result['completed_steps'] or file_hash(complete) != latest['checkpoint_sha256']):
        raise ValueError('Exact complete journaled training state required before export/reuse')
    if inference.exists():
        digest = file_hash(inference)
        saved = torch.load(complete, weights_only=True, mmap=True)
        exported = torch.load(inference, weights_only=True, mmap=True)
        if fingerprint(exported.get('residuals')) != fingerprint(saved['residuals']):
            raise ValueError('Existing inference weights differ from the complete trained state; inspect without overwriting')
        load_inference(inference, plan['base_checkpoint'], plan['tokenizer'], expected_checkpoint_sha256=digest,
            expected_base_sha256=plan['base_checkpoint_sha256'], expected_plan_sha256=file_hash(plan_path),
            expected_training_checkpoint_sha256=latest['checkpoint_sha256'], expected_data_sha256=plan['code_manifest_sha256'])
        receipt = dict(checkpoint=str(inference), checkpoint_sha256=digest, checkpoint_bytes=inference.stat().st_size,
            complete_training_checkpoint_preserved_sha256=file_hash(complete),
            base_checkpoint_sha256=plan['base_checkpoint_sha256'], data_sha256=plan['code_manifest_sha256'],
            run_plan_sha256=file_hash(plan_path), optimizer_or_rng_included=False)
    else:
        receipt = export_inference(complete, plan['base_checkpoint'], plan['tokenizer'], plan_path,
            Path(plan['code_pack']) / 'manifest.json', inference,
            expected_training_checkpoint_sha256=latest['checkpoint_sha256'],
            maximum_bytes=plan['storage']['maximum_inference_bytes'], minimum_free_bytes=plan['storage']['minimum_free_bytes'])
    path = output / 'training-completed.json'
    if path.exists():
        prior = json.loads(path.read_text())
        if (prior.get('latest_checkpoint') != latest or prior.get('inference') != receipt
                or prior.get('training_complete') is not True or prior.get('automatic_candidate_promotion') is not False):
            raise ValueError('Existing completion report differs; inspect without overwriting')
    else:
        report_json(path, dict(completed_at=datetime.now(timezone.utc).isoformat(), **result, inference=receipt,
            code_route_explicit=True, automatic_candidate_promotion=False,
            matched_capability_evaluations_pending=True, high_end_capability_established=False),
            plan['storage']['maximum_report_bytes'])
    return receipt


def controller(plan_path, resume_sha256=None):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    validate_production_plan(plan)
    bundle, guard, bindings = bind_completed_inputs(plan)
    for actor in plan['preceding_processes']:
        if _process(actor['pid'], actor['start_ticks']):
            raise ValueError('Original predecessor remains live; no overlapping GPU stage: ' + actor['role'])
    output = Path(plan['output']).resolve()
    if (not output.is_relative_to(ROOT / 'checkpoints')
            or any(Path(name).resolve().is_relative_to(output) for name in bindings)
            or output in {Path(plan[k]).resolve() for k in PATHS if k != 'output'}):
        raise ValueError('New distinct owned checkpoint output directory required')
    if resume_sha256 is None and output.exists() and any(output.iterdir()):
        raise ValueError('New output or explicitly hashed same-plan resume required')
    storage = plan['storage']
    commitment = storage['minimum_free_bytes'] + 3 * storage['maximum_checkpoint_bytes'] + storage['maximum_inference_bytes'] + storage['maximum_report_bytes']
    if shutil.disk_usage(output.parent if output.parent.exists() else ROOT).free < commitment:
        raise ValueError('Complete latest/previous/pending/inference/report storage commitment is unfunded')
    if resume_sha256 is not None:
        if not _digest(resume_sha256):
            raise ValueError('Explicit exact complete resume hash required')
        ledger = checkpoint_layout(output, file_hash(plan_path))
        if ledger['latest'] is None or ledger['latest']['checkpoint_sha256'] != resume_sha256:
            raise ValueError('Explicit resume must match the exact journaled latest complete checkpoint')
    lock_path = ROOT / 'reports/code-residual-production-v4.gpu.lock'
    with lock_path.open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if resume_sha256 is None and output.exists() and any(output.iterdir()):
            raise ValueError('Output became occupied before the GPU lease; use a new output or explicit same-plan resume')
        memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True).stdout.splitlines()
        if len(memory) != 4 or any(int(value.strip()) > 128 for value in memory):
            raise ValueError('All four reserved GPUs must be idle before the controller launches workers')
        guard.check(full=True)
        output.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        ticks = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()[19]
        owner_path = output / ('controller-' + str(pid) + '-' + ticks + '.json')
        owner = dict(pid=pid, start_ticks=ticks, plan_sha256=file_hash(plan_path),
            parent_argv=sys.argv, parent_cmdline_sha256=file_hash(Path('/proc') / str(pid) / 'cmdline'),
            gpu_lease=str(lock_path), launched_at=datetime.now(timezone.utc).isoformat())
        report_json(owner_path, owner, storage['maximum_report_bytes'])
        argv = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4',
            '-m', 'slm_lab.code_residual_production_v4', '--worker', '--plan', str(plan_path), '--owner', str(owner_path)]
        if resume_sha256 is not None:
            argv += ['--resume-sha256', resume_sha256]
        report_json(output / 'status.json', dict(status='checked controller launching four coding workers',
            controller=owner, run_plan_sha256=file_hash(plan_path), goal_achieved=False), storage['maximum_report_bytes'])
        try:
            subprocess.run(argv, cwd=ROOT, check=True, env={**os.environ, 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
            guard.check(full=True)
            if not (output / 'training-completed.json').exists():
                raise ValueError('Workers ended without a completed training/export report')
            report_json(output / 'status.json', dict(status='coding training/export completed; matched capability evaluation pending',
                run_plan_sha256=file_hash(plan_path), candidate_promoted=False, goal_achieved=False), storage['maximum_report_bytes'])
        except Exception as error:
            report_json(output / 'status.json', dict(status='requires inspection; coding controller stopped',
                error=str(error), run_plan_sha256=file_hash(plan_path), automatic_restart=False,
                candidate_promoted=False, goal_achieved=False), storage['maximum_report_bytes'])
            raise


def worker(plan_path, owner_path, resume_sha256):
    plan_path, owner_path = Path(plan_path).resolve(), Path(owner_path).resolve()
    plan = json.loads(plan_path.read_text())
    options = validate_production_plan(plan)
    owner = json.loads(owner_path.read_text())
    if (owner.get('plan_sha256') != file_hash(plan_path) or not _process(owner['pid'], owner['start_ticks'])
            or file_hash(Path('/proc') / str(owner['pid']) / 'cmdline') != owner.get('parent_cmdline_sha256')):
        raise ValueError('Exact live checked parent controller required')
    if owner_path.parent != Path(plan['output']).resolve():
        raise ValueError('Owned current controller receipt required')
    rank, world, local = (int(os.environ[name]) for name in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 4 or not 0 <= rank < 4 or not 0 <= local < 4 or not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise ValueError('Exactly four actual visible CUDA workers required')
    bundle, initial, bindings = bind_completed_inputs(plan)
    bindings[str(plan_path)] = file_hash(plan_path)
    bindings[str(owner_path)] = file_hash(owner_path)
    code = CompleteCodePack(plan['code_pack'], 'train', purpose='training')
    replay, contract = open_final_source_replay(bundle['replay_folders'], plan['tokenizer_sha256'])
    if fingerprint(contract) != plan['replay_contract_fingerprint']:
        raise ValueError('All actual source manifests/objectives must match the immutable run plan')
    budget = exposure_budget(len(code), options['passes'], world, options['batch_size'], options['gradient_accumulation'])
    if (budget != bundle['prospective_code_budget'] or options['seed'] != bundle['prospective_seed']
            or world * options['replay_batch_size'] * options['replay_accumulation'] != bundle['prospective_global_replay_width']):
        raise ValueError('Actual code/replay draw layout must match the completed source-bundle audit')
    guard = ControllerBoundGuard(FileBindings(bindings), owner)
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise ValueError('Actual BF16 CUDA support required')
    torch.set_num_threads(1)
    random.seed(options['seed'] + rank)
    np.random.seed((options['seed'] + rank) % 2**32)
    torch.manual_seed(options['seed'] + rank)
    dist.init_process_group('nccl')
    try:
        base, metadata = load_checkpoint(plan['base_checkpoint'], device, training_state=False)
        if metadata['tokenizer_sha256'] != plan['tokenizer_sha256'] or base.config != ModelConfig.load(Path(plan['code_pack']) / 'model-config.json'):
            raise ValueError('Exact final own base/tokenizer/architecture required')
        # All residual workers must initialize identically; later per-rank RNG is
        # restored or independently seeded after construction.
        torch.manual_seed(options['seed'])
        model = RoutedCodeResidual(base, width=plan['residual_width'])
        if model.parameter_counts() != MODEL_COUNTS:
            raise ValueError('All own frozen/general/vision/critic/code parameters must match857M budget')
        optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=options['learning_rate'],
            weight_decay=options['weight_decay'], foreach=False)
        tokenizer = Tokenizer.from_file(plan['tokenizer'])
        from .codenet_fact_replay_v1 import CombinedReplay
        joint = replay
        while isinstance(joint, CombinedReplay):
            joint = joint.owners['human_instruction']
        encoder = ExpandedReplayEncoder(tokenizer, model.config, joint.source.parent, options['image_size'])
        def amp():
            return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        sources = dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
            run_plan_sha256=file_hash(plan_path))
        kernel = SourceContractResidualKernel(model, optimizer, code, packed_encoder(code, tokenizer, model.config),
            replay, encoder, options, sources, device, amp, replay_contract=contract,
            expected_contract_fingerprint=plan['replay_contract_fingerprint'], guard=guard)
        output, storage = Path(plan['output']), plan['storage']
        if resume_sha256 is not None:
            kernel.restore(output / 'resume.pt', resume_sha256, allow_completed=True)
        else:
            random.seed(options['seed'] + rank)
            np.random.seed((options['seed'] + rank) % 2**32)
            torch.manual_seed(options['seed'] + rank)
        result = run_updates(kernel, output, maximum_checkpoint_bytes=storage['maximum_checkpoint_bytes'],
            minimum_free_bytes=storage['minimum_free_bytes'], checkpoint_every=options['checkpoint_every'], log_every=options['log_every'],
            maximum_report_bytes=storage['maximum_report_bytes'])
        if not result['training_complete']:
            raise ValueError('Production worker did not finish the complete immutable budget')
        def finish():
            guard.check(full=True)
            return finish_inference(output, plan, plan_path, result)
        rank_zero(finish)
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--resume-sha256')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--owner')
    args = parser.parse_args()
    if args.worker:
        if not args.owner:
            raise ValueError('Workers require the exact live checked controller receipt')
        worker(args.plan, args.owner, args.resume_sha256)
    else:
        if args.owner:
            raise ValueError('Only worker subprocesses accept a controller receipt')
        controller(args.plan, args.resume_sha256)


if __name__ == '__main__':
    main()
