"""Explicit-plan four-L40 capacity and training for original paired visual data.

Admission and completed own-base evidence are checked before CUDA inspection.
Capacity weights are discarded. Production also requires completed native
baselines and the exact capacity proof; neither mode promotes a candidate.
"""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np
from tokenizers import Tokenizer
import torch
from torch import distributed as dist

from .alignment_selection import validate_collection
from .code_residual_collective_checkpoint_v2 import _digest, fingerprint
from .code_residual_lifecycle_v4 import checkpoint_layout, rank_zero, report_json
from .code_residual_production_v4 import ControllerBoundGuard, DONE, _merge, _process, _read_bound
from .codenet_checkpoint_v1 import FileBindings
from .codenet_runner_v1 import bind_evaluations, VerifiedReplayEncoder
from .corpus import file_hash
from .finalized_visual_pairs_v1 import FinalizedVisualPairs
from .finalized_visual_replay_v1 import FinalizedVisualReplay, VISUAL_FAMILIES
from .model import ModelConfig
from .paired_visual_capacity_v1 import select_extremes, exercise_capacity
from .paired_visual_lifecycle_v1 import run_visual_stage, finish_visual_export
from .paired_visual_training_v1 import PairedVisualKernel, freeze_for_visual_stage, INTEGERS, FLOATS, RUNTIME
from .training import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
FORMAT = 'own-paired-visual-production-plan-v1'
CAPACITY_FORMAT = 'own-paired-visual-capacity-plan-v1'
PROOF_FORMAT = 'own-paired-visual-capacity-proof-v1'
MODEL_COUNTS = dict(total=706012672, vision=51624448, language_and_critic=654388224, inference_without_critic=706011136)
RUNTIME_FILES = tuple(sorted(set(RUNTIME) | {'paired_visual_production_v1', 'paired_visual_capacity_v1',
    'paired_visual_lifecycle_v1', 'code_residual_lifecycle_v4', 'code_residual_production_v4',
    'alignment_selection', 'evaluation_provenance'}))


def validate_plan(plan, *, capacity=False):
    opts = plan['training_options']
    if (plan.get('format') != (CAPACITY_FORMAT if capacity else FORMAT) or plan.get('world_size') != 4
            or plan.get('expected_model_counts') != MODEL_COUNTS or plan.get('automatic_candidate_promotion') is not False
            or (capacity and plan.get('candidate_export') is not False)
            or (not capacity and plan.get('training_ready') is not True)
            or set(opts) != INTEGERS | FLOATS
            or any(type(opts[k]) is not int or opts[k] < (0 if k == 'seed' else 1) for k in INTEGERS)
            or any(type(opts[k]) not in {int, float} or not math.isfinite(opts[k]) for k in FLOATS)
            or any(opts[k] <= 0 for k in ['learning_rate', 'gradient_clip', 'beta', 'replay_coefficient'])
            or opts['weight_decay'] < 0 or not 0 <= opts['alpha'] <= 1
            or not 0 < opts['beta'] <= 100 or not 0 <= opts['margin'] <= 10 or opts['image_size'] != 448):
        raise ValueError('Exact own706M/four-worker/full448px explicit visual plan required')
    paths = ['base_checkpoint', 'tokenizer', 'code_pack', 'source_bundle', 'final_base_binding', 'output']
    hashes = ['base_checkpoint', 'tokenizer', 'code_manifest', 'code_admission', 'source_bundle_manifest',
        'final_base_binding', 'paired_manifest']
    if not capacity:
        paths += ['capacity_proof', 'evaluation_binding', 'visual_evaluation_binding']
        hashes += ['capacity_proof', 'evaluation_binding', 'visual_evaluation_binding']
    if (any(not isinstance(plan.get(k), str) or not Path(plan[k]).is_absolute() for k in paths)
            or any(not _digest(plan.get(k + '_sha256')) for k in hashes)
            or any(not _digest(plan.get(k)) for k in ['replay_contract_fingerprint', 'pair_contract_sha256', 'visual_replay_contract_sha256'])):
        raise ValueError('Absolute exact own base/tokenizer/final-source/proof paths and contracts required')
    files = plan.get('source_files_sha256', {})
    if (not isinstance(files, dict) or not {str(PACKAGE / (n + '.py')) for n in RUNTIME_FILES} <= files.keys()
            or any(not Path(p).is_absolute() or not _digest(h) for p, h in files.items())):
        raise ValueError('Complete exact visual controller/kernel/source/lifecycle runtime binding required')
    actors = plan.get('preceding_processes', [])
    if (len(actors) != 4 or {a.get('role') for a in actors} != {'main', 'joint', 'feedback', 'resumer'}
            or any(type(a.get('pid')) is not int or a['pid'] < 1 or not isinstance(a.get('start_ticks'), str)
                or not a['start_ticks'].isdigit() for a in actors)):
        raise ValueError('All original predecessor process birth identities required')
    storage = plan.get('storage', {})
    if (set(storage) != {'maximum_checkpoint_bytes', 'maximum_inference_bytes', 'maximum_report_bytes', 'minimum_free_bytes'}
            or any(type(v) is not int or v < 1 for v in storage.values())
            or storage['maximum_checkpoint_bytes'] < 4 * 1024**3 or storage['maximum_inference_bytes'] < 3 * 1024**3
            or storage['maximum_report_bytes'] < 512 * 1024**2 or storage['minimum_free_bytes'] < 26 * 1024**3):
        raise ValueError('Fund three full4GiB checkpoint slots,3GiB export,512MiB reports and26GiB reserve')
    if any(type(plan.get(k)) is not int or plan[k] < 1 for k in ['checkpoint_every', 'log_every']):
        raise ValueError('Explicit positive checkpoint/logging intervals required')
    return opts


def bind_final_source_base(plan):
    """Use the existing canonical full-source/final-policy evidence formats."""
    path = Path(plan['source_bundle']) / 'manifest.json'
    bundle = _read_bound(path, plan['source_bundle_manifest_sha256'])
    if (bundle.get('format') != 'own-complete-code-final-source-bundle-v4'
            or any(bundle.get(k) is not True for k in ['all30_actual_prepared_families_opened',
                'full_original_paired_admission_completed', 'every_prepared_logical_record_reached'])
            or any(bundle.get(k) != plan[k] for k in ['tokenizer_sha256', 'code_manifest_sha256',
                'code_admission_sha256', 'replay_contract_fingerprint'])):
        raise ValueError('Actual complete finalized30-family source bundle required')
    files = {}
    for added in [plan['source_files_sha256'], bundle['source_files_sha256'], {str(path): plan['source_bundle_manifest_sha256']}]:
        _merge(files, added)
    for name, digest in bundle['artifact_files_sha256'].items():
        if Path(name).name != name:
            raise ValueError('Owned source artifact basenames required')
        _merge(files, {str(path.parent / name): digest})
    for key in ['base_checkpoint', 'tokenizer', 'final_base_binding']:
        _merge(files, {plan[key]: plan[key + '_sha256']})
    for name in ['manifest', 'admission']:
        _merge(files, {str(Path(plan['code_pack']) / (name + '.json')): plan['code_' + name + '_sha256']})
    binding = _read_bound(plan['final_base_binding'], plan['final_base_binding_sha256'])
    if (binding.get('format') != 'own-complete-code-final-base-binding-v4'
            or any(binding.get(k) != plan[k] for k in ['base_checkpoint_sha256', 'tokenizer_sha256',
                'source_bundle_manifest_sha256', 'replay_contract_fingerprint'])
            or any(binding.get(k) is not True for k in ['final_checkpoint_replay_match_verified', 'all_later_queued_sources_covered'])
            or set(binding.get('completion_status_files', {})) != set(DONE)):
        raise ValueError('Completed final own policy and all-later-source correspondence required')
    evidence = binding['files_sha256']; _merge(files, evidence)
    for role, report in binding['completion_status_files'].items():
        if report not in evidence or _read_bound(report, evidence[report]).get('status') != DONE[role]:
            raise ValueError('Original training/evaluation/feedback is incomplete: ' + role)
    selection, coverage = binding['selection_file'], binding['coverage_report']
    if selection not in evidence or coverage not in evidence:
        raise ValueError('Actual guarded final selection and independent coverage evidence required')
    if _read_bound(selection, evidence[selection]).get('selected_checkpoint_sha256') != plan['base_checkpoint_sha256']:
        raise ValueError('Visual base must match the actual final guarded selection')
    cov = _read_bound(coverage, evidence[coverage])
    if (cov.get('matched_checkpoint_sha256') != plan['base_checkpoint_sha256']
            or cov.get('source_bundle_manifest_sha256') != plan['source_bundle_manifest_sha256']
            or cov.get('final_checkpoint_replay_match_verified') is not True
            or cov.get('source_rows_dropped_or_duplicated') != 0 or cov.get('encoder_truncated_sequences') != 0):
        raise ValueError('Complete exact unclipped final-policy/source correspondence required')
    return bundle, files


def bind_visual_baselines(plan):
    binding = _read_bound(plan['visual_evaluation_binding'], plan['visual_evaluation_binding_sha256'])
    if (binding.get('format') != 'own-paired-visual-baseline-binding-v1'
            or binding.get('baseline_checkpoint_sha256') != plan['base_checkpoint_sha256']
            or binding.get('baseline_completed') is not True or set(binding.get('reports', {})) != {'vision', 'ocr', 'objects'}):
        raise ValueError('Completed exact-own-policy caption/OCR/object baselines required')
    files = {plan['visual_evaluation_binding']: plan['visual_evaluation_binding_sha256']}
    _merge(files, binding['files_sha256'])
    reports = {}
    datasets = {'vision': 'data/alignment-curriculum-v1/contrastive.valid.jsonl',
        'ocr': 'data/alignment-curriculum-v1/images.valid.jsonl', 'objects': 'data/object-qa-v1/valid.jsonl'}
    for name, path in binding['reports'].items():
        if path not in files:
            raise ValueError('Every original visual baseline report must be byte-bound')
        reports[name] = _read_bound(path, files[path])
        data = ROOT / datasets[name]
        if reports[name].get('data_sha256') != file_hash(data):
            raise ValueError('Original native visual baseline input bytes differ')
        _merge(files, {str(data): reports[name]['data_sha256']})
        for source, digest in reports[name]['evaluation_provenance']['files_sha256'].items():
            _merge(files, {str((ROOT / source).resolve()): digest})
    if validate_collection(reports, ['vision', 'ocr', 'objects']) != plan['base_checkpoint_sha256']:
        raise ValueError('Visual baselines compare the wrong own policy')
    for name, field, minimum in [('vision', 'samples', 128), ('ocr', 'results', 170), ('objects', 'results', 164)]:
        report = reports[name]
        if (len(report[field]) < minimum or report.get('tokenizer_sha256') != plan['tokenizer_sha256']
                or report.get('image_size') != 448 or len({row['id'] for row in report[field]}) != len(report[field])):
            raise ValueError('Complete original448px visual baseline and exact own tokenizer required')
    return files


def validate_capacity_proof(proof, plan, extremes):
    expected = dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
        pair_contract_sha256=plan['pair_contract_sha256'], visual_replay_contract_sha256=plan['visual_replay_contract_sha256'],
        training_options_fingerprint=fingerprint(plan['training_options']),
        runtime_files_sha256={str(PACKAGE / (n + '.py')): file_hash(PACKAGE / (n + '.py')) for n in RUNTIME_FILES})
    if (proof.get('format') != PROOF_FORMAT or proof.get('passed') is not True or proof.get('actual_world_size') != 4
            or proof.get('model_counts') != MODEL_COUNTS or proof.get('source_extremes') != extremes
            or any(proof.get(k) != v for k, v in expected.items()) or proof.get('candidate_weights_saved') is not False
            or set(proof.get('rank_reports', {})) != {'0', '1', '2', '3'} or not proof.get('files_sha256')):
        raise ValueError('Actual exact-source full706M/four-L40 visual capacity proof required')
    for rank, path in proof['rank_reports'].items():
        if path not in proof['files_sha256']:
            raise ValueError('All actual capacity rank reports must be bound')
        r = _read_bound(path, proof['files_sha256'][path])
        if (type(r.get('rank')) is not int or r['rank'] != int(rank) or r.get('device_index') != int(rank)
                or r.get('device_name') != 'NVIDIA L40' or r.get('precision') != 'bf16'
                or r.get('model_counts') != MODEL_COUNTS or r.get('actual_full_kernel_optimizer_updates') != 1
                or set(r.get('all_eight_visual_families_exercised', [])) != set(VISUAL_FAMILIES)
                or r.get('actual_max_paired_context') != extremes['max_paired_context']
                or r.get('actual_max_paired_total_targets') != extremes['max_paired_total_targets']
                or r.get('actual_replay_maximum_contexts') != extremes['replay_maximum_contexts']
                or (extremes['high_consensus_binary_exists'] and r.get('high_consensus_binary_contrast_exercised') is not True)
                or r.get('original_language_critic_unchanged') is not True or r.get('vision_parameters_changed') is not True
                or r.get('candidate_weights_saved') is not False or r.get('image_size') != 448
                or type(r.get('peak_reserved_bytes')) is not int or not 0 < r['peak_reserved_bytes'] < 46 * 1024**3):
            raise ValueError('CPU fixtures or incomplete source extremes cannot certify full706M capacity')


def bind_inputs(plan, *, capacity=False):
    validate_plan(plan, capacity=capacity)
    bundle, files = bind_final_source_base(plan)
    token = ROOT / 'data/tokenizer-corpus/tokenizer.json'
    if file_hash(token) != plan['tokenizer_sha256']:
        raise ValueError('Exact original own tokenizer bytes required')
    tokenizer = Tokenizer.from_file(plan['tokenizer']); config = ModelConfig.load(ROOT / 'configs/model.json')
    paired_manifest = ROOT / 'data/vqa-real-pairs-v1/manifest.json'
    _merge(files, {str(paired_manifest): plan['paired_manifest_sha256']})
    guard = FileBindings(files)
    pairs = FinalizedVisualPairs(ROOT, tokenizer, config, 448)
    replay = FinalizedVisualReplay(ROOT, tokenizer, config, 448)
    if pairs.contract_sha256 != plan['pair_contract_sha256'] or replay.contract_sha256 != plan['visual_replay_contract_sha256']:
        raise ValueError('Exact finalized native paired/eight-family visual contracts required')
    if pairs.contract['image_size'] != 448 or replay.contract['image_size'] != 448:
        raise ValueError('Both native source interfaces must retain actual448px images')
    pair_positions, replay_positions, extremes = select_extremes(pairs, replay, bundle)
    if not capacity:
        _merge(files, bind_evaluations(plan))
        _merge(files, bind_visual_baselines(plan))
        proof = _read_bound(plan['capacity_proof'], plan['capacity_proof_sha256'])
        validate_capacity_proof(proof, plan, extremes)
        _merge(files, {plan['capacity_proof']: plan['capacity_proof_sha256']})
        _merge(files, proof['files_sha256'])
        guard = FileBindings(files)
    guard.check(full=True)
    return bundle, pairs, replay, (pair_positions, replay_positions, extremes), guard, files


def _plan(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def controller(plan_path, *, capacity=False, resume_sha256=None):
    plan_path = Path(plan_path).resolve(); plan, digest = _plan(plan_path)
    _, _, _, _, _, files = bind_inputs(plan, capacity=capacity)
    _merge(files, {str(plan_path): digest}); guard = FileBindings(files)
    if capacity and resume_sha256 is not None:
        raise ValueError('Disposable capacity weights cannot resume or become a candidate')
    output, storage = Path(plan['output']).resolve(), plan['storage']
    parent = ROOT / ('reports' if capacity else 'checkpoints')
    if output == parent or not output.is_relative_to(parent) or any(Path(p).resolve().is_relative_to(output) for p in files):
        raise ValueError('A distinct owned visual output is required')
    def available():
        for actor in plan['preceding_processes']:
            if _process(actor['pid'], actor['start_ticks']):
                raise ValueError('Original GPU predecessor remains live: ' + actor['role'])
        if resume_sha256 is None and output.exists() and any(output.iterdir()):
            raise ValueError('Visual output became occupied; inspect or explicitly resume')
        if resume_sha256 is not None:
            ledger = checkpoint_layout(output, digest)
            if not _digest(resume_sha256) or ledger['latest'] is None or ledger['latest']['checkpoint_sha256'] != resume_sha256:
                raise ValueError('Explicit visual resume must match the latest complete journaled hash')
        budget = storage['minimum_free_bytes'] + storage['maximum_report_bytes']
        if not capacity:
            budget += 3 * storage['maximum_checkpoint_bytes'] + storage['maximum_inference_bytes']
        if shutil.disk_usage(ROOT).free < budget:
            raise ValueError('Fund all complete visual checkpoints/export/reports above the original reserve')
    available()
    with (ROOT / 'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        available()
        memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True, capture_output=True, check=True).stdout.splitlines()
        if len(memory) != 4 or any(int(v.strip()) > 128 for v in memory):
            raise ValueError('All four L40s must be idle before visual capacity/training launch')
        guard.check(full=True); output.mkdir(parents=True, exist_ok=True)
        pid = os.getpid(); proc = Path('/proc') / str(pid)
        owner = dict(pid=pid, start_ticks=(proc / 'stat').read_text().split(') ', 1)[1].split()[19],
            parent_cmdline_sha256=file_hash(proc / 'cmdline'), plan_sha256=digest, capacity=capacity)
        owner_path = output / ('visual-owner-' + str(pid) + '.json')
        report_json(owner_path, owner, storage['maximum_report_bytes'])
        argv = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4', '-m',
            'slm_lab.paired_visual_production_v1', '--worker', '--plan', str(plan_path), '--owner', str(owner_path)]
        if capacity: argv += ['--capacity']
        if resume_sha256 is not None: argv += ['--resume-sha256', resume_sha256]
        try:
            subprocess.run(argv, cwd=ROOT, check=True, env={**os.environ, 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
            guard.check(full=True)
            result = json.loads((output / ('proof.json' if capacity else 'training-completed.json')).read_text())
            if result.get('passed' if capacity else 'training_complete') is not True:
                raise ValueError('Visual workers ended without a complete result')
        except Exception as error:
            report_json(output / ('failure-' + str(pid) + '.json'), dict(error=str(error),
                automatic_restart=False, candidate_promoted=False, goal_achieved=False), storage['maximum_report_bytes'])
            raise


def worker(plan_path, owner_path, *, capacity=False, resume_sha256=None):
    plan_path, owner_path = Path(plan_path).resolve(), Path(owner_path).resolve()
    plan, digest = _plan(plan_path); owner = json.loads(owner_path.read_text())
    if (owner_path.parent != Path(plan['output']).resolve() or owner.get('capacity') is not capacity
            or owner.get('plan_sha256') != digest or not _process(owner['pid'], owner['start_ticks'])
            or file_hash(Path('/proc') / str(owner['pid']) / 'cmdline') != owner.get('parent_cmdline_sha256')):
        raise ValueError('Exact live checked visual controller and owned receipt required')
    _, pairs, replay, selected, _, files = bind_inputs(plan, capacity=capacity)
    _merge(files, {str(plan_path): digest, str(owner_path): file_hash(owner_path)})
    guard = ControllerBoundGuard(FileBindings(files), owner)
    rank, world, local = [int(os.environ[k]) for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']]
    if world != 4 or not 0 <= rank < 4 or not 0 <= local < 4 or not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise ValueError('Four actual visible CUDA workers required')
    device = torch.device('cuda', local); torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != 'NVIDIA L40' or not torch.cuda.is_bf16_supported():
        raise ValueError('Actual BF16 L40 visual execution required')
    torch.set_num_threads(1); dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    try:
        opts, output, storage = plan['training_options'], Path(plan['output']), plan['storage']
        guard.check(full=True)
        model, meta = load_checkpoint(plan['base_checkpoint'], device, training_state=False)
        if (model.config != ModelConfig.load(ROOT / 'configs/model.json') or model.parameter_counts() != MODEL_COUNTS
                or meta['tokenizer_sha256'] != plan['tokenizer_sha256']):
            raise ValueError('Exact final own706M architecture and tokenizer required')
        optimizer = torch.optim.AdamW(freeze_for_visual_stage(model), lr=opts['learning_rate'], weight_decay=opts['weight_decay'], foreach=False)
        encoder = VerifiedReplayEncoder(Tokenizer.from_file(plan['tokenizer']), model.config, ROOT, 448)
        kernel = PairedVisualKernel(model, optimizer, pairs, replay, encoder, opts,
            dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'], run_plan_sha256=digest),
            device, lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16), guard=guard)
        random.seed(opts['seed'] + rank); np.random.seed((opts['seed'] + rank) % 2**32); torch.manual_seed(opts['seed'] + rank)
        if capacity:
            torch.cuda.reset_peak_memory_stats(device)
            pair_positions, replay_positions, extremes = selected
            proof = exercise_capacity(kernel, pair_positions, replay_positions)
            torch.cuda.synchronize(device)
            proof.update(rank=rank, device_index=local, device_name=torch.cuda.get_device_name(device), precision='bf16', image_size=448,
                model_counts=model.parameter_counts(), peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            ranks = [None] * 4; dist.all_gather_object(ranks, proof)
            def finish():
                guard.check(full=True)
                reports = {str(i): str(output / ('rank-' + str(i) + '.json')) for i in range(4)}
                for r in ranks: report_json(reports[str(r['rank'])], r, storage['maximum_report_bytes'])
                bound = {**files, **{path: file_hash(path) for path in reports.values()}}
                result = dict(format=PROOF_FORMAT, passed=True, completed_at=datetime.now(timezone.utc).isoformat(),
                    actual_world_size=4, model_counts=model.parameter_counts(), source_extremes=extremes,
                    **{k: plan[k] for k in ['base_checkpoint_sha256', 'tokenizer_sha256', 'pair_contract_sha256', 'visual_replay_contract_sha256']},
                    training_options_fingerprint=fingerprint(opts), runtime_files_sha256={str(PACKAGE / (n + '.py')): file_hash(PACKAGE / (n + '.py')) for n in RUNTIME_FILES},
                    rank_reports=reports, files_sha256=bound, candidate_weights_saved=False, goal_achieved=False)
                validate_capacity_proof(result, plan, extremes)
                report_json(output / 'proof.json', result, storage['maximum_report_bytes'])
            rank_zero(finish)
        else:
            result = run_visual_stage(kernel, output, resume_sha256=resume_sha256,
                maximum_checkpoint_bytes=storage['maximum_checkpoint_bytes'], minimum_free_bytes=storage['minimum_free_bytes'],
                maximum_report_bytes=storage['maximum_report_bytes'], checkpoint_every=plan['checkpoint_every'], log_every=plan['log_every'])
            finish_visual_export(kernel, output, result, maximum_inference_bytes=storage['maximum_inference_bytes'],
                minimum_free_bytes=storage['minimum_free_bytes'], maximum_report_bytes=storage['maximum_report_bytes'])
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--capacity', action='store_true')
    parser.add_argument('--resume-sha256'); parser.add_argument('--worker', action='store_true'); parser.add_argument('--owner')
    args = parser.parse_args()
    if args.worker:
        if not args.owner: parser.error('--worker requires --owner')
        worker(args.plan, args.owner, capacity=args.capacity, resume_sha256=args.resume_sha256)
    else:
        if args.owner: parser.error('--owner is only valid with --worker')
        controller(args.plan, capacity=args.capacity, resume_sha256=args.resume_sha256)


if __name__ == '__main__':
    main()
