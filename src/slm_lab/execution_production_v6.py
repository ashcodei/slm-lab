"""Final-source-bound four-L40 Java/Python execution-feedback capacity and production.

Capacity generates the frozen own-policy reference once and discards its probe
weights. Production binds that reference and BOTH the original v4 capacity proof
and the new combined-source auxiliary capacity proof. All admitted Java/Python views are required. The v4 final-source, final-base, baseline,
resource and genuine-feedback prerequisites remain mandatory.
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

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import distributed as dist

from . import code_residual_capacity_v4 as original_capacity
from . import code_residual_production_v4 as original
from .code_residual_branch_v1 import RoutedCodeResidual
from .code_residual_checkpoint_v1 import _digest
from .code_residual_collective_checkpoint_v2 import _same, _validated, fingerprint
from .code_residual_lifecycle_v4 import checkpoint_layout, rank_zero, report_json, run_updates
from .code_residual_source_contract_v4 import REQUIRED_FAMILIES, open_final_source_replay
from .code_residual_stage_v3 import ExpandedReplayEncoder, _EncodedBatches
from .codenet_checkpoint_v1 import FileBindings
from .codenet_fact_replay_v1 import CombinedReplay
from .codenet_runner_v1 import average_gradients
from .codenet_stage_v1 import exposure_budget
from .codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from .corpus import file_hash
from .evaluation_provenance import capture
from .execution_collective_v2 import (feedback_identity, prepare_reference_bundle,
    measure_preferences, backward_preferences)
from .execution_feedback_union_v1 import ExecutionFeedbackUnion
from .execution_training_v5 import (ExecutionResidualKernel, auxiliary_fingerprint,
    global_CE_gradients, reference_from_checkpoint, validate_execution_options)
from .model import ModelConfig
from .recurrent_ablation import state_digest
from .training import load_checkpoint

ROOT, PACKAGE = original.ROOT, original.PACKAGE
FORMAT = 'own-complete-code-execution-production-plan-v6'
CAPACITY_FORMAT = 'own-complete-code-execution-capacity-plan-v6'
PROOF_FORMAT = 'own-complete-code-execution-capacity-proof-v6'
RUNTIME_FILES = original.RUNTIME_FILES | {
    'code_residual_capacity_v4.py', 'execution_production_v6.py',
    'execution_collective_v2.py', 'execution_training_v5.py',
    'execution_feedback_reader_v1.py', 'execution_preference_v1.py',
    'execution_code_feedback_v1.py', 'execution_feedback_union_v1.py',
    'python_feedback_reader_v2.py', 'python_execution_feedback_v2.py', 'python_execution_feedback_v1.py'}


def validate_plan(plan, *, capacity=False):
    if plan.get('format') != (CAPACITY_FORMAT if capacity else FORMAT):
        raise ValueError('Explicit execution-feedback capacity or production plan required')
    projected = {**plan, 'format': original_capacity.FORMAT if capacity else original.FORMAT}
    if capacity:
        original_capacity.validate_plan(projected)
        if plan['storage']['maximum_report_bytes'] < 8 * 1024**2:
            raise ValueError('Bounded8MiB reference/probe report allowance required')
    else:
        original.validate_production_plan(projected)
    validate_execution_options(plan['execution_options'])
    if plan['execution_options']['DPO_alpha'] <= 0:
        raise ValueError('This execution-feedback experiment requires a positive DPO coefficient')
    paths = ['execution_feedback'] + ([] if capacity else ['execution_reference', 'execution_capacity_proof'])
    hashes = ['execution_manifest', 'execution_admission'] + ([] if capacity else ['execution_reference', 'execution_capacity_proof'])
    if (any(not isinstance(plan.get(k), str) or not Path(plan[k]).is_absolute() for k in paths)
            or any(not _digest(plan.get(k + '_sha256')) for k in hashes)
            or not {str(PACKAGE / n) for n in RUNTIME_FILES} <= plan['source_files_sha256'].keys()):
        raise ValueError('Exact admitted feedback and complete old/new runtime bindings required')
    if not capacity and not _digest(plan.get('execution_auxiliary_fingerprint')):
        raise ValueError('Exact original execution reference/options fingerprint required')


def validate_capacity_proof(proof, plan, reference, *, auxiliary_maximum_contexts):
    """Inspect actual rank reports as well as the summary and immutable hashes."""
    expected = dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'],
        tokenizer_sha256=plan['tokenizer_sha256'], replay_contract_fingerprint=plan['replay_contract_fingerprint'],
        execution_manifest_sha256=plan['execution_manifest_sha256'],
        execution_admission_sha256=plan['execution_admission_sha256'],
        training_options_fingerprint=fingerprint(plan['training_options']),
        execution_options_fingerprint=fingerprint(plan['execution_options']),
        execution_reference_sha256=plan['execution_reference_sha256'],
        execution_auxiliary_fingerprint=auxiliary_fingerprint(reference, plan['execution_options']),
        model_counts=original.MODEL_COUNTS, actual_world_size=4,
        runtime_provenance=capture([PACKAGE / name for name in sorted(RUNTIME_FILES)]))
    if (proof.get('format') != PROOF_FORMAT or proof.get('passed') is not True
            or any(proof.get(k) != value for k, value in expected.items())
            or proof.get('actual_reference_pairs_scored') != reference['feedback']['pairs']
            or proof.get('candidate_weights_saved') is not False
            or set(proof.get('rank_reports', {})) != {'0', '1', '2', '3'}
            or not proof.get('files_sha256')):
        raise ValueError('Actual exact four-L40/full857M execution capacity and frozen reference proof required')
    policy = reference['policy']
    if (policy.get('model_counts') != original.MODEL_COUNTS or policy.get('device_type') != 'cuda'
            or policy.get('precision') != 'bf16' or policy.get('base_checkpoint_sha256') != plan['base_checkpoint_sha256']
            or policy.get('tokenizer_sha256') != plan['tokenizer_sha256']):
        raise ValueError('CPU fixtures or a different own policy cannot certify actual857M capacity')
    for rank in range(4):
        path = proof['rank_reports'][str(rank)]
        if path not in proof['files_sha256']:
            raise ValueError('All four actual capacity rank reports must be byte-bound')
        item = original._read_bound(path, proof['files_sha256'][path])
        probes = item.get('execution_extreme_backwards', [])
        actual_maxima = {name: max((p.get(name + '_context_tokens', 0) for p in probes), default=0)
            for name in ['positive', 'chosen', 'rejected']}
        if (item.get('rank') != rank or item.get('device_name') != 'NVIDIA L40'
                or item.get('model_counts') != original.MODEL_COUNTS
                or item.get('actual_full_kernel_optimizer_updates', 0) < 1
                or item.get('actual_maximum_context_backward') != 4096
                or not item.get('actual_image_batch_sizes')
                or any(size != [448, 448] for size in item['actual_image_batch_sizes'])
                or not set(REQUIRED_FAMILIES) <= set(item.get('all_declared_source_families_exercised', []))
                or item.get('all_auxiliary_maximum_contexts_exercised') is not True
                or item.get('auxiliary_maximum_contexts') != auxiliary_maximum_contexts
                or actual_maxima != auxiliary_maximum_contexts
                or item.get('original_backbone_unchanged') is not True
                or type(item.get('peak_allocated_bytes')) is not int or item['peak_allocated_bytes'] < 1
                or not probes or any(p.get('global_execution_pairs') != 4 * plan['execution_options']['pair_batch_size']
                    * plan['execution_options']['pair_accumulation'] or p.get('finite_all_and_only_residual_gradients') is not True for p in probes)):
            raise ValueError('Every actual rank must execute complete production and combined extreme gradients')


def open_feedback(plan):
    """Open exactly the plan's independently admitted complete Java/Python union."""
    folder = Path(plan['execution_feedback'])
    for name in ['manifest', 'admission']:
        original._read_bound(folder / (name + '.json'), plan['execution_' + name + '_sha256'])
    return ExecutionFeedbackUnion(folder, Tokenizer.from_file(plan['tokenizer']), plan['tokenizer'])


def bind_inputs(plan, *, capacity=False):
    validate_plan(plan, capacity=capacity)
    projected = {**plan, 'format': original_capacity.FORMAT if capacity else original.FORMAT}
    if capacity:
        bundle, _, files = original_capacity.bind_inputs(projected)
    else:
        bundle, _, files = original.bind_completed_inputs(projected)
    folder = Path(plan['execution_feedback'])
    original._merge(files, {str(folder / 'manifest.json'): plan['execution_manifest_sha256'],
        str(folder / 'admission.json'): plan['execution_admission_sha256']})
    # This is the real admitted-source reader, never the CPU test subclass.
    feedback = open_feedback(plan)
    original._merge(files, {str(p): h for p, h in feedback.guard.files.items()})
    data = feedback_identity(feedback, plan['tokenizer_sha256'])
    reference = None
    if not capacity:
        reference = original._read_bound(plan['execution_reference'], plan['execution_reference_sha256'])
        if (reference.get('feedback') != data
                or auxiliary_fingerprint(reference, plan['execution_options']) != plan['execution_auxiliary_fingerprint']):
            raise ValueError('Production must retain the exact original admitted feedback/reference/options')
        proof = original._read_bound(plan['execution_capacity_proof'], plan['execution_capacity_proof_sha256'])
        maxima = dict(positive=max(r['context_tokens'] for r in feedback.positives),
            **{name: max(r[name + '_context_tokens'] for r in feedback.pairs) for name in ['chosen', 'rejected']})
        validate_capacity_proof(proof, plan, reference, auxiliary_maximum_contexts=maxima)
        original._merge(files, proof['files_sha256'])
        original._merge(files, {plan[k]: plan[k + '_sha256'] for k in ['execution_reference', 'execution_capacity_proof']})
    guard = FileBindings(files)
    guard.check(full=True)
    return bundle, feedback, reference, guard, files


def auxiliary_extremes(feedback):
    positive_lengths = [len(feedback.positive(i)[0]) for i in range(len(feedback.positives))]
    pair_lengths = [{name: len(value[name][0]) for name in ['chosen', 'rejected']}
        for value in (feedback.example(i) for i in range(len(feedback.pairs)))]
    positive = max(range(len(positive_lengths)), key=positive_lengths.__getitem__)
    pairs = {name: max(range(len(pair_lengths)), key=lambda i: pair_lengths[i][name])
        for name in ['chosen', 'rejected']}
    return positive, list(dict.fromkeys(pairs.values())), dict(positive=positive_lengths[positive],
        **{name: pair_lengths[i][name] for name, i in pairs.items()})


def exercise_execution_extremes(kernel, code_position, image_row, image_field, positive, pairs):
    """Same three CE normalizers and complete DPO backwards at real extremes.

    Adam already exists from the real update. These probes retain all gradient
    buffers, use the exact configured batch/accumulation layout, and are discarded.
    """
    opts, aux = kernel.options, kernel.execution_options
    _same(fingerprint(dict(code=code_position, image=image_row, field=image_field, positive=positive, pairs=pairs)))
    reports = []
    for pair in pairs:
        def prepare():
            groups = [
                [[dict(packed_code_index=code_position)] * opts['batch_size']] * opts['gradient_accumulation'],
                [[image_row] * opts['replay_batch_size']] * opts['replay_accumulation'],
                [[dict(execution_positive_index=positive)] * aux['positive_batch_size']] * aux['positive_accumulation']]
            encoded = [_EncodedBatches(encoder, rows, field, kernel.device) for encoder, rows, field in zip(
                [kernel.code_encoder, kernel.replay_encoder, kernel.positive_encoder], groups, ['response', image_field, 'response'])]
            examples = [kernel.feedback.example(pair) for _ in range(aux['pair_batch_size'] * aux['pair_accumulation'])]
            return encoded, examples
        encoded, examples = _validated(prepare)
        kernel.optimizer.zero_grad(set_to_none=True)
        measured, DPO = measure_preferences(kernel.model.code, examples, kernel.reference_bundle['cache'],
            expected_policy_sha256=kernel.reference_bundle['cache']['policy_sha256'], alpha=aux['DPO_alpha'],
            beta=aux['DPO_beta'], device=kernel.device, amp=kernel.amp)
        metric = global_CE_gradients(kernel.wrapped, [('maximum_complete_code', encoded[0], 1.),
            ('maximum_original_image', encoded[1], opts['replay_coefficient']),
            ('maximum_verified_original', encoded[2], aux['positive_coefficient'])], kernel.model.config, kernel.device, kernel.amp)
        backward_preferences(kernel.model.code, examples, measured, DPO, kernel.device, kernel.amp)
        def finite():
            if (any(p.grad is not None for p in kernel.model.base.parameters())
                    or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in kernel.model.adapter_parameters())):
                raise ValueError('Every extreme requires finite all-and-only residual gradients')
        _validated(finite)
        average_gradients(kernel.wrapped, kernel.device, kernel.world)
        norm = torch.nn.utils.clip_grad_norm_(kernel.model.adapter_parameters(), opts['gradient_clip'], error_if_nonfinite=True)
        reports.append(dict(**metric, execution_DPO=DPO, global_execution_pairs=DPO['global_pairs'],
            positive_context_tokens=len(kernel.feedback.positive(positive)[0]),
            chosen_context_tokens=len(examples[0]['chosen'][0]), rejected_context_tokens=len(examples[0]['rejected'][0]),
            gradient_norm=float(norm), finite_all_and_only_residual_gradients=True))
        kernel.optimizer.zero_grad(set_to_none=True)
    return reports


def controller(plan_path, *, capacity=False, resume_sha256=None):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    _, _, _, guard, files = bind_inputs(plan, capacity=capacity)
    if capacity and resume_sha256 is not None:
        raise ValueError('Disposable capacity weights cannot resume or become candidates')
    output, storage = Path(plan['output']).resolve(), plan['storage']
    parent = ROOT / ('reports' if capacity else 'checkpoints')
    if (not output.is_relative_to(parent) or output == parent
            or any(Path(p).resolve().is_relative_to(output) for p in files)):
        raise ValueError('New distinct owned output required')
    def available():
        for actor in plan['preceding_processes']:
            if original._process(actor['pid'], actor['start_ticks']):
                raise ValueError('Original predecessor remains live: ' + actor['role'])
        if resume_sha256 is None and output.exists() and (capacity or any(output.iterdir())):
            raise ValueError('Output occupied; inspect or explicitly resume exact journaled production state')
        commitment = storage['minimum_free_bytes'] + storage['maximum_report_bytes']
        if not capacity:
            commitment += 3 * storage['maximum_checkpoint_bytes'] + storage['maximum_inference_bytes']
        if shutil.disk_usage(ROOT).free < commitment:
            raise ValueError('Original reserve and complete new state/report commitment required')
        if resume_sha256 is not None:
            ledger = checkpoint_layout(output, file_hash(plan_path))
            if not _digest(resume_sha256) or not ledger['latest'] or ledger['latest']['checkpoint_sha256'] != resume_sha256:
                raise ValueError('Explicit production resume must match the latest complete journaled hash')
    available()
    # Share the original lease so the original and new controllers serialize.
    lock = ROOT / 'reports/code-residual-production-v4.gpu.lock'
    with lock.open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        available()
        values = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            check=True, text=True, capture_output=True).stdout.splitlines()
        if len(values) != 4 or any(int(v.strip()) > 128 for v in values):
            raise ValueError('All four L40s must be idle before an execution-feedback launch')
        guard.check(full=True)
        output.mkdir(parents=True, exist_ok=True)
        pid = os.getpid()
        owner_path = output / ('execution-owner-' + str(pid) + '.json')
        owner = dict(pid=pid, start_ticks=(Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1].split()[19],
            plan_sha256=file_hash(plan_path), parent_cmdline_sha256=file_hash(Path('/proc') / str(pid) / 'cmdline'),
            capacity=capacity)
        report_json(owner_path, owner, storage['maximum_report_bytes'])
        argv = [str(ROOT / '.venv/bin/torchrun'), '--standalone', '--nproc_per_node=4', '-m',
            'slm_lab.execution_production_v6', '--worker', '--plan', str(plan_path), '--owner', str(owner_path)]
        if capacity: argv += ['--capacity']
        if resume_sha256 is not None: argv += ['--resume-sha256', resume_sha256]
        try:
            subprocess.run(argv, cwd=ROOT, check=True, env={**os.environ, 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'})
            guard.check(full=True)
            result_path = output / ('proof.json' if capacity else 'training-completed.json')
            result = json.loads(result_path.read_text())
            if result.get('passed' if capacity else 'training_complete') is not True:
                raise ValueError('Workers ended without a completed capacity or training result')
        except Exception as error:
            report_json(output / 'failure.json', dict(error=str(error), automatic_restart=False,
                candidate_promoted=False, goal_achieved=False), storage['maximum_report_bytes'])
            raise


def worker(plan_path, owner_path, *, capacity=False, resume_sha256=None):
    plan_path, owner_path = Path(plan_path).resolve(), Path(owner_path).resolve()
    plan, owner = json.loads(plan_path.read_text()), json.loads(owner_path.read_text())
    if (owner_path.parent != Path(plan['output']).resolve() or owner.get('capacity') is not capacity
            or owner.get('plan_sha256') != file_hash(plan_path) or not original._process(owner['pid'], owner['start_ticks'])
            or file_hash(Path('/proc') / str(owner['pid']) / 'cmdline') != owner.get('parent_cmdline_sha256')):
        raise ValueError('Exact live checked controller and owned receipt required')
    bundle, feedback, reference, _, files = bind_inputs(plan, capacity=capacity)
    original._merge(files, {str(plan_path): file_hash(plan_path), str(owner_path): file_hash(owner_path)})
    guard = original.ControllerBoundGuard(FileBindings(files), owner)
    rank, world, local = [int(os.environ[k]) for k in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']]
    if world != 4 or not 0 <= rank < 4 or not 0 <= local < 4 or not torch.cuda.is_available() or torch.cuda.device_count() != 4:
        raise ValueError('Four actual visible CUDA workers required')
    code = CompleteCodePack(plan['code_pack'], 'train', purpose='training')
    replay, contract = open_final_source_replay(bundle['replay_folders'], plan['tokenizer_sha256'])
    opts, aux, output, storage = plan['training_options'], plan['execution_options'], Path(plan['output']), plan['storage']
    if (fingerprint(contract) != plan['replay_contract_fingerprint']
            or exposure_budget(len(code), opts['passes'], 4, opts['batch_size'], opts['gradient_accumulation']) != bundle['prospective_code_budget']
            or opts['seed'] != bundle['prospective_seed']
            or 4 * opts['replay_batch_size'] * opts['replay_accumulation'] != bundle['prospective_global_replay_width']):
        raise ValueError('Exact original full-source budget/layout contract required')
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.get_device_name(device) != 'NVIDIA L40':
        raise ValueError('Actual BF16 L40 execution training required')
    torch.set_num_threads(1)
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    try:
        guard.check(full=True)
        base, metadata = load_checkpoint(plan['base_checkpoint'], device, training_state=False)
        if metadata['tokenizer_sha256'] != plan['tokenizer_sha256'] or base.config != ModelConfig.load(code.folder / 'model-config.json'):
            raise ValueError('Exact final own model/tokenizer configuration required')
        torch.manual_seed(opts['seed'])
        model = RoutedCodeResidual(base, width=2048)
        if model.parameter_counts() != original.MODEL_COUNTS:
            raise ValueError('Exact own857M total/frozen/trainable parameter partition required')
        optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=opts['learning_rate'], weight_decay=opts['weight_decay'], foreach=False)
        joint = replay
        while isinstance(joint, CombinedReplay): joint = joint.owners['human_instruction']
        token = Tokenizer.from_file(plan['tokenizer'])
        encoder = ExpandedReplayEncoder(token, model.config, joint.source.parent, 448)
        amp = lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if capacity:
            torch.cuda.reset_peak_memory_stats(device)
            reference = prepare_reference_bundle(model, feedback, base_checkpoint_sha256=plan['base_checkpoint_sha256'],
                tokenizer_sha256=plan['tokenizer_sha256'], device=device, amp=amp, precision='bf16')
        elif resume_sha256 is not None:
            restored_reference = reference_from_checkpoint(output / 'resume.pt', resume_sha256)
            if restored_reference != reference:
                raise ValueError('Complete checkpoint must embed the unchanged originally bound reference')
            reference = restored_reference
        auxfp = auxiliary_fingerprint(reference, aux)
        if not capacity and auxfp != plan['execution_auxiliary_fingerprint']:
            raise ValueError('Frozen reference/options differ from exact production plan')
        kernel = ExecutionResidualKernel(model, optimizer, code, packed_encoder(code, token, model.config), replay,
            encoder, opts, dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
                run_plan_sha256=file_hash(plan_path)), device, amp, replay_contract=contract,
            expected_contract_fingerprint=plan['replay_contract_fingerprint'], guard=guard,
            feedback=feedback, reference_bundle=reference, execution_options=aux, expected_auxiliary_fingerprint=auxfp)
        random.seed(opts['seed'] + rank); np.random.seed((opts['seed'] + rank) % 2**32); torch.manual_seed(opts['seed'] + rank)
        if capacity:
            code_pos, image, field, extremes = original_capacity.capacity_extremes(code, replay, encoder, bundle, contract)
            positive, pairs, maxima = auxiliary_extremes(feedback)
            proof = original_capacity.exercise_capacity(kernel, code_pos, image, field)
            probes = exercise_execution_extremes(kernel, code_pos, image, field, positive, pairs)
            exercised = dict(positive=max(p['positive_context_tokens'] for p in probes),
                chosen=max(p['chosen_context_tokens'] for p in probes), rejected=max(p['rejected_context_tokens'] for p in probes))
            if (exercised != maxima or proof['actual_maximum_context_backward'] != 4096
                    or state_digest(model.base.state_dict()) != reference['policy']['frozen_base_state_sha256']):
                raise ValueError('Every actual auxiliary maximum and original frozen tensor must be exercised/preserved')
            torch.cuda.synchronize(device)
            proof.update(rank=rank, device_name=torch.cuda.get_device_name(device), model_counts=model.parameter_counts(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device), peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                selected_original_extremes=extremes, auxiliary_maximum_contexts=maxima,
                all_auxiliary_maximum_contexts_exercised=True, execution_extreme_backwards=probes)
            ranks=[None]*4; dist.all_gather_object(ranks, proof)
            def finish_capacity():
                guard.check(full=True)
                report_json(output/'reference.json', reference, storage['maximum_report_bytes'])
                reports={str(i):str(output/('rank-'+str(i)+'.json')) for i in range(4)}
                for item in ranks: report_json(reports[str(item['rank'])],item,storage['maximum_report_bytes'])
                bound={**files, str(output/'reference.json'):file_hash(output/'reference.json'),
                    **{p:file_hash(p) for p in reports.values()}}
                report_json(output/'proof.json',dict(format=PROOF_FORMAT,passed=True,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    **{k:plan[k] for k in ['base_checkpoint_sha256','tokenizer_sha256','replay_contract_fingerprint',
                        'execution_manifest_sha256','execution_admission_sha256']},
                    training_options_fingerprint=fingerprint(opts),execution_options_fingerprint=fingerprint(aux),
                    execution_reference_sha256=file_hash(output/'reference.json'),execution_auxiliary_fingerprint=auxfp,
                    model_counts=model.parameter_counts(),actual_world_size=4,actual_reference_pairs_scored=len(feedback.pairs),
                    runtime_provenance=capture([PACKAGE/n for n in sorted(RUNTIME_FILES)]),rank_reports=reports,
                    files_sha256=bound,candidate_weights_saved=False,noncontiguous_probe_weights_discarded=True,
                    goal_achieved=False),storage['maximum_report_bytes'])
            rank_zero(finish_capacity)
        else:
            if resume_sha256 is not None: kernel.restore(output/'resume.pt',resume_sha256,allow_completed=True)
            result=run_updates(kernel,output,maximum_checkpoint_bytes=storage['maximum_checkpoint_bytes'],
                minimum_free_bytes=storage['minimum_free_bytes'],checkpoint_every=opts['checkpoint_every'],
                log_every=opts['log_every'],maximum_report_bytes=storage['maximum_report_bytes'])
            def finish_training():
                guard.check(full=True)
                return original.finish_inference(output,plan,plan_path,result)
            rank_zero(finish_training)
    finally:
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--capacity',action='store_true')
    parser.add_argument('--resume-sha256')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--owner')
    args=parser.parse_args()
    if args.worker:
        if not args.owner: raise ValueError('Workers require their exact checked controller receipt')
        worker(args.plan,args.owner,capacity=args.capacity,resume_sha256=args.resume_sha256)
    else:
        if args.owner: raise ValueError('Only workers accept a controller receipt')
        controller(args.plan,capacity=args.capacity,resume_sha256=args.resume_sha256)


if __name__=='__main__': main()
