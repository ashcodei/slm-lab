"""Bounded four-L40 restart proof; not a full-source training launcher."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
from tokenizers import Tokenizer
import torch
from torch import distributed as dist

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_collective_checkpoint_v2 import (
    fingerprint, restore_collective_residual, save_collective_residual)
from slm_lab.codenet_checkpoint_v1 import FileBindings, capture_rng
from slm_lab.codenet_replay_v1 import FullReplayEncoder
from slm_lab.codenet_runner_v1 import FixedReductionModel, average_gradients
from slm_lab.codenet_stage_v1 import Objective, objective_gradients
from slm_lab.codenet_token_pack_v1 import CompleteCodePack, packed_encoder
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from slm_lab.exclusive_inference_resources_v1 import checkpoint_writers, gpu_memory_for_pid, identity
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/code-residual-collective-cuda-v2'
TEMP = Path('/tmp/slm-code-residual-collective-cuda-v2')
SOURCE = ROOT / 'checkpoints/extended-curriculum-v1/language-pretraining/model.pt'
SOURCE_SHA = '4c05d99b0c3fc775bf89e70c6cbac9c63017eb36f67e17c738b5be268fb191ca'
HOME_RESERVE = 16 * 1024**3
TEMP_RESERVE = 8 * 1024**3
CAP = 2 * 1024**3


def read(path):
    return json.loads(path.read_text())


def status(message, **fields):
    atomic_json(REPORTS / 'status.json', dict(status=message,
        updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))


def lease_guard():
    lease = read(REPORTS / 'lease.json')
    owner = identity(lease['pid'])
    assert owner and owner['ticks'] == lease['ticks'] and owner['state'] not in {'Z', 'X'}
    assert str(Path(__file__).resolve()) in owner['argv']
    assert file_hash(Path(__file__)) == lease['source_sha256']
    allocations = gpu_memory_for_pid(lease['pid'])
    assert len(lease['GPU_UUIDs']) == 4 and all(allocations.get(u, 0) > 128 for u in lease['GPU_UUIDs'])
    assert not checkpoint_writers(), 'Original checkpoint writers take priority'
    assert shutil.disk_usage(ROOT).free >= HOME_RESERVE
    assert shutil.disk_usage(TEMP).free >= TEMP_RESERVE
    for name, digest in lease['stopped_predecessors_sha256'].items():
        assert file_hash(Path(name)) == digest


def sampler(step):
    return dict(updates=step, complete_code_programme_exposures=8 * step, replay_exposures=4 * step)


def worker(phase):
    rank = int(os.environ['RANK'])
    assert int(os.environ['WORLD_SIZE']) == 4
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.35, device)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    try:
        plan_path = REPORTS / 'plan.json'
        plan = read(plan_path)
        guard = FileBindings({**plan['source_files_sha256'], str(plan_path): file_hash(plan_path)})
        lease_guard()
        torch.manual_seed(51113)
        base, metadata = load_checkpoint(SOURCE, 'cpu', training_state=False)
        model = RoutedCodeResidual(base, width=2048).to(device)
        model.code.gradient_checkpointing = True
        assert model.parameter_counts() == dict(total_unique=857007616, frozen_base=706012672, trainable_code=150994944)
        base_before = state_digest(model.base.state_dict())
        pack = CompleteCodePack(ROOT / 'data/codenet-durable-pack-v2', 'train', purpose='training')
        token_path = pack.folder / 'tokenizer.json'
        tokenizer = Tokenizer.from_file(str(token_path))
        assert metadata['tokenizer_sha256'] == file_hash(token_path)
        encoder = packed_encoder(pack, tokenizer, model.config)
        replay_encoder = FullReplayEncoder(tokenizer, model.config, ROOT / 'data/extended-curriculum-v1', 448)
        controls = plan['original_replay_rows']
        image = next(row for row in controls if row.get('image'))
        pixels = replay_encoder.pixels(image)
        rgb = np.rint((pixels.permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == image['model_pixel_sha256_448']
        for row in plan['original_code_rows']:
            ids, labels = pack.sequence(row['local_position'])
            assert hashlib.sha256(np.asarray(ids, dtype='<u2').tobytes()).hexdigest() == row['full_input_ids_sha256']
            assert len(ids) == row['context_tokens'] and sum(y != -100 for y in labels) == row['target_tokens']
        optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=3e-5, weight_decay=.01, foreach=False)
        wrapped = FixedReductionModel(model.code)
        random.seed(61111 + rank)
        np.random.seed(62111 + rank)
        torch.manual_seed(63111 + rank)
        torch.cuda.manual_seed(64111 + rank)
        bindings = dict(base_checkpoint_sha256=SOURCE_SHA, tokenizer_sha256=file_hash(token_path),
            run_plan_sha256=file_hash(plan_path), planned_steps=2, device=device)
        metrics = []
        torch.cuda.reset_peak_memory_stats(device)

        def update(step):
            guard.check()
            lease_guard()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            draws = [random.randrange(1024), int(np.random.randint(1024)),
                int(torch.randint(0, 1024, ())), int(torch.randint(0, 1024, (), device=device))]
            rows = plan['original_code_rows']
            if sum(draws) % 2:
                rows = list(reversed(rows))
            groups = [[dict(packed_code_index=row['local_position'])] for row in rows]
            replay_row = next(row for row in controls if bool(row.get('image')) == (rank % 2 == 0))
            result = objective_gradients(wrapped, [Objective('complete_code', encoder, groups, 'response', 1.),
                Objective('original_replay', replay_encoder, [[replay_row]], 'response', .5)],
                device, lambda: torch.autocast('cuda', dtype=torch.bfloat16), world=4)
            average_gradients(wrapped, device, 4)
            assert not encoder.truncated and not replay_encoder.truncated
            assert all(p.grad is None for p in model.base.parameters())
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.adapter_parameters())
            norm = torch.nn.utils.clip_grad_norm_(model.adapter_parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            metric = dict(step=step, **result, gradient_norm=float(norm))
            metrics.append(metric)
            record = dict(metric=metric, draws=draws, original_code_order=[row['id'] for row in rows],
                replay_id=replay_row['id'], replay_has_image=bool(replay_row.get('image')))
            print('collective_cuda_update', phase, rank, json.dumps(record), flush=True)
            return record

        def exact_state():
            return fingerprint(dict(residuals=model.residuals.state_dict(), optimizer=optimizer.state_dict(),
                rng=capture_rng(device), metrics=metrics, sampler=sampler(2)))

        if phase == 'reference':
            update(1)
            receipt = save_collective_residual(TEMP / 'resume.pt', model, optimizer, **bindings,
                completed_steps=1, sampler_state=sampler(1), metrics=metrics,
                maximum_bytes=CAP, minimum_free_bytes=TEMP_RESERVE)
            if rank == 0:
                atomic_json(REPORTS / 'snapshot.json', receipt)
            dist.barrier()
        else:
            receipt = read(REPORTS / 'snapshot.json')
            step, progress, history = restore_collective_residual(TEMP / 'resume.pt', model, optimizer,
                **bindings, expected_checkpoint_sha256=receipt['checkpoint_sha256'], sampler_state_for_step=sampler)
            assert step == 1 and progress == sampler(1)
            metrics[:] = history
        record = update(2)
        assert state_digest(model.base.state_dict()) == base_before
        digest = exact_state()
        shared = fingerprint(dict(residuals=model.residuals.state_dict(), optimizer=optimizer.state_dict()))
        all_shared = [None] * 4
        dist.all_gather_object(all_shared, shared)
        assert len(set(all_shared)) == 1
        next_draws = [random.random(), float(np.random.random()), float(torch.rand(())), float(torch.rand((), device=device))]
        result = dict(rank=rank, pid=os.getpid(), start_ticks=identity(os.getpid())['ticks'], phase=phase,
            model_counts=model.parameter_counts(), complete_state_sha256=digest, shared_state_sha256=shared,
            next_draws=next_draws, second_update=record, original_frozen_base_sha256=base_before,
            frozen_backbone_unchanged=True, peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(device))
        if phase == 'resume':
            expected = read(REPORTS / f'reference-rank-{rank}.json')
            assert result['pid'] != expected['pid']
            for key in ['complete_state_sha256', 'shared_state_sha256', 'next_draws', 'second_update', 'original_frozen_base_sha256']:
                assert result[key] == expected[key], key
            result['fresh_process_exact_restart_verified'] = True
        guard.check(full=True)
        lease_guard()
        atomic_json(REPORTS / f'{phase}-rank-{rank}.json', result)
    finally:
        dist.destroy_process_group()


def parent():
    if REPORTS.exists() or TEMP.exists():
        raise ValueError('Preserve previous proof/partial checkpoint; no implicit restart')
    assert shutil.disk_usage(ROOT).free >= HOME_RESERVE + 64 * 1024**2
    assert shutil.disk_usage(TEMP.parent).free >= TEMP_RESERVE + CAP
    assert not checkpoint_writers()
    # The already funded original curriculum has priority over this bounded test.
    assert shutil.disk_usage(ROOT).free < 25 * 1024**3 + 256 * 1024**2
    torch.set_num_threads(1)
    buffers = []
    for index in range(4):
        torch.cuda.set_device(index)
        assert torch.cuda.mem_get_info(index)[0] >= 20 * 1024**3
        torch.cuda.set_per_process_memory_fraction(.05, index)
        buffers.append(torch.empty(256 * 1024**2, dtype=torch.uint8, device=f'cuda:{index}'))
    assert not checkpoint_writers()
    REPORTS.mkdir()
    TEMP.mkdir()
    process = None
    try:
        uuids = {int(r[0]): r[1].strip() for r in csv.reader(subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines())}
        actor = identity(os.getpid())
        atomic_json(REPORTS / 'lease.json', dict(pid=os.getpid(), ticks=actor['ticks'], source_sha256=file_hash(Path(__file__)),
            GPU_UUIDs=[uuids[i] for i in range(4)], stopped_predecessors_sha256={str(ROOT / p): file_hash(ROOT / p)
                for p in ['reports/extended-curriculum-v1/status.json', 'reports/dialogue-retention-v1/status.json']}))
        lease_guard()
        old = read(ROOT / 'reports/code-residual-readiness-v1/plan.json')
        ids = {row['id'] for row in old['original_training_control_ids']}
        joint = ROOT / 'data/extended-curriculum-v1/joint.train.jsonl'
        controls = []
        for line in joint.open():
            row = json.loads(line)
            if row['id'] in ids:
                controls.append(row)
            if len(controls) == len(ids):
                break
        assert len(controls) == 2 and sum(bool(row.get('image')) for row in controls) == 1
        files = [Path(__file__), SOURCE, joint, ROOT / 'reports/code-residual-readiness-v1/plan.json',
            ROOT / 'reports/code-residual-collective-checkpoint-v2/independent-CPU-proof-readback.json']
        package = ROOT / 'src/slm_lab'
        for name in ['code_residual_collective_checkpoint_v2.py', 'code_residual_checkpoint_v1.py',
            'code_residual_branch_v1.py', 'codenet_checkpoint_v1.py', 'codenet_runner_v1.py', 'codenet_stage_v1.py',
            'codenet_token_pack_v1.py', 'codenet_replay_v1.py', 'model.py', 'training.py', 'recurrent_ablation.py',
            'corpus.py', 'diverse_images.py', 'exclusive_inference_resources_v1.py']:
            files.append(package / name)
        pack = ROOT / 'data/codenet-durable-pack-v2'
        files += [pack / 'manifest.json', pack / 'admission.json']
        bound = {str(p): file_hash(p) for p in files}
        bound.update({str(pack / name): digest for name, digest in read(pack / 'manifest.json')['files_sha256'].items()})
        assert bound[str(SOURCE)] == SOURCE_SHA
        for row in controls:
            if row.get('image'):
                p = Path(row['image'])
                p = (p if p.is_absolute() else joint.parent / p).resolve()
                bound[str(p)] = file_hash(p)
                assert bound[str(p)] == row['stored_image_sha256']
        atomic_json(REPORTS / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=bound, original_code_rows=old['selected_complete_original_training_rows'],
            original_replay_rows=controls, world_size=4, backend='NCCL', residual_width=2048,
            reference_updates=2, fresh_process_resume_updates=1, logical_final_step=2,
            maximum_checkpoint_bytes=CAP, minimum_remaining_home_bytes=HOME_RESERVE,
            minimum_remaining_temporary_bytes=TEMP_RESERVE, checkpoint=str(TEMP / 'resume.pt'),
            checkpoint_is_ephemeral_capacity_fixture_not_durable_training_progress=True,
            original_curriculum_and_full_source_storage_guards_unchanged=True,
            fixed_reduction='fixed_parameter_order_worker_average_v1', candidate_promoted=False, goal_achieved=False,
            scope='Actual857M/four-L40 complete residual/Adam/per-rank RNG exact continuation mechanics. '
                'Only two original admitted train programs and original text/image replay controls; '
                'not full-source training, final-base integration, capability learning or promotion.'))
        environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', NCCL_ALGO='Ring', NCCL_PROTO='Simple', OMP_NUM_THREADS='1')
        launches = []
        for phase in ['reference', 'resume']:
            lease_guard()
            log = (REPORTS / f'{phase}.log').open('xb')
            args = [str(ROOT / '.venv/bin/python'), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
                str(Path(__file__).resolve()), '--phase', phase]
            process = subprocess.Popen(args, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            launches.append(dict(phase=phase, **identity(process.pid), command=args))
            atomic_json(REPORTS / 'phase-launches.json', dict(phases=launches))
            while process.poll() is None:
                status('four-GPU exact continuation check running', phase=phase, torchrun_pid=process.pid)
                time.sleep(5)
            log.close()
            if process.returncode:
                raise ValueError(f'{phase} workers stopped with exit{process.returncode}; inspect preserved logs')
            assert all((REPORTS / f'{phase}-rank-{i}.json').exists() for i in range(4))
        results = [read(REPORTS / f'resume-rank-{rank}.json') for rank in range(4)]
        assert all(result['fresh_process_exact_restart_verified'] for result in results)
        FileBindings(bound).check(full=True)
        atomic_json(REPORTS / 'results.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
            workers=results, snapshot=read(REPORTS / 'snapshot.json'), world_size=4, actual_GPU_indices=[0, 1, 2, 3],
            all_residuals_Adam_steps_moments_and_per_rank_Python_NumPy_CPU_CUDA_RNG_match=True,
            fresh_worker_processes_verified=True, logical_completed_updates=2, actual_collective_update_rounds=3,
            full_source_training_started=False, final_base_integration=False, functional_capability_gain_assessed=False,
            checkpoint_is_temporary_fixture=True, candidate_promoted=False, goal_achieved=False))
        status('actual857M four-L40 fresh-process exact continuation passed')
    except Exception as error:
        status('four-GPU continuation proof stopped; inspect preserved evidence', error=str(error))
        if process is not None and process.poll() is None:
            process.wait()
        raise
    finally:
        buffers.clear()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['reference', 'resume'])
    args = parser.parse_args()
    if args.phase:
        worker(args.phase)
    else:
        parent()
