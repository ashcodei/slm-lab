"""Check every admitted train-view tensor batch before GPU capacity work.

Waits for exact final admission. Performs no program/neural execution and no
optimizer updates; small loader tests cannot authorize full-model GPU work.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/public-reasoned-code-pack-admission-v1'
OUT = ROOT / 'reports/public-reasoned-code-train-data-CPU-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def live(owner):
    path = Path('/proc') / str(owner['pid']) / 'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == str(owner['birth']) and fields[0] != 'Z'


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread complete train data check required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    tests = ROOT / 'reports/public-reasoned-code-train-data-v1-tests.log'
    assert '4 passed' in tests.read_text()
    predecessor = json.loads((SOURCE / 'owner-plan.json').read_text())
    files = {str(path): sha(path) for path in [Path(__file__).resolve(), SOURCE / 'owner-plan.json', tests,
        ROOT / 'src/slm_lab/public_reasoned_code_train_data_v1.py',
        ROOT / 'tests/test_public_reasoned_code_train_data_v1.py',
        ROOT / 'src/slm_lab/codenet_token_pack_v1.py']}
    OUT.mkdir()

    def emit(name, value):
        with (OUT / name).open('x') as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')

    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    emit('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_predecessor=predecessor['owner'], complete_original_train_view_epoch_required=True,
        epochs=1, global_width=16, source_seed=990231, reserved_sources_forbidden=True,
        all_original_context_response_EOS_and_padding_masks_checked=True,
        original_short_last_batch_not_rounded_or_replayed=True,
        CPU_threads=1, nice=19, scheduler='SCHED_IDLE',
        actual_GPU_capacity_and_recovery_not_granted=True, new_program_or_neural_execution=False,
        new_model_updates=0, original_workers_unchanged=True))
    try:
        while live(predecessor['owner']):
            print(json.dumps(dict(phase='waiting_exact_complete_weak_SFT_admission', predecessor=predecessor['owner'])), flush=True)
            time.sleep(30)
        if (SOURCE / 'failure.json').exists() or not (SOURCE / 'complete.json').exists():
            raise ValueError('Exact final admission lacks successful terminal evidence; never restart it')
        if shutil.disk_usage(ROOT).free < 48 * 1024**3:
            raise ValueError('Preserve48GiB persistent other-writer reserve')
        stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        if int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel')) < 52 * 1024**3:
            raise ValueError('4GiB own plus48GiB other-job nonfile memory reserve required')
        import torch
        from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
        torch.set_num_threads(1)
        assert not torch.cuda.is_initialized()
        data = PublicReasonedCodeSFTData(ROOT, views='both')
        files.update(data.source_files_sha256)
        assert all(sha(path) == value for path, value in files.items())
        entries = data.complete_schedule(epochs=1, width=16, seed=990231)
        budget = data.budget(entries)
        identity = data.identity(entries)
        assert Counter(position for entry in entries for position in entry) == dict.fromkeys(range(len(data)), 1)
        assert len(entries[-1]) == (len(data) % 16 or 16)
        expected_targets = data.pack.manifest['counts']['train']['target_tokens']
        expected_contexts = data.pack.manifest['counts']['train']['context_tokens']
        assert budget['response_EOS_targets'] == expected_targets and budget['complete_context_tokens'] == expected_contexts
        counts = Counter()
        longest_context = max(range(len(data)), key=lambda position: data.information(position)['complete_context_tokens'])
        longest_target = max(range(len(data)), key=lambda position: data.information(position)['response_EOS_targets'])
        emit('execution-plan.json', dict(at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            actual_admitted_train_views=len(data), complete_budget=budget, complete_schedule_identity=identity,
            historical_provenance_files_checked_at_open=len(data.provenance_source_files_sha256),
            actual_runtime_binding_files=len(data.source_files_sha256),
            cold_and_final_full_provenance_checks_not_raw_ancestor_scan_per_batch=True,
            no_throughput_improvement_measured=True,
            longest_original_context=data.information(longest_context), longest_original_target=data.information(longest_target),
            Python=sys.version, torch_version=torch.__version__, actual_GPU_capacity_not_proved=True))
        with (OUT / 'every-original-train-batch.jsonl').open('x') as journal:
            for index, positions in enumerate(entries):
                batch = data.batch(positions, device='cpu')
                assert batch['pixel_values'] is None
                assert batch['input_ids'].dtype == batch['labels'].dtype == torch.long
                assert batch['attention_mask'].dtype == torch.bool
                valid_targets = int((batch['labels'][:,1:] != -100).sum())
                context_count = int(batch['attention_mask'].sum())
                target_count = 0
                for local, position in enumerate(positions):
                    info = data.information(position)
                    row = data.row(position)
                    ids, labels = data.sequence(position)
                    length = len(ids)
                    assert info['source_role'] == row['original_fixed_role']['source_role'] == 'train'
                    assert row['original_fixed_role']['optimization_forbidden'] is False
                    assert row['human_preferences'] is False and row['use_as_RL_reward_oracle_forbidden'] is True
                    assert batch['input_ids'][local,:length].tolist() == ids
                    assert batch['labels'][local,:length].tolist() == labels
                    assert ids[-1] == labels[-1] == 2
                    assert bool(batch['attention_mask'][local,:length].all())
                    assert not bool(batch['attention_mask'][local,length:].any())
                    assert bool((batch['input_ids'][local,length:] == 0).all())
                    assert bool((batch['labels'][local,length:] == -100).all())
                    target_count += info['response_EOS_targets']
                    counts['view:' + info['view']] += 1
                assert valid_targets == target_count
                counts['train_view_presentations'] += len(positions)
                counts['actual_response_EOS_targets'] += valid_targets
                counts['actual_context_positions'] += context_count
                counts['global_batches'] += 1
                journal.write(json.dumps(dict(global_batch=index, exact_original_positions=positions,
                    actual_complete_context_positions=context_count, actual_response_EOS_targets=valid_targets,
                    exact_full_source_and_padding_masks_checked=True, new_model_updates=0), separators=(',', ':')) + '\n')
                if (index+1) % 50 == 0:
                    journal.flush()
                    print(json.dumps(dict(phase='all_admitted_train_tensor_batches', completed=index+1, total=len(entries))), flush=True)
        assert counts['train_view_presentations'] == len(data)
        assert counts['actual_response_EOS_targets'] == expected_targets
        assert counts['actual_context_positions'] == expected_contexts
        assert data.identity(entries) == identity and not torch.cuda.is_initialized()
        data.audit_provenance()
        assert all(sha(path) == value for path, value in files.items())
        emit('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, provenance_identity=data.provenance_identity,
            actual_counts=dict(counts), complete_budget=budget, schedule_identity=identity,
            last_global_batch_examples=len(entries[-1]), all_original_train_views_exactly_once=True,
            all_original_full_targets_and_true_EOS_and_padding_masks_checked=True,
            historical_full_provenance_verified_at_open_and_final=True,
            data_is_public_synthetic_weak_SFT_only=True, reserved_optimization_forbidden=True,
            use_as_RL_reward_oracle_forbidden=True, own_model_human_preference_feedback=False,
            new_program_or_neural_execution=False, new_model_updates=0,
            actual_full_size_GPU_capacity_recovery_and_all_final_evaluations_still_required=True,
            model_quality_gain_established=False, original_workers_unchanged=True, goal_achieved=False))
        print(json.dumps(dict(complete=True, counts=dict(counts))), flush=True)
    except BaseException as error:
        emit('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, new_model_updates=0, original_workers_unchanged=True))
        raise


if __name__ == '__main__':
    main()
