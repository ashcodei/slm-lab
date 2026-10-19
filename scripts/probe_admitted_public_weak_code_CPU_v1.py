"""Exercise every actual admitted weak-SFT train batch without a neural model."""
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/admitted-public-weak-code-interface-CPU-v1'
SOURCE = ROOT / 'reports/public-weak-code-pack-readback-admission-CPU-v1'
PARENT = dict(pid=169730, ticks='56191630')


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread source interface verification required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from record_visual_stage_and_first_repair_arm_v1 import atomic, read, sha
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    if read(SOURCE / 'plan.json')['owner'] != PARENT: raise ValueError('Exact actual weak pack reader/admission birth required')
    paths = [Path(__file__), SOURCE / 'plan.json', ROOT / 'src/slm_lab/public_weak_code_train_data_v1.py']
    files = {str(p): sha(p) for p in paths}
    states = {p: (Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) for p in files}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().split(') ', 1)[1].split()[19])
    OUT.mkdir()
    atomic(OUT / 'plan.json', dict(owner=owner, exact_original_data_admission_owner=PARENT, source_files_sha256=files,
        all_original_weak_train_rows=21739, original_reserved_rows_not_exposed_to_optimization=2509,
        actual_complete_train_batches_to_construct=1359, three_epoch_batches=4077,
        every_original_train_native_ID_mask_EOS_and_padding_checked=True,
        public_synthetic_weak_SFT_only=True, use_as_RL_reward_oracle_forbidden=True,
        models_constructed=0, neural_forwards=0, optimizer_updates=0, GPUleases=0, goal_achieved=False))
    try:
        deadline = time.monotonic() + 7 * 86400
        while not terminal(PARENT):
            if time.monotonic() > deadline: raise ValueError('Source interface observation lifetime expired')
            if any((Path(p).stat().st_ino, Path(p).stat().st_size, Path(p).stat().st_mtime_ns) != s for p, s in states.items()):
                raise ValueError('The exact immutable source interface inputs changed')
            atomic(OUT / 'progress.json', dict(owner=owner, phase='waiting for exact complete terminal weak source admission',
                original_admission_owner_live=True, models_constructed=0, neural_forwards=0))
            time.sleep(15)
        if (SOURCE / 'failure.json').exists() or not (SOURCE / 'complete.json').exists():
            raise ValueError('The exact original weak data admission did not complete; preserve its failure')
        import torch
        from slm_lab.codenet_token_pack_v1 import CompleteCodePack
        from slm_lab.public_weak_code_train_data_v1 import PublicWeakCodeSFTData
        torch.set_num_threads(1)
        data = PublicWeakCodeSFTData(ROOT)
        files.update(data.source_files_sha256)
        entries = data.complete_schedule(epochs=1, width=16)
        three = data.complete_schedule(epochs=3, width=16)
        if (len(entries) != 1359 or len(three) != 4077 or Counter(p for e in entries for p in e) != Counter(data.slots)
                or Counter(p for e in three for p in e) != Counter({p: 3 for p in data.slots})):
            raise ValueError('Every actual original weak train source must occur once per complete declared epoch')
        counts = Counter()
        for step, positions in enumerate(entries):
            if step % 64 == 0: data.guard.check()
            batch = data.batch(positions, 'cpu')
            for j, position in enumerate(positions):
                ids, labels = data.sequence(position)
                if (batch['input_ids'][j, :len(ids)].tolist() != ids or batch['labels'][j, :len(ids)].tolist() != labels
                        or batch['input_ids'][j, 0] != 1 or batch['labels'][j, len(ids)-1] != 2
                        or not bool(batch['attention_mask'][j, :len(ids)].all())
                        or bool(batch['attention_mask'][j, len(ids):].any())
                        or bool((batch['labels'][j, len(ids):] != -100).any())
                        or data.rows[position]['original_fixed_role']['source_role'] != 'train'):
                    raise ValueError('All actual complete original train IDs, labels, true EOS and non-target padding must remain')
                counts['source_presentations'] += 1
                counts['original_context_tokens'] += len(ids)
                counts['response_EOS_targets'] += sum(t != -100 for t in labels)
            counts['actual_complete_batches'] += 1
            if step % 64 == 0:
                atomic(OUT / 'progress.json', dict(owner=owner, phase='all actual admitted weak-SFT native batches',
                    actual_counts=dict(counts), models_constructed=0, neural_forwards=0))
        held_forbidden = oracle_forbidden = False
        try: CompleteCodePack(data.folder, 'validation', purpose='training')
        except ValueError: held_forbidden = True
        try: data.score(0, [2], horizon=512)
        except ValueError: oracle_forbidden = True
        if (not held_forbidden or not oracle_forbidden or counts['source_presentations'] != 21739
                or counts['response_EOS_targets'] != data.pack.manifest['counts']['train']['target_tokens']
                or counts['original_context_tokens'] != data.pack.manifest['counts']['train']['context_tokens']
                or torch.cuda.is_initialized()):
            raise ValueError('Every original weak source/target and strict held-role/oracle restriction must remain')
        data.guard.check(full=True)
        atomic(OUT / 'complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_counts=dict(counts), actual_original_reserved_rows=2509,
            all_actual_native_source_arrays_masks_true_EOS_and_padding_preserved=True,
            reserved_optimization_rejected=True, synthetic_test_reward_oracle_rejected=True,
            actual_one_epoch_batches=1359, actual_three_epoch_batches=4077, actual_three_epoch_source_presentations=65217,
            admitted_use='weak_supervised_coding_only', human_preferences=False, own_model_on_policy=False,
            generated_test_agreement_not_independent_semantic_correctness=True,
            models_constructed=0, neural_forwards=0, new_program_executions=0, optimizer_updates=0,
            full_model_GPUcapacity_training_and_quality_still_required=True, goal_achieved=False))
        print(str(OUT / 'complete.json'), flush=True)
    except BaseException as exc:
        atomic(OUT / 'failure.json', dict(owner=owner, error=repr(exc), automatic_retry=False,
            all_actual_source_observations_preserved=True, no_new_model_or_GPUjob=True, goal_achieved=False))
        raise


if __name__ == '__main__': main()
