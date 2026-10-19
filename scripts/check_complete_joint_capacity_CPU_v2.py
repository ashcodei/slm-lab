"""Four small replicas execute every intact original joint capacity case.

Real code, preferences, generic photos, OCR and complete corrective negatives
retain original source extremes and rank assignment. This CPU diagnostic
does not prove full-size GPU capacity, production training or model quality.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 40 * 1024**3
MAX_OUTPUT = 2 * 1024**3


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(folder, name, value):
    folder = Path(folder)
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if (len(raw) > 32 * 1024**2 or shutil.disk_usage(folder).free - len(raw) < RESERVE
            or sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) + len(raw) > MAX_OUTPUT):
        raise ValueError('The full original-source capacity diagnostic must retain its complete output budget and40GiBreserve')
    pending = folder / (name + '.pending')
    with pending.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    pending.replace(folder / name)


def worker(rank, output, files):
    import random
    import numpy as np
    import torch
    from torch import distributed as dist
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.complete_joint_capacity_plan_v3 import choose
    from slm_lab.complete_joint_capacity_stage_v2 import CompleteJointCapacityStageV2
    from slm_lab.complete_joint_capacity_run_v2 import run
    from slm_lab.complete_joint_training_data_v1 import CompleteJointTrainingDataV1
    from slm_lab.deterministic_positions_v1 import install
    from slm_lab.frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
    from slm_lab.joint_objective_step_v1 import JointReplicaGuardV1
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_finalizer_v1 import finalize
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
    from slm_lab.parallel_function_repair_stage_v1 import value_digest
    from slm_lab.shared_joint_actor_vision_v1 import SharedJointActorVisionV1
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    output = Path(output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19 or os.sched_getscheduler(0) != os.SCHED_IDLE:
        raise ValueError('Every small CPU replica must retain idle scheduling and hidden CUDA')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + str(output / 'rendezvous'), rank=rank, world_size=4, timeout=timedelta(minutes=45))
    try:
        config = ModelConfig(vocab_size=32768, hidden_size=32, intermediate_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
            max_text_length=4096, vision_hidden_size=16, vision_intermediate_size=32, vision_layers=1, vision_heads=4,
            image_size=224, patch_size=16, pool_size=2)
        source = CompleteJointTrainingDataV1(ROOT, config, word_expansion=5)
        visual = json.loads((ROOT / 'reports/joint-visual-actual-capacity-cases-CPU-v1/actual-complete-source-capacity-plan.json').read_text())
        capacity = choose(source, visual)
        own = install(OwnDenseAssistant(new_backbone(config, 996601), reward_hidden=8, word_temporal=4, pointing_rank=8,
            router_hidden=8, detector_hidden=4, word_expansion=5)).eval()
        view = SharedJointActorVisionV1(OwnDenseActor(own, rank=2, seed=996602)).eval()
        reference = FrozenOwnTextReferenceV1(own, sources={'actual_complete_data_sha256': source.identity_sha256}, tokenizer_sha256=source.tokenizer_sha256)
        optimizer = torch.optim.AdamW([p for _, p in view.named_selected()], lr=1e-4, eps=1e-5, weight_decay=.01, foreach=False)
        guard = JointReplicaGuardV1(view, optimizer)
        options = dict(maximum_examples=1, maximum_padded_tokens=16384, gradient_bucket_values=4096, gradient_clip=1., human_weight=.2,
            visual_weight=.3, CTC_weight=1., native_word_CE_weight=1., matching_weight=.1, temperature=.07, correction_weight=.1,
            DPO_beta=.1, chosen_CE_weight=.1, project_conflicts=True, precision='bf16')
        entries = [r['checkpoint_entry'] for r in capacity['cases']]
        sources = dict(text_initializer_sha256=tensor_digest(own.language_named_parameters()),
            visual_initializer_sha256=tensor_digest((n, p) for n, p in own.named_parameters() if n.startswith(('network.vision.', 'heads.pointing.', 'heads.word.'))),
            tokenizer_sha256=source.tokenizer_sha256, run_plan_sha256=value_digest(dict(files=files, options=options, capacity=capacity)),
            source_admission_sha256=source.identity_sha256, complete_schedule_sha256=value_digest(entries), fixed_values_sha256=guard.fixed.sha256)
        budgets = dict(maximum_records=len(entries), maximum_compact_record_bytes=4 * 1024**2, maximum_expanded_record_bytes=4 * 1024**2,
            maximum_logical_bytes=512 * 1024**2, maximum_physical_bytes=512 * 1024**2, minimum_free_bytes=RESERVE)
        initializers = FileBindings(files)
        stage = CompleteJointCapacityStageV2(source, view, reference, optimizer, sources, options, capacity_plan=capacity,
            original_visual_capacity=visual, history_root=output / 'history', history_budgets=budgets,
            initializer_guard=initializers, maximum_attempts=3)
        random.seed(996610 + rank); np.random.seed(996610 + rank); torch.manual_seed(996610 + rank)
        write(output, 'rank-' + str(rank) + '-ready.json', dict(rank=rank, actual_counts=view.actor.counts(), actual_complete_capacity_cases=len(entries)))
        if rank == 0:
            write(output, 'actual-complete-joint-capacity-plan.json', capacity)

        def preflight():
            initializers.check()
            if torch.cuda.is_initialized():
                raise ValueError('CPU source-capacity mechanics must not initialize CUDA')

        def progress(completed, metric):
            if rank == 0:
                write(output, 'progress.json', dict(phase='actual complete original joint capacity sources',
                    completed=completed, planned=len(entries), actual_original_joint_step=metric['gradient_interaction']['actual_original_joint_step'],
                    actual_counts=view.actor.counts(), production_updates=0, full_size_GPU_capacity_passed=False))

        proof = run(stage, output / 'state', maximum_bytes=32 * 1024**2, minimum_free_bytes=RESERVE,
            preflight=preflight, progress=progress, allow_mechanics=True)
        exported = None
        if rank == 0:
            word_step = next(r['original_joint_step'] - 1 for r in capacity['cases']
                if 'visual:words:single_context' in r['source_summary']['limits'])
            photo_step = next(r['original_joint_step'] - 1 for r in capacity['cases']
                if 'visual:images:additional_photo_captions:single_context' in r['source_summary']['limits'])
            word = source.prepare_visual(source.metadata(word_step)['components']['visual'], rank=0, world=4, device='cpu')
            photo = source.prepare_visual(source.metadata(photo_step)['components']['visual'], rank=0, world=4, device='cpu')
            longest = max(source.code.slots, key=lambda p: source.code.information(p)['complete_context_tokens'])

            def one(batch):
                width = int(batch['attention_mask'][0].sum())
                return {k: v[:1] if k == 'pixel_values' else v[:1, :width] for k, v in batch.items()}

            exported = finalize(view.actor, output / 'standalone', source.code.weak.folder / 'tokenizer.json', sources=sources,
                completed_updates=len(entries), probes=[('actual_original_longest_code', source.code.batch([longest])),
                    ('actual_original448photo', one(photo['native'])), ('actual_original_word_native', one(word['native']))],
                word_pixels=word['word']['word_pixels'], maximum_logit_error=1e-5,
                minimum_free_bytes=RESERVE, maximum_checkpoint_bytes=8 * 1024**2)
        write(output, 'rank-' + str(rank) + '-complete.json', dict(proof=proof, native_export=exported))
        preflight(); source.guard.check(full=True); initializers.check(full=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main(output):
    output = Path(output).absolute()
    if (output.exists() or output.resolve() != output or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(output.parent).free < RESERVE + MAX_OUTPUT):
        raise ValueError('A fresh funded CUDA-hidden small four-replica CPU capacity diagnostic is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.local_source_closure_v2 import python_sources
    files = {str(p): sha(p) for p in python_sources(ROOT, [Path(__file__)])}
    visual = ROOT / 'reports/joint-visual-actual-capacity-cases-CPU-v1/actual-complete-source-capacity-plan.json'
    files[str(visual)] = sha(visual)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    output.mkdir()
    write(output, 'plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        explicit_own_small_CPU_capacity_mechanics=True, all_selected_actual_complete_source_cases_required=True,
        original_full_paced_joint_steps=5362, minimum_free_bytes=RESERVE, maximum_output_bytes=MAX_OUTPUT,
        scratch_state_does_not_fund_persistent_GPU_training=True, production_updates=0, goal_achieved=False))
    try:
        from torch.multiprocessing import spawn
        spawn(worker, args=(str(output), files), nprocs=4, join=True)
        rows = [json.loads((output / ('rank-' + str(rank) + '-complete.json')).read_text()) for rank in range(4)]
        proofs = [r['proof'] for r in rows]
        if (len({p['actual_final_checkpoint_sha256'] for p in proofs}) != 1
                or any(not p['complete'] or p['full_size_GPU_capacity_passed'] or p['production_updates'] for p in proofs)):
            raise ValueError('All actual small replicas must finish the same full capacity workload without claiming GPU admission')
        write(output, 'complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_counts=rows[0]['native_export']['actual_counts'],
            actual_complete_original_capacity_cases=proofs[0]['actual_complete_capacity_cases'],
            actual_physical_small_CPU_AdamW_calls=sum(p['actual_physical_AdamW_calls'] for p in proofs),
            exact_next_selected_values_Adam_all_private_RNG_and_full_metric_replay=True,
            all_foreign_rank_private_generators_rejected_before_mutation=True,
            all_original_selected_source_extremes_corrective_negatives_and_real_empty_ranks_preserved=True,
            actual_longest_original_code448photo_and_word_native_export=rows[0]['native_export'],
            actual_rank_reports_sha256={str(rank): sha(output / ('rank-' + str(rank) + '-complete.json')) for rank in range(4)},
            full_size_GPU_capacity_passed=False, production_updates=0, model_quality_gain=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, output=str(output), physical_small_CPU_AdamW_calls=sum(p['actual_physical_AdamW_calls'] for p in proofs))), flush=True)
    except BaseException as exc:
        write(output, 'failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/complete-joint-capacity-CPU-v2')
    main(parser.parse_args().output)
