"""Independently rebuild/export the completed47-case real-source CPU actor.

Wait for the exact owned source run and every original replica to finish.
Reconstruct all47metrics/sparse Adam/private-RNG envelopes into a fresh
small model, then independently cold-load actual code/photo/word outputs.
No GPU context, full-size admission, new optimizer update or quality claim.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 40 * 1024**3
MAX_OUTPUT = 128 * 1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main(source_folder, output):
    source_folder, output = Path(source_folder).resolve(), Path(output).absolute()
    if (output.exists() or output.resolve() != output or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) < 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(output.parent).free < RESERVE + MAX_OUTPUT):
        raise ValueError('Fresh funded one-thread CUDA-hidden CPU export preparation is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    files = {str(p): sha(p) for p in python_sources(ROOT, [Path(__file__)])}
    producer_plan = read(source_folder / 'plan.json')
    if producer_plan.get('explicit_own_small_CPU_capacity_mechanics') is not True or producer_plan.get('original_full_paced_joint_steps') != 5362:
        raise ValueError('Only the actual complete original47-case own-small source run is eligible')
    files[str(source_folder / 'plan.json')] = sha(source_folder / 'plan.json')
    for p, h in producer_plan['source_files_sha256'].items():
        if p in files and files[p] != h:
            raise ValueError('Actual original and new export runtime sources cannot disagree')
        files[p] = h
    guard = FileBindings(files)
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    output.mkdir()

    def write(name, value):
        raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if (len(raw) > 32 * 1024**2 or shutil.disk_usage(output).free - len(raw) < RESERVE
                or sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) + len(raw) > MAX_OUTPUT):
            raise ValueError('Complete native export/evidence must fit above the unchanged40GiBscratch reserve')
        pending = output / (name + '.pending')
        with pending.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        pending.replace(output / name)

    write('plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), exact_original_source_owner=producer_plan['owner'],
        source_files_sha256=files, actual_original_capacity_cases=47, original_small_research_values=1092386,
        original_small_native_values=1090146, maximum_output_bytes=MAX_OUTPUT, minimum_free_bytes=RESERVE,
        scratch_metadata_and_export_do_not_fund_full_size_persistent_training=True,
        GPU_contexts=0, optimizer_updates=0, production_training_admitted=False, goal_achieved=False))
    try:
        deadline = time.monotonic() + 6 * 3600
        while not terminal(producer_plan['owner']):
            guard.check()
            write('progress.json', dict(phase='waiting for exact original complete47-case source owner', source_owner_live=True,
                source_observation_timeout_not_terminal=True, GPU_contexts=0, new_optimizer_updates=0))
            if time.monotonic() > deadline:
                raise ValueError('The CPUexport observation lease expired;the original source process is not relabelled terminal')
            time.sleep(15)
        done = read(source_folder / 'complete.json')
        if ((source_folder / 'failure.json').exists() or done.get('complete') is not True
                or done.get('actual_complete_original_capacity_cases') != 47 or done.get('actual_physical_small_CPU_AdamW_calls') != 192
                or done.get('full_size_GPU_capacity_passed') is not False or done.get('production_updates') != 0):
            raise ValueError('All47original cases/all192small optimizer calls must complete successfully before independent export')
        for rank, digest in done['actual_rank_reports_sha256'].items():
            path = source_folder / ('rank-' + rank + '-complete.json')
            if sha(path) != digest:
                raise ValueError('Every actual original four-replica completion report must retain its bytes')
            files[str(path)] = digest
        files[str(source_folder / 'complete.json')] = sha(source_folder / 'complete.json')
        import torch
        from slm_lab.complete_joint_training_data_v1 import CompleteJointTrainingDataV1
        from slm_lab.complete_joint_capacity_plan_v3 import choose
        from slm_lab.deterministic_positions_v1 import install
        from slm_lab.joint_indexed_durable_state_v3 import read_latest
        from slm_lab.joint_selected_native_export_v1 import export
        from slm_lab.model import ModelConfig
        from slm_lab.own_dense_actor_v1 import OwnDenseActor
        from slm_lab.own_dense_growth_v1 import OwnDenseAssistant, new_backbone
        from slm_lab.shared_joint_actor_vision_v1 import SharedJointActorVisionV1
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        shared, receipt, pointer = read_latest(source_folder / 'state')
        if shared['common']['cursor']['completed'] != 47 or shared['common']['world_size'] != 4:
            raise ValueError('The actual full47-case state and all original four private generators are required')
        config = ModelConfig(**shared['common']['model_metadata']['configuration'])
        source = CompleteJointTrainingDataV1(ROOT, config, word_expansion=5)
        visual = read(ROOT / 'reports/joint-visual-actual-capacity-cases-CPU-v1/actual-complete-source-capacity-plan.json')
        capacity = choose(source, visual); entries = [r['checkpoint_entry'] for r in capacity['cases']]
        own = install(OwnDenseAssistant(new_backbone(config, 996601), reward_hidden=8, word_temporal=4, pointing_rank=8,
            router_hidden=8, detector_hidden=4, word_expansion=5)).eval()
        view = SharedJointActorVisionV1(OwnDenseActor(own, **shared['common']['model_metadata']['actor_settings'])).eval()
        if view.actor.counts()['actual_unique'] != 1092386 or own.counts()['total_unique'] != 1090146:
            raise ValueError('This diagnostic may rebuild only the exact original own-small model,never a different full-size initializer')
        group = shared['common']['optimizer']['param_groups'][0]
        keys = ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'foreach', 'maximize', 'capturable', 'differentiable', 'fused']
        optimizer = torch.optim.AdamW([p for _, p in view.named_selected()], **{k: group[k] for k in keys})
        for mapping in [source.files, {str(source_folder / 'state/LATEST.json'): pointer,
                receipt['checkpoint']: receipt['checkpoint_sha256']}]:
            for path, digest in mapping.items():
                if path in files and files[path] != digest:
                    raise ValueError('The complete actual private state,source and exporter bindings must agree')
                files[path] = digest
        guard = FileBindings(files)
        word_step = next(r['original_joint_step'] - 1 for r in capacity['cases'] if 'visual:words:single_context' in r['source_summary']['limits'])
        photo_step = next(r['original_joint_step'] - 1 for r in capacity['cases'] if 'visual:images:additional_photo_captions:single_context' in r['source_summary']['limits'])
        word = source.prepare_visual(source.metadata(word_step)['components']['visual'], rank=0, world=4, device='cpu')
        photo = source.prepare_visual(source.metadata(photo_step)['components']['visual'], rank=0, world=4, device='cpu')
        longest = max(source.code.slots, key=lambda p: source.code.information(p)['complete_context_tokens'])

        def one(batch):
            width = int(batch['attention_mask'][0].sum())
            return {k: v[:1] if k == 'pixel_values' else v[:1, :width] for k, v in batch.items()}

        write('closed-source-bindings.json', dict(source_files_sha256=files, original_capacity_plan_sha256=sha(source_folder / 'actual-complete-joint-capacity-plan.json')))
        report = export(view, optimizer, source_folder / 'state', output / 'standalone', source.code.weak.folder / 'tokenizer.json',
            sources=shared['common']['sources'], entries=entries, source_files=files, expected_world=4, expected_completed=47,
            expected_pointer_sha256=pointer, mode='capacity', probes=[('actual_complete_longest_source_code', source.code.batch([longest])),
                ('actual448photo', one(photo['native'])), ('actual_word_native', one(word['native']))], word_pixels=word['word']['word_pixels'],
            working_budget_bytes=256 * 1024**2, maximum_checkpoint_bytes=8 * 1024**2, minimum_free_bytes=RESERVE,
            maximum_logit_error=1e-5, allow_mechanics=True)
        guard.check(full=True)
        if torch.cuda.is_initialized():
            raise ValueError('Independent CPU export must never acquire a GPU context')
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            actual_original_capacity_cases_reconstructed=47, actual_original_Adam_states_validated=report['actual_reconstruction']['actual_Adam_states_validated'],
            actual_independent_native_export=report, all_original_source_records_and_complete_selected_weights_reconstructed=True,
            original_four_private_generators_retained_but_not_restored=True, original_source_checkpoint_and_private_RNG_unchanged=True,
            GPU_contexts=0, optimizer_updates=0, full_size_GPU_cold_or_training_or_quality_proved=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, output=str(output), actual_original_capacity_cases_reconstructed=47)), flush=True)
    except BaseException as exc:
        write('failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); main(args.source, args.output)
