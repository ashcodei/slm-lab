"""Materialize the fixed completed own human-DPO initializer on CPU/scratch.

The text donor is fixed before any joint training result. Its own unchanged
public-pretraining foundation supplies vision/OCR. No old worker, GPU lease,
training schedule or assessment is modified. This cannot admit GPU training.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

ROOT = Path(__file__).resolve().parents[1]
RESERVE = 40 * 1024**3
NATIVE = 5 * 1024**3
METADATA = 512 * 1024**2
WORKING = 48 * 1024**3


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def completed_sources(root, observed=lambda row: None):
    """Close the actually consumed model/state evidence, without importing Torch."""
    from slm_lab.exact_waiting_process_retirement_v1 import terminal
    root = Path(root).resolve(); human = root / 'reports/phase-human-preference-v8'
    public = root / 'reports/public-pretraining-v1'
    paths = [human / n for n in ['plan.json', 'complete.json', 'children.json', 'restored.json',
        'human_DPO-training-complete.json', 'train-human_DPO-resume-distributed-state.json']]
    paths += [public / 'plan.json', public / 'training-complete.json']
    if any(not p.is_file() or p.is_symlink() for p in paths) or (human / 'failure.json').exists():
        raise ValueError('Every exact completed human source and original public-training receipt must remain')
    plan, done, children, restored, trained, receipt, foundation, pretraining = [read(p) for p in paths]
    if (not terminal(plan['owner']) or any(not terminal(c) for c in children)
            or restored.get('all_registered_point_actors_terminal') is not True
            or done.get('all_original_human_train_pairs_once_each_arm') is not True
            or done.get('all_held_human_pairs_and_native_development_assessments_complete') is not True
            or done.get('actual_source_epoch_updates_per_arm') != 5362
            or trained.get('actual_updates') != 5362 or trained.get('human_train_pairs') != 85786
            or trained.get('all_original_human_pairs_once') is not True
            or plan['training_options']['steps'] != 5362 or plan['training_options']['epochs'] != 1
            or plan['human_train_pairs'] != 85786 or plan['actual_counts']['actual_unique'] != 999706442
            or plan['actual_dense_counts']['total_unique'] != 984399690 or plan['actor_settings']['rank'] != 16):
        raise ValueError('The fixed original full human-DPO epoch, all original assessments and terminal actors are required')
    if (not terminal(foundation['owner']) or foundation['planned_steps'] != 4447
            or pretraining.get('actual_updates') != 4447 or pretraining.get('all_new_admitted_raw_train_windows_once') is not True
            or pretraining.get('all826verified_train_functions_once') is not True
            or foundation['tokenizer_sha256'] != plan['tokenizer_sha256']):
        raise ValueError('The same completed own public foundation and native vocabulary must supply the visual donor')
    checkpoint = root / 'checkpoints/phase-human-preference-v8/human_DPO-actor.pt'
    base = root / 'checkpoints/public-pretraining-v1/trained/model.pt'
    token = base.parent / 'tokenizer.json'; inputs = plan['foundation_inputs']
    if (Path(inputs['checkpoint']).resolve() != base or Path(inputs['plan']).resolve() != public / 'plan.json'
            or inputs['plan_sha256'] != sha(public / 'plan.json')
            or plan['source_files_sha256'].get(str(base)) != inputs['checkpoint_sha256']
            or receipt.get('completed') != 5362 or receipt.get('complete_distributed_state_requires_every_private_file') is not True
            or [row['rank'] for row in receipt.get('all_four_private_states', [])] != [0, 1, 2, 3]):
        raise ValueError('The actual fixed foundation and every private final human state must remain bound')
    expected = {str(base): inputs['checkpoint_sha256'], str(token): plan['tokenizer_sha256']}
    for row in receipt['all_four_private_states']:
        path = Path(row['path'])
        if (row['actual_completed'] != 5362 or path.parent != checkpoint.parent
                or path.name != 'train-human_DPO-resume-slot-' + str(receipt['committed_generation'] % 2) + '-rank-' + str(row['rank']) + '.pt'):
            raise ValueError('Only all four original committed final human private files may supply source evidence')
        expected[str(path)] = row['sha256']
    paths += [checkpoint, base, token, *[Path(row['path']) for row in receipt['all_four_private_states']]]
    files = {}
    for p in paths:
        if not p.is_file() or p.is_symlink() or p.resolve() != p:
            raise ValueError('Every actual original source must remain a direct regular file')
        observed(dict(phase='hashing fixed completed model and private-state evidence', path=str(p)))
        digest = sha(p)
        if str(p) in expected and digest != expected[str(p)]:
            raise ValueError('An original actual completed source byte differs from its committed identity')
        files[str(p)] = digest
    return dict(fixed_text_arm='human_DPO', checkpoint=str(checkpoint), checkpoint_sha256=files[str(checkpoint)],
        plan=str(human / 'plan.json'), plan_sha256=files[str(human / 'plan.json')], tokenizer=str(token),
        tokenizer_sha256=files[str(token)], own_visual_foundation=str(base), original_final_private_receipt=receipt,
        source_files_sha256=files, original_full_human_and_public_source_epochs_complete=True,
        original_training_corpus_and_preference_likelihoods_not_reread=True,
        quality_selection_or_GPU_admission=False)


def main(output):
    output = Path(output).absolute()
    if (output.exists() or output.resolve() != output or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'])
            or shutil.disk_usage(output.parent).free < RESERVE + NATIVE + METADATA):
        raise ValueError('A fresh funded one-thread CUDA-hidden CPU initializer workspace is required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    files = {str(p): sha(p) for p in python_sources(ROOT, [Path(__file__)])}
    owner = dict(pid=os.getpid(), ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    output.mkdir()
    def write(name, value):
        raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if len(raw) > 32 * 1024**2 or shutil.disk_usage(output).free - len(raw) < RESERVE:
            raise ValueError('Complete metadata must retain its unchanged40GiBactual-filesystem reserve')
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) + len(raw) > NATIVE + METADATA:
            raise ValueError('Every actual initial native and complete metadata byte must fit the declared output budget')
        pending = output / (name + '.pending')
        with pending.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        pending.replace(output / name)
    write('plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        fixed_text_source='completed original phase-human-preference-v8 human_DPO; no future quality selection',
        fixed_visual_source='same own completed public-pretraining-v1 foundation', actual_CPU_working_budget_bytes=WORKING,
        unchanged_other_worker_memory_headroom_bytes=64 * 1024**3, maximum_native_bytes=NATIVE, maximum_metadata_bytes=METADATA,
        minimum_free_bytes=RESERVE, all_original_workers_and_experiments_preserved=True,
        GPU_contexts=0, new_optimizer_updates=0, full_size_GPU_training_admitted=False, goal_achieved=False))
    try:
        admission = completed_sources(ROOT, lambda row: write('progress.json', row))
        write('source-admission.json', admission)
        files.update(admission['source_files_sha256'])
        for path in [output / 'plan.json', output / 'source-admission.json']:files[str(path)] = sha(path)
        guard = FileBindings(files)
        def preflight():
            guard.check()
            stats = {k: int(v) for k, v in (line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())}
            available = int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(stats[k] for k in ['anon', 'shmem', 'kernel'])
            if (available < WORKING + 64 * 1024**3 or shutil.disk_usage(output).free < RESERVE
                    or os.sched_getscheduler(0) != os.SCHED_IDLE or os.nice(0) != 19):
                raise ValueError('Actual working memory, other-worker headroom, idle ownership and40GiBreserve must remain funded')
        preflight()
        write('progress.json', dict(phase='loading fixed completed own human-DPO model', new_optimizer_updates=0))
        import torch
        from slm_lab.completed_human_native_initializer_v1 import verify_private_states, materialize
        from slm_lab.complete_joint_training_data_v1 import CompleteJointTrainingDataV1
        from slm_lab.phase_human_policy_v3 import load, options, sources as human_sources
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        actor, _ = load(admission['checkpoint'], admission['plan'], 'human_DPO', expected_checkpoint_sha256=admission['checkpoint_sha256'],
            expected_plan_sha256=admission['plan_sha256'], device='cpu')
        actor.eval(); plan = read(admission['plan'])
        write('progress.json', dict(phase='independently reading all four final human actor/Adam/metric states', new_optimizer_updates=0))
        def states():
            for row in admission['original_final_private_receipt']['all_four_private_states']:
                preflight(); yield torch.load(row['path'], map_location='cpu', weights_only=True, mmap=True)
        verified = verify_private_states(actor, states(), options=options(plan, 'human_DPO'), population=85786,
            sources=human_sources(plan, admission['plan_sha256']))
        write('actual-completed-human-state-readback.json', verified)
        preflight(); write('progress.json', dict(phase='preparing actual complete code/photo/word cold inputs', new_optimizer_updates=0))
        source = CompleteJointTrainingDataV1(ROOT, actor.config, word_expansion=5)
        files.update(source.files)
        capacity_path = ROOT / 'reports/complete-joint-actual-capacity-cases-CPU-v3/actual-complete-source-capacity-plan.json'
        capacity = read(capacity_path); files[str(capacity_path)] = sha(capacity_path)
        guard = FileBindings(files)
        photo_step = next(r['original_joint_step'] - 1 for r in capacity['cases']
            if 'visual:images:additional_photo_captions:single_context' in r['source_summary']['limits'])
        word_step = next(r['original_joint_step'] - 1 for r in capacity['cases'] if 'visual:words:single_context' in r['source_summary']['limits'])
        photo = source.prepare_visual(source.metadata(photo_step)['components']['visual'], rank=0, world=4, device='cpu')
        word = source.prepare_visual(source.metadata(word_step)['components']['visual'], rank=0, world=4, device='cpu')
        shortest = min(source.code.slots, key=lambda p: source.code.information(p)['complete_context_tokens'])
        def one(batch):
            width = int(batch['attention_mask'][0].sum())
            return {k: v[:1] if k == 'pixel_values' else v[:1, :width] for k, v in batch.items()}
        probes = [('actual_complete_original_short_code', source.code.batch([shortest])),
            ('actual_complete448photo', one(photo['native'])), ('actual_complete_word_native', one(word['native']))]
        identities = dict(completed_human_policy_sha256=admission['checkpoint_sha256'], completed_human_plan_sha256=admission['plan_sha256'],
            own_visual_foundation_sha256=plan['foundation_inputs']['checkpoint_sha256'],
            completed_source_admission_sha256=sha(output / 'source-admission.json'),
            actual_four_private_state_readback_sha256=sha(output / 'actual-completed-human-state-readback.json'))
        write('progress.json', dict(phase='assembling and independently cold-loading full984Mnative', new_optimizer_updates=0))
        result = materialize(actor, output / 'native', admission['tokenizer'], sources=identities, verified=verified,
            probes=probes, word_pixels=word['word']['word_pixels'], working_budget_bytes=WORKING,
            maximum_checkpoint_bytes=NATIVE, minimum_free_bytes=RESERVE, preflight=preflight)
        guard.check(full=True); source.guard.check(full=True); preflight()
        write('complete.json', dict(result, owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            original_full4096text_GPUcold_still_required=True, actual_native_output_filesystem=str(output),
            persistent_final_export_and_full_joint_workspace_funding_pending=True))
        print(json.dumps(dict(complete=True, output=str(output), actual_counts=result['actual_counts'], joint_updates=0)), flush=True)
    except BaseException as exc:
        write('failure.json', dict(owner=owner, error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False,
            original_workers_and_partial_outputs_preserved=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/slm-completed-joint-initializer-CPU-v1'))
    main(parser.parse_args().output)
