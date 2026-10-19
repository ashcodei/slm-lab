"""Wait for real capacity, then resume the unchanged original frozen curriculum once."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json
from train_multimodal_pilot import complete_stage

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    'scripts/train_extended_curriculum.py': '3d5dcca3651f2ced8515f4ecb90c981bd136947ed67473081e6d912b3c8ac35d',
    'src/slm_lab/extended_curriculum_recovery.py': '4908676721efc81b82870d3e7f0e71d55bb517ae42d3096ba4bdfe136d84590d',
    'reports/extended-curriculum-v1/plan.json': '7e684dfd378036bc5a9ae0e00866c2d636739f5a4735a74a7cc61d0be8b9ae7e',
    'scripts/train_dialogue_retention.py': 'd0da435b96ed8b702429fe2f45f05e867cdfe8aec2a00c1f702a15fd03cf9afa',
    'src/slm_lab/model.py': 'd94fa54c63c2cb7a89a72ac33016d7d7dcc70fc91d13ad739b95c26220d22d1e',
    'src/slm_lab/training.py': '2b0c4583841a6d82cfa33c39b3a586dc9480ef35d426feb6c37b090fabdda04a',
}


def no_other_training_controller():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
            state = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[0]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if state in {'Z', 'X'}:
            continue
        if any(a.endswith(b'scripts/train_extended_curriculum.py')
                or a.endswith(b'scripts/train_dialogue_retention.py')
                or a == b'slm_lab.training' for a in args):
            raise ValueError('Another original training/controller process is live; never launch duplicates')


def main():
    reports = ROOT / 'reports/curriculum-disk-cleanup-resumption-v1'
    if reports.exists():
        raise ValueError('Inspect any earlier resumption; no implicit restart')
    bound = {str(ROOT / name): digest for name, digest in EXPECTED.items()}
    bound[str(Path(__file__))] = file_hash(Path(__file__))
    plan = json.loads((ROOT / 'reports/extended-curriculum-v1/plan.json').read_text())
    completed = []
    for spec in plan['stages'][:3]:
        directory = ROOT / 'checkpoints/extended-curriculum-v1' / spec['name']
        if not complete_stage(directory, spec):
            raise ValueError('All three preceding stages must actually have completed their frozen budgets')
        model = directory / 'model.pt'
        digest = file_hash(model)
        if spec['name'] == 'verified-standalone-code' and digest != '639a73762684d93067b9c4084b652c3c8a50af23201f09e0913aaae2820dc988':
            raise ValueError('Exact newly completed standalone-code model required')
        bound[str(model)] = digest
        for name in ['training.json', 'tokenizer.json']:
            bound[str(directory / name)] = file_hash(directory / name)
        metadata = json.loads((directory / 'training.json').read_text())
        completed.append(dict(stage=spec['name'], steps=metadata['steps'], planned_steps=metadata['planned_steps'],
            checkpoint=str(model), checkpoint_sha256=digest,
            reported_truncated_sequences=metadata['truncated_sequences']))
    for name in ['expanded-everyday-photos-v1', 'expanded-scene-words-v1', 'clevr-reasoning-v1']:
        manifest = ROOT / 'data' / name / 'manifest.json'
        bound[str(manifest)] = file_hash(manifest)
    guard = FileBindings(bound)
    no_other_training_controller()
    previous = json.loads((ROOT / 'reports/extended-curriculum-v1/status.json').read_text())
    if previous.get('error') != 'Insufficient persistent space for atomic curriculum checkpoints':
        raise ValueError('Inspect any different original stop condition before scheduling a resumption')
    reports.mkdir()
    shutil.copy2(ROOT / 'reports/extended-curriculum-v1/status.json', reports / 'preceding-main-status.json')
    shutil.copy2(ROOT / 'reports/dialogue-retention-v1/status.json', reports / 'preceding-joint-status.json')

    def status(message, **fields):
        atomic_json(reports / 'status.json', dict(status=message,
            updated_at=datetime.now(timezone.utc).isoformat(), goal_achieved=False, **fields))

    atomic_json(reports / 'plan.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=bound, verified_completed_stages=completed,
        next_frozen_stage=plan['stages'][3], minimum_home_free_bytes=25 * 1024**3 + 256 * 1024**2,
        original_per_stage_preflight_bytes=25 * 1024**3, maximum_initial_GPU_memory_used_MiB=128,
        polling_seconds=15, user_authorization='Continue original training; user offered to clear disk space.',
        existing_data_models_controllers_and_guards_modified=False,
        maximum_main_launches=1, no_timeout_based_restarts=True, goal_achieved=False))
    try:
        while True:
            guard.check()
            no_other_training_controller()
            free = shutil.disk_usage(ROOT).free
            memory = [int(s) for s in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True).splitlines()]
            if free >= 25 * 1024**3 + 256 * 1024**2 and len(memory) == 4 and max(memory) <= 128:
                break
            status('waiting for disk cleanup and idle GPUs before one checked frozen-curriculum resumption',
                free_home_bytes=free, required_home_bytes=25 * 1024**3 + 256 * 1024**2,
                GPU_memory_used_MiB=memory, original_training_restarted=False)
            time.sleep(15)
        guard.check(full=True)
        no_other_training_controller()
        if shutil.disk_usage(ROOT).free < 25 * 1024**3 + 256 * 1024**2:
            raise ValueError('Actual capacity changed before launch; inspect the waiting evidence')
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
        python = str(ROOT / '.venv/bin/python')

        def launch(name, args):
            log = reports / (name + '.log')
            argv = [python, '-u', *args]
            with log.open('xb') as stream:
                process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
            stat = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
            receipt = dict(launched_at=datetime.now(timezone.utc).isoformat(), pid=process.pid,
                start_ticks=stat[19], argv=argv, log=str(log))
            atomic_json(reports / (name + '-launch.json'), receipt)
            return process, receipt

        main_process, main_receipt = launch('main', ['scripts/train_extended_curriculum.py', '--resume-frozen-plan'])
        # Finalized collector manifests are bound above. Zero explicitly denotes
        # no live collector; it avoids reusing unrelated historical process IDs.
        paired = Path('/proc/130149/cmdline')
        paired_manifest = ROOT / 'data/vqa-real-pairs-v1/manifest.json'
        if not paired_manifest.exists() and (not paired.exists() or b'slm_lab.vqa_pairs' not in paired.read_bytes().split(b'\0')):
            raise ValueError('Original paired-photo collector must be live or finalized before joint scheduling')
        joint_process, joint_receipt = launch('joint', ['scripts/train_dialogue_retention.py',
            '--preceding-pid', str(main_process.pid), '--photos-pid', '0', '--words-pid', '0',
            '--clevr-pid', '0', '--vqa-pairs-pid', '130149'])
        status('original frozen curriculum resumed once; guarded joint continuation scheduled',
            main=main_receipt, joint=joint_receipt)
        code = main_process.wait()
        guard.check(full=True)
        status('original resumed main process is terminal; inspect its authoritative final state',
            main=main_receipt, main_exit_code=code,
            main_status=json.loads((ROOT / 'reports/extended-curriculum-v1/status.json').read_text()),
            automatic_relaunch_permitted=False)
    except Exception as error:
        status('resumption stopped; inspect preserved evidence', error=str(error))
        raise


if __name__ == '__main__':
    main()
