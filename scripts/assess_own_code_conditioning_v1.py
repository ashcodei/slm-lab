"""Full train-only question/response conditioning matrices on the shared986M model."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import train_point_grounding_joint_v1 as owner
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/own-code-conditioning-v1'
MANIFEST = ROOT / 'checkpoints/own-assistant-candidate-v1/manifest.json'
MAXIMUM_SECONDS = 2 * 60 * 60
owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS


def previous_ended():
    for folder in ['own-code-generation-repair-v1', 'frozen-base-pointer-v1', 'own-view-contrastive-v1']:
        previous = ROOT / 'reports' / folder
        plan = json.loads((previous / 'plan.json').read_text())
        actors = json.loads((previous / 'children.json').read_text())
        if owner.resources.same(plan['owner']) or any(owner.resources.same(actor) for actor in actors):
            raise ValueError('Every exact preceding owner and registered actor must actually end before resource queries')
        if not (previous / 'restored.json').exists():
            raise ValueError('Completed preceding owner must have restored the original waiter')


def registered(shard):
    plan = json.loads((REPORTS / 'plan.json').read_text()); owner.owned(plan)
    actual = owner.resources.identity(os.getpid())
    while True:
        owner.owned(plan)
        actors = json.loads((REPORTS / 'children.json').read_text())
        if any(a['pid'] == actual['pid'] and a['ticks'] == actual['ticks'] and a['subject'] == 'shard-' + str(shard) for a in actors):
            return plan
        time.sleep(.05)


def worker(shard):
    plan = registered(shard)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(shard):
        raise ValueError('Exact isolated physical L40 required before CUDA initialization')
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    import torch
    from tokenizers import Tokenizer
    from slm_lab.own_assistant_v1 import load_manifest
    from slm_lab.own_code_conditioning_v1 import sequences, score
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    torch.set_num_threads(1); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.65, 0)
    if torch.cuda.get_device_name(0) != 'NVIDIA L40':
        raise ValueError('Actual original L40 required')
    model, metadata = load_manifest(MANIFEST, expected_manifest_sha256=plan['manifest_sha256'], device='cuda:0')
    tokenizer = Tokenizer.from_file(plan['tokenizer'])
    rows = list(map(json.loads, (REPORTS / 'calibration.jsonl').read_text().splitlines()))
    matrix = sequences(rows, tokenizer)
    assigned = {row['id'] for index, row in enumerate(rows) if index % 4 == shard}
    before = {name: state_digest(part) for name, part in [('base', model.language.base), ('code', model.language.parts),
        ('query_projection', model.grounding.query_projection), ('patch_projection', model.grounding.patch_projection)]}
    versions = [(id(p), p._version, p.data_ptr()) for p in model.parameters()]
    path = REPORTS / f'scores-{shard}.jsonl'
    with path.open('x') as stream:
        for sequence in matrix:
            if sequence['target_id'] not in assigned:
                continue
            guard.check(); owner.owned(plan)
            for route in plan['routes']:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    record = score(model.language.route(route), sequence, device='cuda:0', checked=lambda: owner.owned(plan))
                record['route'] = route
                stream.write(json.dumps(record, allow_nan=False) + '\n'); stream.flush()
            if sequence['condition'] == 'proper':
                print(json.dumps(dict(id=sequence['target_id'], condition='proper', actual_route=route,
                    complete_response_EOS_NLL=record['whole_response_EOS_mean_NLL'], target_tokens=record['target_tokens'])), flush=True)
        os.fsync(stream.fileno())
    guard.check(full=True); owner.owned(plan)
    after = {name: state_digest(part) for name, part in [('base', model.language.base), ('code', model.language.parts),
        ('query_projection', model.grounding.query_projection), ('patch_projection', model.grounding.patch_projection)]}
    if before != after or versions != [(id(p), p._version, p.data_ptr()) for p in model.parameters()]:
        raise ValueError('All actual original/code/head weights, pointers and versions must remain unchanged')
    atomic_json(REPORTS / f'complete-{shard}.json', dict(actual_counts=model.counts(), scores_sha256=file_hash(path),
        component_digests_before=before, component_digests_after=after, all_weights_and_versions_unchanged=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0), actual_GPU=torch.cuda.get_device_name(0),
        model_updates=0, candidate_promoted=False, goal_achieved=False))


def combine(plan):
    from tokenizers import Tokenizer
    from slm_lab.own_code_conditioning_v1 import sequences, summarize
    tokenizer = Tokenizer.from_file(plan['tokenizer'])
    rows = list(map(json.loads, (REPORTS / 'calibration.jsonl').read_text().splitlines()))
    matrix = sequences(rows, tokenizer); sequence_map = {s['sequence_sha256']: s for s in matrix}
    records = []
    for shard in range(4):
        path = REPORTS / f'scores-{shard}.jsonl'; done = json.loads((REPORTS / f'complete-{shard}.json').read_text())
        if file_hash(path) != done['scores_sha256'] or not done['all_weights_and_versions_unchanged'] or done['actual_counts']['total_unique'] != 986450944:
            raise ValueError('Every complete unchanged actual full-model GPU worker required')
        part = list(map(json.loads, path.read_text().splitlines()))
        expected = [(s['sequence_sha256'], route) for s in matrix
            if s['target_id'] in {r['id'] for i, r in enumerate(rows) if i % 4 == shard} for route in plan['routes']]
        if [(r['sequence_sha256'], r['route']) for r in part] != expected:
            raise ValueError('All original disjoint source matrix cells must reconstruct exactly once')
        for record in part:
            original = sequence_map[record['sequence_sha256']]
            if record['target_ids'] != original['input_ids'][original['prefix_tokens']:]:
                raise ValueError('Every actual complete human-response/EOS target must reconstruct')
        records.extend(part)
    summary = summarize(records, rows, routes=plan['routes'])
    atomic_json(REPORTS / 'comparison.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        actual_unique_parameters=986450944, complete_original_training_programs=16,
        actual_complete_question_target_route_scores=len(records), routes=summary,
        qualification='Teacher-forced train likelihood, not greedy functional accuracy or verified wrong-answer labels. Other complete questions differ in length/position; semantic equivalence is not certified.',
        all_original_target_EOS_losses_argmax_and_matrix_entries_reconstructed=True,
        no_external_teacher_or_model=True, model_updates=0, candidate_promoted=False, goal_achieved=False))


def parent():
    if REPORTS.exists():
        raise ValueError('Preserve previous attempts and generations; no implicit retry')
    previous_ended(); owner.idle_original()
    from tokenizers import Tokenizer
    from slm_lab.own_view_contrastive_data_v1 import prepare
    from slm_lab.evaluation_provenance import capture
    from slm_lab.apps_published_cases_v2 import implementation_sources
    spec = json.loads(MANIFEST.read_text()); tokenizer_path = (MANIFEST.parent / spec['tokenizer']).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); rows = prepare(ROOT, tokenizer)
    files = {str(ROOT / name): sha for name, sha in capture()['files_sha256'].items()}
    files.update(implementation_sources())
    names = ['scripts/assess_own_code_conditioning_v1.py', 'src/slm_lab/own_code_conditioning_v1.py', 'scripts/train_point_grounding_joint_v1.py',
        'scripts/assess_joint_java_comparison_v1.py', 'src/slm_lab/own_view_contrastive_v1.py',
        'src/slm_lab/own_view_contrastive_data_v1.py', 'src/slm_lab/own_assistant_v1.py',
        'src/slm_lab/own_code_delta_v1.py', 'src/slm_lab/own_code_delta_inference_v1.py',
        'src/slm_lab/own_code_training_inference_v1.py', 'src/slm_lab/own_code_training_stage_v1.py',
        'src/slm_lab/frozen_base_pointer_stage_v1.py', 'src/slm_lab/frozen_base_pointer_inference_v1.py',
        'src/slm_lab/point_grounding_head_v1.py', 'src/slm_lab/point_grounding_data_v1.py',
        'src/slm_lab/point_grounding_evaluation_v1.py', 'src/slm_lab/codenet_checkpoint_v1.py',
        'tests/test_own_code_conditioning_v1.py', 'reports/own-code-conditioning-v1-tests.log',
        'tests/test_own_code_conditioning_driver_v1.py', 'reports/own-code-conditioning-driver-v1-tests.log',
        'reports/own-assistant-candidate-v1/validation-complete.json',
        'reports/own-code-generation-repair-endpoint-v2/complete.json',
        'reports/frozen-base-pointer-v1/complete-head-source-readback.json',
        'reports/frozen-base-pointer-v1/matched-attention-and-native-comparison.json',
        'reports/frozen-base-pointer-full-CPU-v1/complete.json',
        'data/APPS-verified-generation-v1/admission.json', 'data/APPS-verified-generation-v1/manifest.json',
        'data/APPS-verified-generation-v1/original-rows.jsonl.gz',
        'data/localized-pointer-candidate-v1/admission.json']
    for name in names:
        files[str(ROOT / name)] = file_hash(ROOT / name)
    for name in ['base_checkpoint', 'tokenizer', 'code_checkpoint', 'code_initial_plan', 'code_training_plan', 'grounding_head', 'grounding_plan']:
        path = (MANIFEST.parent / spec[name]).resolve(); files[str(path)] = spec[name + '_sha256']
    files[str(MANIFEST)] = file_hash(MANIFEST)
    for row in rows:
        files.update(row['original_files_sha256'])
    if '3 passed' not in (ROOT / 'reports/own-code-conditioning-v1-tests.log').read_text():
        raise ValueError('Actual shifted whole-target scores and complete matrix reconstruction checks required')
    if '2 passed' not in (ROOT / 'reports/own-code-conditioning-driver-v1-tests.log').read_text():
        raise ValueError('Actual active predecessor/waiter rejection before resource queries required')
    guard = FileBindings(files)
    with (ROOT / 'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous_ended(); owner.idle_original()
        memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(memory) != 4 or any(int(x.strip()) > 128 for x in memory):
            raise ValueError('All four GPUs must be idle before this bounded pilot')
        if shutil.disk_usage(ROOT).free < 26 * 1024**3 + 128 * 1024**2:
            raise ValueError('Keep the original26GiB persistent reserve plus128MiB pilot commitment')
        guard.check(full=True); REPORTS.mkdir()
        with (REPORTS / 'calibration.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row) + '\n')
        files[str(REPORTS / 'calibration.jsonl')] = file_hash(REPORTS / 'calibration.jsonl')
        guard = FileBindings(files)
        expires = datetime.now(timezone.utc) + timedelta(seconds=MAXIMUM_SECONDS)
        plan = dict(created_at=datetime.now(timezone.utc).isoformat(), owner=owner.resources.identity(os.getpid()),
            waiter=owner.WAITER, source_files_sha256=files, manifest_sha256=file_hash(MANIFEST),
            tokenizer=str(tokenizer_path), training_problems=16, routes=['general', 'code'], conditions=['proper', 'other_question', 'no_problem'],
            maximum_complete_response_EOS_targets=512, arithmetic='CUDA BF16 autocast, complete original human-program shifted target/EOS scores',
            source_selection='Eight stdin and eight function original verified publisher-train problems, shortest-reference pack, targets<=512, whole prompts+512<=4096; fixed SHA256 seed ordering, no model-result selection',
            selection_seed='own-view-contrastive-train-calibration-v1',
            selection_gate='No model or decoder selection; full same-interface8x8question/target matrices plus generic-prompt controls',
            diagnostic_not_a_new_method_or_novelty_claim=True,
            method='Full teacher-forced original human program/EOS likelihood under every same-interface original question and generic prompt; report exact per-target losses and argmax; no gradients',
            expires_at_unix=expires.timestamp(), maximum_seconds=MAXIMUM_SECONDS,
            human_preference_RLHF=False, model_updates=0, candidate_promoted=False, goal_achieved=False)
        atomic_json(REPORTS / 'plan.json', plan); atomic_json(REPORTS / 'children.json', [])
        watcher = subprocess.Popen(['.venv/bin/python', '-u', str(Path(__file__).resolve()), '--watchdog'], cwd=ROOT,
            stdout=(REPORTS / 'watchdog.log').open('xb'), stderr=subprocess.STDOUT, start_new_session=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Owned question/response conditioning diagnostic interrupted')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        children = []
        try:
            owner.idle_original(); os.kill(owner.WAITER['pid'], signal.SIGSTOP); until = time.monotonic() + 5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in {'T', 't'}:
                if time.monotonic() > until:
                    raise ValueError('Original idle waiter did not enter owned hold')
                time.sleep(.05)
            if (Path('/proc') / str(owner.WAITER['pid']) / 'task' / str(owner.WAITER['pid']) / 'children').read_text().strip():
                raise ValueError('Original waiter began work before hold; do not launch')
            actors = []
            for shard in range(4):
                with (REPORTS / f'shard-{shard}.log').open('xb') as stream:
                    process = subprocess.Popen(['.venv/bin/python', '-u', str(Path(__file__).resolve()), '--worker', str(shard)],
                        cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(shard), 'OMP_NUM_THREADS': '1'})
                children.append(process); actual = owner.resources.identity(process.pid)
                if actual is None:
                    raise ValueError('Worker ended before parent birth registration')
                actors.append(dict(**actual, subject='shard-' + str(shard))); atomic_json(REPORTS / 'children.json', actors)
            while any(p.poll() is None for p in children):
                owner.owned(plan)
                if any(p.poll() not in [None, 0] for p in children):
                    raise ValueError('An owned pilot worker failed; preserve generations without retry')
                atomic_json(REPORTS / 'status.json', dict(status='original-train question/response conditioning matrices active',
                    observed_at=datetime.now(timezone.utc).isoformat(), workers=[dict(pid=p.pid, exit_code=p.poll()) for p in children],
                    model_updates=0, goal_achieved=False)); time.sleep(2)
            if any(p.returncode != 0 for p in children):
                raise ValueError('All four complete calibration workers must succeed')
            owner.owned(plan); combine(plan); guard.check(full=True)
            atomic_json(REPORTS / 'status.json', dict(status='train conditioning matrices complete; inspect measured likelihood controls', goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS / 'failure.json', dict(error=str(error), automatic_retry=False, goal_achieved=False)); raise
        finally:
            owner.restore_owned(plan, 'Question/response conditioning diagnostic ended; exact original waiter restored')
            watcher.wait(timeout=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=range(4)); parser.add_argument('--watchdog', action='store_true')
    args = parser.parse_args()
    if args.watchdog:
        owner.watchdog()
    elif args.worker is not None:
        worker(args.worker)
    else:
        parent()
