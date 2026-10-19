"""Queue all3088 original visual questions at eight fixed own-model endpoints.

Runs only after the exact full text-QA predecessor and all its workers finish,
with restoration receipts and the original GPUs idle. Each question receives
proper/gray/fixed-wrong images. No training, retry, selection or promotion.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from slm_lab.code_first_endpoints_v2 import ENDPOINTS,COMPARISONS,descriptors,load_native
import train_whole_human_SFT_v1 as legacy
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/full-AI2D-pipeline-v3'
QUEUE = ROOT / 'reports/full-AI2D-pipeline-v3-queue.json'
PREVIOUS = ROOT / 'reports/full-held-QA-pipeline-v3'
PREVIOUS_QUEUE = ROOT / 'reports/full-held-QA-pipeline-v3-queue.json'
PREPARATION = ROOT / 'reports/ai2d-full-native-evaluation-v1-preparation/complete.json'
owner = legacy.owner


def configure():
    owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS; legacy.REPORTS = REPORTS


def previous_ready(identity):
    if owner.resources.same(identity): return False
    children = PREVIOUS / 'children.json'
    if children.exists() and any(owner.resources.same(p) for p in json.loads(children.read_text())): return False
    if ((PREVIOUS / 'failure.json').exists()
            or any(not (PREVIOUS / n).exists() for n in ['plan.json', 'complete.json', 'restored.json', 'paired-comparisons.json'])):
        raise ValueError('Complete exact original text-QA predecessor, all assessments and restoration required')
    plan = json.loads((PREVIOUS / 'plan.json').read_text()); done = json.loads((PREVIOUS / 'complete.json').read_text())
    if (done.get('all_eight_predeclared_endpoints_assessed') is not True or done.get('original_questions_each_endpoint') != 16961
            or done.get('total_original_question_endpoint_assessments') != 135688
            or done.get('all_raw_deliveries_and_question_verdicts_retained') is not True
            or plan.get('full_QA_questions_each_endpoint') != 16961 or set(plan.get('endpoints', {})) != set(ENDPOINTS)):
        raise ValueError('Every original text-QA endpoint and complete full-population receipt required')
    for endpoint in ENDPOINTS:
        if not (PREVIOUS / endpoint / 'full-QA.json').is_file(): raise ValueError('Every complete native text-QA report required')
        for gpu in range(4):
            receipt = json.loads((PREVIOUS / endpoint / f'QA-shard-{gpu}-complete.json').read_text())
            if (receipt.get('endpoint') != endpoint or receipt.get('physical_GPU') != gpu
                    or receipt.get('actual_original_questions') != (4241 if gpu == 0 else 4240)
                    or receipt.get('all_loaded_values_unchanged') is not True or receipt.get('model_updates') != 0):
                raise ValueError('All32 exact complete original text-QA shard receipts required')
    return True


def worker(endpoint, gpu):
    plan = legacy.registered('AI2D-' + endpoint + '-' + str(gpu), gpu)
    import torch
    from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D, evaluate_shard
    from slm_lab.dense_foundation_inference_v1 import DenseNativeView
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    torch.set_num_threads(1); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.85, 0)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.get_device_name(0) != 'NVIDIA L40': raise ValueError('Only the original physical L40 is admitted')
    guard = FileBindings(plan['source_files_sha256']); guard.check(full=True); spec = plan['endpoints'][endpoint]
    native, metadata = load_native(spec,device='cuda:0')
    actual_count = sum(p.numel() for p in native.parameters())
    signature = [(id(p), p._version, p.data_ptr()) for p in native.parameters()]
    population = FullAI2D(ROOT)
    if metadata['tokenizer_sha256'] != population.tokenizer_sha256: raise ValueError('Exact own native image/text tokenizer required')
    folder = REPORTS / endpoint; folder.mkdir(exist_ok=True); journal = folder / f'AI2D-shard-{gpu}-actual-deliveries.jsonl'
    with journal.open('x') as output:
        count = 0
        def record(row):
            nonlocal count
            output.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'); output.flush(); count += 1
            if count % 25 == 0:
                os.fsync(output.fileno()); atomic_json(folder / f'AI2D-shard-{gpu}-progress.json',
                    dict(actual_original_questions=count, actual_question_condition_assessments=3 * count,
                        original_selected_questions=772, last_original_id=row['id'], model_updates=0, goal_achieved=False))
        with inference_cache():
            report = evaluate_shard(native, population.tokenizer, population, folder / f'AI2D-shard-{gpu}.json', rank=gpu, world=4,
                checkpoint_sha256=spec['checkpoint_sha256'], checked=lambda: owner.owned(plan), record=record)
        os.fsync(output.fileno())
    if signature != [(id(p), p._version, p.data_ptr()) for p in native.parameters()] or any(p.grad is not None for p in native.parameters()):
        raise ValueError('Every native model value must stay unchanged and all gradients absent')
    owner.owned(plan); guard.check(full=True)
    atomic_json(folder / f'AI2D-shard-{gpu}-complete.json', dict(physical_GPU=gpu, endpoint=endpoint,
        actual_original_questions=len(report['results']), actual_question_condition_assessments=3 * len(report['results']),
        actual_student_values=actual_count, actual_delivery_journal_sha256=file_hash(journal),
        all_loaded_values_unchanged=True, model_updates=0, goal_achieved=False))


def paired(before, after):
    from slm_lab.ai2d_full_native_evaluation_v1 import CONDITIONS
    for key in ['original_population_sha256', 'protocol', 'generation_budget', 'conditions', 'inference_device_type', 'inference_precision']:
        if before[key] != after[key]: raise ValueError('Exact full original visual protocol must match between endpoints')
    if len(before['results']) != 3088 or len(after['results']) != 3088: raise ValueError('All3088 original visual questions required at both endpoints')
    result = {condition: dict(questions=0, before_strict_correct=0, after_strict_correct=0, gained=[], lost=[]) for condition in CONDITIONS}
    for a, b in zip(before['results'], after['results']):
        for key in ['id', 'original_native_prompt_ids', 'generation_budget', 'wrong_source_question_id', 'original_ABC_labels']:
            if a[key] != b[key]: raise ValueError('Exact paired source questions and whole native prompts required')
        for condition in CONDITIONS:
            x, y = a['conditions'][condition], b['conditions'][condition]
            if (x['actual_model_input_tensor_sha256_448'] != y['actual_model_input_tensor_sha256_448']
                    or x['supplied_image_id'] != y['supplied_image_id']): raise ValueError('Every actual paired visual input must match')
            p, q = x['verdict']['strict_correct'], y['verdict']['strict_correct']; item = result[condition]
            item['questions'] += 1; item['before_strict_correct'] += p; item['after_strict_correct'] += q
            if not p and q: item['gained'].append(a['id'])
            if p and not q: item['lost'].append(a['id'])
    return dict(by_image_condition=result, candidate_promoted=False, goal_achieved=False)


def parent():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or REPORTS.exists() or QUEUE.exists(): raise ValueError('Fresh CPU-hidden visual evaluation queue required')
    from slm_lab.ai2d_full_native_evaluation_v1 import FullAI2D, reconcile
    from slm_lab.local_source_closure_v2 import python_sources
    population = FullAI2D(ROOT); preparation = json.loads(PREPARATION.read_text())
    if (preparation['original_population_sha256'] != population.identity_sha256 or preparation['actual_native448_variants_checked'] != 1555
            or preparation['original_questions'] != 3088 or preparation['original_nonempty_diagrams'] != 814):
        raise ValueError('Complete original native image/text preflight and corrected literal inventory required')
    previous = json.loads(PREVIOUS_QUEUE.read_text()); me = owner.resources.identity(os.getpid())
    files = dict(preparation['source_files_sha256']); files.update(population.bindings)
    extra = [PREPARATION, PREVIOUS_QUEUE, ROOT / 'reports/ai2d-full-native-evaluation-v1-tests.log',
        ROOT / 'reports/state-code-first-full-assessment-drivers-v1-tests.log', *python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_full_AI2D_pipeline_v3.py'])]
    files.update({str(p): file_hash(p) for p in extra}); guard = FileBindings(files); guard.check(full=True)
    expires = time.time() + 14 * 24 * 3600
    atomic_json(QUEUE, dict(status='waiting for exact complete eight-endpoint text QA and restoration', owner=me, predecessor=previous['owner'],
        endpoints=descriptors(ROOT), original_questions_each_endpoint=3088, native_image_conditions=['proper_image', 'gray_image', 'wrong_image'],
        actual_neural_assessments=0, all_eight_endpoints_predeclared_not_score_selected=True,
        original_population_sha256=population.identity_sha256, source_files_sha256=files, expires_at_unix=expires,
        text_image_and_semantic_exposure_audit_pending=True, not_an_official_leaderboard_score=True,
        automatic_retry=False, model_updates=0, candidate_promoted=False, goal_achieved=False))
    while not previous_ready(previous['owner']):
        if time.time() > expires: raise ValueError('Declared visual evaluation queue lifetime exceeded; no automatic retry')
        guard.check(full=False); time.sleep(2)
    specs = descriptors(ROOT)
    for spec in specs.values():
        spec['checkpoint_sha256'] = file_hash(spec['checkpoint']); spec['plan_sha256'] = file_hash(spec['plan'])
        files[spec['checkpoint']] = spec['checkpoint_sha256']; files[spec['plan']] = spec['plan_sha256']
    for path in [PREVIOUS / 'complete.json', PREVIOUS / 'restored.json']: files[str(path)] = file_hash(path)
    with (ROOT / 'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); owner.idle_original()
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
        if len(used) != 4 or any(int(value) > 128 for value in used): raise ValueError('All four original GPUs must be idle before acquisition')
        if shutil.disk_usage(ROOT).free < 27 * 1024**3: raise ValueError('Preserve26GiB persistent reserve and the complete visual report budget')
        guard = FileBindings(files); guard.check(full=True); REPORTS.mkdir()
        plan = dict(owner=me, waiter=owner.WAITER, endpoints=specs, source_files_sha256=files,
            expires_at_unix=time.time() + 72 * 3600, original_questions_each_endpoint=3088,
            full_population_sha256=population.identity_sha256, all_four_L40s_each_endpoint_in_sequence=True,
            proper_gray_and_fixed_wrong_native448_images_each_question=True, automatic_retry=False,
            text_image_and_semantic_exposure_audit_pending=True, not_an_official_leaderboard_score=True,
            model_updates=0, candidate_promoted=False, goal_achieved=False)
        atomic_json(REPORTS / 'plan.json', plan); atomic_json(REPORTS / 'children.json', [])
        def interrupted(signum, frame): raise KeyboardInterrupt('Owned all-original visual QA interrupted')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        watcher = None; driver = str(Path(__file__).resolve())
        try:
            owner.idle_original(); os.kill(owner.WAITER['pid'], signal.SIGSTOP); stop = time.monotonic() + 5
            while (owner.resources.same(owner.WAITER) or {}).get('state') not in ['T', 't']:
                if time.monotonic() > stop: raise ValueError('Exact original waiter hold failed')
                time.sleep(.05)
            watcher = subprocess.Popen(['.venv/bin/python', '-u', driver, '--watchdog'], cwd=ROOT,
                stdout=(REPORTS / 'watchdog.log').open('x'), stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
            reports = {}
            for endpoint in ENDPOINTS:
                children = [owner.launch(plan, ['-u', driver, '--worker', endpoint, '--gpu', str(gpu)],
                    'AI2D-' + endpoint + '-' + str(gpu), str(gpu)) for gpu in range(4)]
                owner.wait_children(plan, children)
                reports[endpoint] = reconcile(population, [REPORTS / endpoint / f'AI2D-shard-{gpu}.json' for gpu in range(4)],
                    REPORTS / endpoint / 'full-AI2D.json')
            comparisons = {a + '__to__' + b: paired(reports[a], reports[b]) for a, b in COMPARISONS}
            atomic_json(REPORTS / 'paired-comparisons.json', comparisons); guard.check(full=True)
            atomic_json(REPORTS / 'complete.json', dict(all_eight_predeclared_endpoints_assessed=True, original_questions_each_endpoint=3088,
                full_original_questions_and_options_and_budgets_preserved=True, native_image_conditions=['proper_image', 'gray_image', 'wrong_image'],
                total_question_endpoint_assessments=len(ENDPOINTS) * 3088, total_question_endpoint_condition_assessments=len(ENDPOINTS) * 3088 * 3,
                all_actual_native_pixel_identities_and_delivered_tokens_and_likelihoods_retained=True,
                strict_and_exact_duplicate_equivalent_verdicts_separate=True, automatic_selection_or_promotion=False,
                independent_output_readback_and_training_exposure_audit_still_required=True,
                not_an_official_leaderboard_score=True, model_updates=0, goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS / 'failure.json', dict(error=repr(error), automatic_retry=False, model_updates=0, goal_achieved=False)); raise
        finally:
            owner.restore_owned(plan, 'All-original visual QA ended; original work restored')
            if watcher is not None: watcher.wait(timeout=20)


if __name__ == '__main__':
    configure(); parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--worker', choices=ENDPOINTS); parser.add_argument('--gpu', type=int, choices=range(4)); args = parser.parse_args()
    if args.watchdog: owner.watchdog()
    elif args.worker: worker(args.worker, args.gpu)
    else: parent()
