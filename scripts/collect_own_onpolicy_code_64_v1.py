"""Collect64whole own-policy train generations and original finite-case execution signals."""
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
REPORTS = ROOT / 'reports/own-onpolicy-code-64-v1'
MANIFEST = ROOT / 'checkpoints/own-assistant-candidate-v1/manifest.json'
MAXIMUM_SECONDS = 2 * 60 * 60
owner.REPORTS = REPORTS; owner.resources.REPORTS = REPORTS


def previous_ended():
    for folder in ['own-code-generation-repair-v1', 'frozen-base-pointer-v1', 'own-view-contrastive-v1', 'own-code-conditioning-v1']:
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
        raise ValueError('Exact isolated physical L40 required before importing/initializing CUDA')
    guard = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    import torch
    from tokenizers import Tokenizer
    from slm_lab.own_assistant_v1 import load_manifest
    from slm_lab.own_view_contrastive_v1 import generate
    from slm_lab.apps_published_cases_v2 import execute
    from slm_lab.code_quality import instruction_code
    from slm_lab.point_grounding_data_v1 import PointerData
    from slm_lab.point_grounding_evaluation_v1 import state_digest
    torch.set_num_threads(1); torch.cuda.set_device(0); torch.cuda.set_per_process_memory_fraction(.65, 0)
    if torch.cuda.get_device_name(0) != 'NVIDIA L40':
        raise ValueError('Actual original L40 inference hardware required')
    model, metadata = load_manifest(MANIFEST, expected_manifest_sha256=plan['manifest_sha256'], device='cuda:0')
    tokenizer = Tokenizer.from_file(plan['tokenizer'])
    rows = list(map(json.loads, (REPORTS / 'calibration.jsonl').read_text().splitlines()))
    data = PointerData(ROOT / 'data/localized-pointer-candidate-v1', 'train', 'audit')
    original_probe = json.loads((ROOT / 'reports/frozen-base-pointer-full-CPU-v1/complete.json').read_text())
    batch = data.batch([original_probe['original_source_position']], 'cuda:0')
    digest_before = {name: state_digest(part) for name, part in [('base', model.language.base), ('code', model.language.parts),
        ('query_projection', model.grounding.query_projection), ('patch_projection', model.grounding.patch_projection)]}
    versions = [(id(p), p._version, p.data_ptr()) for p in model.parameters()]
    def checked():
        owner.owned(plan)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        ids, pixels = batch['input_ids'], batch['pixel_values']
        general = model.generate(ids, pixels, route='general', max_new_tokens=2, eos_token_id=2)
        code = model.generate(ids, pixels, route='code', max_new_tokens=2, eos_token_id=2)
        if (not torch.equal(general, model.language.base.generate(ids, pixels, max_new_tokens=2, eos_token_id=2))
                or not torch.equal(code, generate(model.language, ids, pixels, strength=0., max_new_tokens=2)['tokens'])):
            raise ValueError('Actual shared GPU general/code generation must equal original explicit routes')
        state = model.language.code.prefill(ids, pixels)
        expected = model.language.code.decode(torch.tensor([[4]], device='cuda:0'), state['cache'])['logits'].clone()
        located = model.locate(ids, pixels, attention_mask=batch['attention_mask'], valid_cells=batch['valid_cells'])
        if (not bool(torch.isfinite(located['logits']).all()) or not torch.equal(expected,
                model.language.code.decode(torch.tensor([[4]], device='cuda:0'), state['cache'])['logits'])):
            raise ValueError('Grounding must stay finite and preserve the existing GPU code cache')
    atomic_json(REPORTS / f'capacity-{shard}.json', dict(actual_model_counts=model.counts(),
        actual_GPU=torch.cuda.get_device_name(0), precision='CUDA BF16 autocast', actual_pixels=list(pixels.shape),
        original_train_query_id=data.rows[original_probe['original_source_position']]['id'],
        code_and_general_route_generation_parity=True, grounding_and_preexisting_code_cache_parity=True,
        actual_all_three_trained_routes_exercised=True, one_shared_original_backbone=True,
        model_updates=0, goal_achieved=False))
    result_path = REPORTS / f'generations-{shard}.jsonl'
    with result_path.open('x') as stream:
        for index, row in enumerate(rows):
            if index % 4 != shard:
                continue
            if row['split'] != 'train' or row['publisher_split'] != 'train':
                raise ValueError('Held data may not enter this calibration experiment')
            guard.check(); checked()
            reference = execute(row['original_human_program'], row['cases'], seconds=20)
            if not reference['all_finite_published_cases_pass']:
                raise ValueError('Every source reference must still pass all original cases')
            atomic_json(REPORTS / f'reference-{index}.json', reference)
            prefix = torch.tensor([row['prefix_ids']], device='cuda:0')
            for strength in plan['strengths']:
                started = time.monotonic()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    output = generate(model.language, prefix, strength=strength, plausibility=plan['plausibility'],
                        max_new_tokens=plan['generation_budget'], checked=checked)
                response_ids = output['tokens'][0, prefix.shape[1]:].tolist()
                text = tokenizer.decode(response_ids, skip_special_tokens=True)
                code = instruction_code(text)
                execution = (execute(code, row['cases'], seconds=20) if code else dict(syntax_valid=False,
                    all_finite_published_cases_pass=False, all_cases_observed=False, total_cases=row['cases']['total_cases'],
                    passed_cases=0, status='empty_response'))
                record = dict(id=row['id'], original_problem=row['original_problem'], split='train', publisher_split='train',
                    strength=strength, plausibility=plan['plausibility'], prefix_ids=row['prefix_ids'],
                    response_ids=response_ids, raw_response=text, code=code, execution=execution,
                    trace=output['trace'], stopped_on_all_EOS=output['stopped_on_all_EOS'],
                    model_and_execution_seconds=time.monotonic() - started,
                    additional_neural_parameters=0, human_preference_or_RLHF=False)
                stream.write(json.dumps(record, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
                print(json.dumps(dict(id=row['id'], strength=strength, syntax=execution.get('syntax_valid', False),
                    passed=execution['all_finite_published_cases_pass'], status=execution['status'],
                    generated_tokens=len(response_ids))), flush=True)
    guard.check(full=True); checked()
    digest_after = {name: state_digest(part) for name, part in [('base', model.language.base), ('code', model.language.parts),
        ('query_projection', model.grounding.query_projection), ('patch_projection', model.grounding.patch_projection)]}
    if digest_before != digest_after or versions != [(id(p), p._version, p.data_ptr()) for p in model.parameters()]:
        raise ValueError('All original/code/head tensor bytes, pointers and versions must stay unchanged')
    atomic_json(REPORTS / f'complete-{shard}.json', dict(actual_counts=model.counts(),
        generations_sha256=file_hash(result_path), component_digests_before=digest_before, component_digests_after=digest_after,
        original_code_head_weights_and_versions_unchanged=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        model_updates=0, candidate_promoted=False, goal_achieved=False))


def combine(plan):
    from tokenizers import Tokenizer
    from slm_lab.apps_published_cases_v2 import execute, compare
    from slm_lab.code_quality import instruction_code
    tokenizer = Tokenizer.from_file(plan['tokenizer'])
    rows = list(map(json.loads, (REPORTS / 'calibration.jsonl').read_text().splitlines()))
    source = {r['id']: r for r in rows}; records = []; completion = []
    for shard in range(4):
        done = json.loads((REPORTS / f'complete-{shard}.json').read_text())
        path = REPORTS / f'generations-{shard}.jsonl'
        if file_hash(path) != done['generations_sha256'] or not done['original_code_head_weights_and_versions_unchanged']:
            raise ValueError('All complete unchanged GPU workers and generation bytes required')
        part = list(map(json.loads, path.read_text().splitlines()))
        expected = [(r['id'], strength) for index, r in enumerate(rows) if index % 4 == shard for strength in plan['strengths']]
        if [(r['id'], r['strength']) for r in part] != expected:
            raise ValueError('Every original disjoint train problem and fixed decoder setting exactly once required')
        records.extend(part); completion.append(done)
    metrics = {}
    for strength in plan['strengths']:
        selected = [r for r in records if r['strength'] == strength]
        for row in selected:
            original = source[row['id']]; observed = row['execution']; cases = original['cases']
            if (row['split'] != 'train' or row['publisher_split'] != 'train' or row['prefix_ids'] != original['prefix_ids']
                    or tokenizer.decode(row['response_ids'], skip_special_tokens=True) != row['raw_response']
                    or instruction_code(row['raw_response']) != row['code'] or len(row['trace']) != len(row['response_ids'])
                    or len(row['response_ids']) > plan['generation_budget']
                    or [t['token_ids'][0] for t in row['trace']] != row['response_ids']):
                raise ValueError('Every original prefix and delivered own-token code must reconstruct')
            if observed['all_cases_observed']:
                raw = observed['raw_results']['results']
                passes = [compare(r['value'], expected, cases['mode']) for r, expected in zip(raw, cases['outputs'])]
                if (len(raw) != cases['total_cases'] or observed['case_passes'] != passes
                        or observed['passed_cases'] != sum(passes) or observed['all_finite_published_cases_pass'] != all(passes)):
                    raise ValueError('Every complete observed raw case comparison must reconstruct')
            elif observed['all_finite_published_cases_pass']:
                raise ValueError('Incomplete execution cannot count as a complete functional pass')
            if observed['all_finite_published_cases_pass'] and not execute(row['code'], cases, seconds=20)['all_finite_published_cases_pass']:
                raise ValueError('Every claimed new pass must independently reexecute on all original cases')
        metrics[str(strength)] = dict(training_problems=len(selected),
            syntax=sum(r['execution'].get('syntax_valid', False) for r in selected),
            all_finite_cases_pass=sum(r['execution']['all_finite_published_cases_pass'] for r in selected),
            all_cases_observed=sum(r['execution']['all_cases_observed'] for r in selected),
            EOS_completions=sum(r['stopped_on_all_EOS'] for r in selected),
            total_generated_tokens=sum(len(r['response_ids']) for r in selected),
            execution_status=dict(Counter(r['execution']['status'] for r in selected)))
    baseline = metrics['0.0']
    eligible = [s for s in plan['strengths'][1:] if metrics[str(s)]['all_finite_cases_pass'] > baseline['all_finite_cases_pass']
        and metrics[str(s)]['syntax'] >= baseline['syntax']]
    choice = min(eligible, key=lambda s: (-metrics[str(s)]['all_finite_cases_pass'], -metrics[str(s)]['syntax'], s)) if eligible else None
    atomic_json(REPORTS / 'comparison.json', dict(completed_at=datetime.now(timezone.utc).isoformat(),
        metrics=metrics, calibration_choice_for_later_new_sealed_held_experiment=choice,
        choice_gate='One unchanged native greedy policy only; no decoder/model selection; data not yet admitted to training',
        train_calibration_is_not_held_accuracy=True, no_held_problem_or_answer_used_for_selection=True,
        all_three_actual_trained_GPU_routes_and_preexisting_code_cache_exercised=True,
        actual_unique_parameters=986450944, additional_neural_parameters=0, model_updates=0,
        full_original_finite_cases_not_algorithmic_or_efficiency_proof=True,
        unchanged_native_greedy_policy_only=True,
        fresh_genuine_human_alignment_not_completed=True, candidate_promoted=False, goal_achieved=False))


def parent():
    if REPORTS.exists():
        raise ValueError('Preserve previous attempts and generations; no implicit retry')
    previous_ended(); owner.idle_original()
    from tokenizers import Tokenizer
    from slm_lab.own_onpolicy_code_data_v1 import prepare
    from slm_lab.evaluation_provenance import capture
    from slm_lab.apps_published_cases_v2 import implementation_sources
    spec = json.loads(MANIFEST.read_text()); tokenizer_path = (MANIFEST.parent / spec['tokenizer']).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); rows = prepare(ROOT, tokenizer)
    files = {str(ROOT / name): sha for name, sha in capture()['files_sha256'].items()}
    files.update(implementation_sources())
    names = ['scripts/collect_own_onpolicy_code_64_v1.py', 'src/slm_lab/own_onpolicy_code_data_v1.py', 'scripts/train_point_grounding_joint_v1.py',
        'scripts/assess_joint_java_comparison_v1.py', 'src/slm_lab/own_view_contrastive_v1.py',
        'src/slm_lab/own_view_contrastive_data_v1.py', 'src/slm_lab/own_assistant_v1.py',
        'src/slm_lab/own_code_delta_v1.py', 'src/slm_lab/own_code_delta_inference_v1.py',
        'src/slm_lab/own_code_training_inference_v1.py', 'src/slm_lab/own_code_training_stage_v1.py',
        'src/slm_lab/frozen_base_pointer_stage_v1.py', 'src/slm_lab/frozen_base_pointer_inference_v1.py',
        'src/slm_lab/point_grounding_head_v1.py', 'src/slm_lab/point_grounding_data_v1.py',
        'src/slm_lab/point_grounding_evaluation_v1.py', 'src/slm_lab/codenet_checkpoint_v1.py',
        'tests/test_own_view_contrastive_v1.py', 'reports/own-view-contrastive-v1-tests.log',
        'tests/test_own_onpolicy_code_driver_v1.py', 'reports/own-onpolicy-code-driver-v1-tests.log',
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
    if '3 passed' not in (ROOT / 'reports/own-view-contrastive-v1-tests.log').read_text():
        raise ValueError('Actual independent score/full-prefix/unchanged native generation checks required')
    if '2 passed' not in (ROOT / 'reports/own-onpolicy-code-driver-v1-tests.log').read_text():
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
            tokenizer=str(tokenizer_path), training_problems=64, strengths=[0.], plausibility=.1,
            generation_budget=512, arithmetic='CUDA BF16 autocast, original448px train route probe',
            source_selection='Thirty-two stdin and thirty-two function original verified publisher-train problems, shortest-reference pack, targets<=512, whole prompts+512<=4096; fixed SHA256 seed ordering, no model-result selection',
            selection_seed='own-onpolicy-code-64-source-v1',
            selection_gate='No decoder or model selection. All64original train generations retained; later feedback admission requires complete observed wrong values and genuine EOS completion.',
            native_own_policy_generation_no_new_decoding_method=True,
            method='Original native cached greedy own-code generation512tokens, all original finite cases, complete original human-reference execution. Zero neural updates.',
            expires_at_unix=expires.timestamp(), maximum_seconds=MAXIMUM_SECONDS,
            human_preference_RLHF=False, model_updates=0, candidate_promoted=False, goal_achieved=False)
        atomic_json(REPORTS / 'plan.json', plan); atomic_json(REPORTS / 'children.json', [])
        watcher = subprocess.Popen(['.venv/bin/python', '-u', str(Path(__file__).resolve()), '--watchdog'], cwd=ROOT,
            stdout=(REPORTS / 'watchdog.log').open('xb'), stderr=subprocess.STDOUT, start_new_session=True)
        def interrupted(signum, frame):
            raise KeyboardInterrupt('Owned shared-view decoding pilot interrupted')
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
                atomic_json(REPORTS / 'status.json', dict(status='original train own-policy generation and complete-case execution active',
                    observed_at=datetime.now(timezone.utc).isoformat(), workers=[dict(pid=p.pid, exit_code=p.poll()) for p in children],
                    model_updates=0, goal_achieved=False)); time.sleep(2)
            if any(p.returncode != 0 for p in children):
                raise ValueError('All four complete calibration workers must succeed')
            owner.owned(plan); combine(plan); guard.check(full=True)
            atomic_json(REPORTS / 'status.json', dict(status='all64original train own-policy generations complete; feedback admission pending', goal_achieved=False))
        except BaseException as error:
            atomic_json(REPORTS / 'failure.json', dict(error=str(error), automatic_retry=False, goal_achieved=False)); raise
        finally:
            owner.restore_owned(plan, 'Own-policy64source generation ended; exact original waiter restored')
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
