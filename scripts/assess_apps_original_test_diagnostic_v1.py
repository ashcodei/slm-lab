"""Matched own-model APPS diagnostic only in idle, checked GPU lease windows.

Preselects all supported function tasks and a source-hash sample of sixteen
stdin tasks per difficulty. The full original test inventory remains separate.
No test feedback changes training, the cohort, prompts or the stopping budget.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

import train_apps_full_language_v1 as owner
from slm_lab.apps_published_cases_v2 import contract, execute, implementation_sources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

DATA = owner.ROOT/'data/public-apps-publisher-test-evaluation-v1'
FOLDER = owner.ROOT/'reports/APPS-original-test-diagnostic-v1'
FENCE = re.compile(r'\A\s*```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```\s*\Z', re.S|re.I)


def extract(text):
    if not isinstance(text, str) or not text.strip():
        return None
    if text.lstrip().startswith('```'):
        match = FENCE.fullmatch(text)
        return match[1] if match else None
    return text.strip()


def selected_rows(rows):
    eligible = [row for row in rows if row['supported_by_declared_local_protocol']]
    chosen = [row for row in eligible if row['mode'] == 'function']
    for difficulty in ['introductory', 'interview', 'competition']:
        population = [row for row in eligible if row['mode'] == 'stdin' and row['difficulty'] == difficulty]
        population.sort(key=lambda row:hashlib.sha256(('APPS-original-test-diagnostic-v1:'+row['id']).encode()).hexdigest())
        if len(population) < 16:
            raise ValueError('Full predeclared source-hash sample required')
        chosen.extend(population[:16])
    chosen.sort(key=lambda row:row['id'])
    if len({row['id'] for row in chosen}) != len(chosen):
        raise ValueError('Distinct original diagnostic tasks required')
    return chosen


def idle_window(plan):
    owner.ownership(plan)
    if json.loads((owner.REPORTS/'status.json').read_text()).get('phase') != 'candidate':
        raise ValueError('Owned candidate assessment window is not active')
    validation = owner.REPORTS/'APPS-validation/candidate-complete.json'
    if not validation.exists() or not all((owner.REPORTS/'candidate'/f'suite-{i}-complete.json').exists() for i in [1,2,3]):
        raise ValueError('All original GPU1-3 assessments and full validation must finish first')
    actors = json.loads((owner.REPORTS/'children.json').read_text())
    for actor in actors:
        if (actor['subject'] in {'candidate-assessment-1','candidate-assessment-2','candidate-assessment-3'}
                or actor['subject'].startswith('candidate-full-APPS-validation-shard-')) and owner.resources.same(actor):
            raise ValueError('An earlier assigned GPU assessment process remains live')


def worker(shard):
    def interrupted(signum, frame):
        # Let run_bounded's finally block stop a currently owned sandbox child
        # when the main lease watchdog asks this GPU process to terminate.
        raise KeyboardInterrupt('Owned diagnostic window ended')
    signal.signal(signal.SIGTERM, interrupted)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(shard+1):
        raise ValueError('Exact isolated physical GPU assignment required')
    parent = owner.register('read-only-original-APPS-test-diagnostic-shard-'+str(shard))
    idle_window(parent)
    unused = subprocess.run(['nvidia-smi', '-i', str(shard+1), '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout.strip()
    if unused:
        raise ValueError('Assigned physical GPU has a live compute process; preserve queued work')
    plan_path = FOLDER/'plan.json'
    plan = json.loads(plan_path.read_text())
    guard = FileBindings({**plan['source_files_sha256'], str(plan_path):file_hash(plan_path)})
    guard.check(full=True)
    rows = list(map(json.loads, (DATA/'evaluation-records.jsonl').read_text().splitlines()))
    selected = selected_rows(rows)
    if [r['id'] for r in selected] != plan['selected_original_test_ids']:
        raise ValueError('Exact source-preselected cohort required')
    import torch
    from tokenizers import Tokenizer
    from slm_lab.training import load_checkpoint
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    tokenizer = Tokenizer.from_file(str(owner.TOKEN))
    for phase, checkpoint in [('baseline', owner.BASE), ('candidate', owner.OUTPUT/'model.pt')]:
        idle_window(parent)
        target = FOLDER/f'{phase}-{shard}.json'
        journal = FOLDER/f'{phase}-{shard}.jsonl'
        if target.exists() or journal.exists():
            raise ValueError('Preserve earlier generations; no implicit duplicate assessment')
        checkpoint_sha = file_hash(checkpoint)
        if phase == 'baseline' and checkpoint_sha != plan['baseline_checkpoint_sha256']:
            raise ValueError('Exact own completed systems baseline required')
        model, meta = load_checkpoint(checkpoint, 'cuda:0', training_state=False)
        model.eval()
        if meta['tokenizer_sha256'] != plan['tokenizer_sha256']:
            raise ValueError('Own original tokenizer/checkpoint binding required')
        records = []
        with journal.open('x') as stream, torch.no_grad():
            for row in selected[shard::3]:
                idle_window(parent)
                guard.check()
                path = DATA/'original-test'/row['problem']/'input_output.json'
                if file_hash(path) != row['original_tests_sha256']:
                    raise ValueError('Original complete held test bytes changed')
                cases = contract(path.read_bytes())
                if cases['total_cases'] != row['complete_published_case_count'] or cases['mode'] != row['mode']:
                    raise ValueError('Exact complete held case set required')
                prefix = [tokenizer.token_to_id('<bos>')]+tokenizer.encode('User: '+row['complete_prompt']+'\nAssistant:').ids
                prefix_sha = hashlib.sha256(json.dumps(prefix,separators=(',',':')).encode()).hexdigest()
                if prefix_sha != row['complete_prompt_input_ids_sha256'] or len(prefix)+1024 > model.config.max_text_length:
                    raise ValueError('Unclipped original own-BPE prompt/generation span required')
                ids = torch.tensor([prefix], device='cuda:0')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    generated = model.generate(ids, max_new_tokens=1024, eos_token_id=tokenizer.token_to_id('<eos>'), use_cache=True)
                tokens = generated[0, len(prefix):].tolist()
                response = tokenizer.decode(tokens)
                code = extract(response)
                outcome = execute(code, cases, seconds=20) if code else dict(status='incomplete_code_delivery',
                    syntax_valid=False, total_cases=cases['total_cases'], all_cases_observed=False,
                    all_finite_published_cases_pass=False, passed_cases=0)
                result = dict(id=row['id'], problem=row['problem'], difficulty=row['difficulty'], mode=row['mode'],
                    original_tests_sha256=row['original_tests_sha256'], prompt_input_ids_sha256=prefix_sha,
                    generation_token_ids=tokens, raw_response=response, extracted_code=code, execution=outcome)
                records.append(result)
                stream.write(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
                stream.flush()
                print(phase, 'shard', shard, 'complete_original_tasks', len(records), '/', len(selected[shard::3]), flush=True)
        idle_window(parent)
        guard.check(full=True)
        if file_hash(checkpoint) != checkpoint_sha:
            raise ValueError('Assessed fixed own checkpoint changed')
        atomic_json(target, dict(phase=phase, shard=shard, checkpoint_sha256=checkpoint_sha,
            tokenizer_sha256=meta['tokenizer_sha256'], plan_sha256=file_hash(plan_path),
            journal_sha256=file_hash(journal), tasks=len(records), model_updates=0, goal_achieved=False))
        del model
        gc.collect()
        torch.cuda.empty_cache()


def combine(plan):
    comparisons = {}
    for phase in ['baseline','candidate']:
        certificates = [json.loads((FOLDER/f'{phase}-{i}.json').read_text()) for i in range(3)]
        if len({r['checkpoint_sha256'] for r in certificates}) != 1:
            raise ValueError('Same fixed checkpoint across all original task shards required')
        rows = []
        for i, certificate in enumerate(certificates):
            path = FOLDER/f'{phase}-{i}.jsonl'
            if file_hash(path) != certificate['journal_sha256'] or certificate['plan_sha256'] != file_hash(FOLDER/'plan.json'):
                raise ValueError('Every complete generation journal must match its frozen plan')
            rows.extend(map(json.loads, path.read_text().splitlines()))
        if sorted(r['id'] for r in rows) != plan['selected_original_test_ids']:
            raise ValueError('Every preselected original test problem exactly once required')
        comparisons[phase] = dict(checkpoint_sha256=certificates[0]['checkpoint_sha256'], tasks=len(rows),
            syntax_valid=sum(r['execution'].get('syntax_valid',False) for r in rows),
            all_finite_published_cases_pass=sum(r['execution']['all_finite_published_cases_pass'] for r in rows),
            outcomes=dict(Counter(r['execution']['status'] for r in rows)),
            by_interface={mode:dict(tasks=sum(r['mode']==mode for r in rows),
                passed=sum(r['mode']==mode and r['execution']['all_finite_published_cases_pass'] for r in rows)) for mode in ['stdin','function']})
    atomic_json(FOLDER/'comparison.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
        **comparisons, plan_sha256=file_hash(FOLDER/'plan.json'),
        diagnostic_not_full_APPS_or_certified_independent_score=True,
        finite_tests_not_algorithmic_or_efficiency_proof=True, used_for_gradients_stopping_or_promotion=False,
        candidate_promoted=False, goal_achieved=False))


def coordinator():
    parent = owner.register('read-only-original-APPS-test-diagnostic-coordinator')
    if FOLDER.exists():
        raise ValueError('No implicit duplicate diagnostic; preserve prior observations')
    manifest = json.loads((DATA/'manifest.json').read_text())
    if manifest['training_admitted'] or not manifest['optimization_forbidden'] or manifest['model_generations']:
        raise ValueError('Original held evaluation-only source required')
    readback_path = DATA/'independent-evaluation-readback.json'
    readback = json.loads(readback_path.read_text())
    if (readback['source_manifest_sha256'] != file_hash(DATA/'manifest.json')
            or readback['original_test_problems'] != 5000
            or not readback['all_original_test_files_reconciled_to_separate_archive_inventory']
            or not readback['every_complete_prompt_and_own_BPE_span_independently_reconstructed']):
        raise ValueError('Actual complete independent original held-source readback required')
    rows = list(map(json.loads, (DATA/'evaluation-records.jsonl').read_text().splitlines()))
    selected = selected_rows(rows)
    files = implementation_sources()
    for p in [Path(__file__).resolve(), DATA/'manifest.json', DATA/'evaluation-records.jsonl', readback_path, owner.TOKEN,
            owner.ROOT/'src/slm_lab/model.py', owner.ROOT/'src/slm_lab/training.py',
            owner.ROOT/'tests/test_apps_original_test_diagnostic_v1.py', owner.ROOT/'reports/APPS-original-test-diagnostic-v1-tests.log']:
        files[str(p)] = file_hash(p)
    if '2 passed' not in (owner.ROOT/'reports/APPS-original-test-diagnostic-v1-tests.log').read_text():
        raise ValueError('Exact delivery/cohort and idle-window rejection tests required')
    guard = FileBindings(files)
    FOLDER.mkdir()
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        baseline_checkpoint_sha256=file_hash(owner.BASE), tokenizer_sha256=manifest['tokenizer_sha256'],
        parent_owner=parent['owner'], selected_original_test_ids=[r['id'] for r in selected],
        selected_tasks=len(selected), by_interface=dict(Counter(r['mode'] for r in selected)),
        by_difficulty=dict(Counter(r['difficulty'] for r in selected)),
        selection='All supported original function tasks plus16source-hash-selected stdin tasks per difficulty',
        max_new_tokens=1024, greedy=True, precision='BF16', complete_context_window=4096,
        full_original_case_sets=True, model_updates=0, diagnostic_not_full_APPS=True,
        preserve_partial_generations_if_owner_window_ends=True, automatic_retry=False,
        candidate_promoted=False, goal_achieved=False)
    atomic_json(FOLDER/'plan.json',plan)
    while True:
        owner.ownership(parent)
        guard.check()
        if (owner.REPORTS/'APPS-validation/candidate-complete.json').exists():
            idle_window(parent)
            break
        time.sleep(5)
    guard.check(full=True)
    children = []
    for shard in range(3):
        with (FOLDER/f'shard-{shard}.log').open('xb') as stream:
            child = subprocess.Popen(['.venv/bin/python','-u',str(Path(__file__).resolve()),'--worker',str(shard)],
                cwd=owner.ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(shard+1),'OMP_NUM_THREADS':'1'})
        children.append(child)
    atomic_json(FOLDER/'launches.json',dict(actors=[owner.resources.identity(p.pid) for p in children],
        model_updates=0,goal_achieved=False))
    while any(p.poll() is None for p in children):
        idle_window(parent)
        if any(p.poll() not in [None,0] for p in children):
            raise ValueError('A diagnostic worker failed; preserve partial observations without retry')
        time.sleep(2)
    if any(p.returncode != 0 for p in children):
        raise ValueError('All diagnostic workers must complete')
    combine(plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=int, choices=[0,1,2])
    args = parser.parse_args()
    if args.worker is None:
        coordinator()
    else:
        worker(args.worker)
