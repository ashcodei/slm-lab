"""Full857M matched CE/source-association pilot on two idle physical L40s."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import assess_joint_java_comparison_v1 as resources
from slm_lab.codenet_checkpoint_v1 import FileBindings
from slm_lab.corpus import file_hash
from slm_lab.diverse_images import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/code-prompt-matching-pilot-v1'
OUTPUT = ROOT / 'checkpoints/code-prompt-matching-pilot-v1'
BASE = ROOT / 'checkpoints/extended-curriculum-v1/joint/model.pt'
TOKEN = BASE.parent / 'tokenizer.json'
SELECTION = ROOT / 'reports/code-residual-lr-pilot-v1/selected-original-rows.json'
PAIR = ROOT / 'data/code-prompt-pairs-v1'
WAITER = dict(pid=2754743, ticks='27602681')
PROFILES = [dict(name='CE_control', alpha=0., physical_gpu=1),
    dict(name='CE_plus_source_matching', alpha=.1, physical_gpu=2)]
RESERVE, MAX_CHECKPOINT, EXTRA = 26 * 1024**3, 2 * 1024**3, 256 * 1024**2
FOLDERS = {k: ROOT / 'data' / v for k, v in [
    ('joint', 'codenet-replay-v1'), ('facts', 'codenet-fact-replay-v1'),
    ('dialogue', 'codenet-dialogue-replay-v1'), ('images', 'codenet-image-replay-v1'),
    ('later', 'codenet-later-instruction-replay-v2'), ('packed', 'codenet-packed-pretraining-replay-v1')]}
resources.REPORTS = REPORTS


def now():
    return datetime.now(timezone.utc).isoformat()


def status(directory, message, **fields):
    atomic_json(directory / 'status.json', dict(status=message, updated_at=now(),
        candidate_promoted=False, goal_achieved=False, **fields))


def ownership(plan):
    waiter = resources.same(plan['waiter'])
    if not resources.same(plan['owner']) or not waiter or waiter['state'] not in {'T', 't'}:
        raise ValueError('Exact pilot owner and held original idle waiter required')


def await_registration(plan, name):
    me = resources.identity(os.getpid())
    while True:
        if not resources.same(plan['owner']):
            raise ValueError('Pilot owner ended before CUDA worker registration')
        children = json.loads((REPORTS / 'children.json').read_text())
        if any(c['pid'] == me['pid'] and c['ticks'] == me['ticks'] and c['subject'] == name for c in children):
            ownership(plan)
            return
        time.sleep(.1)


def worker(name):
    plan = json.loads((REPORTS / 'plan.json').read_text())
    await_registration(plan, name)
    profile = next(p for p in plan['profiles'] if p['name'] == name)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(profile['physical_gpu']):
        raise ValueError('Exact isolated physical pilot GPU required')
    # Registration and ownership are verified before any CUDA initialization.
    import gc
    import random
    import shutil
    import numpy as np
    from tokenizers import Tokenizer
    import torch
    from slm_lab.code_prompt_matching_v1 import scores
    from slm_lab.code_prompt_pair_corpus_v1 import SourcePairs
    from slm_lab.code_prompt_pilot_v1 import (validate_selection, update, verify_saved_state,
        FrozenPilotLoader, evaluator_loader)
    from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
    from slm_lab.code_residual_checkpoint_v1 import make_training_state
    from slm_lab.code_residual_stage_v3 import open_expanded_replay, ExpandedReplayEncoder
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.codenet_fact_replay_v1 import CombinedReplay
    from slm_lab.codenet_stage_v1 import EpochOrder
    from slm_lab.recurrent_ablation import state_digest
    from slm_lab.training import load_checkpoint
    from slm_lab import code_quality, knowledge_evaluation

    report, output = REPORTS / name, OUTPUT / name
    report.mkdir()
    output.mkdir()
    bindings = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    def guard(full=False):
        ownership(plan)
        bindings.check(full=full)
        if shutil.disk_usage(ROOT).free < RESERVE:
            raise ValueError('Unchanged26GiB persistent reserve required')
    guard(True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0] < 40 * 1024**3:
        raise ValueError('Initially40GiB free on an owned BF16 L40 required')
    torch.cuda.set_per_process_memory_fraction(.85, device)
    random.seed(920041); np.random.seed(920042); torch.manual_seed(920043); torch.cuda.manual_seed_all(920044)
    status(report, 'opening complete original source pairs and all29 rehearsal families')
    sources = {role: SourcePairs(PAIR, role) for role in ['train', 'validation']}
    selection = json.loads(SELECTION.read_text())
    validate_selection(selection, sources)
    replay = open_expanded_replay(FOLDERS, plan['tokenizer_sha256'])
    joint = replay
    while isinstance(joint, CombinedReplay):
        joint = joint.owners['human_instruction']
    base, metadata = load_checkpoint(BASE, 'cpu', training_state=False)
    if metadata['tokenizer_sha256'] != plan['tokenizer_sha256']:
        raise ValueError('Own exact completed joint policy and own BPE required')
    model = RoutedCodeResidual(base, width=2048)
    initial = dict(model_counts=model.parameter_counts(), residual_state_sha256=state_digest(model.residuals.state_dict()),
        frozen_backbone_state_sha256=state_digest(model.base.state_dict()), zero_initialized_up_projections=
        all(not bool(torch.any(r.up.weight)) for r in model.residuals))
    if initial['model_counts'] != dict(total_unique=857007616, frozen_base=706012672, trainable_code=150994944):
        raise ValueError('Exact full857M model required')
    atomic_json(report / 'initial-policy.json', initial)
    model.to(device)
    model.code.gradient_checkpointing = True
    tokenizer = Tokenizer.from_file(str(TOKEN))
    shutil.copy2(TOKEN, output / 'tokenizer.json')
    encoder = ExpandedReplayEncoder(tokenizer, model.config, joint.source.parent, 448)
    orders = {family: EpochOrder(replay.size(family), 920100 + i) for i, family in enumerate(replay.families)}
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    examples = {role: [sources[role].example(r['local_position']) for r in selection[role]] for role in selection}

    @torch.no_grad()
    def assess():
        model.eval()
        result = {}
        for role in ['train', 'validation']:
            values = []
            for example in examples[role]:
                guard()
                value = scores(model.code, [example], device, amp)[0]
                values.append(dict(id=example['id'], language=example['language'], role=role,
                    negative_source_id=example['negative_source_id'],
                    targets=sum(y != -100 for y in example['positive'][1]), **value))
            total = sum(v['targets'] for v in values)
            paired = [v for v in values if v['mismatched'] is not None]
            result[role] = dict(examples=len(values), target_tokens=total,
                matched_token_mean_NLL=sum(v['targets'] * v['matched'] for v in values) / total,
                matching_gap_per_token=sum(v['targets'] * (v['mismatched'] - v['matched']) for v in paired)
                    / sum(v['targets'] for v in paired),
                source_association_accuracy=sum(v['matched'] < v['mismatched'] for v in paired) / len(paired),
                rows=values, correctness_metric=False)
        return result

    status(report, 'measuring identical initial full-source selection before neural updates')
    initial_scores = assess()
    atomic_json(report / 'initial-source-scores.json', initial_scores)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=3e-5, weight_decay=.01, foreach=False)
    metrics = []
    started = time.monotonic()
    for step in range(128):
        guard()
        family = replay.families[step % len(replay.families)]
        occurrence = step // len(replay.families)
        position = orders[family].at(occurrence)
        before = time.monotonic()
        record = update(model, optimizer, examples['train'][step * 4:(step + 1) * 4],
            encoder, replay.row(family, position), family, step, profile['alpha'], device, amp)
        record.update(replay_position=position, replay_occurrence=occurrence, step_seconds=time.monotonic() - before)
        metrics.append(record)
        atomic_json(report / 'training-metrics.json', metrics)
        if step < 2 or (step + 1) % 4 == 0:
            status(report, 'training full857M source-matching pilot', completed_steps=step + 1, planned_steps=128, latest=record)
            print('source_matching_update', name, json.dumps(record), flush=True)
        if (step + 1) % 32 == 0:
            guard(True)
            if state_digest(model.base.state_dict()) != initial['frozen_backbone_state_sha256']:
                raise ValueError('Any frozen own backbone tensor change invalidates training')
            bundle = make_training_state(model, optimizer, base_checkpoint_sha256=plan['base_checkpoint_sha256'],
                tokenizer_sha256=plan['tokenizer_sha256'], run_plan_sha256=file_hash(REPORTS / 'plan.json'),
                completed_steps=step + 1, planned_steps=128, device=device, metrics=metrics)
            bundle.update(stage='code_prompt_matching_pilot_v1', profile=profile,
                source_pair_manifest_sha256=file_hash(PAIR / 'manifest.json'),
                source_selection_sha256=file_hash(SELECTION), fresh_optimizer=True,
                sampler_next_step=step + 1, replay_family_population_sizes={f: replay.size(f) for f in replay.families},
                clipped_sequences=encoder.truncated)
            atomic_checkpoint(output / 'latest.pt', bundle, maximum_bytes=MAX_CHECKPOINT,
                minimum_free_bytes=RESERVE + 2 * MAX_CHECKPOINT + EXTRA)
            del bundle
            gc.collect()
            atomic_json(report / 'checkpoint.json', dict(completed_steps=step + 1,
                checkpoint_sha256=file_hash(output / 'latest.pt'), complete_Adam_RNG=True, candidate_promoted=False))
    model.eval()
    guard(True)
    receipt = verify_saved_state(model, output / 'latest.pt', device=device,
        plan_sha256=file_hash(REPORTS / 'plan.json'), base_sha256=plan['base_checkpoint_sha256'],
        tokenizer_sha256=plan['tokenizer_sha256'], completed=128)
    atomic_json(report / 'frozen-final-policy.json', dict(**receipt, sealed_before_benchmark_evaluation=True,
        training_seconds=time.monotonic() - started, candidate_promoted=False, goal_achieved=False))
    status(report, 'fixed128-update policy sealed; measuring source diagnostics')
    atomic_json(report / 'final-source-scores.json', assess())
    # Release Adam on the GPU; retain every complete moment/RNG byte on disk.
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    loader = FrozenPilotLoader(model, output / 'latest.pt', dict(metadata, stage='code_prompt_matching_pilot_v1',
        steps=128, planned_steps=128), receipt, device, lambda: guard())
    for mode in ['completion', 'instruction']:
        status(report, 'evaluating fixed128-update policy with unchanged local24-code protocol', prompt_mode=mode)
        with evaluator_loader(code_quality, loader):
            code_quality.evaluate(output / 'latest.pt', report / ('code-' + mode + '.json'),
                device=str(device), max_new_tokens=512, prompt_mode=mode)
    status(report, 'evaluating fixed128-update policy with unchanged559-question knowledge protocol')
    with evaluator_loader(knowledge_evaluation, loader):
        knowledge_evaluation.evaluate(output / 'latest.pt', ROOT / 'data/knowledge-eval-v1/questions.jsonl',
            report / 'knowledge.json', device=str(device), max_examples=1000, max_new_tokens=32)
    loader.check()
    guard(True)
    atomic_json(report / 'evaluation-binding.json', dict(checkpoint=receipt, loader_calls=loader.calls,
        source_files_sha256=plan['source_files_sha256'],
        unchanged_prompt_generation_and_scoring=True, held_gold_used_for_gradients=False,
        complete_world_one_checkpoint_retained=True, optimizer_loaded_on_evaluation_GPU=False,
        candidate_promoted=False, goal_achieved=False))
    status(report, 'bounded source-matching training and capability assessment completed', completed_steps=128)


def parent():
    import shutil
    if REPORTS.exists() or OUTPUT.exists():
        raise ValueError('Preserve prior pilot artifacts; no implicit restart')
    waiter = resources.same(WAITER)
    if not waiter or waiter['state'] != 'S':
        raise ValueError('Exact original queued trainer must be idle and sleeping')
    status_path = ROOT / 'reports/dialogue-retention-v1/status.json'
    if json.loads(status_path.read_text())['status'] != 'waiting for verified live extended curriculum':
        raise ValueError('Original later trainer has left its verified idle wait')
    children_path = Path('/proc') / str(WAITER['pid']) / 'task' / str(WAITER['pid']) / 'children'
    if children_path.read_text().strip():
        raise ValueError('Original queued trainer already owns children')
    main = resources.identity(2754742)
    if not main or main['ticks'] != '27602681':
        raise ValueError('Exact original main evaluator required before holding idle trainer')
    if not json.loads((ROOT / 'reports/extended-curriculum-v1/status.json').read_text())['status'].startswith('evaluating'):
        raise ValueError('Original main must have completed training and handed off to evaluation')
    usage = subprocess.run(['nvidia-smi', '--id=1,2', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout.splitlines()
    if len(usage) != 2 or any(int(s.strip()) > 128 for s in usage):
        raise ValueError('Both assigned physical GPUs must be idle')
    if shutil.disk_usage(ROOT).free < RESERVE + 4 * MAX_CHECKPOINT + EXTRA:
        raise ValueError('Fund both latest+pending complete checkpoints and reports above26GiB')
    from slm_lab.evaluation_provenance import capture
    files = {str(ROOT / p): digest for p, digest in capture()['files_sha256'].items()}
    paths = [Path(__file__), Path(resources.__file__), BASE, TOKEN, SELECTION,
        ROOT / 'reports/code-residual-lr-pilot-v1/plan.json', PAIR / 'manifest.json',
        ROOT / 'reports/code-prompt-matching-v1-focused-tests.log',
        ROOT / 'reports/code-prompt-pilot-v1-focused-tests.log',
        ROOT / 'reports/code-prompt-pairs-v1-independent-audit.json',
        ROOT / 'scripts/audit_code_prompt_pairs_v1.py',
        ROOT / 'tests/test_code_prompt_matching_v1.py', ROOT / 'tests/test_code_prompt_pilot_v1.py']
    names = ['code_prompt_matching_v1', 'code_prompt_pair_corpus_v1', 'code_prompt_pilot_v1',
        'code_residual_branch_v1', 'code_residual_checkpoint_v1', 'code_residual_pilot_v1',
        'code_residual_stage_v3', 'codenet_runner_v1', 'codenet_stage_v1', 'codenet_token_pack_v1',
        'codenet_checkpoint_v1', 'codenet_replay_v1', 'codenet_fact_replay_v1', 'codenet_replay_schedule_v1',
        'codenet_dialogue_replay_v1', 'codenet_image_replay_v1', 'codenet_later_instruction_replay_v2',
        'codenet_packed_replay_v1', 'recurrent_ablation']
    import importlib
    for name in names:
        module = importlib.import_module('slm_lab.' + name)
        paths.append(Path(module.__file__))
    import sys
    paths.extend(Path(m.__file__) for n, m in list(sys.modules.items())
        if n.startswith('slm_lab.') and getattr(m, '__file__', None))
    for folder in FOLDERS.values():
        paths.extend(p for p in folder.iterdir() if p.is_file())
    pair_manifest = json.loads((PAIR / 'manifest.json').read_text())
    files.update(pair_manifest['source_files_sha256'])
    files.update({str(PAIR / name): digest for name, digest in pair_manifest['artifact_files_sha256'].items()})
    files.update({str(p): file_hash(p) for p in paths})
    if files[str(BASE)] != '2dacd490e4a6ed20527a8393629d560e7230c1b6fdec54c1a49dc2eedf7f4e49':
        raise ValueError('Exact own completed joint policy required')
    old_plan = json.loads((ROOT / 'reports/code-residual-lr-pilot-v1/plan.json').read_text())
    if files[str(SELECTION)] != old_plan['selection_sha256']:
        raise ValueError('Unchanged prior original source selection required')
    audit = json.loads((ROOT / 'reports/code-prompt-pairs-v1-independent-audit.json').read_text())
    if (audit.get('passed') is not True or audit.get('manifest_sha256') != files[str(PAIR / 'manifest.json')]
            or audit.get('original_full_programmes_read_back') != {'train':112609, 'validation':5415}):
        raise ValueError('Independent complete original pairing/source audit required')
    FileBindings(files).check(full=True)
    REPORTS.mkdir()
    OUTPUT.mkdir()
    plan = dict(format='own-full857M-source-matching-pilot-v1', created_at=now(),
        owner=resources.identity(os.getpid()), waiter=WAITER, original_main=main,
        profiles=PROFILES, base_checkpoint=str(BASE), base_checkpoint_sha256=files[str(BASE)],
        tokenizer_sha256=files[str(TOKEN)], source_files_sha256=files,
        identical_fresh_zero_up_residuals_and_original_data_order=True, updates_per_profile=128,
        original_training_programmes_per_profile=512, original_training_languages=54,
        validation_programmes=128, validation_languages=53, train_validation_group_overlap=0,
        complete_source_pair_training_population=112609, complete_source_pair_validation_population=5415,
        replay_families=29, replay_examples_per_update=1, replay_coefficient=.5, image_size=448,
        source_objective='target-token-mean CE + alpha * token-weighted softplus(10*(matchedNLL-mismatchedNLL)+.1)',
        mismatched_prefix='Whole question of a different original same-language/role problem; every original code/EOS target preserved.',
        label_scope='Source association only, not execution-certified incorrect programs or human preferences.',
        learning_rate='3e-5 peak,16-update linear warmup', weight_decay=.01, gradient_clip=1.,
        batch_size=1, code_accumulation=4, fresh_Adam_per_profile=True,
        compute_matched=False, extra_matching_backward_passes_only_in_experimental_profile=True,
        maximum_checkpoint_bytes=MAX_CHECKPOINT, retained_checkpoint_slots_per_profile=1,
        funded_pending_slots_per_profile=1, minimum_free_bytes=RESERVE,
        final_budget_checkpoint_fixed_before_benchmarks=True, no_benchmark_feedback_for_training=True,
        full_final_source_or_final_policy_integration=False, candidate_promoted=False, goal_achieved=False,
        scope='Bounded hypothesis test on actual857M models. Full112609-program production, all30 final source families, final own base, full fresh baselines and genuine-feedback continuation remain required.')
    atomic_json(REPORTS / 'plan.json', plan)
    atomic_json(REPORTS / 'children.json', [])
    watcher = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()), '--watchdog'],
        cwd=ROOT, start_new_session=True)
    processes, registry = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Pilot owner interrupted')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        os.kill(WAITER['pid'], signal.SIGSTOP)
        deadline = time.monotonic() + 5
        while (resources.same(WAITER) or {}).get('state') not in {'T', 't'}:
            if time.monotonic() > deadline:
                raise ValueError('Exact idle waiter did not stop')
            time.sleep(.05)
        if children_path.read_text().strip():
            raise ValueError('Original waiter spawned a child before hold')
        for profile in PROFILES:
            with (REPORTS / (profile['name'] + '.log')).open('xb') as log:
                process = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()), '--profile', profile['name']],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(profile['physical_gpu']),
                        'OMP_NUM_THREADS': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
            registry.append({**resources.identity(process.pid), 'subject': profile['name']})
            atomic_json(REPORTS / 'children.json', registry)
            processes.append(process)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in {None, 0} for p in processes):
                raise ValueError('A pilot worker stopped; retain state/logs and restore original waiter')
            status(REPORTS, 'full857M matched source-objective experiment active',
                children=[dict(**a, exit_code=p.poll()) for a, p in zip(registry, processes)],
                original_idle_waiter_held=(resources.same(WAITER) or {}).get('state') in {'T', 't'})
            time.sleep(2)
        if any(p.returncode != 0 for p in processes):
            raise ValueError('One or more full-model pilots stopped; no automatic retry')
        status(REPORTS, 'both fixed128-update source-objective experiments and code/knowledge assessments completed')
    except BaseException as error:
        atomic_json(REPORTS / 'failure.json', dict(error=str(error), automatic_retry=False, goal_achieved=False))
        raise
    finally:
        resources.restore(plan, 'pilot workers ended or stopped; original exact queued trainer restored')
        watcher.wait(timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--profile', choices=[p['name'] for p in PROFILES])
    args = parser.parse_args()
    if args.watchdog:
        resources.watchdog()
    elif args.profile:
        worker(args.profile)
    else:
        parent()
