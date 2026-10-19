"""Parameter-matched own857M FFN-only versus FFN/attention CE experiment.

This bounded128-update hypothesis test preserves the final all30-source/base/
baseline production requirements. It holds only the exact childless image-data
waiter, with the established restoration watchdog, and never preempts training.
"""
import argparse
from datetime import datetime, timezone
import fcntl
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
REPORTS = ROOT / 'reports/attention-allocation-pilot-v1'; OUTPUT = ROOT / 'checkpoints/attention-allocation-pilot-v1'
BASE = ROOT / 'checkpoints/extended-curriculum-v1/joint/model.pt'; TOKEN = BASE.parent / 'tokenizer.json'
PAIR = ROOT / 'data/code-prompt-pairs-v1'; FEEDBACK = ROOT / 'data/execution-feedback-union-v1'
SELECTION = ROOT / 'reports/code-residual-lr-pilot-v1/selected-original-rows.json'
WAITER = dict(pid=2754743, ticks='27602681')
WAIT_STATE = 'waiting for finalized original paired real-photo questions'
PROFILES = [dict(name='FFN_only_control', alpha=0., physical_gpu=1, attention=False),
    dict(name='FFN_plus_attention', alpha=0., physical_gpu=2, attention=True)]
RESERVE, MAX_CHECKPOINT, EXTRA = 26 * 1024**3, 2 * 1024**3, 1024**3
FOLDERS = {k: ROOT / 'data' / v for k, v in [('joint', 'codenet-replay-v1'), ('facts', 'codenet-fact-replay-v1'),
    ('dialogue', 'codenet-dialogue-replay-v1'), ('images', 'codenet-image-replay-v1'),
    ('later', 'codenet-later-instruction-replay-v2'), ('packed', 'codenet-packed-pretraining-replay-v1')]}
resources.REPORTS = REPORTS


def status(folder, message, **fields):
    atomic_json(folder / 'status.json', dict(status=message, updated_at=datetime.now(timezone.utc).isoformat(),
        candidate_promoted=False, goal_achieved=False, **fields))


def ownership(plan):
    waiter = resources.same(plan['waiter'])
    if not resources.same(plan['owner']) or not waiter or waiter['state'] not in {'T', 't'}:
        raise ValueError('Exact pilot owner and held original childless data waiter required')


def worker(name):
    plan = json.loads((REPORTS / 'plan.json').read_text()); me = resources.identity(os.getpid())
    while True:
        if not resources.same(plan['owner']): raise ValueError('Pilot owner ended before worker registration')
        registry = json.loads((REPORTS / 'children.json').read_text())
        if any(c['pid'] == me['pid'] and c['ticks'] == me['ticks'] and c['subject'] == name for c in registry): break
        time.sleep(.1)
    ownership(plan)
    profile = next(p for p in plan['profiles'] if p['name'] == name)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(profile['physical_gpu']):
        raise ValueError('Exact isolated physical GPU required')
    import gc
    import random
    import shutil
    import numpy as np
    from tokenizers import Tokenizer
    import torch
    from slm_lab import code_quality, knowledge_evaluation
    from slm_lab.code_prompt_pair_corpus_v1 import SourcePairs
    from slm_lab.code_prompt_pilot_v1 import validate_selection, verify_saved_state, FrozenPilotLoader, evaluator_loader
    from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
    from slm_lab.code_attention_branch_v1 import RoutedCodeAttention
    from slm_lab.code_residual_checkpoint_v1 import make_training_state
    from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
    from slm_lab.code_residual_stage_v3 import open_expanded_replay, ExpandedReplayEncoder
    from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
    from slm_lab.codenet_fact_replay_v1 import CombinedReplay
    from slm_lab.codenet_stage_v1 import EpochOrder
    from slm_lab.execution_code_pilot_v1 import update
    from slm_lab.execution_feedback_union_v1 import ExecutionFeedbackUnion
    from slm_lab.execution_preference_v1 import reference_scores
    from slm_lab.recurrent_ablation import state_digest
    from slm_lab.training import load_checkpoint

    report = REPORTS / name; report.mkdir()
    bound = FileBindings({**plan['source_files_sha256'], str(REPORTS / 'plan.json'): file_hash(REPORTS / 'plan.json')})
    def guard(full=False):
        ownership(plan); bound.check(full=full)
        if shutil.disk_usage(ROOT).free < RESERVE: raise ValueError('Retain original26GiB persistent reserve')
    guard(True)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    if torch.cuda.get_device_name(device) != 'NVIDIA L40' or not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info(device)[0] < 40 * 1024**3:
        raise ValueError('Initially40GiB free on the exact owned BF16 L40 required')
    torch.cuda.set_per_process_memory_fraction(.85, device)
    output = OUTPUT / name; output.mkdir()
    random.seed(920041); np.random.seed(920042); torch.manual_seed(920043); torch.cuda.manual_seed_all(920044)
    tokenizer = Tokenizer.from_file(str(TOKEN)); feedback = ExecutionFeedbackUnion(FEEDBACK, tokenizer, TOKEN)
    if len(feedback.positives) != 195 or len(feedback.pairs) != 374: raise ValueError('Complete actual Java/Python union required')
    sources = {role: SourcePairs(PAIR, role) for role in ['train', 'validation']}
    selection = json.loads(SELECTION.read_text()); validate_selection(selection, sources)
    replay = open_expanded_replay(FOLDERS, plan['tokenizer_sha256']); joint = replay
    while isinstance(joint, CombinedReplay): joint = joint.owners['human_instruction']
    base, metadata = load_checkpoint(BASE, 'cpu', training_state=False)
    if metadata['tokenizer_sha256'] != plan['tokenizer_sha256']: raise ValueError('Exact own completed joint/BPE required')
    model = RoutedCodeAttention(base, width=1728, rank=96, scale=1.) if profile['attention'] else RoutedCodeResidual(base, width=2048)
    architecture = model.architecture() if profile['attention'] else dict(format='own-code-FFN-only-v1', residual_width=2048)
    initial = dict(model_counts=model.parameter_counts(), residual_state_sha256=state_digest(model.residuals.state_dict()),
        frozen_backbone_state_sha256=state_digest(model.base.state_dict()))
    if initial['model_counts'] != dict(total_unique=857007616, frozen_base=706012672, trainable_code=150994944):
        raise ValueError('Actual own full857M model required')
    atomic_json(report / 'initial-policy.json', dict(**initial, architecture=architecture)); model.to(device); model.code.gradient_checkpointing = True
    shutil.copy2(TOKEN, output / 'tokenizer.json')
    encoder = ExpandedReplayEncoder(tokenizer, model.config, joint.source.parent, 448)
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    policy = dict(base_checkpoint_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'],
        **initial, architecture=architecture, precision='bf16', device_type='cuda')
    policy_sha = fingerprint(policy)
    status(report, 'scoring all374complete frozen own-reference pairs before updates')
    model.eval()
    reference = reference_scores(model.code, [feedback.example(i) for i in range(374)], policy_sha256=policy_sha, device=device, amp=amp)
    if state_digest(model.residuals.state_dict()) != initial['residual_state_sha256']: raise ValueError('Reference policy changed')
    atomic_json(report / 'reference.json', dict(policy=policy, cache=reference, cache_fingerprint=fingerprint(reference),
        source_manifest_sha256=file_hash(FEEDBACK / 'manifest.json'), source_admission_sha256=file_hash(FEEDBACK / 'admission.json')))
    from slm_lab.code_prompt_matching_v1 import scores
    diagnostics = [sources['validation'].example(r['local_position']) for r in selection['validation']]
    def source_diagnostic(phase):
        model.eval()
        measured = scores(model.code, diagnostics, device, amp)
        atomic_json(report / ('source-conditioning-' + phase + '.json'), dict(phase=phase,
            source_ids=[e['id'] for e in diagnostics], measurements=measured,
            original_code_and_EOS_unchanged=True, used_for_gradients_or_endpoint_selection=False,
            correctness_or_human_feedback=False))
    source_diagnostic('before')
    examples = [sources['train'].example(r['local_position']) for r in selection['train']]
    replay_orders = {n: EpochOrder(replay.size(n), 920100 + i) for i, n in enumerate(replay.families)}
    positives_order, pair_order = EpochOrder(195, 920200), EpochOrder(374, 920300)
    optimizer = torch.optim.AdamW(model.adapter_parameters(), lr=3e-5, weight_decay=.01, foreach=False)
    metrics = []; tick = time.monotonic(); positive_seen = set(); pair_seen = set()
    for step in range(128):
        guard(); feedback.guard.check()
        family = replay.families[step % len(replay.families)]; occurrence = step // len(replay.families)
        position = replay_orders[family].at(occurrence)
        pp = [positives_order.at(step * 2 + i) for i in range(2)]; ep = [pair_order.at(step * 4 + i) for i in range(4)]
        positives = [{**feedback.positives[i], 'sequence': feedback.positive(i)} for i in pp]
        pairs = [feedback.example(i) for i in ep]
        record = update(model, optimizer, examples[step * 4:(step + 1) * 4], positives, pairs, reference,
            policy_sha256=policy_sha, replay_encoder=encoder, replay_row=replay.row(family, position), family=family,
            step=step, alpha=profile['alpha'], device=device, amp=amp)
        positive_seen.update(pp); pair_seen.update(ep)
        record.update(replay_position=position, replay_occurrence=occurrence, positive_positions=pp, pair_positions=ep)
        metrics.append(record); atomic_json(report / 'training-metrics.json', metrics)
        if step < 2 or (step + 1) % 4 == 0:
            status(report, 'training actual857M parameter-matched attention allocation', completed_steps=step + 1,
                planned_steps=128, latest_objective=record['summed_objective'])
            print('execution_update', name, step + 1, record['summed_objective'], flush=True)
        if (step + 1) % 32 == 0:
            guard(True)
            if state_digest(model.base.state_dict()) != initial['frozen_backbone_state_sha256']:
                raise ValueError('Every own frozen backbone tensor must remain identical')
            bundle = make_training_state(model, optimizer, base_checkpoint_sha256=plan['base_checkpoint_sha256'],
                tokenizer_sha256=plan['tokenizer_sha256'], run_plan_sha256=file_hash(REPORTS / 'plan.json'),
                completed_steps=step + 1, planned_steps=128, device=device, metrics=metrics)
            bundle.update(stage='attention_allocation_pilot_v1', architecture=architecture, profile=profile, execution_reference=reference,
                execution_reference_policy=policy, execution_reference_fingerprint=fingerprint(reference),
                feedback_manifest_sha256=file_hash(FEEDBACK / 'manifest.json'), feedback_admission_sha256=file_hash(FEEDBACK / 'admission.json'),
                sampler_state=dict(updates=step + 1, code_draws=4 * (step + 1), positive_draws=2 * (step + 1),
                    pair_draws=4 * (step + 1), replay_draws=step + 1, replay_family_populations={n: replay.size(n) for n in replay.families},
                    seeds=dict(positive=920200, pair=920300, replay_base=920100)), clipped_sequences=encoder.truncated)
            atomic_checkpoint(output / 'latest.pt', bundle, maximum_bytes=MAX_CHECKPOINT, minimum_free_bytes=RESERVE + 2 * MAX_CHECKPOINT + EXTRA)
            del bundle; gc.collect()
    if len(positive_seen) != 195 or len(pair_seen) != 374 or encoder.truncated:
        raise ValueError('Every verified original/pair must actually enter training, without clipping')
    model.eval(); guard(True)
    receipt = verify_saved_state(model, output / 'latest.pt', device=device, plan_sha256=file_hash(REPORTS / 'plan.json'),
        base_sha256=plan['base_checkpoint_sha256'], tokenizer_sha256=plan['tokenizer_sha256'], completed=128)
    if torch.load(output / 'latest.pt', map_location='cpu', weights_only=True, mmap=True)['architecture'] != architecture:
        raise ValueError('Exact architecture/rank/scale must be saved with all adapter tensors')
    atomic_json(report / 'frozen-final-policy.json', dict(**receipt, architecture=architecture, sealed_before_benchmark_evaluation=True,
        training_seconds=time.monotonic() - tick, all195positives_and374pairs_exposed=True, candidate_promoted=False, goal_achieved=False))
    source_diagnostic('after')
    del optimizer; gc.collect(); torch.cuda.empty_cache()
    loader = FrozenPilotLoader(model, output / 'latest.pt', dict(metadata, stage='attention_allocation_pilot_v1', steps=128, planned_steps=128), receipt, device, lambda: guard())
    for mode in ['completion', 'instruction']:
        status(report, 'evaluating fixed128-update explicit code route with unchanged local24-code protocol', prompt_mode=mode)
        with evaluator_loader(code_quality, loader):
            code_quality.evaluate(output / 'latest.pt', report / ('code-' + mode + '.json'), device=str(device), max_new_tokens=512, prompt_mode=mode)
    status(report, 'evaluating fixed128-update explicit code route on all559knowledge questions')
    with evaluator_loader(knowledge_evaluation, loader):
        knowledge_evaluation.evaluate(output / 'latest.pt', ROOT / 'data/knowledge-eval-v1/questions.jsonl', report / 'knowledge.json',
            device=str(device), max_examples=1000, max_new_tokens=32)
    loader.check(); guard(True)
    atomic_json(report / 'evaluation-binding.json', dict(checkpoint=receipt, loader_calls=loader.calls,
        unchanged_native_prompts_and_scoring=True, held_gold_used_for_gradients=False, complete_reference_Adam_RNG_retained=True,
        candidate_promoted=False, goal_achieved=False))
    status(report, 'bounded attention-allocation experiment and native code/knowledge assessment completed', completed_steps=128)


def parent():
    import shutil
    if REPORTS.exists() or OUTPUT.exists(): raise ValueError('Preserve prior experiment artifacts; no implicit restart')
    child_path = Path('/proc') / str(WAITER['pid']) / 'task' / str(WAITER['pid']) / 'children'
    def idle():
        actor = resources.same(WAITER)
        if (not actor or actor['state'] != 'S' or child_path.read_text().strip()
                or json.loads((ROOT / 'reports/dialogue-retention-v1/status.json').read_text())['status'] != WAIT_STATE):
            raise ValueError('Original trainer must be in its exact childless finalized-image wait; never preempt training/preparation')
    idle()
    if '6 passed' not in (ROOT / 'reports/code-attention-branch-v1-release-tests.log').read_text():
        raise ValueError('Current full/cached/gradient/complete-recovery architecture checks required')
    if '1 passed' not in (ROOT / 'reports/attention-allocation-pilot-v1-driver-tests.log').read_text():
        raise ValueError('Current active-original-trainer prelaunch rejection check required')
    if shutil.disk_usage(ROOT).free < RESERVE + 4 * MAX_CHECKPOINT + EXTRA: raise ValueError('Fund all complete checkpoint slots above26GiB')
    from tokenizers import Tokenizer
    from slm_lab.execution_feedback_union_v1 import ExecutionFeedbackUnion
    old = json.loads((ROOT / 'reports/execution-code-pilot-v1/plan.json').read_text())
    files = dict(old['source_files_sha256'])
    feedback = ExecutionFeedbackUnion(FEEDBACK, Tokenizer.from_file(str(TOKEN)), TOKEN)
    files.update({str(p): h for p, h in feedback.guard.files.items()})
    for p in [Path(__file__).resolve(), ROOT / 'src/slm_lab/code_attention_branch_v1.py',
            ROOT / 'src/slm_lab/execution_code_pilot_v1.py', ROOT / 'tests/test_code_attention_branch_v1.py',
            ROOT / 'reports/code-attention-branch-v1-release-tests.log',
            ROOT / 'tests/test_attention_allocation_pilot_v1.py', ROOT / 'reports/attention-allocation-pilot-v1-driver-tests.log']:
        files[str(p)] = file_hash(p)
    import sys
    for name, module in list(sys.modules.items()):
        if name.startswith('slm_lab.') and getattr(module, '__file__', None): files[str(Path(module.__file__).resolve())] = file_hash(module.__file__)
    FileBindings(files).check(full=True)
    with (ROOT / 'reports/code-residual-production-v4.gpu.lock').open('a+') as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); idle()
        memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.splitlines()
        if len(memory) != 4 or any(int(v.strip()) > 128 for v in memory): raise ValueError('All four physical GPUs must be idle')
        REPORTS.mkdir(); OUTPUT.mkdir()
        plan = dict(format='own-full857M-parameter-matched-attention-pilot-v1', created_at=datetime.now(timezone.utc).isoformat(),
            owner=resources.identity(os.getpid()), waiter=WAITER, profiles=PROFILES, source_files_sha256=files,
            base_checkpoint_sha256=file_hash(BASE), tokenizer_sha256=file_hash(TOKEN), updates_per_training_arm=128,
            complete_original_CodeNet_programmes=512, code_languages=54, verified_original_programmes=195,
            complete_assertion_preferences=374, reference_all374scored_before_updates=True,
            code_microbatches=4, positive_microbatches=2, pair_exposures_per_update=4,
            replay_families=29, replay_examples_per_update=1, full_replay_pass_completed=False,
            replay_coefficient=.5, positive_CE_coefficient=1., DPO_beta=.1, image_size=448,
            identical_original_backbone_initial_function_data_order_and_fresh_Adam=True,
            identical_adapter_tensor_layout=False, total_unique_parameter_budget_matched=True,
            fixed_updates_and_data_exposures_matched=True, compute_matched=False,
            architecture_hypothesis='Allocate23592960of150994944code parameters to rank96QKVO updates; reduce FFN width2048to1728. Both alpha0CE, no DPO gradient.',
            source_validation_diagnostic_all128=True, held_diagnostic_used_for_selection=False,
            fixed_budget_sealed_before_benchmarks=True, human_preference_RLHF=False,
            original_waiter_state_before_hold=WAIT_STATE, maximum_checkpoint_bytes=MAX_CHECKPOINT,
            minimum_free_bytes=RESERVE, maximum_seconds=10800,
            final_all30_source_and_final_own_policy_production=False, candidate_promoted=False, goal_achieved=False,
            scope='Bounded actual857M FFN-only versus FFN/attention CE hypothesis test; same parameters and update/data budget, different compute. Not final full112609-code/all30 replay production or frontier capability.')
        atomic_json(REPORTS / 'plan.json', plan); atomic_json(REPORTS / 'children.json', [])
        watcher = subprocess.Popen(['.venv/bin/python', '-u', str(Path(__file__).resolve()), '--watchdog'], cwd=ROOT, start_new_session=True)
        children = []; registry = []; started = time.monotonic()
        def interrupted(signum, frame): raise KeyboardInterrupt('Pilot owner interrupted')
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
        try:
            idle(); os.kill(WAITER['pid'], signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while (resources.same(WAITER) or {}).get('state') not in {'T', 't'}:
                if time.monotonic() > deadline: raise ValueError('Exact original idle waiter did not stop')
                time.sleep(.05)
            if child_path.read_text().strip() or json.loads((ROOT / 'reports/dialogue-retention-v1/status.json').read_text())['status'] != WAIT_STATE:
                raise ValueError('Original waiter advanced before hold; restore without launching')
            for profile in PROFILES:
                with (REPORTS / (profile['name'] + '.log')).open('xb') as stream:
                    child = subprocess.Popen(['.venv/bin/python', '-u', str(Path(__file__).resolve()), '--profile', profile['name']], cwd=ROOT,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(profile['physical_gpu']), 'OMP_NUM_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
                registry.append({**resources.identity(child.pid), 'subject': profile['name']})
                atomic_json(REPORTS / 'children.json', registry); children.append(child)
            while any(p.poll() is None for p in children):
                if time.monotonic() - started > plan['maximum_seconds']: raise ValueError('Bounded experiment lease expired; preserve state and restore original waiter')
                if any(p.poll() not in {None, 0} for p in children): raise ValueError('An experiment worker stopped; inspect preserved state before restart')
                status(REPORTS, 'actual857M parameter-matched attention-allocation experiment active',
                    children=[dict(**a, exit_code=p.poll()) for a, p in zip(registry, children)], original_idle_waiter_held=True)
                time.sleep(2)
            if any(p.returncode != 0 for p in children): raise ValueError('An experiment worker did not complete')
            status(REPORTS, 'both128-update attention-allocation experiments and native assessments completed')
        except BaseException as error:
            atomic_json(REPORTS / 'failure.json', dict(error=str(error), automatic_retry=False, goal_achieved=False)); raise
        finally:
            resources.restore(plan, 'attention-allocation experiment workers ended; exact original image-data waiter restored')
            watcher.wait(timeout=15)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--watchdog', action='store_true')
    parser.add_argument('--profile', choices=[p['name'] for p in PROFILES]); args = parser.parse_args()
    if args.watchdog: resources.watchdog()
    elif args.profile: worker(args.profile)
    else: parent()
