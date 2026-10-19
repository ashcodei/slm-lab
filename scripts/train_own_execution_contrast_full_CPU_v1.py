"""Controlled full-size CPU learning from own execution-rejected programs.

Wait for the exact conditional CPU trial, then compare40 updates of original
code CE against the same CE plus length-normalized own-negative preferences.
Both keep an own-initial-policy cached KL anchor. Full native development
code assessments follow for initial and both endpoints, without GPU use.
"""
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'reports/conditional-prefix-full-CPU-v1'
REPORTS = ROOT / 'reports/own-execution-contrast-full-CPU-v1'
OUTPUT = ROOT / 'checkpoints/own-execution-contrast-full-CPU-v1'
QUEUE = ROOT / 'reports/own-execution-contrast-full-CPU-v1-queue.json'
ARMS = ('anchor_CE', 'anchor_contrast')


def same(owner):
    try: fields = (Path('/proc') / str(owner['pid']) / 'stat').read_text().split(') ', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError): return None
    return fields[0] if fields[19] == owner['ticks'] and fields[0] not in ('Z', 'X') else None


def pair_order():
    import random
    result = []
    for epoch in range(2):
        indices = list(range(20)); random.Random(970217 + epoch).shuffle(indices); result += indices
    return result


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(p.exists() for p in [REPORTS, OUTPUT, QUEUE]):
        raise ValueError('Fresh CPU-hidden controlled own-negative trial required; no automatic retry')
    from slm_lab.codenet_checkpoint_v1 import FileBindings, atomic_checkpoint
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.evaluation_provenance import capture
    from slm_lab.local_source_closure_v2 import python_sources
    previous_plan = json.loads((PREVIOUS / 'plan.json').read_text()); prior = previous_plan['owner']
    owner = dict(pid=os.getpid(), ticks=(Path('/proc') / str(os.getpid()) / 'stat').read_text().split(') ', 1)[1].split()[19])
    data_path = ROOT / 'data/own-execution-negative-pairs-CPU-v1/manifest.json'
    mechanics_path = ROOT / 'reports/own-execution-contrast-CPU-v1-mechanics/complete.json'
    proof = json.loads(mechanics_path.read_text())
    if (not proof['next_actual_update_weights_Adam_and_private_CPU_RNG_bitexact']
            or not proof['changed_pair_restore_rejected_before_mutation'] or proof['current_full_size_model_or_GPU_training']):
        raise ValueError('Actual small native full-pair recovery/failure proof required')
    files = [Path(__file__), PREVIOUS / 'plan.json', data_path, mechanics_path,
        ROOT / 'reports/own-execution-contrast-CPU-v1-tests.log', ROOT / 'reports/own-execution-contrast-policy-CPU-v1-tests.log',
        ROOT / 'reports/own-execution-contrast-full-CPU-v1-driver-final-tests.log',
        ROOT / 'src/slm_lab/native_code_shards_v1.py', ROOT / 'src/slm_lab/code_quality.py', ROOT / 'src/slm_lab/inference_rotary_cache_v1.py',
        ROOT / 'src/slm_lab/native_prefix_rollout_CPU_v1.py', ROOT / 'tests/test_own_execution_contrast_CPU_v1.py',
        ROOT / 'tests/test_own_execution_contrast_policy_CPU_v1.py', ROOT / 'tests/test_own_execution_contrast_full_CPU_v1_driver.py']
    files += [ROOT / name for name in capture()['files_sha256']]
    files += python_sources(ROOT, files[:1] + [p for p in files if p.suffix == '.py' and p != Path(__file__)])
    bindings = {str(p): file_hash(p) for p in files}
    atomic_json(QUEUE, dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, predecessor=prior,
        status='waiting for the exact full-size conditional CPU comparison and its native cold reloads',
        source_files_sha256=bindings, fixed_pair_order=pair_order(), original_train_sources=4, pairs=20,
        planned_updates_each_arm=40, physical_optimizer_calls_each_arm=41, CPU_only=True,
        future_native_development_code_cases_each_endpoint=48, future_native_endpoints=3,
        automatic_retry=False, model_updates=0, candidate_promoted=False, goal_achieved=False))
    deadline = time.monotonic() + 86400
    while same(prior):
        if (PREVIOUS / 'failure.json').exists(): raise ValueError('Predecessor conditional CPU trial failed')
        if time.monotonic() > deadline: raise TimeoutError('Exact predecessor still live; no restart or preemption')
        time.sleep(2)
    completed_previous = PREVIOUS / 'complete.json'
    if not completed_previous.exists() or (PREVIOUS / 'failure.json').exists(): raise ValueError('Complete exact predecessor required')
    prior_result = json.loads(completed_previous.read_text())
    if not prior_result['native_student_only_text_image_cold_reload_exact'] or any(a['logical_updates'] != 3 for a in prior_result['arms'].values()):
        raise ValueError('Both predecessor full native endpoints and private recovery must finish first')
    import copy
    import random
    import numpy as np
    import torch
    from slm_lab.active_execution_input_bindings_v1 import input_bindings, merge_bindings
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.fast_joint_inference_v2 import load as load_base
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.native_code_shards_v1 import evaluate_shard, reconcile
    from slm_lab.native_prefix_rollout_CPU_v1 import sample
    from slm_lab.own_dense_actor_checkpoint_v1 import make_state, restore_state
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_execution_contrast_CPU_v1 import ExecutionContrastPairs, batch, update
    from slm_lab.own_execution_contrast_policy_CPU_v1 import export, load as load_trial, sources
    from slm_lab.own_tokenizer_package_v1 import ensure
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    data = ExecutionContrastPairs(data_path.parent); corpus = ExpandedExecutionCorpus(ROOT)
    merge_bindings(bindings, input_bindings(ROOT, corpus)); merge_bindings(bindings, data.guard.files)
    inputs = previous_plan['base_inputs']
    merge_bindings(bindings, {str(completed_previous): file_hash(completed_previous),
        str(Path(inputs['checkpoint'])): inputs['checkpoint_sha256'], str(Path(inputs['plan'])): inputs['plan_sha256']})
    base, metadata = load_base(inputs['checkpoint'], inputs['plan'], inputs['arm'],
        expected_checkpoint_sha256=inputs['checkpoint_sha256'], expected_plan_sha256=inputs['plan_sha256'], device='cpu')
    actor_settings = dict(rank=16, seed=970117, parameter_limit=10**9)
    actor = OwnDenseActor(base, **actor_settings).eval()
    if actor.counts()['actual_unique'] != 999706442 or torch.cuda.is_initialized(): raise ValueError('Actual own full-size under1B CPU actor required')
    initial_parts = tensor_digest(actor.parts.named_parameters()); base_digest = tensor_digest(base.named_parameters())
    REPORTS.mkdir(); OUTPUT.mkdir()
    ensure(ROOT / 'data/tokenizer-corpus/tokenizer.json', OUTPUT, expected_sha256=data.manifest['original_own_tokenizer_sha256'], expected_vocab_size=32768)
    own_cache = {}
    with torch.no_grad():
        for source_id, index in data.by_source.items():
            own_cache[source_id] = actor(batch(data.rows[index], 'chosen')['input_ids'])['logits'][:, :-1].float().log_softmax(-1).detach().clone()
    cache_path = OUTPUT / 'own-initial-full-context-logps.pt'
    atomic_checkpoint(cache_path, dict(initial_parts_sha256=initial_parts, own_base_sha256=base_digest, logps=own_cache,
        source_to_complete_native_input_ids={k: data.rows[v]['complete_chosen_ids'] for k, v in data.by_source.items()},
        external_teacher=False, own_reference_distributions_not_extra_neural_weights=True), 1024**3, 26 * 1024**3)
    merge_bindings(bindings, {str(cache_path): file_hash(cache_path)})
    order = pair_order(); order_digest = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    plan = dict(at=datetime.now(timezone.utc).isoformat(), owner=owner, predecessor=prior, base_inputs=inputs, actual_base_updates=2048,
        complete_human_alignment_or_source_epoch=False, pair_manifest_sha256=file_hash(data_path), pairs=20, original_train_sources=4,
        actor_settings=actor_settings, actual_counts=actor.counts(), initial_parts_sha256=initial_parts,
        own_initial_reference_cache_sha256=file_hash(cache_path), own_initial_cached_distribution_values=sum(t.numel() for t in own_cache.values()),
        additional_resident_reference_neural_parameters=0, active_registered_student_values=999706442,
        checkpoint_recovery_copies_and_Adam_are_training_only=True, planned_updates_each_arm=40,
        physical_optimizer_calls_each_arm=41, two_complete_passes_over_fixed20_pairs=True,
        all3721_public_code_sources_consumed=False, fixed_pair_order=order, fixed_pair_order_sha256=order_digest,
        learning_rate=1e-4, betas=[.9, .95], weight_decay=.01, chosen_CE_weight=1., preference_weight=.2,
        beta=2., margin=.5, own_initial_full_context_KL_weight=.01, CPU_threads=2,
        all_four_original_training_source_greedy_action_budgets=512, native_development_code_frames=['completion', 'instruction'],
        native_development_code_cases_each_endpoint=48, native_development_code_endpoints=['initial', *ARMS],
        native_development_code_budget=512, inference_and_training_precision='CPU_FP32',
        machine_preferences_not_human_RLHF=True, paper_url='https://arxiv.org/abs/2405.14734',
        custom_censored_negative_and_chosen_CE_plus_own_KL_extension_not_exact_paper_replication=True,
        source_files_sha256=bindings, automatic_retry=False, full_new_GPU_capacity_still_required=True,
        candidate_promoted=False, goal_achieved=False)
    atomic_json(REPORTS / 'plan.json', plan); digest = file_hash(REPORTS / 'plan.json'); native_sources = sources(plan, digest)
    guard = FileBindings(bindings)
    probe = torch.tensor([data.rows[0]['original_native_prefix_ids'][:16]])
    image = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    def native_probe(model):
        with torch.no_grad():
            return {name: tensor_digest([('native_logits', model(probe, pixel_values=pixels)['logits'])]) for name, pixels in [('text', None), ('image', image)]}
    native_digests = {'initial': native_probe(actor)}
    atomic_checkpoint(OUTPUT / 'initial-actor.pt', export(actor, plan, digest, 'initial', 0), 96 * 1024**2, 26 * 1024**3)
    def equal(a, b):
        if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    arm_results = {}
    for arm in ARMS:
        if arm != ARMS[0]: actor = OwnDenseActor(base, **actor_settings).eval()
        if tensor_digest(actor.parts.named_parameters()) != initial_parts: raise ValueError('Every arm must start from the exact original neutral actor')
        parameters = [p for _, p in actor.named_selected(('actor',))]
        optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(.9, .95), weight_decay=.01, foreach=False)
        random.seed(970317); np.random.seed(970317); torch.manual_seed(970317); metrics = []
        def cursor(count): return dict(method=arm, next_update=count, fixed_pair_order_sha256=order_digest, own_initial_reference_cache_sha256=plan['own_initial_reference_cache_sha256'])
        def step(index, label):
            row = data.rows[order[index]]; guard.check(full=False)
            result = update(actor, optimizer, row, own_cache[row['source_id']], method=arm)
            result.update(step=index + 1); metrics.append(result)
            atomic_json(REPORTS / (arm + '-' + label + '.json'), result)
            atomic_json(REPORTS / (arm + '-progress.json'), dict(actual_logical_updates=len(metrics), planned_updates=40,
                latest=result, machine_execution_not_human_RLHF=True, goal_achieved=False))
            print(json.dumps(dict(arm=arm, step=index + 1, CE=result['original_reference_CE'], preference=result['length_normalized_preference_loss'])), flush=True)
            return result
        step(0, 'first'); saved = OUTPUT / (arm + '-after-first.pt')
        atomic_checkpoint(saved, make_state(actor, optimizer, groups=('actor',), sources=native_sources, completed=1, planned=40,
            cursor=cursor(1), metrics=metrics, device='cpu'), 384 * 1024**2, 26 * 1024**3)
        second = step(1, 'expected-second'); expected = actor.export(native_sources); moments = copy.deepcopy(optimizer.state_dict()); rng = capture_rng('cpu')
        state = torch.load(saved, map_location='cpu', weights_only=True, mmap=True)
        bad = copy.deepcopy(state); bad['model']['sources']['own_initial_reference_cache_sha256'] = 'f' * 64
        versions = [(id(p), p._version, p.data_ptr()) for p in actor.parameters()]
        try: restore_state(actor, optimizer, bad, groups=('actor',), expected_sources=native_sources, planned=40, expected_cursor=cursor(1), device='cpu')
        except ValueError: pass
        else: raise ValueError('Changed own reference cache must reject before mutation')
        if versions != [(id(p), p._version, p.data_ptr()) for p in actor.parameters()]: raise ValueError('Rejected restore changed actor')
        _, metrics = restore_state(actor, optimizer, state, groups=('actor',), expected_sources=native_sources, planned=40, expected_cursor=cursor(1), device='cpu')
        replayed = step(1, 'replayed-second')
        if not all([equal(second, replayed), equal(expected, actor.export(native_sources)), equal(moments, optimizer.state_dict()), equal(rng, capture_rng('cpu'))]):
            raise ValueError('Actual full-size next-update weights/Adam/private CPU RNG must replay exactly')
        del expected, moments, state, bad
        for index in range(2, 40): step(index, 'step-' + str(index + 1))
        guard.check(full=True)
        if (len(optimizer.state) != 561 or any(float(s['step']) != 40 for s in optimizer.state.values())
                or tensor_digest(base.named_parameters()) != base_digest or tensor_digest(actor.parts.named_parameters()) == initial_parts):
            raise ValueError('Every561 selected Adam clock, changed actor and unchanged own base required')
        atomic_checkpoint(OUTPUT / (arm + '-final-state.pt'), make_state(actor, optimizer, groups=('actor',), sources=native_sources,
            completed=40, planned=40, cursor=cursor(40), metrics=metrics, device='cpu'), 384 * 1024**2, 26 * 1024**3)
        atomic_checkpoint(OUTPUT / (arm + '-actor.pt'), export(actor, plan, digest, arm, 40), 96 * 1024**2, 26 * 1024**3)
        native_digests[arm] = native_probe(actor)
        atomic_json(REPORTS / (arm + '-training-complete.json'), dict(actual_logical_updates=40, physical_optimizer_calls=41,
            all561_Adam_clocks=40, next_actual_update_weights_Adam_and_CPU_RNG_exact=True, own_reference_cache_tamper_rejected_before_mutation=True,
            own_base_vision_other_heads_and_critic_fixed=True, original_native_probe_logits_sha256=native_digests[arm],
            all_original_pair_positions_each_seen_twice=True, final_native_assessments_pending=True, goal_achieved=False))
        arm_results[arm] = metrics
        del actor, optimizer, parameters; gc.collect()
    del base; gc.collect()
    # Evaluate fresh student-only loads one at a time; no second resident neural
    # reference or teacher is needed, and cached KL data are not loaded by IO.
    outcomes = {}
    for endpoint in ['initial', *ARMS]:
        checkpoint = OUTPUT / (endpoint + '-actor.pt')
        actor, info = load_trial(checkpoint, REPORTS / 'plan.json', endpoint,
            expected_checkpoint_sha256=file_hash(checkpoint), expected_plan_sha256=digest)
        if native_probe(actor) != native_digests[endpoint]: raise ValueError('Actual student-only native cold text/image logits changed')
        folder = REPORTS / endpoint; folder.mkdir(); source_rows = []
        with inference_cache():
            for source_id, position in data.by_source.items():
                row = data.rows[position]
                actual = sample(actor.policy, corpus.tokenizer, row['original_native_prefix_ids'], [], horizon=512, sampler='greedy', seed=970417)
                reward = corpus.score(row['source_position'], actual['complete_candidate_response'], stopped_on_eos=actual['stopped_on_generated_EOS'])
                source_rows.append(dict(source_id=source_id, actual_rollout=actual, actual_original_execution=reward,
                    public_training_source_not_held=True, model_updates=0))
                atomic_json(folder / 'actual-unassisted-training-source-evaluation.json', source_rows)
            for frame in ['completion', 'instruction']:
                evaluate_shard(actor.policy, dict(metadata, tokenizer_sha256=plan['source_files_sha256'][str(ROOT / 'data/tokenizer-corpus/tokenizer.json')]
                    if str(ROOT / 'data/tokenizer-corpus/tokenizer.json') in plan['source_files_sha256'] else data.manifest['original_own_tokenizer_sha256'],
                    training_options=dict(image_size=448)), checkpoint, folder, rank=0, world=1, frame=frame, device='cpu')
                reconcile([folder / ('code-' + frame + '-shard-0.json')], folder / ('code-' + frame + '.json'))
        outcomes[endpoint] = dict(public_train_unassisted_full_finite_case_passes=sum(r['actual_original_execution']['reward'] == 1. for r in source_rows),
            public_train_sources=4, native_code={frame: {key: json.loads((folder / ('code-' + frame + '.json')).read_text())[key]
                for key in ['task_count', 'syntax_pass_rate', 'functional_pass_at_1', 'efficiency_stress_pass_rate']} for frame in ['completion', 'instruction']},
            native_cold_text_and_synthetic_image_probe_bitexact=True)
        atomic_json(folder / 'complete.json', dict(**outcomes[endpoint], development_suite_not_independent_final_benchmark=True, goal_achieved=False))
        del actor; gc.collect(); guard.check(full=True)
    if torch.cuda.is_initialized(): raise ValueError('CPU-only experiment initialized CUDA')
    atomic_json(REPORTS / 'complete.json', dict(at=datetime.now(timezone.utc).isoformat(), complete=True,
        actual_student_values_each_endpoint=999706442, actual_updates_each_arm=40, actual_physical_optimizer_calls_each_arm=41,
        actual_native_development_code_cases_each_endpoint=48, actual_source_unassisted_cases_each_endpoint=4,
        actual_outcomes=outcomes, native_student_only_cold_text_image_probes_exact=True,
        own_initial_reference_cache_excluded_from_inference=True, additional_reference_neural_parameters=0,
        all_original20_pairs_used_twice_per_arm=True, original_unique_train_problems=4,
        no_claim_of_full_source_epoch_human_RLHF_or_frontier_parity=True, full_new_GPU_admission_still_required=True,
        independent_raw_state_objective_and_generated_program_readback_pending=True, candidate_promoted=False, goal_achieved=False))


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        from slm_lab.diverse_images import atomic_json
        target = REPORTS / 'failure.json' if REPORTS.exists() else ROOT / 'reports/own-execution-contrast-full-CPU-v1-queue-failure.json'
        atomic_json(target, dict(error=repr(error), traceback=traceback.format_exc(), automatic_retry=False, goal_achieved=False))
        raise
