"""Measure exact scalar prompt reuse on the real counted warm-up policy.

CPU-only, short sampled responses, longest original public training prompt.
This measures neither full-horizon GPU speed nor generalization.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/execution-prefix-reuse-full-CPU-v5'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Hide all GPUs for this CPU measurement')
    if REPORTS.exists():
        raise ValueError('Preserve all earlier measurements')
    import torch
    from slm_lab.bounded_code_reward_catalog_v1 import BoundedPublicCodeReward
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.execution_group_bounded_v4 import BoundedExecutionGroup
    from slm_lab.execution_group_prefix_reuse_v5 import sample_group
    from slm_lab.fast_joint_inference_v2 import fresh
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run = ROOT / 'reports/verified-views-execution-v1'
    checkpoint = ROOT / 'checkpoints/verified-views-execution-v1/warmup-actor.pt'
    files = [Path(__file__), run / 'plan.json', checkpoint,
             ROOT / 'src/slm_lab/execution_group_prefix_reuse_v5.py',
             ROOT / 'src/slm_lab/execution_group_bounded_v4.py',
             ROOT / 'src/slm_lab/bounded_code_reward_catalog_v1.py',
             ROOT / 'reports/bounded-code-probes-v1/admission.json',
             ROOT / 'reports/execution-group-prefix-reuse-v5-tests.log',
             ROOT / 'reports/execution-group-prefix-four-CPU-v5/complete.json']
    bindings = {str(p): file_hash(p) for p in files}
    guard = FileBindings(bindings)
    plan = json.loads((run / 'plan.json').read_text())
    started = time.monotonic()
    base, _ = fresh(plan, 'cpu')
    model = OwnDenseActor(base)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    model.import_export(payload, expected_sources=payload['sources'])
    if model.counts()['actual_unique'] != 999706442:
        raise ValueError('Use the actual full counted own policy')
    del payload
    optimizer = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))], lr=1e-5)
    stage = BoundedExecutionGroup(model, optimizer, ROOT, method='execution_hybrid',
                                  group_size=2, max_new_tokens=4)
    candidates = []
    for p in range(len(stage.pack)):
        ids, labels = stage.pack.sequence(p)
        start = next(i for i, y in enumerate(labels) if y != -100)
        candidates.append((start, p, ids[:start]))
    length, position, prefix = max(candidates)
    source_id = stage.rows[position]['parent_source']['id']
    before = tensor_digest(model.named_parameters())
    original = model.policy.prefill
    counts = []
    def prefill(*args, **kwargs):
        counts.append(1)
        return original(*args, **kwargs)
    model.policy.prefill = prefill
    torch.manual_seed(958911)
    rng = torch.get_rng_state()
    begin = time.monotonic()
    old = stage.sample(prefix, source_id)
    old_seconds = time.monotonic() - begin
    old_calls = len(counts)
    expected_rng = torch.get_rng_state()
    counts.clear()
    torch.set_rng_state(rng)
    begin = time.monotonic()
    new = sample_group(stage, prefix, source_id)
    new_seconds = time.monotonic() - begin
    new_calls = len(counts)
    if (old_calls != 2 or new_calls != 1 or not torch.equal(torch.get_rng_state(), expected_rng)
            or any(a != {k: b[k] for k in a} for a, b in zip(old, new))):
        raise ValueError('Every actual action, probability, reward and final RNG must remain exact')
    model.policy.prefill = original
    if before != tensor_digest(model.named_parameters()) or optimizer.state:
        raise ValueError('Sampling must not update any model value or Adam state')
    guard.check(full=True)
    REPORTS.mkdir()
    atomic_json(REPORTS / 'complete.json', dict(
        at=datetime.now(timezone.utc).isoformat(), actual_unique_parameters=model.counts()['actual_unique'],
        original_source_position=position, original_source_id=source_id,
        complete_longest_original_prefix_tokens=length, group_size=2, maximum_new_tokens=4,
        actual_trajectories=new, baseline_prefills=old_calls, shared_prefills=new_calls,
        baseline_seconds=old_seconds, shared_seconds=new_seconds,
        short_CPU_wall_time_ratio=old_seconds/new_seconds,
        exact_actions_behavior_probabilities_rewards_and_final_CPU_RNG=True,
        all_model_values_exact_and_no_optimizer_calls=True, source_files_sha256=bindings,
        CPU_threads=2, trials_per_method=1, timings_include_execution_reward=True,
        execution_reward=BoundedPublicCodeReward(ROOT).provenance,
        full_horizon_GPU_throughput_not_measured=True, held_quality_not_measured=True,
        production_updates=0, seconds=time.monotonic()-started, goal_achieved=False))
    print(json.dumps(dict(prefix_tokens=length, baseline_prefills=old_calls,
                          shared_prefills=new_calls, baseline_seconds=old_seconds,
                          shared_seconds=new_seconds, exact=True)), flush=True)


if __name__ == '__main__':
    main()
