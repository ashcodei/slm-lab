"""Full own-policy BF16 inference parity and fixed-source timing control.

Only complete original training sources are used. No optimizer, new weights,
GPU context or change to the existing FP32 assessment is involved.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1')
OUT = Path('/tmp/slm-broad-CPU-BF16-generation-probe-v1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def memory():
    stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    live = sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel'))
    return dict(live_non_file_bytes=live, limit_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()))


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 one-thread inference probe required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import torch
    from slm_lab.CPU_frozen_cast_cache_v1 import CPUFrozenCastCache
    from slm_lab.broad_anchored_execution_stage_v1 import source_order
    from slm_lab.broad_anchored_policy_v3 import fresh
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.inference_rotary_cache_v1 import inference_cache
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    budget = memory()
    if budget['limit_bytes'] - budget['live_non_file_bytes'] < 60 * 1024**3:
        raise ValueError('12GiB private working allowance plus48GiB other-job memory headroom required')
    if shutil.disk_usage(OUT.parent).free < 42 * 1024**3 or torch.cuda.is_initialized():
        raise ValueError('CPU-only probe must retain40GiB scratch floor and2GiB output headroom')
    receipt = json.loads((SNAPSHOT / 'complete.json').read_text())
    plan = json.loads((SNAPSHOT / 'plan.json').read_text())
    if not receipt['complete'] or receipt['completed'] != 512 or plan['planned_steps'] != 3721:
        raise ValueError('Exact captured own512/3721 source lineage required')
    files = dict(receipt['source_files_sha256'])
    files[str(SNAPSHOT / 'complete.json')] = sha(SNAPSHOT / 'complete.json')
    files.update({str(p): sha(p) for p in python_sources(ROOT, [Path(__file__)])})
    guard = FileBindings(files)
    guard.check(full=True)
    OUT.mkdir()
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    write(OUT / 'owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        policy='complete own human_DPO starting policy', CPU_threads=1, CPU_affinity=sorted(os.sched_getaffinity(0)),
        nice=19, scheduler='SCHED_IDLE', device='CPU', precision='BF16 autocast over unchanged FP32 parameters',
        source_seed=963117, source_selection='first two complete original train positions in fixed source order',
        generation_source='first selected original train prompt only', generation_budget=512,
        timed_order=['uncached_cast_1', 'cached_cast', 'uncached_cast_2'], greedy=True,
        derived_cast_copies_counted_separately_from_learned_parameters=True, memory_admission=budget,
        working_allowance_bytes=12 * 1024**3, minimum_other_job_memory_bytes=48 * 1024**3,
        optional_BF16_assessment_admission=dict(minimum_observed_actions=256, minimum_cached_actions_per_second=10.,
            minimum_speedup_over_both_uncached_controls=1.5),
        no_parameter_updates_or_private_training_RNG_resume=True, original_FP32_comparison_not_changed=True,
        quality_gain_not_measured=True, goal_achieved=False))
    cache = CPUFrozenCastCache()
    try:
        print(json.dumps(dict(phase='load_actual_own_starting_policy', owner=owner)), flush=True)
        load_plan = copy.deepcopy(plan)
        load_plan['starting_policy_inputs']['checkpoint'] = str(SNAPSHOT / 'starting-human-DPO-actor.pt')
        model, _ = fresh(load_plan, 'cpu')
        if model.counts()['actual_unique'] != 999706442:
            raise ValueError('Every actual native/actor/critic parameter must remain under1B')
        corpus = ExpandedExecutionCorpus(ROOT)
        positions = source_order(corpus, 963117)[:2]
        batches = [corpus.batch(p, 'cpu') for p in positions]
        sources = [dict(position=p, id=corpus.row(p)['id'], complete_context_positions=corpus.row(p)['complete_context_positions'],
            prefix_positions=corpus.row(p)['prefix_positions']) for p in positions]
        parameter_signature = [(id(p), p._version, p.data_ptr(), p.dtype, p.requires_grad) for p in model.parameters()]
        base_sha = tensor_digest(model.base.named_parameters())
        parts_sha = tensor_digest(model.parts.named_parameters())
        cpu_rng = torch.get_rng_state().clone()
        def intact():
            if (parameter_signature != [(id(p), p._version, p.data_ptr(), p.dtype, p.requires_grad) for p in model.parameters()]
                    or any(p.grad is not None for p in model.parameters()) or torch.cuda.is_initialized()
                    or not torch.equal(cpu_rng, torch.get_rng_state())):
                raise ValueError('Every parameter value/version/flag, absent gradient and CPU RNG must stay unchanged')
            cache.assert_current()
            guard.check()
        with torch.no_grad(), inference_cache():
            expected = []
            for batch in batches:
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])['logits']
                if logits.dtype != torch.bfloat16:
                    raise ValueError('Actual full-source logits must use CPU BF16')
                expected.append(logits.clone())
            cache.attach(model.policy)
            for batch, wanted in zip(batches, expected):
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    actual = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])['logits']
                if not torch.equal(wanted, actual):
                    raise ValueError('Every complete original source logit must be bit-exact with explicit cached casts')
            parity_inventory = cache.inventory()
            cache.close()
            del expected, logits, actual, wanted
            intact()
            write(OUT / 'full-source-parity.json', dict(complete=True, sources=sources, actual_counts=model.counts(),
                all_complete_original_source_logits_bit_exact=True, derived_cast_inventory=parity_inventory,
                precision='CPU BF16', no_new_model_quality_claim=True))
            prompt = batches[0]['input_ids'][:, :sources[0]['prefix_positions']].clone()
            rows = []
            for name in ('uncached_cast_1', 'cached_cast', 'uncached_cast_2'):
                if name == 'cached_cast':
                    cache.attach(model.policy)
                print(json.dumps(dict(phase='whole_native_generation', profile=name, original_prompt_tokens=prompt.shape[1])), flush=True)
                begin = time.monotonic()
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    generated = model.generate(prompt, max_new_tokens=512, eos_token_id=2, use_cache=True)
                seconds = time.monotonic() - begin
                ids = generated[0, prompt.shape[1]:].tolist()
                if (not torch.equal(prompt, generated[:, :prompt.shape[1]]) or not 0 < len(ids) <= 512
                        or 2 in ids[:-1] or rows and ids != rows[0]['actual_response_ids']):
                    raise ValueError('Every actual greedy action/EOS must agree across all three complete native runs')
                intact()
                row = dict(profile=name, actual_complete_prompt_ids=prompt[0].tolist(), actual_response_ids=ids,
                    seconds=seconds, actual_actions=len(ids), actions_per_second=len(ids)/seconds,
                    stopped_on_EOS=ids[-1] == 2, ended_on_budget=len(ids) == 512 and ids[-1] != 2,
                    derived_cast_inventory=cache.inventory())
                write(OUT / (name + '.json'), row)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ('profile', 'actual_actions', 'seconds', 'actions_per_second')}), flush=True)
                if name == 'cached_cast':
                    cache.close()
        intact()
        if (tensor_digest(model.base.named_parameters()) != base_sha
                or tensor_digest(model.parts.named_parameters()) != parts_sha):
            raise ValueError('Every original native/actor/critic value must remain exact after all probe calls')
        guard.check(full=True)
        corpus.guard.check(full=True)
        speedup = min(rows[0]['seconds'], rows[2]['seconds']) / rows[1]['seconds']
        admit = rows[1]['actual_actions'] >= 256 and rows[1]['actions_per_second'] >= 10. and speedup >= 1.5
        write(OUT / 'complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(),
            source_files_sha256=files, actual_counts=model.counts(), sources=sources,
            full_source_BF16_cast_parity=True, all_three_complete_native_deliveries_bit_exact=True,
            every_original_parameter_and_CPU_RNG_unchanged=True, timed_runs=rows,
            minimum_observed_uncached_over_cached_speedup=speedup,
            separate_BF16_assessment_timing_admitted=admit,
            single_fixed_training_prompt_shared_host_timing_not_general_throughput_proof=True,
            original_FP32_comparison_not_changed=True, model_quality_gain_verified=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, minimum_speedup=speedup, separate_BF16_timing_admitted=admit)), flush=True)
    except BaseException as error:
        write(OUT / 'failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            original_workers_and_inputs_not_changed=True, partial_outputs_retained=True, goal_achieved=False))
        raise
    finally:
        cache.close()


if __name__ == '__main__':
    main()
