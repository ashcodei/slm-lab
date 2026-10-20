"""Full own-policy FP32 component parity and explicit Q8 inference probe."""
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/tmp/slm-broad-code-step512-snapshot-CPU-v1')
RUNTIME = Path('/tmp/slm-compiled-CPU-backend-v1')
OUT = Path('/tmp/slm-own-compiled-CPU-full-probe-v1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden one-thread low-priority probe required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    import gguf
    import llama_cpp
    import numpy as np
    import torch
    from torch.nn import functional as F
    from slm_lab.broad_anchored_execution_stage_v1 import source_order
    from slm_lab.broad_anchored_policy_v3 import fresh
    from slm_lab.codenet_checkpoint_v1 import FileBindings
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.local_source_closure_v2 import python_sources
    from slm_lab.own_decoder_gguf_v1 import export_text_component
    from slm_lab.own_decoder_gguf_quantize_v1 import quantize_text_component
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tests = Path('/tmp/slm-own-decoder-GGUF-v1-with-Q8-tests.log')
    if ('3 passed' not in tests.read_text() or llama_cpp.llama_supports_gpu_offload()
            or not Path(llama_cpp.__file__).resolve().is_relative_to(RUNTIME)
            or torch.cuda.is_initialized()):
        raise ValueError('Actual completed compiled controls and the isolated CPU-only runtime required')
    receipt = json.loads((SNAPSHOT / 'complete.json').read_text())
    plan = json.loads((SNAPSHOT / 'plan.json').read_text())
    assert receipt['complete'] and receipt['completed'] == 512
    files = dict(receipt['source_files_sha256'])
    paths = [SNAPSHOT / 'complete.json', tests, RUNTIME / 'runtime-info.json', RUNTIME / 'wheel-hashes.json',
        RUNTIME / 'requirements-installed.txt', ROOT / 'requirements-compiled-cpu-v1.txt']
    paths += python_sources(ROOT, [Path(__file__), ROOT / 'tests/test_own_decoder_gguf_v1.py'])
    paths += list((RUNTIME / 'wheels').glob('*.whl'))
    for module in (gguf, llama_cpp, np):
        paths += [p for p in Path(module.__file__).resolve().parent.rglob('*') if p.suffix in {'.py', '.so'}]
    for path in set(paths):
        digest = sha(path)
        if str(path) in files and files[str(path)] != digest:
            raise ValueError('Original immutable source bytes differ before probe')
        files[str(path)] = digest
    guard = FileBindings(files)

    def checked(full=False, initial=False):
        stat = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
        live = sum(int(stat[k]) for k in ('anon', 'shmem', 'kernel'))
        if (int(Path('/sys/fs/cgroup/memory.max').read_text()) - live < (60 if initial else 48) * 1024**3
                or shutil.disk_usage('/tmp').free < (52 if initial else 40) * 1024**3
                or torch.cuda.is_initialized()):
            raise ValueError('12GiBworking budget,48GiB other-job memory and40GiBscratch reserve required')
        guard.check(full=full)

    checked(True, initial=True)
    OUT.mkdir()
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    write('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        policy='complete own human_DPO starting policy', CPU_threads=1, CPU_affinity=sorted(os.sched_getaffinity(0)),
        nice=19, scheduler='SCHED_IDLE', original_FP32_model=True, component_precisions=['GGUF F32', 'GGUF Q8_0'],
        native_model_values=999706442, working_allowance_bytes=12 * 1024**3,
        source_seed=963117, full_original_train_sources=2, generation_source='first complete original train prompt',
        generation_budget=512, generation_EOS=2, cached_greedy=True, runtime_KV_precision='F32',
        F32_logit_parity_atol=.001, F32_logit_parity_rtol=.001, F32_response_NLL_maximum_absolute_drift=.001,
        optional_full_coding_assessment_gates=dict(minimum_actual_actions=256, minimum_actions_per_second=10.,
            maximum_Q8_response_NLL_absolute_drift=.05),
        component_only_not_full_multimodal_delivery=True, no_optimizer_or_training_RNG_restore=True,
        original_training_and_assessment_not_changed=True, goal_achieved=False))
    try:
        print(json.dumps(dict(phase='load_complete_own_model', owner=owner)), flush=True)
        load_plan = copy.deepcopy(plan)
        load_plan['starting_policy_inputs']['checkpoint'] = str(SNAPSHOT / 'starting-human-DPO-actor.pt')
        model, _ = fresh(load_plan, 'cpu')
        model.eval()
        assert model.counts()['actual_unique'] == 999706442
        write('actual-model-constructed.json', dict(actual_counts=model.counts(), complete_own_model=True))
        original_digest = tensor_digest(model.named_parameters())
        rng = torch.get_rng_state().clone()
        provenance = dict(snapshot_receipt_sha256=sha(SNAPSHOT / 'complete.json'),
            starting_actor_sha256=sha(SNAPSHOT / 'starting-human-DPO-actor.pt'), tokenizer_sha256=sha(SNAPSHOT / 'tokenizer.json'))
        print(json.dumps(dict(phase='export_actual_complete_text_decoder_F32')), flush=True)
        f32 = export_text_component(model, SNAPSHOT / 'tokenizer.json', OUT / 'own-F32.gguf', sources=provenance)
        checked()
        print(json.dumps(dict(phase='derive_Q8_matrices_and_preserve_F32_norms')), flush=True)
        q8 = quantize_text_component(OUT / 'own-F32.gguf', OUT / 'own-Q8.gguf')
        corpus = ExpandedExecutionCorpus(ROOT)
        positions = source_order(corpus, 963117)[:2]
        batches = [corpus.batch(p, 'cpu') for p in positions]
        assert [batch['input_ids'].shape[1] for batch in batches] == [553, 754]
        source_rows = [dict(position=p, id=corpus.row(p)['id'], prefix_positions=corpus.row(p)['prefix_positions']) for p in positions]

        def engine(path):
            return llama_cpp.Llama(model_path=str(path), n_ctx=1536, n_batch=768, n_ubatch=768,
                n_gpu_layers=0, n_threads=1, n_threads_batch=1, logits_all=True,
                type_k=llama_cpp.GGML_TYPE_F32, type_v=llama_cpp.GGML_TYPE_F32,
                flash_attn=False, offload_kqv=False, op_offload=False, verbose=False, seed=963117)

        def nll(scores, batch):
            targets = batch['labels'][0, 1:]
            values = torch.from_numpy(np.array(scores[:-1], copy=True))
            return float(F.cross_entropy(values, targets, ignore_index=-100)), int((targets != -100).sum())

        diagnostics = []
        compiled = engine(OUT / 'own-F32.gguf')
        try:
            for source, batch in zip(source_rows, batches):
                checked()
                print(json.dumps(dict(phase='whole_source_F32_operator_parity', source=source)), flush=True)
                with torch.no_grad():
                    expected = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])['logits'][0].numpy()
                compiled.reset()
                compiled.eval(batch['input_ids'][0].tolist())
                actual = compiled.scores[:len(expected)]
                max_error = float(np.abs(actual - expected).max())
                native_loss, targets = nll(expected, batch)
                compiled_loss, actual_targets = nll(actual, batch)
                assert targets == actual_targets
                assert np.allclose(actual, expected, atol=.001, rtol=.001)
                assert abs(native_loss - compiled_loss) <= .001
                diagnostics.append(dict(**source, complete_context_positions=len(expected), response_EOS_targets=targets,
                    native_F32_NLL=native_loss, compiled_F32_NLL=compiled_loss, F32_maximum_abs_logit_error=max_error))
                del expected
        finally:
            compiled.close()
        gc.collect()
        compiled = engine(OUT / 'own-Q8.gguf')
        try:
            for row, batch in zip(diagnostics, batches):
                print(json.dumps(dict(phase='whole_source_Q8_precision_observation', position=row['position'])), flush=True)
                compiled.reset()
                compiled.eval(batch['input_ids'][0].tolist())
                loss, targets = nll(compiled.scores[:batch['input_ids'].shape[1]], batch)
                assert targets == row['response_EOS_targets']
                row['compiled_Q8_NLL'] = loss
                row['Q8_NLL_absolute_drift'] = abs(loss - row['native_F32_NLL'])
            write('whole-original-source-diagnostics.json', dict(complete=True, observations=diagnostics,
                source_role='train', not_a_held_or_model_quality_assessment=True))
            checked()
            prefix = batches[0]['input_ids'][0, :source_rows[0]['prefix_positions']].tolist()
            assert len(prefix) == 433
            print(json.dumps(dict(phase='whole_native_Q8_cached_greedy', prompt_positions=len(prefix), budget=512)), flush=True)
            compiled.reset()
            begin = time.monotonic()
            compiled.eval(prefix)
            actions = []
            for position in range(512):
                token = int(compiled.scores[compiled.n_tokens - 1].argmax())
                actions.append(token)
                if token == 2:
                    break
                if position + 1 < 512:
                    compiled.eval([token])
            elapsed = time.monotonic() - begin
            generation = dict(original_complete_prompt_ids=prefix, actual_response_ids=actions,
                actual_actions=len(actions), seconds=elapsed, actions_per_second=len(actions) / elapsed,
                stopped_on_EOS=actions[-1] == 2, ended_on_budget=len(actions) == 512 and actions[-1] != 2,
                precision='GGUF Q8_0 matrices / F32 norms and KV', warm_runtime_single_fixed_train_prompt=True)
            write('native-Q8-generation.json', generation)
        finally:
            compiled.close()
        checked(True)
        corpus.guard.check(full=True)
        assert original_digest == tensor_digest(model.named_parameters()) and torch.equal(rng, torch.get_rng_state())
        gates = dict(minimum_actual_actions=len(actions) >= 256, minimum_actions_per_second=len(actions) / elapsed >= 10.,
            Q8_response_NLL_drift=max(row['Q8_NLL_absolute_drift'] for row in diagnostics) <= .05)
        write('complete.json', dict(complete=True, owner=owner, source_files_sha256=files, actual_counts=model.counts(),
            F32_component=f32, Q8_component=q8, complete_source_diagnostics=diagnostics,
            original_model_and_CPU_RNG_unchanged=True, native_Q8_generation=generation,
            optional_full_coding_assessment_gates=gates, optional_full_coding_assessment_admitted=all(gates.values()),
            two_whole_training_sources_not_independent_quality_evidence=True,
            text_component_only_not_full_multimodal_delivery=True, trained_reward_feature_basis_not_certified=True,
            no_cross_protocol_speedup_or_bit_exact_Q8_action_claim=True,
            original_training_and_assessments_not_changed=True, model_quality_gain_verified=False, goal_achieved=False))
        print(json.dumps(dict(complete=True, Q8_actions_per_second=generation['actions_per_second'], assessment_gates=gates)), flush=True)
    except BaseException as error:
        write('failure.json', dict(owner=owner, error=repr(error), traceback=traceback.format_exc(),
            partial_outputs_retained=True, original_workers_and_inputs_not_changed=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
