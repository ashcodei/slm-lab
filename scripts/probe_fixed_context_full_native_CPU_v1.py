"""Full984M native CPU cold/inventory proof for plain positional extrapolation.

Uses original-context probes, including full448px photo and16whole word crops.
It does not allocate a full-size16K forward/backward, train, or admit GPUs.
The source is the completed own pre-PPO dense native, not an unfused policy.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/tmp/slm-completed-joint-initializer-CPU-v3')
OUT = Path('/tmp/slm-native-fixed-context-full-CPU-v1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or 'torch' in sys.modules or any(os.environ.get(k) != '1' for k in
                ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'))):
        raise ValueError('Fresh CUDA-hidden nice19 single-thread owned CPU proof required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    memory = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    available = int(Path('/sys/fs/cgroup/memory.max').read_text()) - sum(int(memory[k]) for k in ('anon', 'shmem', 'kernel'))
    if available < 80 * 1024**3 or shutil.disk_usage(OUT.parent).free < 56 * 1024**3:
        raise ValueError('32GiB own/48GiB other-job memory and16GiB own/40GiB scratch budgets required')
    receipt = json.loads((SOURCE / 'complete.json').read_text())
    if (receipt.get('complete') is not True or receipt.get('actual_full_size_CPU_initializer') is not True
            or receipt['actual_counts']['total_unique'] != 984399690
            or receipt.get('all_native_values_and_actual_text_photo_word_cold_outputs_exact') is not True):
        raise ValueError('Only the exact completed own984Mnative initializer may supply weights')
    path = Path(receipt['checkpoint']); token_path = path.parent / 'tokenizer.json'
    if sha(path) != receipt['checkpoint_sha256'] or sha(token_path) != receipt['tokenizer_sha256']:
        raise ValueError('Both complete original native and packaged tokenizer bytes must match')
    from slm_lab.local_source_closure_v2 import python_sources
    paths = python_sources(ROOT, [Path(__file__).resolve()])
    tests = ROOT / 'reports/fixed-text-context-v1-tests.log'
    assert '11 passed' in tests.read_text()
    files = {str(p): sha(p) for p in [*paths, SOURCE / 'complete.json', path, token_path,
        tests, ROOT / 'tests/test_fixed_text_context_v1.py']}
    OUT.mkdir(); started = time.monotonic()
    owner = dict(pid=os.getpid(), birth=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19])
    write('owner-plan.json', dict(owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
        exact_source_checkpoint_sha256=receipt['checkpoint_sha256'], scope='original-context full-native CPU inventory and cold proof',
        extended_profile='extrapolation4096to16384', full_size16Kcapacity_not_exercised=True,
        GPU_contexts=0, optimizer_updates=0, original_workers_unchanged=True, goal_achieved=False))
    try:
        import torch
        from tokenizers import Tokenizer
        from slm_lab.fixed_context_native_checkpoint_v1 import export, load
        from slm_lab.fixed_text_context_v1 import Profile, install
        from slm_lab.own_native_query_v1 import inspect_package, load_package
        from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        package = inspect_package(path, expected_sha256=receipt['checkpoint_sha256'],
            expected_tokenizer_sha256=receipt['tokenizer_sha256'], expected_sources=receipt['sources'])
        rng = torch.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]): model, _ = load_package(package)
        assert torch.equal(rng, torch.get_rng_state()) and model.counts()['total_unique'] == 984399690
        tokenizer = Tokenizer.from_file(str(token_path))
        text = 'User: Write a Python function that returns the square of its integer argument.\nAssistant:'
        ids = torch.tensor([[1, *tokenizer.encode(text, add_special_tokens=False).ids]], dtype=torch.long)
        photo = torch.linspace(-1, 1, 3*448*448).reshape(1, 3, 448, 448)
        words = torch.linspace(-1, 1, 16*3*64*448).reshape(16, 3, 64, 448)
        torch.save(dict(ids=ids, full_original_prompt=text, photo=photo, words=words), OUT / 'inputs.pt')
        weights = tensor_digest(model.named_parameters()); signature = [(id(p), p.data_ptr(), p._version, p.requires_grad) for p in model.parameters()]

        def probe(actual):
            with torch.no_grad():
                plain = actual(ids); image = actual(ids, photo)
                word_logits = actual.word_logits(words)
                reward = actual.preference_score(ids)
                actions = actual.generate(ids, max_new_tokens=8, eos_token_id=2, use_cache=True)
            assert word_logits.shape == (16, 140, 257)
            return dict(text_logits=plain['logits'], text_critic=plain['values'], photo_logits=image['logits'],
                photo_critic=image['values'], full16word_logits=word_logits, original_context_reward=reward,
                cached_greedy_tokens=actions)

        before = probe(model); torch.save(before, OUT / 'outputs-before.pt')
        write('progress.json', dict(phase='original984Mfull_photo_and16word_probes_complete', elapsed=time.monotonic()-started))
        install(model, Profile('extrapolation', 4096, 16384))
        after = probe(model); torch.save(after, OUT / 'outputs-extended-original-context.pt')
        assert all(torch.equal(before[k], after[k]) for k in before)
        assert weights == tensor_digest(model.named_parameters())
        assert signature == [(id(p), p.data_ptr(), p._version, p.requires_grad) for p in model.parameters()]
        sources = {**receipt['sources'], 'context_source_checkpoint_sha256': receipt['checkpoint_sha256']}
        shutil.copyfile(token_path, OUT / 'tokenizer.json')
        payload = export(model, sources=sources, tokenizer_sha256=receipt['tokenizer_sha256'], completed_updates=0)
        destination = OUT / 'model.pt'; torch.save(payload, destination)
        assert destination.stat().st_size <= 5*1024**3
        del payload
        write('progress-export.json', dict(phase='full_native_profile_export_complete', elapsed=time.monotonic()-started))
        cold, metadata = load(destination, expected_checkpoint_sha256=sha(destination), expected_sources=sources)
        assert weights == tensor_digest(cold.named_parameters())
        assert not ({p.data_ptr() for p in cold.parameters()} & {p.data_ptr() for p in model.parameters()})
        reloaded = probe(cold); torch.save(reloaded, OUT / 'outputs-cold.pt')
        assert all(torch.equal(before[k], reloaded[k]) for k in before)
        assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
        assert all(sha(p) == h for p, h in files.items())
        outputs = {str(p): sha(p) for p in OUT.iterdir() if p.is_file()}
        write('complete.json', dict(complete=True, owner=owner, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            output_files_sha256=outputs, actual_registered_parameters=984399690, actual_context_limit=16384,
            actual_original_text_probe_tokens=ids.shape[1], actual_photo_shape=list(photo.shape), actual_word_shape=list(words.shape),
            actual_word_output_shape=[16, 140, 257], all_original_native_values_unchanged=True,
            original_parameter_identity_trainability_versions_unchanged=True,
            all_original_context_text_photo_word_critic_reward_and_cached_greedy_outputs_exact=True,
            complete_native_cold_independent_storage_and_all_values_exact=True, CPU_rng_unchanged=True,
            metadata=metadata, source_pre_PPO_dense_native_not_unfused_policy_reward=True,
            full_size_original4096_or_extended16384_text_capacity_not_exercised=True,
            PI_YaRN_full_model_neural_outputs_not_exercised=True, GPU_contexts=0, optimizer_updates=0,
            full_source_training_admission_or_model_quality_gain=False, elapsed_seconds=time.monotonic()-started, goal_achieved=False))
        print(json.dumps(dict(complete=True, actual_registered_parameters=984399690,
            all_original_context_outputs_and_cold_values_exact=True, full_size16Kcapacity_proved=False)), flush=True)
    except BaseException as error:
        write('failure.json', dict(error=repr(error), traceback=traceback.format_exc(), partial_outputs_retained=True,
            original_workers_unchanged=True, optimizer_updates=0, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
