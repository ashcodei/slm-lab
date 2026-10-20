"""Freeze one real coding checkpoint and prove complete standalone CPU loading.

This takes no GPU, does not alter a live checkpoint, and does not turn an
interim checkpoint into a completed epoch or a quality result.
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
RUN = Path('/tmp/slm-supplemental-public-code-full-GPU-v1')
OUT = Path('/tmp/slm-supplemental-coding-policy-standalone-CPU-v1')
GIB = 1024**3


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()


def write(name, value):
    path = OUT/name
    pending = path.with_suffix(path.suffix+'.pending')
    with pending.open('x') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending,path)


def main():
    started = time.monotonic()
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('A fresh CUDA-hidden evidence destination is required')
    os.nice(10)
    os.sched_setaffinity(0, {22})
    from slm_lab.local_source_closure_v2 import python_sources
    run_plan = read(RUN/'plan.json')
    latest = read(RUN/'plain_tokens/latest-state.json')
    if latest['completed'] < 16 or latest['planned'] != 230:
        raise ValueError('An actual durable coding checkpoint with at least sixteen complete updates is required')
    state_path = Path(latest['checkpoint'])
    if sha(state_path) != latest['checkpoint_sha256']:
        raise ValueError('The selected actual committed private checkpoint must retain its exact bytes')
    memory = dict(line.split()[:2] for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
    available = int(Path('/sys/fs/cgroup/memory.max').read_text())-sum(int(memory[k]) for k in ['anon','shmem','kernel'])
    if available < 80*GIB or shutil.disk_usage(OUT.parent).free < 56*GIB:
        raise ValueError('Both full-model CPU controls and complete artifacts must fit above unchanged reserves')
    files = {str(p):sha(p) for p in python_sources(ROOT,[Path(__file__).resolve()])}
    for path in [RUN/'plan.json', ROOT/'reports/own-coding-policy-standalone-v1-tests.log',
            ROOT/'tests/test_own_coding_policy_standalone_v1.py']:
        files[str(path)] = sha(path)
    if '3 passed' not in (ROOT/'reports/own-coding-policy-standalone-v1-tests.log').read_text():
        raise ValueError('Trained cold-loading, corruption and fresh-process query checks must pass first')
    native = run_plan['native']
    for path in [Path(native['checkpoint']),Path(native['checkpoint']).parent/'tokenizer.json']:
        files[str(path)] = sha(path)
    if files[native['checkpoint']] != native['checkpoint_sha256']:
        raise ValueError('The complete original frozen native must retain its expected bytes')
    OUT.mkdir()
    write('plan.json', dict(at=datetime.now(timezone.utc).isoformat(),source_checkpoint=latest,
        run_plan_sha256=sha(RUN/'plan.json'), source_files_sha256=files, GPU_usage_permitted=False,
        CPU_affinity=[22], CPU_threads=1, own_nice=10, copied_checkpoint_is_interim=latest['completed']!=230,
        full_source_epoch_or_quality_gain_claim=False, candidate_promoted=False, goal_achieved=False))
    try:
        frozen = OUT/'original-private-state.pt'
        shutil.copyfile(state_path,frozen)
        if sha(frozen) != latest['checkpoint_sha256'] or sha(state_path) != latest['checkpoint_sha256']:
            raise ValueError('Freeze the exact complete selected state without any live-file mutation')
        import torch
        from tokenizers import Tokenizer
        from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint
        from slm_lab.own_coding_policy_standalone_v1 import export,load
        from slm_lab.own_dense_actor_v1 import OwnDenseActor
        from slm_lab.own_native_query_v1 import inspect_package,load_package
        from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        package = inspect_package(native['checkpoint'],expected_sha256=native['checkpoint_sha256'],
            expected_tokenizer_sha256=native['tokenizer_sha256'],expected_sources=native['sources'])
        with torch.random.fork_rng(devices=[]):
            base,_ = load_package(package)
            model = OwnDenseActor(base,rank=16,seed=1003311).eval()
        state = torch.load(frozen,map_location='cpu',weights_only=True,mmap=True)
        if state['completed'] != latest['completed'] or state['planned'] != 230 or state['cursor'] != {'next_global_batch':latest['completed']}:
            raise ValueError('The original complete selected source cursor must remain exact')
        bound = state['model']['sources']
        if bound['run_plan_sha256'] != sha(RUN/'plan.json') or bound['initial_native_sha256'] != native['checkpoint_sha256']:
            raise ValueError('The original own native and exact actual training plan must bind this selected policy')
        model.import_export(state['model'],expected_sources=bound)
        if model.counts()['actual_unique'] != 999706442:
            raise ValueError('The actual complete model must contain exactly999706442unique parameters')
        shutil.copyfile(Path(native['checkpoint']).parent/'tokenizer.json',OUT/'tokenizer.json')
        checkpoint = OUT/'model.pt'
        payload = export(model,sources=bound,tokenizer_sha256=native['tokenizer_sha256'],completed_updates=latest['completed'],
            provenance=dict(objective='plain_tokens',full_planned_updates=230,source_private_state_sha256=sha(frozen),
                exact_interim_checkpoint=True,complete_source_epoch=latest['completed']==230,model_quality_unproved=True))
        atomic_checkpoint(checkpoint,payload,5*GIB,40*GIB)
        del payload,state
        write('progress.json',dict(phase='complete_native_and_unfused_policy_exported',checkpoint_sha256=sha(checkpoint),completed_updates=latest['completed']))
        rng = torch.get_rng_state().clone()
        cold,metadata = load(checkpoint,expected_checkpoint_sha256=sha(checkpoint),expected_sources=bound)
        if not torch.equal(rng,torch.get_rng_state()) or tensor_digest(model.named_parameters()) != tensor_digest(cold.named_parameters()):
            raise ValueError('Every native/policy/critic value and caller RNG must reproduce exactly')
        tokenizer = Tokenizer.from_file(str(OUT/'tokenizer.json'))
        short = torch.tensor([[1,*tokenizer.encode('User: Write a Python function to add two integers.\nAssistant:').ids]])
        full = torch.tensor([[1]+[4]*4094+[2]])
        photo = torch.linspace(-1,1,3*448*448).reshape(1,3,448,448)
        words = torch.linspace(-1,1,16*3*64*448).reshape(16,3,64,448)
        outputs = {}
        with torch.no_grad():
            for name,ids,pixels in [('short_text',short,None),('full4096_synthetic_text',full,None),('photo448',short,photo)]:
                a,b = model(ids,pixel_values=pixels),cold(ids,pixel_values=pixels)
                if any(not torch.equal(a[k],b[k]) for k in ['logits','values']):
                    raise ValueError('Complete native text/photo/critic cold outputs must reproduce exactly')
                outputs[name] = dict(shapes={k:list(b[k].shape) for k in ['logits','values']},
                    tensor_sha256={k:tensor_digest([(k,b[k])]) for k in ['logits','values']})
                del a,b
                write('progress.json',dict(phase='cold_outputs',completed=list(outputs),completed_updates=latest['completed']))
            a,b = model.base.word_logits(words),cold.base.word_logits(words)
            if list(a.shape) != [16,140,257] or not torch.equal(a,b):
                raise ValueError('Every complete16word/140frame native output must reproduce exactly')
            outputs['whole_words'] = dict(shape=list(b.shape),tensor_sha256=tensor_digest([('logits',b)]))
            for name,pixels in [('text',None),('photo',photo)]:
                a,b = model.generate(short,pixels,max_new_tokens=8,eos_token_id=2),cold.generate(short,pixels,max_new_tokens=8,eos_token_id=2)
                if not torch.equal(a,b): raise ValueError('Actual bounded cached greedy actions must reproduce exactly')
                outputs[name+'_actions'] = b[0,short.shape[1]:].tolist()
        if torch.cuda.is_initialized() or not torch.equal(rng,torch.get_rng_state()) or any(sha(p)!=h for p,h in files.items()):
            raise ValueError('No CUDA use, caller RNG change or original/runtime source change is allowed')
        write('complete.json',dict(complete=True,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
            tokenizer_sha256=native['tokenizer_sha256'],sources=bound,completed_updates=latest['completed'],planned_updates=230,
            actual_registered_parameters=metadata['actual_counts']['actual_unique'],all_actual_source_and_cold_values_exact=True,
            actual_full4096_FP32_text448_photo16_whole_words_and_cached_actions_exact=True,outputs=outputs,
            full_context_probe_is_synthetic_not_quality=True,source_checkpoint_is_interim=latest['completed']!=230,
            source_checkpoint_untouched=True,caller_CPU_RNG_unchanged=True,GPU_initialized=False,
            no_ancestor_needed_for_inference=True,unfused_operators_preserved=True,
            CUDA_BF16_cold_parity_and_complete_matched_quality_still_required=True,
            full_epoch_or_quality_improvement_proved=False,elapsed_seconds=time.monotonic()-started,
            candidate_promoted=False,goal_achieved=False))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),traceback=traceback.format_exc(),partial_outputs_retained=True,goal_achieved=False))
        raise


if __name__ == '__main__': main()
