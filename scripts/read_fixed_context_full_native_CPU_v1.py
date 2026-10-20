"""Read every source/exported native tensor and saved full-probe output.

Waits on the exact CPU probe owner without Torch. After successful terminal
evidence, independently checks weights/geometry/ties and all saved output
equalities. It does not replay neural forwards or establish16Kcapacity.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/tmp/slm-native-fixed-context-full-CPU-v1')
OUT = ROOT / 'reports/fixed-context-full-native-readback-CPU-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def live(owner):
    try: fields = Path(f'/proc/{owner["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError: return False
    return fields[19] == owner['birth'] and fields[0] not in ('Z', 'X')


def write(name, value):
    with (OUT / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    if (OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.nice(0) != 19
            or 'torch' in sys.modules):
        raise ValueError('Fresh CUDA-hidden nice19 source-only observer required')
    os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    original_plan = SOURCE / 'owner-plan.json'; plan = json.loads(original_plan.read_text())
    predecessor = plan['owner']; plan_sha = sha(original_plan)
    OUT.mkdir(); write('owner-plan.json', dict(at=datetime.now(timezone.utc).isoformat(),
        exact_predecessor=predecessor, exact_predecessor_plan_sha256=plan_sha,
        source_sha256=sha(__file__), model_updates=0, neural_forward_replay=False, GPU_contexts=0))
    try:
        while live(predecessor):
            if sha(original_plan) != plan_sha: raise ValueError('The exact original probe owner/plan changed')
            print(json.dumps(dict(waiting_for_exact_original_CPU_probe=predecessor)), flush=True)
            time.sleep(15)
        if (SOURCE / 'failure.json').exists() or not (SOURCE / 'complete.json').is_file():
            raise ValueError('The exact original CPU probe ended without successful complete evidence')
        done = json.loads((SOURCE / 'complete.json').read_text())
        if (done.get('complete') is not True or done.get('owner') != predecessor
                or done.get('actual_registered_parameters') != 984399690
                or done.get('full_size_original4096_or_extended16384_text_capacity_not_exercised') is not True):
            raise ValueError('Complete actual984Moriginal-context native proof with unchanged scope required')
        files = {**done['source_files_sha256'], **done['output_files_sha256']}
        files[str(SOURCE / 'complete.json')] = sha(SOURCE / 'complete.json')
        files[str(Path(__file__).resolve())] = sha(__file__)
        if any(sha(p) != h for p, h in files.items()): raise ValueError('Every closed original source/output byte must remain exact')
        import torch
        from tokenizers import Tokenizer
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        receipt = json.loads(Path('/tmp/slm-completed-joint-initializer-CPU-v3/complete.json').read_text())
        original = torch.load(receipt['checkpoint'], map_location='cpu', weights_only=True, mmap=True)['complete_dense']
        package = torch.load(SOURCE / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
        assert package['format'] == 'own-complete-native-fixed-text-context-v1'
        assert package['text_context_protocol'] == 'own-fixed-text-context-rope-profiles-v1'
        profile = package['fixed_text_context_profile']
        assert profile == dict(mode='extrapolation',original_text_length=4096,maximum_text_length=16384,
            rope_theta=10000.,alpha=1.,beta=32.)
        fingerprint = hashlib.sha256(json.dumps(dict(protocol=package['text_context_protocol'], **profile),
            sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        assert fingerprint == package['profile_fingerprint']
        saved = package['complete_original_context_native']
        assert original['config'] == saved['config'] and saved['config']['max_text_length'] == 4096
        assert original['settings'] == saved['settings'] and original['actual_counts'] == saved['actual_counts'] == package['actual_counts']
        assert saved['sources'] == {**receipt['sources'], 'context_source_checkpoint_sha256': receipt['checkpoint_sha256']}
        values = 0; tensors = 0
        for family in ['network', 'heads']:
            assert original[family].keys() == saved[family].keys()
            for name, tensor in original[family].items():
                result = saved[family][name]
                assert tensor.shape == result.shape and tensor.dtype == result.dtype == torch.float32
                assert torch.equal(tensor, result) and bool(torch.isfinite(result).all())
                tensors += 1
                if family != 'network' or name != 'lm_head.weight': values += result.numel()
        assert values == 984399690
        embedding = saved['network']['embedding.weight']; head = saved['network']['lm_head.weight']
        assert embedding.data_ptr() == head.data_ptr() and embedding.shape == head.shape
        inputs = torch.load(SOURCE / 'inputs.pt', map_location='cpu', weights_only=True)
        assert list(inputs['photo'].shape) == [1,3,448,448] and list(inputs['words'].shape) == [16,3,64,448]
        tokenizer = Tokenizer.from_file(str(SOURCE / 'tokenizer.json'))
        assert inputs['ids'].tolist() == [[1, *tokenizer.encode(inputs['full_original_prompt'], add_special_tokens=False).ids]]
        expected_keys = {'text_logits','text_critic','photo_logits','photo_critic','full16word_logits',
            'original_context_reward','cached_greedy_tokens'}
        outputs = [torch.load(SOURCE / name, map_location='cpu', weights_only=True) for name in
            ['outputs-before.pt','outputs-extended-original-context.pt','outputs-cold.pt']]
        for row in outputs:
            assert set(row) == expected_keys and row['full16word_logits'].shape == (16,140,257)
            assert all(torch.equal(value, row[key]) for key,value in outputs[0].items())
            assert all(bool(torch.isfinite(value).all()) for value in row.values())
            tokens = row['cached_greedy_tokens']; length = inputs['ids'].shape[1]
            assert torch.equal(tokens[:,:length], inputs['ids']) and 1 <= tokens.shape[1]-length <= 8
        assert not torch.cuda.is_initialized() and all(sha(p) == h for p,h in files.items())
        write('complete.json', dict(complete=True, at=datetime.now(timezone.utc).isoformat(), source_files_sha256=files,
            every_original_native_tensor_checked=tensors, every_unique_native_value_checked=values,
            tied_saved_vocabulary_storage_verified=True, original_geometry_and_profile_fingerprint_reconstructed=True,
            all_saved_text_photo_full16word_critic_reward_and_cached_action_outputs_exact=True,
            actual_word_logits_shape=[16,140,257], original_prompt_BPE_reconstructed=True,
            new_model_or_neural_forward_or_optimizer_updates=0, full_size16Kcapacity_or_model_quality_proved=False,
            source_pre_PPO_dense_native_not_unfused_policy_reward=True, GPU_contexts=0, goal_achieved=False))
        print(json.dumps(dict(complete=True, native_values_checked=values, tensors_checked=tensors)), flush=True)
    except BaseException as error:
        write('failure.json', dict(error=repr(error), traceback=traceback.format_exc(), partial_outputs_retained=True,
            no_automatic_retry=True, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
