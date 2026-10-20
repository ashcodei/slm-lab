"""Actual small PPO-origin/SFT policy-profile packages in fresh cold processes.

All three profiles retain one shared original reward encoder, nonzero learned
critic and every modality. This is explicit small synthetic CPU mechanics,
not the full999Mmodel, human-alignment quality or GPU capacity.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fixed-context-policy-reward-cold-CPU-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def outputs(model, data):
    import torch
    model.eval()
    with torch.no_grad():
        short = model(data['short']); long = model(data['long']); photo = model(data['short'], data['photo'])
        return dict(short_logits=short['logits'], short_critic=short['values'], long_logits=long['logits'],
            long_critic=long['values'], photo_logits=photo['logits'], photo_critic=photo['values'],
            words=model.word_logits(data['words']), original_reward=model.preference_score(data['short']),
            greedy_across_original_boundary=model.generate(data['long'][:,:model.profile.original_text_length-2],
                max_new_tokens=8, use_cache=True))


def cold(folder):
    import torch
    from slm_lab.fixed_context_policy_reward_v1 import load
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    spec = json.loads((folder / 'cold-request.json').read_text()); rng = torch.get_rng_state().clone()
    model, metadata = load(folder / 'model.pt', expected_checkpoint_sha256=spec['checkpoint_sha256'],
        expected_sources=spec['sources'], allow_mechanics=True)
    data = torch.load(folder / 'inputs.pt', map_location='cpu', weights_only=True)
    observed = outputs(model, data); torch.save(observed, folder / 'cold-outputs.pt')
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert not any(name.startswith(('test_', 'pytest')) for name in sys.modules)
    result = dict(complete=True, metadata=metadata, actual_registered_values=sum(p.numel() for p in model.parameters()),
        actor_parameters_sha256=tensor_digest(model.actor.named_parameters()),
        reward_parameters_sha256=tensor_digest(model.reward.named_parameters()),
        policy_and_reward_share_one_base=model.base is model.reward.base,
        nonzero_inherited_critic=bool(model.actor.parts['critic'].weight.count_nonzero()),
        caller_CPU_RNG_unchanged=True, ancestor_or_test_fixture_modules_loaded=False,
        additional_reward_encoder_values=0, mechanics_only=True, full_size_GPU_capacity_or_quality_proved=False)
    (folder / 'cold-complete.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


def parent():
    if OUT.exists() or os.environ.get('CUDA_VISIBLE_DEVICES') != '' or 'torch' in sys.modules:
        raise ValueError('Fresh CUDA-hidden source-owned CPU mechanics proof required')
    from slm_lab.local_source_closure_v2 import python_sources
    paths = python_sources(ROOT, [Path(__file__).resolve()])
    tests = ROOT / 'reports/fixed-context-policy-reward-v1-tests.log'; assert '6 passed' in tests.read_text()
    sources = {str(p): sha(p) for p in [*paths, tests, ROOT / 'tests/test_fixed_context_policy_reward_v1.py']}
    OUT.mkdir(); (OUT / 'plan.json').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),
        source_files_sha256=sources, actual_small_synthetic_PPO_controls=True,
        full_size_model_or_human_alignment_or_GPU_admission=False), indent=2) + '\n')
    import torch
    from slm_lab.fixed_context_policy_reward_v1 import FixedContextPolicyReward, export
    from slm_lab.fixed_text_context_v1 import Profile
    from slm_lab.own_policy_reward_standalone_v1 import load as original_load
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_own_policy_reward_standalone_v1 import package
    torch.set_num_threads(1); records = []
    for mode in ['extrapolation','linear_PI','YaRN']:
        folder = OUT / mode; folder.mkdir(); _,bound,initial = package(folder / 'source-control')
        original,_ = original_load(initial, expected_checkpoint_sha256=sha(initial), expected_sources=bound, allow_mechanics=True)
        short = torch.tensor([[1,4,7,2]]); before = original.preference_score(short).clone()
        native = tensor_digest(original.base.named_parameters()); critic = original.actor.parts['critic'].weight.detach().clone()
        old = original.config.max_text_length
        model = FixedContextPolicyReward(original.actor, original.reward, Profile(mode,old,old*4,original.config.rope_theta), allow_mechanics=True)
        long = (torch.arange(old+8) % (model.config.vocab_size-4) + 4)[None]; long[0,-1] = 2
        labels = long.clone(); labels[:,:4] = -100
        optimizer = torch.optim.AdamW([p for _,p in model.actor.named_selected(('actor',))],lr=.001,weight_decay=0.)
        model.train(); model(long,labels=labels)['loss'].backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True); model.eval()
        assert tensor_digest(model.base.named_parameters()) == native
        assert torch.equal(critic,model.actor.parts['critic'].weight) and bool(critic.count_nonzero())
        assert torch.equal(before,model.preference_score(short))
        portable = folder / 'portable'; portable.mkdir()
        (portable / 'tokenizer.json').write_bytes((initial.parent / 'tokenizer.json').read_bytes())
        data = dict(short=short,long=long,
            photo=torch.linspace(-1,1,3*model.config.max_image_size**2).reshape(1,3,model.config.max_image_size,model.config.max_image_size),
            words=torch.linspace(-1,1,3*8*32).reshape(1,3,8,32))
        expected = outputs(model,data); torch.save(expected,folder / 'expected-outputs.pt'); torch.save(data,portable / 'inputs.pt')
        raw = export(model,sources=bound,provenance=dict(small_source_PPO_updates=3,new_long_SFT_updates=1),
            completed_updates=1,tokenizer_sha256=bound['tokenizer_sha256'])
        torch.save(raw,portable / 'model.pt')
        request = dict(sources=bound,checkpoint_sha256=sha(portable / 'model.pt'))
        (portable / 'cold-request.json').write_text(json.dumps(request,indent=2) + '\n')
        with (folder / 'cold.log').open('x') as log:
            child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--cold',str(portable)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
            if child.wait() != 0: raise ValueError('Preserve a failed exact cold process without automatic retry')
        observation = torch.load(portable / 'cold-outputs.pt',map_location='cpu',weights_only=True)
        proof = json.loads((portable / 'cold-complete.json').read_text())
        assert set(observation) == set(expected) and all(torch.equal(v,observation[k]) for k,v in expected.items())
        assert proof['actor_parameters_sha256'] == tensor_digest(model.actor.named_parameters())
        assert proof['reward_parameters_sha256'] == tensor_digest(model.reward.named_parameters())
        records.append(dict(profile=mode,actual_counts=model.counts(),all_fresh_cold_values_and_modal_outputs_exact=True,
            original_frozen_reward_and_inherited_nonzero_critic_preserved_after_long_SFT=True,
            actual_long_SFT_input_tokens=long.shape[1],cold_process_proof_sha256=sha(portable / 'cold-complete.json')))
        print(json.dumps(records[-1]),flush=True)
    assert all(sha(p) == h for p,h in sources.items())
    result = dict(complete=True,at=datetime.now(timezone.utc).isoformat(),source_files_sha256=sources,
        profiles=records,models_are_small_synthetic_CPU_controls=True,
        full_size999Mmodel_GPU_capacity_or_human_alignment_quality_proved=False,
        new_full_size_model_updates=0,goal_achieved=False)
    (OUT / 'complete.json').write_text(json.dumps(result,indent=2,allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run',action='store_true'); parser.add_argument('--cold',type=Path)
    args = parser.parse_args()
    if args.cold: cold(args.cold)
    elif args.run: parent()
    else: parser.error('Use explicit --run or a fresh controlled --cold package')
