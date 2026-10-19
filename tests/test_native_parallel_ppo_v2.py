"""Four actual CPU processes exercise real collectives and private recovery."""
import argparse
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
from torch import distributed as dist

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.dense_ppo_v1 import action_outputs
from slm_lab.native_parallel_ppo_checkpoint_v2 import make_state, restore_state, finish
from slm_lab.native_parallel_ppo_capacity_v2 import stress
from slm_lab.native_parallel_ppo_export_v2 import materialize
from slm_lab.native_parallel_ppo_v2 import Collective, ParallelNativePPO
from slm_lab.native_shared_ppo_v1 import NativeSharedPPOStage
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from test_native_shared_ppo_v1 import stage as old_stage, BUDGETS
from test_own_aligned_assistant_v1 import original_signature
from test_own_alignment_checkpoint_v1 import assert_nested_equal


def kernel(root):
    torch.manual_seed(999671); original = old_stage(root, start=False)
    old = original.data.rows; sequence = original.data.sequence
    original.data.rows = [copy.deepcopy(old[i % len(old)]) for i in range(17)]
    original.data.sequence = lambda position, field: sequence(position % len(old), field)
    for i, row in enumerate(original.data.rows): row['id'] += ':distinct-CPU-source-'+str(i)
    options = {**original.options, 'steps': 9}
    return NativeSharedPPOStage(original.model, original.optimizer, original.data, original.reward,
        options, original.sources, allow_mechanics=True)


def worker(root, rank):
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method=(root/'group').as_uri(), rank=rank, world_size=4,
        timeout=timedelta(seconds=90))
    try:
        local = kernel(root); actual = ParallelNativePPO(local, Collective('cpu', control_group=dist.group.WORLD))
        directory = root/f'rank-{rank}'; directory.mkdir()
        capacity = stress(actual)
        assert capacity['all_parameters_Adam_and_RNG_unchanged'] and not capacity['full_size_original_L40_horizon_capacity_passed']
        assert all(r['actual_fixed_nonEOS_actions'] == 8 for r in capacity['maximum_two_whole_original_prompts'])
        actual.start_history(directory/'first', BUDGETS)
        torch.manual_seed(999771+rank); fixed = original_signature(local.model.base)
        first = actual.update(); saved = make_state(actual); pristine = copy.deepcopy(saved)
        torch.save(saved, directory/'step1.pt'); dist.barrier()
        expected = actual.update(); weights = local.model.export(local.sources)
        adam = copy.deepcopy(local.optimizer.state_dict()); rng = value_digest(capture_rng('cpu'))
        before = original_signature(local.model)
        broken = copy.deepcopy(saved)
        if rank == 2: broken['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))
        with pytest.raises(ValueError, match='Coordinated'):
            restore_state(actual, broken, expected_common_sha256=saved['common_sha256'],
                expected_private_rng_sha256=saved['private_RNG_sha256'], expected_completed=1)
        assert before == original_signature(local.model)
        assert_nested_equal(weights, local.model.export(local.sources)); assert_nested_equal(adam, local.optimizer.state_dict())
        assert value_digest(capture_rng('cpu')) == rng
        wrong = copy.deepcopy(saved)
        if rank == 1:
            foreign = torch.load(root/'rank-0'/'step1.pt', map_location='cpu', weights_only=True)
            wrong['private_RNG'] = foreign['private_RNG']; wrong['private_RNG_sha256'] = foreign['private_RNG_sha256']
        with pytest.raises(ValueError, match='Coordinated'):
            restore_state(actual, wrong, expected_common_sha256=saved['common_sha256'],
                expected_private_rng_sha256=saved['private_RNG_sha256'], expected_completed=1)
        assert before == original_signature(local.model); assert value_digest(capture_rng('cpu')) == rng
        restored = restore_state(actual, saved, expected_common_sha256=saved['common_sha256'],
            expected_private_rng_sha256=saved['private_RNG_sha256'], expected_completed=1)
        assert restored['complete_updates'] == 1 and restored['original_prompt_draws'] == 8
        actual.start_history(directory/'replayed', BUDGETS)
        assert actual.update() == expected
        assert_nested_equal(weights, local.model.export(local.sources)); assert_nested_equal(adam, local.optimizer.state_dict())
        assert value_digest(capture_rng('cpu')) == rng; assert_nested_equal(saved, pristine)
        third = actual.update(); completed = finish(actual); final_state = make_state(actual)
        torch.save(final_state,directory/'final.pt')
        assert fixed == original_signature(local.model.base)
        assert completed['complete'] and completed['actual_updates'] == 3 and not completed['full_original_four_rank_PPO_recipe_complete']
        (directory/'complete.json').write_text(json.dumps(dict(complete=completed,
            every_actual_peer_rejected_bad_rank2_Adam_before_mutation=True,
            every_actual_peer_rejected_foreign_rank1_generator_before_mutation=True,
            exact_own_actions_common_Adam_and_private_RNG_replayed=True, shared_native_values_unchanged=True,
            final_actual_local_prompt_count=len(actual.positions(2, rank)),
            final_state_artifact=dict(path=str(directory/'final.pt'),common_sha256=final_state['common_sha256'],
                private_RNG_sha256=final_state['private_RNG_sha256']),
            actual_full_horizon_small_CPU_capacity=capacity,
            no_GPU_admission=True), indent=2)+'\n')
        if rank == 0:
            (root/'all-records.json').write_text(json.dumps([first, expected, third]))
            torch.save(dict(first=saved['policy'], second=weights,
                final=local.model.export(local.sources), final_Adam=local.optimizer.state_dict()), root/'actual-values.pt')
    finally: dist.destroy_process_group()


@pytest.fixture(scope='module')
def actual_four(tmp_path_factory):
    root = tmp_path_factory.mktemp('native-parallel-PPO-v2'); children = []; streams = []
    try:
        for rank in range(4):
            stream = (root/f'rank-{rank}.log').open('wb'); streams.append(stream)
            children.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--worker', str(rank), '--root', str(root)], stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}))
        deadline = time.monotonic()+150
        while any(p.poll() is None for p in children):
            if time.monotonic() > deadline or any(p.poll() not in (None, 0) for p in children):
                pytest.fail('Actual four-process PPO failed: '+str(root)+'\n'+'\n'.join(p.read_text()[-5000:] for p in sorted(root.glob('rank-*.log'))))
            time.sleep(.1)
        assert all(p.returncode == 0 for p in children)
        yield root
    finally:
        for child in children:
            if child.poll() is None: child.terminate()
        for child in children:
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=5)
        for stream in streams: stream.close()


def test_actual_four_process_global_whitening_and_genuine_empty_tail(actual_four):
    rows = json.loads((actual_four/'all-records.json').read_text())
    assert [len(row['source_positions']) for row in rows] == [8, 8, 1]
    assert sorted(p for row in rows for p in row['source_positions']) == list(range(17))
    assert rows[-1]['rank_source_positions'][1:] == [[], [], []]
    assert rows[-1]['rank_action_tokens'][1:] == [0, 0, 0]
    for row in rows:
        raw = torch.tensor([x for r in row['rollouts'] for x in r['raw_advantages']])
        wanted = (raw-raw.mean())/(raw.std(unbiased=False)+1e-8)
        got = torch.tensor([x for r in row['rollouts'] for x in r['advantages']])
        assert torch.equal(wanted, got)
        assert row['gradients_summed_after_global_token_weighting']
    for rank in range(4):
        v = json.loads((actual_four/f'rank-{rank}'/'complete.json').read_text())
        assert v['final_actual_local_prompt_count'] == (1 if rank == 0 else 0)
        assert v['every_actual_peer_rejected_bad_rank2_Adam_before_mutation']
        assert v['every_actual_peer_rejected_foreign_rank1_generator_before_mutation']
        assert v['exact_own_actions_common_Adam_and_private_RNG_replayed']
        assert v['shared_native_values_unchanged'] and v['no_GPU_admission']


def test_actual_all_four_states_agree_on_common_values_and_keep_distinct_private_generators(actual_four):
    saved = [torch.load(actual_four/f'rank-{r}'/'step1.pt', map_location='cpu', weights_only=True) for r in range(4)]
    assert len({s['common_sha256'] for s in saved}) == 1
    assert len({s['private_RNG_sha256'] for s in saved}) == 4
    assert all(s['no_resident_trajectory_history'] and len(str(s['history'])) < 2000 for s in saved)
    for value in saved[1:]:
        assert_nested_equal(saved[0]['policy'], value['policy']); assert_nested_equal(saved[0]['optimizer'], value['optimizer'])


def test_actual_global_actor_critic_Adam_matches_independent_unsharded_joint_oracle(actual_four):
    local = kernel(actual_four); model, options = local.model, local.options
    rows = json.loads((actual_four/'all-records.json').read_text())
    actual = torch.load(actual_four/'actual-values.pt', map_location='cpu', weights_only=True)
    for step, record in enumerate(rows):
        logps, values = [], []
        for row in record['rollouts']:
            ids = torch.tensor([row['prompt_ids']+row['response_ids']])
            lp, v = action_outputs(model, ids, len(row['prompt_ids']), local.amp)
            logps.append(lp); values.append(v)
        logps, values = torch.cat(logps), torch.cat(values)
        old, advantage, previous_value, returns = [torch.tensor([v for row in record['rollouts'] for v in row[k]])
            for k in ['old_logps', 'advantages', 'old_values', 'returns']]
        ratio = torch.exp(logps-old); clipped_value = previous_value+(values-previous_value).clamp(-.2, .2)
        loss = -torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage).mean()
        loss += .5*torch.maximum((values-returns)**2, (clipped_value-returns)**2).mean()/2
        loss.backward(inputs=local.parameters); torch.nn.utils.clip_grad_norm_(local.parameters, 1.)
        local.optimizer.step(); local.optimizer.zero_grad(set_to_none=True)
        wanted = actual[['first', 'second', 'final'][step]]['parts']
        for name, p in model.parts.state_dict().items():
            assert torch.allclose(p, wanted[name], atol=4e-7, rtol=5e-5), (step, name, float((p-wanted[name]).abs().max()))
    for a, b in zip(local.optimizer.state_dict()['state'].values(), actual['final_Adam']['state'].values()):
        assert torch.allclose(a['exp_avg'], b['exp_avg'], atol=4e-7, rtol=5e-5)
        assert torch.allclose(a['exp_avg_sq'], b['exp_avg_sq'], atol=4e-7, rtol=5e-5)


def test_independent_CPU_reconstruction_reads_all_four_full_histories_before_copy_and_preserves_RNG(actual_four):
    from slm_lab.corpus import file_hash
    from slm_lab.own_dense_actor_fold_v1 import fold_actor
    from slm_lab.own_dense_standalone_v1 import export, load
    artifacts=[]
    for rank in range(4):
        row=json.loads((actual_four/f'rank-{rank}'/'complete.json').read_text())['final_state_artifact']
        artifacts.append({**row,'sha256':file_hash(row['path'])})
    local=kernel(actual_four);rng=value_digest(capture_rng('cpu'));fixed=original_signature(local.model.base)
    proof=materialize(local,artifacts,expected_completed=3)
    assert proof['every_actual_private_state_and_original_trajectory_checked'] and proof['CPU_RNG_unchanged']
    assert [r['original_prompt_draws'] for r in proof['actual_complete_prefixes']]==[17]*4
    assert value_digest(capture_rng('cpu'))==rng and fixed==original_signature(local.model.base)
    original=torch.load(actual_four/'actual-values.pt',map_location='cpu',weights_only=True)['final']
    assert_nested_equal(local.model.export(local.sources),original)
    folded,_=fold_actor(local.model.eval(),sources=local.sources,working_budget_bytes=512*1024**2)
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    vocab={name:i for i,name in enumerate(['<pad>','<bos>','<eos>']+[f'control-{i}' for i in range(folded.config.vocab_size-3)])}
    token_path=actual_four/'tokenizer.json';Tokenizer(WordLevel(vocab=vocab,unk_token='<pad>')).save(str(token_path))
    path=actual_four/'standalone.pt'
    torch.save(export(folded,sources=local.sources,provenance=dict(actual_four_small_PPO_updates=3),
        completed_updates=3,tokenizer_sha256=file_hash(token_path)),path)
    cold,_=load(path,expected_checkpoint_sha256=file_hash(path),expected_sources=local.sources)
    ids=torch.tensor([[1,4,7,2]])
    with torch.no_grad():
        assert torch.equal(folded(ids)['logits'],cold(ids)['logits'])
        assert torch.equal(folded(ids)['values'],cold(ids)['values'])
    bad=copy.deepcopy(artifacts);bad[2]['sha256']='0'*64
    fresh=kernel(actual_four);before=original_signature(fresh.model)
    with pytest.raises(ValueError,match='checkpoint byte'):materialize(fresh,bad,expected_completed=3)
    assert before==original_signature(fresh.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--worker', type=int, required=True); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args(); worker(args.root, args.worker)
