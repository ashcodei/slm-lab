"""Actual own-small optimizer/replica/recovery evidence; no corpus admission."""
import copy
from datetime import timedelta
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.complete_response_stopping_loss_v1 import PROFILES
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import new_backbone, OwnDenseAssistant
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.parallel_function_repair_stage_v1 import manifest, value_digest
from slm_lab.parallel_public_reasoned_code_SFT_stage_v1 import ParallelPublicReasonedCodeSFTStage
from slm_lab import parallel_repair_durable_state_v1 as durable
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_public_reasoned_code_train_data_v1 import packet

OPTIONS = dict(learning_rate=.001, gradient_clip=1., maximum_microbatch_examples=2,
    maximum_padded_tokens=4096, gradient_bucket_values=4096, epochs=1, width=4, seed=990231)
SOURCES = dict(own_small_initializer_sha256='a'*64, controlled_loader_fixture_sha256='b'*64)
ENTRIES = [[0,1,2],[1,2],[2]]


def own_model():
    torch.manual_seed(991104)
    config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=32, vision_hidden_size=16,
        vision_intermediate_size=32, vision_layers=1, vision_heads=4, image_size=16, patch_size=8, pool_size=2)
    base = OwnDenseAssistant(new_backbone(config, 991001), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4)
    return OwnDenseActor(base, rank=2)


def setup(root, profile='plain_tokens', *, rank=0, world=1, control=None, options=None):
    data = PublicReasonedCodeSFTData(root)
    model = own_model()
    adam = torch.optim.AdamW([p for _,p in model.named_selected(('actor',))], lr=.001, weight_decay=0., foreach=False)
    return ParallelPublicReasonedCodeSFTStage(model, adam, data, copy.deepcopy(ENTRIES), profile,
        options or OPTIONS, SOURCES, mode='mechanics', rank=rank, world=world,
        control_group=control, gradient_group=control)


def live(stage):
    return dict(parts=tensor_digest(stage.model.parts.named_parameters()), base=tensor_digest(stage.model.base.named_parameters()),
        optimizer=value_digest(stage.optimizer.state_dict()), RNG=value_digest(capture_rng('cpu')),
        cursor=stage.completed, metrics=value_digest(stage.metrics), failed=stage.failed,
        tensor_identities=[(id(p), k, id(t), t.data_ptr(), t._version)
            for p, row in stage.optimizer.state.items() for k, t in row.items()])


def draws():
    return [random.random(), float(np.random.random()), *torch.rand(3).tolist()]


@pytest.mark.parametrize('profile', PROFILES)
def test_actual_actor_gradients_and_update_match_independent_per_token_serial_objective(tmp_path, profile):
    packet(tmp_path)
    torch.use_deterministic_algorithms(True)
    stage, serial = setup(tmp_path, profile), setup(tmp_path, profile)
    factor = 1 if profile == 'plain_tokens' else 4
    denominator = 7+(factor-1)*3
    serial.model.train()
    serial.model.policy.gradient_checkpointing = True
    original_sums, stopping_sums = [], []
    for position in ENTRIES[0]:
        ids, labels = serial.data.sequence(position)
        logits = serial.model(torch.tensor([ids]), attention_mask=torch.ones(1,len(ids),dtype=torch.bool))['logits']
        scores, stopping = [], []
        for index, token in enumerate(labels[1:], 1):
            if token == -100:
                continue
            ce = -logits[0,index-1].log_softmax(-1)[token]
            scores.append(ce)
            if token == 2:
                stopping.append(ce)
        raw, eos = torch.stack(scores).sum(), torch.stack(stopping).sum()
        ((raw+(factor-1)*eos)/denominator).backward()
        original_sums.append(float(raw.detach()))
        stopping_sums.append(float(eos.detach()))
    norm = float(torch.nn.utils.clip_grad_norm_(serial.parameters, 1., error_if_nonfinite=True))
    expected_grads = [p.grad.clone() for p in serial.parameters]
    serial.optimizer.step()
    actual_grads = []
    ordinary = stage.optimizer.step
    def observed(*args, **kwargs):
        actual_grads.extend(p.grad.clone() for p in stage.parameters)
        return ordinary(*args, **kwargs)
    stage.optimizer.step = observed
    initial = live(stage)
    row = stage.update()
    maximum = max(float((a-b).abs().max()) for a,b in zip(actual_grads,expected_grads))
    assert maximum < 3e-6
    assert row['gradient_norm_before_clip'] == pytest.approx(norm, abs=3e-6, rel=3e-6)
    assert row['original_token_mean_NLL'] == pytest.approx(sum(original_sums)/7, abs=2e-6)
    assert row['EOS_token_mean_NLL'] == pytest.approx(sum(stopping_sums)/3, abs=2e-6)
    assert row['non_EOS_token_mean_NLL'] == pytest.approx((sum(original_sums)-sum(stopping_sums))/4, abs=2e-6)
    assert row['weighted_token_mean_CE'] == pytest.approx((sum(original_sums)+(factor-1)*sum(stopping_sums))/denominator, abs=2e-6)
    for actual, expected in zip(stage.parameters, serial.parameters):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    assert row['source_metadata']['response_EOS_targets'] == 7
    assert row['source_metadata']['genuine_EOS_targets'] == 3
    assert row['source_metadata']['actual_global_weighted_targets'] == denominator
    assert {r['view'] for r in row['source_metadata']['complete_sources']} == {'code_only','reasoning'}
    assert live(stage)['base'] == initial['base'] and live(stage)['parts'] != initial['parts']
    assert all(float(state['step']) == 1 for state in stage.optimizer.state.values())
    assert all(p.grad is None for p in stage.model.parameters()) and not torch.cuda.is_initialized()


def test_small_claimed_GPU_capacity_altered_scope_settings_and_source_reject_before_mutation(tmp_path):
    packet(tmp_path)
    for options in [dict(OPTIONS, maximum_padded_tokens=512), dict(OPTIONS, epochs=True), dict(OPTIONS, gradient_clip=float('nan'))]:
        with pytest.raises(ValueError): setup(tmp_path, options=options)
    for mode in ['train','capacity']:
        model = own_model()
        adam = torch.optim.AdamW([p for _,p in model.named_selected(('actor',))], lr=.001, weight_decay=0., foreach=False)
        with pytest.raises(ValueError):
            ParallelPublicReasonedCodeSFTStage(model, adam, PublicReasonedCodeSFTData(tmp_path, purpose='capacity' if mode == 'capacity' else 'optimization'),
                ENTRIES, 'plain_tokens', OPTIONS, SOURCES, mode=mode)
        assert not adam.state
    stage = setup(tmp_path)
    before = live(stage)
    for attribute, value in [('role','reserved'),('views','code_only'),('human_preferences',True),('use_as_RL_reward_oracle_forbidden',False)]:
        old = getattr(stage.data, attribute)
        setattr(stage.data, attribute, value)
        try:
            with pytest.raises(ValueError): stage.update()
            assert live(stage) == before
        finally: setattr(stage.data, attribute, old)


def worker(rank, folder):
    folder = Path(folder)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://'+str(folder/'rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=5))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=5))
    try:
        for profile in PROFILES:
            stage = setup(folder/'source', profile, rank=rank, world=4, control=control)
            initial = live(stage)
            first = stage.update()
            assert first['actual_rank_observations'][3]['microbatches'] == []
            durable.commit(stage, folder/profile, maximum_bytes=8*1024**2, minimum_free_bytes=0)
            state, receipt, pointer = durable.read_latest(folder/profile)
            wanted_draws = draws()
            wanted_row = stage.update()
            wanted = live(stage)
            durable.restore_latest(stage, folder/profile, expected_pointer_sha256=pointer)
            assert draws() == wanted_draws and stage.update() == wanted_row
            actual = live(stage)
            assert all(actual[k] == wanted[k] for k in ['parts','base','optimizer','RNG','cursor','metrics'])
            before = live(stage)
            for corruption in ['peer_RNG','peer_cursor','Adam_clock','EOS_count','EOS_weight','EOS_metric','critic','nonfinite']:
                bad = copy.deepcopy(state)
                if corruption == 'peer_RNG': bad['private_workers'][2]['rng']['torch'] = torch.zeros(1,dtype=torch.uint8)
                if corruption == 'peer_cursor': bad['private_workers'][2]['local_source_cursor']['rank'] = 1
                if corruption == 'Adam_clock': bad['optimizer']['state'][0]['step'].fill_(77)
                if corruption == 'EOS_count': bad['metrics'][0]['source_metadata']['genuine_EOS_targets'] += 1
                if corruption == 'EOS_weight': bad['metrics'][0]['actual_rank_observations'][0]['microbatches'][0]['actual_EOS_weight'] += 1
                if corruption == 'EOS_metric': bad['metrics'][0]['EOS_token_mean_NLL'] += 1
                if corruption == 'critic': bad['model']['parts']['critic.weight'].add_(.1)
                if corruption == 'nonfinite': next(iter(bad['model']['parts'].values())).fill_(float('nan'))
                for retained in [receipt['semantic_manifest'], manifest(bad)]:
                    with pytest.raises(ValueError): stage.resume(bad, expected_manifest=retained)
                    assert live(stage) == before
            tail = stage.update()
            assert [row['actual_positions'] for row in tail['actual_rank_observations']] == [[2],[],[],[]]
            assert [row['microbatches'] for row in tail['actual_rank_observations']][1:] == [[],[],[]]
            assert tail['source_metadata']['actual_global_weighted_targets'] == (2 if profile == 'plain_tokens' else 5)
            durable.commit(stage, folder/profile, maximum_bytes=8*1024**2, minimum_free_bytes=0)
            before = live(stage)
            with pytest.raises(ValueError): durable.restore_latest(stage, folder/profile)
            assert live(stage) == before
            cold = setup(folder/'source', profile, rank=rank, world=4, control=control)
            durable.restore_latest(cold, folder/profile, allow_completed=True)
            assert all(live(cold)[k] == before[k] for k in ['parts','base','optimizer','RNG','cursor','metrics'])
            assert live(stage)['base'] == initial['base'] and live(stage)['parts'] != initial['parts']
            fault = setup(folder/'source', profile, rank=rank, world=4, control=control)
            prior = live(fault)
            hook = None
            if rank == 1:
                def poison(module, args, output): return dict(output, logits=output['logits']*float('nan'))
                hook = fault.model.register_forward_hook(poison)
            with pytest.raises(ValueError, match='before reduction'): fault.update()
            if hook is not None: hook.remove()
            after = live(fault)
            assert all(after[k] == prior[k] for k in ['parts','base','optimizer','RNG','cursor','metrics'])
            assert after['failed'] and all(p.grad is None for p in fault.model.parameters())
            assert fault.failure_observations[-1]['actual_rank_errors'][1] is not None
            assert not torch.cuda.is_initialized()
            (folder/(profile+'-rank-'+str(rank)+'.json')).write_text(json.dumps(dict(
                profile=profile, rank=rank, actual_small_model_parameters=stage.model.counts()['actual_unique'],
                selected_Adam_states=len(stage.parameters), actual_three_global_updates=stage.metrics,
                first_gradient_sha256=first['actual_unclipped_global_actor_gradient_sha256'],
                actual_four_replica_SUM=True, bit_exact_next_weights_Adam_private_RNG_and_metrics=True,
                cold_reload_exact=True, peer_fault_clears_all_gradients_before_Adam=True,
                true_single_source_tail_three_empty_ranks=True, original_7target_first_batch_preserved=True,
                production_GPU_capacity=False, production_training=False, quality_improvement_established=False), indent=2))
    finally:
        dist.destroy_process_group()


def test_actual_four_replicas_shared_Adam_recovery_empty_ranks_and_coordinated_faults(tmp_path):
    packet(tmp_path/'source')
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    records = [json.loads((tmp_path/(profile+'-rank-'+str(rank)+'.json')).read_text()) for profile in PROFILES for rank in range(4)]
    for profile in PROFILES:
        assert len({r['first_gradient_sha256'] for r in records if r['profile'] == profile}) == 1
    assert records[0]['first_gradient_sha256'] != records[4]['first_gradient_sha256']
