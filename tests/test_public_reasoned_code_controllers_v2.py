"""Connected four-CPU capacity/readback/training and interrupted replay."""
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

from slm_lab.batched_reasoning_capacity_controller_v1 import live
from slm_lab.bounded_gzip_observation_stream_v1 import lines
from slm_lab.public_reasoned_code_capacity_controller_v2 import run_capacity, require_full_capacity_reports
from slm_lab.public_reasoned_code_training_controller_v2 import read_capacity, run_training
from slm_lab.parallel_repair_durable_state_v1 import read_latest
from slm_lab.complete_response_stopping_loss_v1 import PROFILES
from slm_lab.model import ModelConfig
from slm_lab.own_dense_growth_v1 import new_backbone, OwnDenseAssistant
from slm_lab.own_dense_actor_v1 import OwnDenseActor
from slm_lab.parallel_public_reasoned_code_SFT_stage_v1 import ParallelPublicReasonedCodeSFTStage
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from test_parallel_public_reasoned_code_SFT_stage_v1 import ENTRIES, OPTIONS, SOURCES
from test_public_reasoned_code_train_data_v1 import packet


def own_model():
    torch.manual_seed(991104)
    config = ModelConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_layers=2,
        num_heads=4, num_kv_heads=2, max_text_length=32, vision_hidden_size=16,
        vision_intermediate_size=32, vision_layers=1, vision_heads=4, image_size=32, patch_size=8, pool_size=2)
    base = OwnDenseAssistant(new_backbone(config,991001), reward_hidden=8, word_temporal=4,
        pointing_rank=8, router_hidden=8, detector_hidden=4)
    return OwnDenseActor(base,rank=2)


def execute(stage, folder, plan, reports, readbacks, *, progress=None, pointer=None, opt_in=True, journal='attempt.jsonl.gz'):
    return run_training(stage, folder/'durable', folder/journal, capacity_plan=plan,
        reports=reports, readbacks=readbacks, checkpoint_every=1, maximum_state_bytes=8*1024**2,
        minimum_free_bytes=0, maximum_journal_raw_bytes=8*1024**2,
        preflight=lambda:stage.data.guard.check(), progress=progress,
        expected_resume_pointer_sha256=pointer, allow_mechanics=opt_in)


def worker(rank, folder):
    folder = Path(folder)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://'+str(folder/'rendezvous'), rank=rank, world_size=4,
        timeout=timedelta(minutes=5))
    control = dist.new_group(backend='gloo', timeout=timedelta(minutes=5))
    try:
        text = torch.tensor([[1,4,5,2]])
        inputs = dict(text=dict(input_ids=text, attention_mask=torch.ones_like(text,dtype=torch.bool)),
            photo=dict(input_ids=text, attention_mask=torch.ones_like(text,dtype=torch.bool),
                pixel_values=torch.linspace(-1,1,3*32*32).reshape(1,3,32,32)),
            word=dict(word_pixels=torch.linspace(-1,1,16*3*64*64).reshape(16,3,64,64)))
        def independent():
            with torch.random.fork_rng(devices=[]): return own_model()
        for profile in PROFILES:
            def fresh():
                model = own_model()
                adam = torch.optim.AdamW([p for _,p in model.named_selected(('actor',))], lr=.001,
                    weight_decay=0., foreach=False)
                stage = ParallelPublicReasonedCodeSFTStage(model,adam,PublicReasonedCodeSFTData(folder/'source'),
                    copy.deepcopy(ENTRIES),profile,OPTIONS,SOURCES,mode='mechanics',rank=rank,world=4,
                    control_group=control,gradient_group=control)
                random.seed(991115+rank)
                np.random.seed(991115+rank)
                torch.manual_seed(991115+rank)
                return stage
            plan = dict(capacity_entries=copy.deepcopy(ENTRIES), mechanics_only=True, complete_source_selection=False,
                full_original_capacity_plan_not_used_by_this_explicit_small_fixture=True)
            cap_dir = folder/profile/'capacity'
            cap = fresh()
            before = live(cap)
            with pytest.raises(ValueError): run_capacity(cap, cap_dir, capacity_plan=plan, independent_actor=independent,
                cold_inputs=inputs, maximum_bytes=8*1024**2, minimum_free_bytes=0)
            assert live(cap) == before
            result = run_capacity(cap, cap_dir, capacity_plan=plan, independent_actor=independent,
                cold_inputs=inputs, maximum_bytes=8*1024**2, minimum_free_bytes=0, allow_mechanics=True)
            assert result['actual_complete_capacity_updates'] == 3 and result['actual_physical_optimizer_calls'] == 4
            assert result['actual_source_views'] == ['code_only','reasoning']
            assert result['actual_source_budget']['source_presentations'] == 6
            assert result['actual_source_budget']['response_EOS_targets'] == 14
            assert result['independent_complete_actor_cold']['complete_text_photo_word_outputs_bit_exact']
            assert result['independent_complete_actor_cold']['actual_word_batch_size'] == 16
            assert result['independent_complete_actor_cold']['actual_word_readout_shape'][0] == 16
            assert not result['full_size_GPU_capacity_passed'] and result['production_updates'] == 0
            reports = cap.gather(result)
            with pytest.raises(ValueError): require_full_capacity_reports(reports, capacity_plan=plan, profile=profile,
                expected_sources=SOURCES, expected_options=OPTIONS)
            reader = fresh()
            before = live(reader)
            wrong = copy.deepcopy(reports)
            for r in wrong: r['final_actual_pointer_sha256'] = 'e'*64
            with pytest.raises(ValueError): read_capacity(reader, cap_dir, reports=wrong, capacity_plan=plan,
                independent_actor=independent, cold_inputs=inputs, allow_mechanics=True)
            assert live(reader) == before
            proof = read_capacity(reader, cap_dir, reports=reports, capacity_plan=plan,
                independent_actor=independent, cold_inputs=inputs, allow_mechanics=True)
            assert proof['new_optimizer_calls'] == 0 and proof['initial_actor_sha256'] != proof['actual_actor_sha256']
            readbacks = reader.gather(proof)
            trained = fresh()
            before = live(trained)
            broken = copy.deepcopy(readbacks)
            if rank == 2: broken[1]['actual_private_RNG_sha256'] = 'c'*64
            with pytest.raises(ValueError): execute(trained, folder/profile/'forbidden', plan, reports, broken)
            assert live(trained) == before and not (folder/profile/'forbidden').exists()
            with pytest.raises(ValueError): execute(trained, folder/profile/'no-opt-in', plan, reports, readbacks, opt_in=False)
            assert live(trained) == before
            stale = fresh()
            saved, _, _ = read_latest(cap_dir)
            with torch.no_grad():
                for name,p in stale.named: p.copy_(saved['model']['parts'][name])
            stale.intact()
            before_stale = live(stale)
            with pytest.raises(ValueError): execute(stale, folder/profile/'stale-actor', plan, reports, readbacks)
            assert live(stale) == before_stale
            complete = execute(trained, folder/profile/'complete', plan, reports, readbacks)
            expected = live(trained)
            assert complete['actual_complete_updates'] == complete['actual_new_optimizer_calls'] == 3
            assert complete['original_complete_training_budget']['response_EOS_targets'] == 14
            assert complete['actual_complete_journal']['records'] == 3 and complete['production_updates'] == 0
            interrupted = fresh()
            attempts = folder/profile/'recovery'
            def stop_after_second(row, elapsed):
                if row['step'] == 2: raise OSError('controlled root-only observer failure after actual Adam')
            with pytest.raises(ValueError, match='retain every complete actual training update'):
                execute(interrupted, attempts, plan, reports, readbacks, progress=stop_after_second)
            assert interrupted.failed and interrupted.completed == 2
            saved, _, pointer = read_latest(attempts/'durable')
            failed_rows = [r for _,r in lines(attempts/'attempt.jsonl.gz')]
            assert saved['completed'] == 1 and [r['actual_complete_updates'] for r in failed_rows] == [1,2]
            resumed = fresh()
            before = live(resumed)
            with pytest.raises(ValueError): execute(resumed, attempts, plan, reports, readbacks,
                pointer='d'*64, journal='bad-pin.jsonl.gz')
            assert live(resumed) == before and not (attempts/'bad-pin.jsonl.gz').exists()
            recovered = execute(resumed, attempts, plan, reports, readbacks,
                pointer=pointer, journal='second-attempt.jsonl.gz')
            actual = live(resumed)
            assert recovered['actual_attempt_started_at_update'] == 1 and recovered['actual_new_optimizer_calls'] == 2
            assert all(actual[k] == expected[k] for k in ['actor','Adam','RNG','metrics','completed','failed'])
            assert [r['actual_complete_updates'] for _,r in lines(attempts/'second-attempt.jsonl.gz')] == [2,3]
            assert all(p.grad is None for p in resumed.model.parameters()) and not torch.cuda.is_initialized()
            (folder/(profile+'-controller-rank-'+str(rank)+'.json')).write_text(json.dumps(dict(
                capacity=result, fresh_capacity_readback=proof, complete_training=complete, resumed_training=recovered,
                root_failure_preserved_both_attempts=True, exact_uninterrupted_weights_Adam_RNG_metrics_after_recovery=True,
                stale_capacity_actor_not_reused_as_training_initializer=True, altered_peer_readback_rejected_before_mutation=True,
                production_capacity_or_training=False, quality_gain_established=False), indent=2, allow_nan=False))
    finally:
        dist.destroy_process_group()


def test_connected_capacity_fresh_readback_whole_training_and_interrupted_attempt_recovery(tmp_path):
    packet(tmp_path/'source')
    spawn(worker, args=(str(tmp_path),), nprocs=4, join=True)
    for profile in PROFILES:
        rows = [json.loads((tmp_path/(profile+'-controller-rank-'+str(rank)+'.json')).read_text()) for rank in range(4)]
        assert len({row['complete_training']['actual_final_actor_sha256'] for row in rows}) == 1
