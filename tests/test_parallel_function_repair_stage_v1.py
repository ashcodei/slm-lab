"""Actual small neural shared-state replay; full-size four-L40 proof is separate."""
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

from slm_lab.codenet_checkpoint_v1 import capture_rng,atomic_checkpoint
from slm_lab.parallel_function_repair_stage_v1 import ParallelFunctionRepairStage,manifest,value_digest
from slm_lab.parallel_function_repair_v1 import signature
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_full_function_repair_stage_v1 import fixture

SOURCES={'unit_complete_own_base_sha256':'a'*64,'unit_source_plan_sha256':'b'*64}
ENTRIES=[[0,1,2],[1],[0,2]]


def setup(profile='weighted_semantic_repair',precision='bf16',rank=0,world=1):
    torch.manual_seed(983113);model,data,weights,options=fixture()
    optimizer=torch.optim.AdamW([p for _,p in model.named_selected(('actor',))],lr=options['learning_rate'],betas=(.9,.95),weight_decay=.01,foreach=False)
    random.seed(983113+rank);np.random.seed(983113+rank);torch.manual_seed(983113+rank)
    stage=ParallelFunctionRepairStage(model,optimizer,data,ENTRIES,profile,options,SOURCES,weights,
        rank=rank,world=world,mode='capacity',precision=precision)
    return stage


def draws():return [random.random(),float(np.random.random()),*torch.rand(4).tolist()]


def live(stage):
    return dict(model_signature=signature(stage.model.parameters()),model_parts=tensor_digest(stage.model.parts.named_parameters()),
        optimizer=value_digest(stage.optimizer.state_dict()),rng=value_digest(capture_rng('cpu')),
        Adam_tensor_identity=[(id(p),k,id(t),t.data_ptr(),t._version) for p,row in stage.optimizer.state.items() for k,t in row.items()],
        completed=stage.completed,metrics=value_digest(stage.metrics),failed=stage.failed)


@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_actual_shared_serial_snapshot_restores_next_draws_source_update_weights_Adam_and_RNG(tmp_path,precision):
    stage=setup(precision=precision);frozen=tensor_digest(stage.model.base.named_parameters());stage.update()
    state,receipt=stage.snapshot();path=tmp_path/'shared.pt';atomic_checkpoint(path,state,128*1024**2,0)
    wanted_draws=draws();wanted=stage.update();wanted_values=live(stage)
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    assert stage.resume(saved,expected_manifest=receipt)==1 and draws()==wanted_draws and stage.update()==wanted
    assert stage.completed==2 and tensor_digest(stage.model.parts.named_parameters())==wanted_values['model_parts']
    assert value_digest(stage.optimizer.state_dict())==wanted_values['optimizer'] and value_digest(capture_rng('cpu'))==wanted_values['rng']
    assert frozen==tensor_digest(stage.model.base.named_parameters())
    assert not torch.cuda.is_initialized()


def test_resealed_wrong_masks_clocks_unused_critic_sources_and_foreign_CPU_rng_reject_before_mutation():
    stage=setup();stage.update();state,receipt=stage.snapshot();stage.update();before=live(stage)
    corruptions=[]
    bad=copy.deepcopy(state);bad['metrics'][0]['response_EOS_targets']-=1;corruptions.append(bad)
    bad=copy.deepcopy(state);bad['optimizer']['state'][0]['step'].fill_(42);corruptions.append(bad)
    bad=copy.deepcopy(state);bad['model']['parts']['critic.weight'].add_(1);corruptions.append(bad)
    bad=copy.deepcopy(state);bad['model']['sources']['unit_source_plan_sha256']='c'*64;corruptions.append(bad)
    bad=copy.deepcopy(state);bad['private_workers'][0]['rng']['cuda']=torch.zeros(24,dtype=torch.uint8);corruptions.append(bad)
    bad=copy.deepcopy(state);bad['private_workers'][0]['local_source_cursor']['rank']=3;corruptions.append(bad)
    for bad in corruptions:
        with pytest.raises(ValueError):stage.resume(bad,expected_manifest=manifest(bad))
        assert live(stage)==before
    stage.data.purpose='source_preparation'
    with pytest.raises(ValueError):stage.resume(state,expected_manifest=receipt)
    assert live(stage)==before


def worker(rank,folder):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=4))
    proofs=[]
    try:
        for precision in ['fp32','bf16']:
            for profile in ['plain_repeat','semantic_repair','weighted_semantic_repair']:
                stage=setup(profile,precision,rank,4);base=tensor_digest(stage.model.base.named_parameters());stage.update()
                state,receipt=stage.snapshot();path=Path(folder)/(profile+'-'+precision+'.pt')
                if rank==0:atomic_checkpoint(path,state,128*1024**2,0);(Path(folder)/(profile+'-'+precision+'-manifest.json')).write_text(json.dumps(receipt))
                dist.barrier();saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                assert (state is not None)==(rank==0)
                assert len({value_digest(w['rng']) for w in saved['private_workers']})==4
                assert saved['global_cursor']['all_consumed_global_source_batches']==[[0,1,2]]
                wanted_draws=draws();wanted=stage.update();before=live(stage)
                # A valid other worker's RNG is wrong for this retained commit,
                # even if the modified state remains a legal RNG container.
                bad=copy.deepcopy(saved)
                if rank==2:bad['private_workers'][0]['rng']=copy.deepcopy(saved['private_workers'][1]['rng'])
                with pytest.raises(ValueError):stage.resume(bad,expected_manifest=receipt)
                assert live(stage)==before
                # A peer-only malformed RNG must be inspected by every rank;
                # resealing a corrupt shared payload cannot bypass shape checks.
                bad=copy.deepcopy(saved);bad['private_workers'][3]['rng']['torch']=torch.zeros(1,dtype=torch.uint8)
                with pytest.raises(ValueError):stage.resume(bad,expected_manifest=manifest(bad))
                assert live(stage)==before
                bad=copy.deepcopy(saved);bad['private_workers'][3]['local_source_cursor']['all_consumed_local_source_batches']=[[0]]
                with pytest.raises(ValueError):stage.resume(bad,expected_manifest=manifest(bad))
                assert live(stage)==before
                bad=copy.deepcopy(saved);bad['optimizer']['state'][0]['exp_avg_sq'].fill_(float('nan'))
                with pytest.raises(ValueError):stage.resume(bad,expected_manifest=manifest(bad))
                assert live(stage)==before
                assert stage.resume(saved,expected_manifest=receipt)==1
                assert draws()==wanted_draws and stage.update()==wanted
                assert tensor_digest(stage.model.parts.named_parameters())==before['model_parts']
                assert value_digest(stage.optimizer.state_dict())==before['optimizer']
                assert value_digest(capture_rng('cpu'))==before['rng']
                if rank==1:stage.failed=True
                preflight_before=live(stage)
                with pytest.raises(ValueError):stage.update()
                assert live(stage)==preflight_before
                if rank==1:stage.failed=False
                stage.update();final,final_receipt=stage.snapshot()
                if rank==0:atomic_checkpoint(path,final,128*1024**2,0)
                dist.barrier();final=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
                with pytest.raises(ValueError):stage.resume(final,expected_manifest=final_receipt)
                stage.resume(final,expected_manifest=final_receipt,allow_completed=True)
                # Completed source epochs must never silently extend.
                before_invalid=live(stage)
                with pytest.raises(ValueError):stage.update()
                assert live(stage)==before_invalid
                assert base==tensor_digest(stage.model.base.named_parameters()) and not torch.cuda.is_initialized()
                proofs.append(dict(rank=rank,profile=profile,precision=precision,logical_updates=3,physical_optimizer_calls=4,
                    shared_policy_and_Adam_file_only_written_by_rank0=True,all_four_distinct_private_RNGs_retained=True,
                    exact_actual_next_draws_source_update_actor_values_Adam_and_private_RNG=True,
                    peer_only_wrong_valid_rng_and_resealed_malformed_peer_rng_cursor_and_Adam_rejected_before_mutation=True,
                    single_unhealthy_peer_rejected_before_different_collectives_or_any_mutation=True,
                    true_single_example_global_tail_record=stage.metrics[1],completed_resume_requires_explicit_allow_completed=True,
                    actual_four_GPU_or984M_capacity_not_proved=True))
        (Path(folder)/('rank-'+str(rank)+'.json')).write_text(json.dumps(proofs))
    finally:dist.destroy_process_group()


def test_actual_four_workers_replay_shared_actor_Adam_and_all_private_sources_and_RNGs(tmp_path):
    spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    records=[json.loads((tmp_path/('rank-'+str(rank)+'.json')).read_text()) for rank in range(4)]
    assert all(len(rows)==6 for rows in records)
    assert all(r['true_single_example_global_tail_record']['actual_rank_source_shards']==[[1],[],[],[]] for rows in records for r in rows)
