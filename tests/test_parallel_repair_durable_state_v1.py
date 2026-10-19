"""Real neural two-slot commits and fault recovery; all fixtures are CPU-small."""
from datetime import timedelta
import json
from pathlib import Path

import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab import parallel_repair_durable_state_v1 as durable
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.corpus import file_hash
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_parallel_function_repair_stage_v1 import setup,live

MAXIMUM=128*1024**2


def commit(stage,path):return durable.commit(stage,path,maximum_bytes=MAXIMUM,minimum_free_bytes=0)


def test_three_real_generations_preserve_old_mmap_and_cold_complete_recovery(tmp_path):
    stage=setup();folder=tmp_path/'durable';stage.update();first=commit(stage,folder)
    old,_,old_pointer=durable.read_latest(folder);old_parts=tensor_digest(old['model']['parts'].items())
    stage.update();second=commit(stage,folder);stage.update();third=commit(stage,folder)
    assert [first['generation'],second['generation'],third['generation']]==[1,2,3]
    assert [first['slot'],second['slot'],third['slot']]==[1,0,1]
    assert tensor_digest(old['model']['parts'].items())==old_parts and old['completed']==1
    expected_parts=tensor_digest(stage.model.parts.named_parameters());expected_Adam=value_digest(stage.optimizer.state_dict());expected_rng=value_digest(capture_rng('cpu'))
    cold=setup()
    with pytest.raises(ValueError):durable.restore_latest(cold,folder,expected_pointer_sha256=old_pointer)
    before=live(cold)
    with pytest.raises(ValueError):durable.restore_latest(cold,folder)
    assert live(cold)==before
    receipt=durable.restore_latest(cold,folder,allow_completed=True)
    assert receipt['completed']==cold.completed==3
    assert tensor_digest(cold.model.parts.named_parameters())==expected_parts
    assert value_digest(cold.optimizer.state_dict())==expected_Adam and value_digest(capture_rng('cpu'))==expected_rng


@pytest.mark.parametrize('fail_at',['receipt','pointer'])
def test_injected_root_IO_failure_keeps_previous_complete_generation_and_every_live_value(tmp_path,monkeypatch,fail_at):
    stage=setup();folder=tmp_path/'durable';stage.update();first=commit(stage,folder);first_pointer=file_hash(folder/'LATEST.json')
    stage.update();before=live(stage);writer=durable.durable_json
    def fail(path,value,reserve):
        if (fail_at=='receipt' and Path(path).name=='receipt-slot-0.json') or (fail_at=='pointer' and Path(path).name=='LATEST.json'):
            raise OSError('Deliberate complete-generation publication fault')
        return writer(path,value,reserve)
    monkeypatch.setattr(durable,'durable_json',fail)
    with pytest.raises(ValueError,match='publication'):commit(stage,folder)
    assert live(stage)==before and file_hash(folder/'LATEST.json')==first_pointer
    saved,receipt,_=durable.read_latest(folder)
    assert saved['completed']==receipt['completed']==1 and receipt['checkpoint_sha256']==first['checkpoint_sha256']
    monkeypatch.setattr(durable,'durable_json',writer)
    second=commit(stage,folder);assert second['completed']==2 and second['generation']==2


def test_corrupt_external_receipt_and_state_bytes_reject_before_neural_or_private_mutation(tmp_path):
    stage=setup();folder=tmp_path/'durable';stage.update();receipt=commit(stage,folder);stage.update();before=live(stage)
    path=folder/'receipt-slot-1.json';raw=path.read_bytes();path.write_bytes(raw+b' ')
    with pytest.raises(ValueError):durable.restore_latest(stage,folder)
    assert live(stage)==before;path.write_bytes(raw)
    path=Path(receipt['checkpoint']);original=path.read_bytes();path.write_bytes(original[:-1]+bytes([original[-1]^1]))
    with pytest.raises(ValueError):durable.restore_latest(stage,folder)
    assert live(stage)==before


def worker(rank,folder):
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=4))
    try:
        stage=setup(rank=rank,world=4);directory=Path(folder)/'durable';stage.update();first=commit(stage,directory)
        stage.update();before=live(stage);writer=durable.durable_json
        def fail(path,value,reserve):
            if Path(path).name=='receipt-slot-0.json':raise OSError('Actual root-only injected I/O failure')
            return writer(path,value,reserve)
        if rank==0:durable.durable_json=fail
        with pytest.raises(ValueError,match='publication'):commit(stage,directory)
        assert live(stage)==before
        if rank==0:durable.durable_json=writer
        second=commit(stage,directory);assert second['generation']==2 and second['completed']==2
        stage.update();third=commit(stage,directory)
        all_receipts=[None]*4;dist.all_gather_object(all_receipts,third);assert all(v==third for v in all_receipts)
        wanted_model=tensor_digest(stage.model.parts.named_parameters());wanted_Adam=value_digest(stage.optimizer.state_dict());wanted_rng=value_digest(capture_rng('cpu'))
        # Deliberately disturb only selected actor values, then recover the
        # actual completed shared file into all four live own models.
        with torch.no_grad():next(iter(stage.parameters)).add_(.25)
        durable.restore_latest(stage,directory,allow_completed=True)
        assert tensor_digest(stage.model.parts.named_parameters())==wanted_model
        assert value_digest(stage.optimizer.state_dict())==wanted_Adam and value_digest(capture_rng('cpu'))==wanted_rng
        assert not torch.cuda.is_initialized()
        (Path(folder)/('rank-'+str(rank)+'.json')).write_text(json.dumps(dict(rank=rank,actual_global_updates=3,
            root_only_IO_failure_coordinated_to_all_four_replicas=True,previous_generation_retained_on_failure=True,
            actual_actor_Adam_private_RNG_recovered_from_completed_shared_durable_file=True,
            final_generation=third['generation'],final_checkpoint_sha256=third['checkpoint_sha256'],
            four_GPU_or_actual984M_capacity_not_proved=True)))
    finally:dist.destroy_process_group()


def test_actual_four_workers_coordinate_root_IO_failure_and_restore_one_shared_complete_file(tmp_path):
    spawn(worker,args=(str(tmp_path),),nprocs=4,join=True)
    proofs=[json.loads((tmp_path/('rank-'+str(rank)+'.json')).read_text()) for rank in range(4)]
    assert len({p['final_checkpoint_sha256'] for p in proofs})==1 and all(p['actual_global_updates']==3 for p in proofs)
