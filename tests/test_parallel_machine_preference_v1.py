from copy import deepcopy

import pytest
import torch

from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.machine_code_preference_stage_v1 import gradients
from slm_lab.parallel_machine_preference_v1 import local_gradients, ParallelMachinePreferenceStage
from test_machine_code_preference_stage_v1 import data, trained, settings, SOURCES
from test_own_alignment_checkpoint_v1 import assert_nested_equal


@pytest.mark.parametrize('method', ['chosen_CE', 'machine_DPO'])
def test_actual_unequal_whole_program_denominators_and_empty_replicas_match_serial_gradient(method):
    torch.set_num_threads(1)
    original = trained(); d = data(2); options = settings(method)
    # Unequal chosen response/EOS lengths exercise token vs pair normalization.
    d.rows[1]['chosen_ids'].insert(-1, 13); d.rows[1]['chosen_labels'].insert(-1, 13)
    oracle = deepcopy(original); q = FrozenOwnActorReference(oracle, sources=SOURCES)
    gradients(oracle, d, [0,1], q, method=method, chosen_CE_weight=options['chosen_CE_weight'])
    replicas = []; records = []
    for rank in range(4):
        m = deepcopy(original); ref = FrozenOwnActorReference(m, sources=SOURCES)
        rows = local_gradients(m,d,[0,1],ref,options,rank=rank,world=4)
        replicas.append([p.grad for _,p in m.named_selected(('actor',))]); records.extend(rows)
        assert all(p.grad is None for p in m.base.parameters())
        assert m.parts['critic'].weight.grad is None
        if rank >= 2:
            assert not rows and all(torch.count_nonzero(g) == 0 for g in replicas[-1])
    assert sorted(r['original_train_position'] for r in records)==[0,1]
    assert sum(r['chosen_response_EOS_targets'] for r in records)==7
    for index,(_,parameter) in enumerate(oracle.named_selected(('actor',))):
        assert torch.allclose(sum(row[index] for row in replicas),parameter.grad,atol=3e-7,rtol=3e-5)


def test_private_replica_cursor_and_actual_next_update_recover_and_bad_metadata_rejects(tmp_path):
    m=trained();d=data();o=settings();q=FrozenOwnActorReference(m,sources=SOURCES)
    opt=torch.optim.AdamW([p for _,p in m.named_selected(('actor',))],lr=o['learning_rate'],
                         betas=tuple(o['betas']),weight_decay=o['weight_decay'],foreach=False)
    stage=ParallelMachinePreferenceStage(m,opt,d,q,o,SOURCES)
    stage.update();path=tmp_path/'worker.pt';torch.save(stage.state(),path)
    expected=stage.update();weights=m.export(SOURCES);moments=deepcopy(opt.state_dict());rng=torch.get_rng_state()
    saved=torch.load(path,weights_only=True,mmap=True)
    for change in [lambda s:s.update(replica_rank=1),
                   lambda s:s['core']['core']['metrics'][0].update(per_replica_original_pairs=[0]),
                   lambda s:s['core']['core']['metrics'][0].update(human_preferences=True)]:
        bad=deepcopy(saved);change(bad)
        with pytest.raises(ValueError):stage.resume(bad)
        assert_nested_equal(weights,m.export(SOURCES));assert_nested_equal(moments,opt.state_dict())
        assert torch.equal(rng,torch.get_rng_state())
    stage.resume(saved);assert stage.update()==expected
    assert_nested_equal(weights,m.export(SOURCES));assert_nested_equal(moments,opt.state_dict())
    assert torch.equal(rng,torch.get_rng_state())
