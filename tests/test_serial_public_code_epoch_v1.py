import copy

import pytest
import torch

from slm_lab.serial_public_code_epoch_v1 import reconcile, schedule, train
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_parallel_public_reasoned_code_SFT_stage_v1 import own_model
from test_public_reasoned_code_train_data_v1 import packet


@pytest.mark.parametrize('profile', ['plain_tokens', 'eos_weight4'])
def test_actual_complete_small_epoch_preserves_original_targets_and_checkpoints_real_tail(tmp_path, profile):
    torch.set_num_threads(1)
    packet(tmp_path)
    data = PublicReasonedCodeSFTData(tmp_path)
    model = own_model()
    adam = torch.optim.AdamW([p for _, p in model.named_selected(('actor',))], lr=.001, foreach=False)
    fixed = tensor_digest(model.base.named_parameters())
    recorded, saved = [], []
    report = train(model, adam, data, profile, checked=lambda: None,
        record=lambda step, row: recorded.append(copy.deepcopy(row)),
        checkpoint=lambda step, entries, rows: saved.append((step, copy.deepcopy(entries), copy.deepcopy(rows))),
        allow_mechanics=True)
    assert report['complete'] and report['all_train_views_consumed_exactly_once']
    assert report['original_complete_training_budget']['response_EOS_targets'] == 7
    assert report['original_complete_training_budget']['complete_context_tokens'] == 14
    assert report['actual_final_batch_sources'] == 3 and saved[0][0] == 1
    assert fixed == tensor_digest(model.base.named_parameters())
    assert reconcile(data, schedule(data), recorded, profile) == report
    for field, bad in [('actual_positions', [0]), ('actual_Adam_step', 2)]:
        changed = copy.deepcopy(recorded)
        changed[0][field] = bad
        with pytest.raises(ValueError): reconcile(data, schedule(data), changed, profile)
    with pytest.raises(ValueError): reconcile(data, schedule(data), [], profile)
    with pytest.raises(ValueError):
        train(model, adam, data, profile, checked=lambda: None, record=lambda *a: None,
            checkpoint=lambda *a: None, allow_mechanics=True)


def test_capacity_or_incomplete_views_cannot_supply_production_epoch(tmp_path):
    packet(tmp_path)
    for kwargs in [dict(purpose='capacity'), dict(views='code_only'), dict(views='reasoning')]:
        with pytest.raises(ValueError): schedule(PublicReasonedCodeSFTData(tmp_path, **kwargs))
