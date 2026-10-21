import copy

import pytest
import torch

from slm_lab.own_dense_checkpoint_v1 import make_state
from slm_lab.public_reasoned_code_train_data_v1 import PublicReasonedCodeSFTData
from slm_lab.serial_dense_code_epoch_v2 import train,reconcile,schedule
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_growth_v1 import model as own_model
from test_public_reasoned_code_train_data_v1 import packet


@pytest.mark.parametrize('rate',[3e-6,1e-5])
def test_actual_small_full_language_epoch_keeps_complete_tail_and_real_Adam_checkpoint(tmp_path,rate):
    torch.set_num_threads(1);packet(tmp_path);data=PublicReasonedCodeSFTData(tmp_path);model,_=own_model()
    parameters=[p for _,p in model.language_named_parameters()];chosen={id(p) for p in parameters}
    fixed=tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in chosen)
    before=tensor_digest(model.language_named_parameters());optimizer=torch.optim.AdamW(parameters,lr=rate,foreach=False,fused=False)
    rows=[];states=[]
    def checkpoint(step,entries,records):
        states.append(make_state(model,optimizer,sources={'test':'e'*64},completed=step,planned=len(entries),
            cursor={'next_global_batch':step},metrics=records,device='cpu'))
    report=train(model,optimizer,data,checked=lambda:None,record=lambda _,r:rows.append(copy.deepcopy(r)),
        checkpoint=checkpoint,allow_mechanics=True)
    assert report['complete'] and report['original_complete_training_budget']['response_EOS_targets']==7
    assert report['original_complete_training_budget']['complete_context_tokens']==14 and report['actual_final_batch_sources']==3
    assert len(states)==1 and states[0]['completed']==states[0]['planned']==1
    assert len(states[0]['optimizer']['state'])==len(parameters) and before!=tensor_digest(model.language_named_parameters())
    assert fixed==tensor_digest((n,p) for n,p in model.named_parameters() if id(p) not in chosen)
    assert reconcile(data,schedule(data),rows)==report
    for key,value in [('actual_positions',[0]),('actual_Adam_step',99),('private_RNG_unchanged',False)]:
        changed=copy.deepcopy(rows);changed[0][key]=value
        with pytest.raises(ValueError):reconcile(data,schedule(data),changed)
    with pytest.raises(ValueError):train(model,optimizer,data,checked=lambda:None,record=lambda *a:None,checkpoint=lambda *a:None,allow_mechanics=True)
