"""Independent combined-loss Adam control for the actual joint dispatcher."""
import copy
import math

import torch
from torch.nn import functional as F

from slm_lab.joint_objective_step_v1 import update
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_joint_batched_human_preferences_v1 import oracle_loss as human_loss
from test_joint_objective_step_v1 import ROOT,Trial,visual_packet


def test_joint_code_human_and_complete_word_update_matches_independent_total_loss_and_Adam():
    data=MixedCodingSFTData(ROOT);trial=Trial(data);oracle=copy.deepcopy(trial.view)
    options={**trial.options,'project_conflicts':False}
    packet=visual_packet('word',0,1);source=packet['source'];word=packet['word'];native=packet['native']
    actual=update(trial.view,trial.reference,trial.optimizer,trial.guard,
        code=dict(data=data,positions=trial.positions,profile='balanced_sources'),human=dict(data=trial.human,positions=[0,1],DPO_weight=1),
        visual=packet,options=options,source_entry_sha256='e'*64)
    denominator=math.fsum(data.information(p)['response_EOS_targets']*data.weight('balanced_sources',p) for p in trial.positions)
    total=0.
    for p in trial.positions:
        batch=data.batch([p]);target=batch['labels'][:,1:];mask=target!=-100
        output=oracle.text(**{k:v for k,v in batch.items() if k!='labels'})['logits'][:,:-1]
        total=total+F.cross_entropy(output.float()[mask],target[mask],reduction='sum')*data.weight('balanced_sources',p)/denominator
    total=total+options['human_weight']*human_loss(oracle,trial.human,1,coefficient=options['chosen_CE_weight'])
    visual_loss=0.
    for i,text in enumerate(word['texts']):
        logits=oracle.word_logits(word['word_pixels'][i:i+1]).float().log_softmax(-1).transpose(0,1)
        raw=F.ctc_loss(logits,torch.tensor(list(text.encode())),torch.tensor([4]),torch.tensor([len(text.encode())]),blank=256,reduction='sum')
        visual_loss=visual_loss+raw/len(text.encode())/2
        batch={k:v[i:i+1] for k,v in native.items()}
        visual_loss=visual_loss+oracle.native_image(**batch)['loss']/2
    total=total+options['visual_weight']*visual_loss
    named=oracle.named_selected();gradients=torch.autograd.grad(total,[p for _,p in named],allow_unused=True)
    for (_,p),g in zip(named,gradients):p.grad=g
    other=torch.optim.AdamW([p for _,p in named],lr=1e-4,eps=1e-5,weight_decay=.01,foreach=False)
    norm=float(torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True));other.step();other.zero_grad(set_to_none=True)
    assert abs(norm-actual['gradient_norm_before_clip'])<5e-6
    for (name,p),(expected_name,q) in zip(trial.view.named_selected(),oracle.named_selected()):
        assert name==expected_name and float((p.detach()-q.detach()).abs().max())<5e-6,name
    assert tensor_digest(trial.view.actor.base.language_named_parameters())==tensor_digest(oracle.actor.base.language_named_parameters())
    trial.reference.assert_bound(full=True)
    assert all(p.grad is None for p in trial.view.parameters()) and not torch.cuda.is_initialized()
