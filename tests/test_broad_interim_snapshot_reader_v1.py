"""Actual small CPU update/readback; never a distributed/GPU capacity proof."""
import copy
import json

import pytest
import torch

from slm_lab.broad_anchored_execution_stage_v1 import BroadAnchoredExecutionStage
from slm_lab.broad_anchored_policy_v3 import sources,reference_sources
from slm_lab.broad_interim_snapshot_reader_v1 import materialize,equal
from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_own_dense_actor_v1 import actor
from test_own_aligned_assistant_v1 import original_signature


class Tokens:
    def decode(self,ids):return ' '.join(map(str,ids))


class Corpus:
    tokenizer=Tokens();provenance={'explicit_small_CPU_format_control':True}
    rows=[dict(id=f'control-{i}',kind='documented',complete_context_positions=4,prefix_positions=2,response_EOS_targets=2,
               full_training_action_budget=4) for i in range(3)]
    def __len__(self):return len(self.rows)
    def row(self,i):return self.rows[i]
    def sequence(self,i):return [1,4+i,7,2],[-100,-100,7,2]
    def batch(self,i,device):
        ids,labels=self.sequence(i);return dict(input_ids=torch.tensor([ids],device=device),labels=torch.tensor([labels],device=device))
    def score(self,i,response,*,stopped_on_eos):return dict(reward=0.,status='explicit_CPU_format_control_without_execution_quality_claim')


def fixture(tmp_path,monkeypatch):
    from slm_lab import anchored_execution_step_v2 as kernel
    monkeypatch.setattr(kernel,'ExpandedExecutionCorpus',Corpus)
    model=actor();baseline=copy.deepcopy(model);corpus=Corpus();options=dict(seed=963117,learning_rate=1e-3,betas=[.9,.95],weight_decay=.01,KL_weight=.01,anchor_weight=1.)
    plan=dict(planned_steps=3,training_options=options,actual_counts=model.counts(),
        starting_policy_inputs=dict(checkpoint_sha256='a'*64,plan_sha256='b'*64),
        APPS_execution_admission_sha256='c'*64,documented_probe_admission_sha256='d'*64,
        cache_calibration=dict(bound=.001,sha256='e'*64),CPU_mechanics_horizon=2)
    reference=FrozenOwnActorReference(model,sources=reference_sources(plan));plan['own_reference_binding']=reference.binding()
    params=[p for _,p in model.named_selected(('actor',))]
    optimizer=torch.optim.AdamW(params,lr=options['learning_rate'],betas=tuple(options['betas']),weight_decay=options['weight_decay'],foreach=False)
    stage=BroadAnchoredExecutionStage(model,optimizer,corpus,reference,root=tmp_path,method='anchor_CE',seed=options['seed'],
        sources=sources(plan,'f'*64),cache_absolute_bound=.001,cache_calibration_sha256='e'*64,CPU_mechanics_horizon=2)
    record=stage.update();state=stage.state();path=tmp_path/'private.pt';torch.save(state,path)
    journal=tmp_path/'journal.jsonl';journal.write_text(json.dumps(record)+'\n')
    return model,baseline,corpus,plan,state,path,journal


def test_actual_CPU_update_readback_matches_all_values_and_logits_without_restoring_RNG_or_Adam(tmp_path,monkeypatch):
    trained,baseline,corpus,plan,state,path,journal=fixture(tmp_path,monkeypatch);saved=copy.deepcopy(state);rng=torch.get_rng_state().clone()
    base=original_signature(baseline.base);proof=materialize(baseline,corpus,plan,'f'*64,[path],journal,expected_completed=1,allow_small_cpu_control=True)
    assert proof['completed_updates']==1 and proof['original_source_epoch_is_incomplete'] and not proof['optimizer_constructed_or_resumed']
    assert equal(saved,state) and torch.equal(rng,torch.get_rng_state()) and base==original_signature(baseline.base)
    assert tensor_digest(baseline.parts.named_parameters())==tensor_digest(trained.parts.named_parameters())
    with torch.no_grad():assert torch.equal(baseline(torch.tensor([[1,4,7]]))['logits'],trained(torch.tensor([[1,4,7]]))['logits'])
    with pytest.raises(ValueError,match='four-rank'):
        materialize(baseline,corpus,plan,'f'*64,[path],journal,expected_completed=1)


@pytest.mark.parametrize('damage',['Adam','critic','RNG','journal_actions','journal_prompt','missing_rank'])
def test_corrupt_private_or_original_journal_rejects_before_any_policy_or_global_RNG_mutation(tmp_path,monkeypatch,damage):
    _,baseline,corpus,plan,state,path,journal=fixture(tmp_path,monkeypatch);paths=[path]
    if damage=='Adam':state['core']['optimizer']['state'][0]['exp_avg_sq'].fill_(-1)
    if damage=='critic':state['core']['model']['parts']['critic.weight'].add_(1)
    if damage=='RNG':state['core']['rng']['torch']=torch.zeros(3,dtype=torch.uint8)
    if damage.startswith('journal'):
        row=json.loads(journal.read_text())
        if damage=='journal_actions':row['trajectories'][0]['response_ids'][0]=3
        else:row['complete_prompt_ids'][0]=5
        journal.write_text(json.dumps(row)+'\n')
    if damage=='missing_rank':paths=[]
    torch.save(state,path);before=original_signature(baseline);rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):materialize(baseline,corpus,plan,'f'*64,paths,journal,expected_completed=1,allow_small_cpu_control=True)
    assert before==original_signature(baseline) and torch.equal(rng,torch.get_rng_state())
