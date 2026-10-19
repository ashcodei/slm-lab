"""Independent global gradients and actual four-worker all30-source recovery."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from slm_lab.code_residual_branch_v1 import RoutedCodeResidual
from slm_lab.code_residual_collective_checkpoint_v2 import fingerprint
from slm_lab.code_residual_source_contract_v4 import build_required_contract,COMPONENT_NAMES
from slm_lab.code_residual_stage_v3 import ExpandedReplayEncoder
from slm_lab.code_residual_export_v2 import export_inference,load_inference
from slm_lab.code_residual_lifecycle_v4 import FORMAT as LEDGER_FORMAT,checkpoint_layout,rank_zero,run_updates
from slm_lab.code_residual_production_v4 import finish_inference
from slm_lab.codenet_checkpoint_v1 import capture_rng
from slm_lab.codenet_runner_v1 import average_gradients
from slm_lab.codenet_token_pack_v1 import packed_encoder
from slm_lab.codenet_stage_v1 import EpochOrder
from slm_lab.corpus import file_hash
from slm_lab.execution_collective_v2 import execution_gradients,prepare_reference_bundle
from slm_lab.execution_feedback_reader_v1 import ExecutionFeedback
from slm_lab.execution_preference_v1 import reference_scores
from slm_lab.execution_training_v5 import (ExecutionResidualKernel,auxiliary_fingerprint,
    global_CE_gradients,reference_from_checkpoint)
from slm_lab.execution_production_v5 import auxiliary_extremes,exercise_execution_extremes
from slm_lab.model import TinyMultimodalLM
from slm_lab.recurrent_ablation import state_digest
from slm_lab.training import load_checkpoint

from test_code_residual_stage_v2 import Guard
from test_code_residual_source_contract_v4 import Code,Replay
from test_codenet_runner_v1 import small_config,Tokenizer,options
from test_execution_preference_v1 import OwnCausalFixture,examples as pair_examples


class Feedback(ExecutionFeedback):
    """Explicit artificial CPU fixture; never a real-source admission claim."""
    def __init__(self,folder):
        self.folder=folder/'feedback';self.guard=Guard();self.held=False
        self.pairs=[dict(id='p'+str(i),split='train',human_preferences=False) for i in range(7)]
        self.positives=[dict(id='j'+str(i),split='train',human_preferences=False) for i in range(5)]
        self.manifest=json.loads((self.folder/'manifest.json').read_text())
        self.admission=json.loads((self.folder/'admission.json').read_text())
    def positive(self,i):
        body=[20+i]*(45 if i==0 else 2+i)+[2]
        return [1,10]+body,[-100,-100]+body
    def example(self,i):
        positive=self.positive(i%5)
        body=[31+i]*(41 if i==0 else 1+i%3)+[2]
        return dict(id='p'+str(i),split='validation' if self.held else 'train',
            chosen=positive,rejected=([1,10]+body,[-100,-100]+body),
            feedback_source='machine_execution_of_original_publisher_tests',
            human_preferences=False,finite_execution_verified=True)


class DoublePolicy(OwnCausalFixture):
    def forward(self,input_ids,labels,**ignored):
        return super().forward(input_ids,labels)


def gradient_oracles(rank):
    torch.manual_seed(441);model=DoublePolicy();direct=deepcopy(model);reference=deepcopy(model)
    with torch.no_grad():reference.head.weight.add_(.004)
    populations=[]
    for r in range(4):
        current=[]
        for i in range(r+1):
            e=deepcopy(pair_examples()[i%2]);e['id']=f'r{r}:e{i}';current.append(e)
        populations.append(current)
    all_pairs=[e for rows in populations for e in rows]
    cache=reference_scores(reference,all_pairs,policy_sha256='a'*64,device='cpu',amp=nullcontext)
    objective=0.
    for e in all_pairs:
        def lp(name):
            ids,labels=e[name];n=sum(v!=-100 for v in labels[1:])
            return -n*direct(torch.tensor([ids]),torch.tensor([labels]))['loss']
        ref=cache['records'][e['id']]
        objective+=.1*(-F.logsigmoid(.1*((lp('chosen')-lp('rejected'))-(ref['chosen_logp']-ref['rejected_logp']))))/10
    objective.backward()
    measured=execution_gradients(model,populations[rank],cache,expected_policy_sha256='a'*64,
        alpha=.1,beta=.1,device='cpu',amp=nullcontext)
    average_gradients(model,'cpu',4)
    assert measured['global_pairs']==10 and measured['weighted_DPO']==pytest.approx(float(objective.detach()),abs=1e-12)
    for actual,expected in zip(model.parameters(),direct.parameters()):
        torch.testing.assert_close(actual.grad,expected.grad,atol=1e-12,rtol=1e-12)
    # Two separately normalized CE streams, with unequal target counts on all
    # workers; direct autograd sums all actual programs before one backward.
    model.zero_grad(set_to_none=True);direct.zero_grad(set_to_none=True)
    def batch(r,j):
        body=[11+j]*(1+r+j)+[2];ids=[1,4]+body
        return dict(input_ids=torch.tensor([ids]),labels=torch.tensor([[-100,-100]+body]))
    complete=0.
    for j,coefficient in enumerate([1.,.3]):
        total=sum(2+r+j for r in range(4))
        for r in range(4):complete+=coefficient*(2+r+j)/total*direct(**batch(r,j))['loss']
    complete.backward()
    values=global_CE_gradients(model,[(str(j),SimpleNamespace(batches=[batch(rank,j)]),c)
        for j,c in enumerate([1.,.3])],SimpleNamespace(max_text_length=64,vocab_size=32),'cpu',nullcontext)
    average_gradients(model,'cpu',4)
    assert values['CE_summed_objective']==pytest.approx(float(complete.detach()),abs=1e-12)
    for actual,expected in zip(model.parameters(),direct.parameters()):
        torch.testing.assert_close(actual.grad,expected.grad,atol=1e-12,rtol=1e-12)


def worker(rank,rendezvous,directory):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=4,timeout=timedelta(seconds=120))
    try:
        gradient_oracles(rank)
        folder=Path(directory);config=small_config();tokenizer=Tokenizer();replay=Replay(folder);code=Code(folder)
        token=folder/'tokenizer.json';source=folder/'base.pt';plan=folder/'plan.json';data=folder/'data-manifest.json'
        if rank==0:
            token.write_bytes(b'own-fixture-tokenizer')
            torch.manual_seed(50189)
            torch.save(dict(config=asdict(config),model=TinyMultimodalLM(config).state_dict(),tokenizer_sha256=file_hash(token)),source)
            f=folder/'feedback';f.mkdir()
            (f/'manifest.json').write_text(json.dumps(dict(fixture=True,real_source_admission=False,tokenizer_sha256=file_hash(token))))
            (f/'admission.json').write_text(json.dumps(dict(fixture=True,real_source_admission=False,passed=True,
                source_roles_only_train=True,machine_feedback_not_human_preferences=True,
                candidate_manifest_sha256=file_hash(f/'manifest.json'),finite_assertion_pairs=7,complete_passing_originals=5)))
        dist.barrier();feedback=Feedback(folder)
        opts={**options(),'gradient_accumulation':1,'replay_accumulation':1,'image_size':16,'log_every':9}
        aux=dict(positive_batch_size=1,positive_accumulation=1,pair_batch_size=1,pair_accumulation=2,
            positive_coefficient=.3,DPO_alpha=.1,DPO_beta=.1)
        contract=build_required_contract(replay,file_hash(token),{n:hashlib.sha256(('fixture:'+n).encode()).hexdigest() for n in COMPONENT_NAMES})
        def fresh():
            torch.manual_seed(50189)
            return RoutedCodeResidual(TinyMultimodalLM(config),width=12)
        reference_model=fresh()
        bundle=prepare_reference_bundle(reference_model,feedback,base_checkpoint_sha256=file_hash(source),
            tokenizer_sha256=file_hash(token),device='cpu',amp=nullcontext,precision='fp32')
        del reference_model
        auxfp=auxiliary_fingerprint(bundle,aux)
        if rank==0:
            (folder/'reference.json').write_text(json.dumps(bundle))
            data.write_text(json.dumps(dict(fixture=True,real_source_admission=False,code=120,replay_families=30,execution_pairs=7)))
            plan.write_text(json.dumps(dict(fixture=True,production_stage_ready=False,base_checkpoint_sha256=file_hash(source),
                tokenizer_sha256=file_hash(token),code_manifest_sha256=file_hash(data),execution_auxiliary_fingerprint=auxfp)))
        dist.barrier()
        sources=dict(base_checkpoint_sha256=file_hash(source),tokenizer_sha256=file_hash(token),run_plan_sha256=file_hash(plan))
        def construct(reference):
            model=fresh();optimizer=torch.optim.AdamW(model.adapter_parameters(),lr=opts['learning_rate'],weight_decay=opts['weight_decay'],foreach=False)
            kernel=ExecutionResidualKernel(model,optimizer,code,packed_encoder(code,tokenizer,config),replay,
                ExpandedReplayEncoder(tokenizer,config,folder,image_size=16),opts,sources,'cpu',nullcontext,
                replay_contract=contract,expected_contract_fingerprint=fingerprint(contract),guard=Guard(),
                feedback=feedback,reference_bundle=reference,execution_options=aux,expected_auxiliary_fingerprint=auxfp)
            return model,optimizer,kernel
        model,optimizer,kernel=construct(bundle);assert kernel.steps==60
        base_before=state_digest(model.base.state_dict())
        random.seed(5101+rank);np.random.seed(5201+rank);torch.manual_seed(5301+rank)
        def state():
            return fingerprint(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng('cpu'),
                completed=kernel.completed,metrics=kernel.metrics,sampler=kernel.sampler_state(kernel.completed)))
        def update():
            draws=[random.random(),float(np.random.random()),float(torch.rand(()))]
            return draws,kernel.update()
        before=state();feedback.held=rank==2
        with pytest.raises(ValueError,match='Collective preflight failed'):kernel.update()
        feedback.held=False;assert state()==before
        if rank==3:feedback.positives[0]['split']='validation'
        with pytest.raises(ValueError,match='Collective preflight failed'):kernel.update()
        feedback.positives[0]['split']='train';assert state()==before
        for _ in range(31):update()
        partial=kernel.snapshot(folder/'resume.pt',maximum_bytes=2_000_000,minimum_free_bytes=0)
        before=state()
        if rank==1:kernel.reference_bundle['cache']['records']['p0']['chosen_logp']+=.001
        with pytest.raises(ValueError,match='Collective preflight failed'):
            kernel.snapshot(folder/'must-not-write.pt',maximum_bytes=2_000_000,minimum_free_bytes=0)
        assert not (folder/'must-not-write.pt').exists()
        kernel.reference_bundle=deepcopy(bundle);assert state()==before
        continuation=[update() for _ in range(31,60)]
        expected=state();next_rng=[random.random(),float(np.random.random()),float(torch.rand(()))]
        if rank==0:(folder/'reference.json').unlink()
        dist.barrier()
        recovered=reference_from_checkpoint(folder/'resume.pt',partial['checkpoint_sha256'])
        assert recovered==bundle
        model,optimizer,kernel=construct(recovered)
        assert kernel.restore(folder/'resume.pt',partial['checkpoint_sha256'])[0]==31
        assert [update() for _ in range(31,60)]==continuation and state()==expected
        assert [random.random(),float(np.random.random()),float(torch.rand(()))]==next_rng
        assert state_digest(model.base.state_dict())==base_before
        assert all(p.grad is None for p in model.parameters())
        progress=kernel.sampler_state(60)
        assert progress['code_examples']==progress['replay_examples']==240
        assert progress['execution_source_cursors']['positive']['examples']==240
        assert progress['execution_source_cursors']['pair']['examples']==480
        assert progress['execution_source_cursors']['pair']['epoch_and_cursor']==[68,4]
        assert progress['execution_reference_bundle']==bundle
        assert {m['replay_family'] for m in kernel.metrics}==set(replay.families)
        assert all(sum(m['replay_family']==n for m in kernel.metrics)==2 for n in replay.families)
        for step,metric in enumerate(kernel.metrics):
            positions=[p for r in range(4) for group in EpochOrder(5,opts['seed']+2000).rank_groups(step,r,4,1,1) for p in group]
            targets=sum(sum(v!=-100 for v in feedback.positive(p)[1]) for p in positions)
            assert metric['objectives']['verified_original_program']['global_target_tokens']==targets
            assert metric['global_execution_pairs']==8 and len(metric['execution_DPO']['records'])==8
            assert metric['execution_reference_fingerprint']==fingerprint(bundle)
        kernel.snapshot(folder/'complete.pt',maximum_bytes=2_000_000,minimum_free_bytes=0)
        before=state()
        with pytest.raises(ValueError,match='Collective preflight failed'):kernel.update()
        assert state()==before
        # Use the actual production loop, including rotation, a bounded stop,
        # failed save, explicit recovery and an idempotent inference export.
        output=folder/'lifecycle'
        if rank==0:
            output.mkdir()
            shutil.copy2(folder/'resume.pt',output/'resume.pt')
            (output/'checkpoint-ledger.json').write_text(json.dumps(dict(format=LEDGER_FORMAT,
                run_plan_sha256=sources['run_plan_sha256'],latest=dict(checkpoint=str(output/'resume.pt'),
                    checkpoint_sha256=partial['checkpoint_sha256'],checkpoint_bytes=partial['checkpoint_bytes'],steps=31),previous=None)))
            (folder/'code-pack').mkdir()
            shutil.copy2(data,folder/'code-pack/manifest.json')
        dist.barrier()
        def restore_loop(digest):
            reference=reference_from_checkpoint(output/'resume.pt',digest)
            m,o,k=construct(reference)
            k.restore(output/'resume.pt',digest)
            return m,o,k
        class Caller:
            def __getattr__(self,name):return getattr(kernel,name)
            def update(self):return update()[1]
        loop_options=dict(maximum_checkpoint_bytes=2_000_000,minimum_free_bytes=0,
            checkpoint_every=5,log_every=7,maximum_report_bytes=2_000_000)
        model,optimizer,kernel=restore_loop(partial['checkpoint_sha256'])
        bounded=run_updates(Caller(),output,until_step=43,**loop_options)
        assert not bounded['training_complete'] and bounded['completed_steps']==43
        ledger=rank_zero(lambda:checkpoint_layout(output,sources['run_plan_sha256']))
        assert ledger['latest']['steps']==43 and ledger['previous']['steps']==40
        retained={name:file_hash(output/name) for name in ['resume.pt','previous.pt','checkpoint-ledger.json']}
        with pytest.raises(ValueError,match='Collective snapshot failed'):
            run_updates(Caller(),output,until_step=44,**{**loop_options,'maximum_checkpoint_bytes':100})
        assert all(file_hash(output/name)==digest for name,digest in retained.items())
        assert not (output/'pending.pt').exists()
        model,optimizer,kernel=restore_loop(retained['resume.pt'])
        completed=run_updates(Caller(),output,**loop_options)
        assert completed['training_complete'] and state()==expected
        ledger=rank_zero(lambda:checkpoint_layout(output,sources['run_plan_sha256']))
        assert ledger['latest']['steps']==60 and ledger['previous']['steps']==55
        assert {p.name for p in output.glob('*.pt')}=={'resume.pt','previous.pt'}
        inline_plan=dict(base_checkpoint=str(source),tokenizer=str(token),
            base_checkpoint_sha256=sources['base_checkpoint_sha256'],code_manifest_sha256=file_hash(data),
            code_pack=str(folder/'code-pack'),storage=dict(maximum_inference_bytes=2_000_000,
                minimum_free_bytes=0,maximum_report_bytes=2_000_000))
        receipt=rank_zero(lambda:finish_inference(output,inline_plan,plan,completed))
        retained={name:file_hash(output/name) for name in ['resume.pt','previous.pt','training-completed.json','inference.pt']}
        assert rank_zero(lambda:finish_inference(output,inline_plan,plan,completed))==receipt
        assert all(file_hash(output/name)==digest for name,digest in retained.items())
        for metric in kernel.metrics:
            assert metric['total_target_tokens']==metric['CE_total_target_tokens']+sum(
                r['chosen_targets']+r['rejected_targets'] for r in metric['execution_DPO']['records'])
        positive,pairs,maxima=auxiliary_extremes(feedback)
        before=state()
        probes=exercise_execution_extremes(kernel,0,replay.row(replay.families[-1],0),'response',positive,pairs)
        assert {key:max(p[key+'_context_tokens'] for p in probes) for key in ['positive','chosen','rejected']}==maxima
        assert all(p['global_execution_pairs']==8 and p['finite_all_and_only_residual_gradients'] for p in probes)
        assert state()==before and all(p.grad is None for p in model.parameters())
        (folder/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,actual_reference_updates=60,
            repeated_exact_resume_updates=29,all30_replay_families_executed_twice=True,
            unequal_global_pair_and_CE_target_means_match_independent_FP64_autograd=True,
            complete_positive_response_exceeds_old_half_window=True,
            repeated_source_exposures_across_epoch_boundaries_retained=True,
            exact_Adam_RNG_full_history_reference_scores_and_all_cursors_resume=True,
            reference_recovered_from_checkpoint_after_external_cache_deleted=True,
            one_rank_held_pair_and_positive_role_reject_before_update=True,
            one_rank_cache_change_rejects_snapshot_without_write=True,frozen_backbone_identical=True,
            actual_production_loop_stop43_failure44_recovery43_to60_exact=True,
            bounded_latest60_previous55_checkpoint_rotation_and_idempotent_export=True,
            actual_combined_maximum_code_image_positive_and_complete_DPO_backward=True,
            model_counts=model.parameter_counts(),
            scope='Own8688-parameter CPU replica and explicit artificial source fixtures. No real-source admission, full857M run or model capability result.')))
    finally:dist.destroy_process_group()


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    folder=tmp_path_factory.mktemp('complete-execution-training-v5')
    Image.new('RGB',(8,4),(21,55,90)).save(folder/'image.png')
    Image.new('RGB',(8,4),(220,85,10)).save(folder/'other.png')
    mp.spawn(worker,args=(str(folder/'rendezvous'),str(folder)),nprocs=4,join=True)
    return folder


def test_four_actual_workers_preserve_all30_sources_and_complete_auxiliary_state_through_resume(artifacts):
    for rank in range(4):
        proof=json.loads((artifacts/f'rank-{rank}.json').read_text())
        assert proof['all30_replay_families_executed_twice']
        assert proof['exact_Adam_RNG_full_history_reference_scores_and_all_cursors_resume']


def test_unequal_worker_pair_and_token_normalizers_match_independent_autograd(artifacts):
    for rank in range(4):
        assert json.loads((artifacts/f'rank-{rank}.json').read_text())['unequal_global_pair_and_CE_target_means_match_independent_FP64_autograd']


def test_checkpoint_contains_original_reference_and_exact_auxiliary_cursors(artifacts):
    state=torch.load(artifacts/'complete.pt',weights_only=True)
    assert state['steps']==state['planned_steps']==60 and len(state['worker_rng_states'])==4
    progress=state['sampler_state']
    assert progress['execution_source_cursors']['pair']['examples']==480
    assert len(progress['execution_reference_bundle']['cache']['records'])==7
    assert not (artifacts/'reference.json').exists()


def test_completed_execution_state_exports_without_discarding_complete_training_state(artifacts):
    folder=artifacts;checkpoint=folder/'complete.pt';before=file_hash(checkpoint)
    receipt=export_inference(checkpoint,folder/'base.pt',folder/'tokenizer.json',folder/'plan.json',folder/'data-manifest.json',
        folder/'inference.pt',expected_training_checkpoint_sha256=before,maximum_bytes=2_000_000,minimum_free_bytes=0)
    assert file_hash(checkpoint)==before
    policy,_=load_inference(folder/'inference.pt',folder/'base.pt',folder/'tokenizer.json',
        expected_checkpoint_sha256=receipt['checkpoint_sha256'],expected_base_sha256=file_hash(folder/'base.pt'),
        expected_plan_sha256=file_hash(folder/'plan.json'),expected_training_checkpoint_sha256=before,
        expected_data_sha256=file_hash(folder/'data-manifest.json'))
    base,_=load_checkpoint(folder/'base.pt','cpu',training_state=False)
    direct=RoutedCodeResidual(base,width=12);direct.load_adapter_state(torch.load(checkpoint,weights_only=True)['residuals'])
    direct.eval();ids=torch.tensor([[1,10,20,2]])
    with torch.no_grad():torch.testing.assert_close(policy(ids)['logits'],direct(ids,route='code')['logits'],rtol=0,atol=0)
