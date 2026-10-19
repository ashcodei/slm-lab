"""Actual four-replica complete objective/recovery controls, not GPU admission."""
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch
from torch import distributed as dist
from torch.multiprocessing import spawn

from slm_lab.codenet_checkpoint_v1 import atomic_checkpoint,capture_rng
from slm_lab.corpus import file_hash
from slm_lab.frozen_own_text_reference_v1 import FrozenOwnTextReferenceV1
from slm_lab.joint_objective_step_v1 import JointReplicaGuardV1,update
from slm_lab.mixed_coding_SFT_data_v1 import MixedCodingSFTData
from slm_lab.own_dense_actor_fold_v1 import fold_actor,signature
from slm_lab.own_dense_standalone_v1 import export,load
from slm_lab.parallel_function_repair_stage_v1 import value_digest
from slm_lab import shared_joint_durable_state_v2 as durable
from slm_lab import shared_joint_checkpoint_v2 as private
from slm_lab.shared_joint_gradient_reduce_v2 import reduce
from slm_lab.whole_human_data_v2 import WholeHumanData
from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
from test_dense_human_preference_v1 import pairs
from test_joint_task_gradients_v1 import view,code_positions,photo_fixture,word_fixture,correction_fixture

ROOT=Path(__file__).resolve().parents[1]
OPTIONS=dict(maximum_examples=1,maximum_padded_tokens=16384,gradient_bucket_values=4096,gradient_clip=1.,human_weight=.2,visual_weight=.3,
    CTC_weight=1.,native_word_CE_weight=1.,matching_weight=.1,temperature=.7,correction_weight=.1,
    DPO_beta=.1,chosen_CE_weight=.1,project_conflicts=True,precision='fp32')


def visual_packet(kind,rank,world):
    if kind=='photo':
        native,matching,source=photo_fixture();word=None;negatives={}
        # Retain a complete three-source photo batch, including an actual
        # empty fourth task rank. Global negatives/positives remain intact.
        native={k:v[:3] for k,v in native.items()};matching={k:v[:3] for k,v in matching.items()}
        for key in ['positions','context_lengths','response_EOS_targets','negative_context_lengths','negative_response_EOS_targets','pixel_keys','captions']:source[key]=source[key][:3]
    elif kind=='word':native,word,source=word_fixture();matching=None;negatives={}
    else:native,negatives,source=correction_fixture();word=matching=None
    local=list(range(len(source['positions'])))[rank::world]
    selected={k:v[local] for k,v in native.items()} if local else None
    words=None if word is None or not local else dict(word_pixels=word['word_pixels'][local],texts=[word['texts'][i] for i in local])
    matches=None if matching is None or not local else {k:v[local] for k,v in matching.items()}
    negative={local.index(i):value for i,value in negatives.items() if i in local}
    return dict(kind=kind,native=selected,word=words,matching=matches,negatives=negative,source=source)


class Trial:
    def __init__(self,data,precision='fp32',rank=0,world=1):
        self.view=view();self.data=data;self.positions=code_positions(data);self.rank,self.world=rank,world
        self.human=pairs();self.human.batch=WholeHumanData.batch.__get__(self.human,WholeHumanData)
        self.reference=FrozenOwnTextReferenceV1(self.view.actor.base,sources={'synthetic_fixture': 'a'*64},tokenizer_sha256='b'*64)
        self.optimizer=torch.optim.AdamW([p for _,p in self.view.named_selected()],lr=1e-4,eps=1e-5,weight_decay=.01,foreach=False)
        self.guard=JointReplicaGuardV1(self.view,self.optimizer);self.options={**OPTIONS,'precision':precision};self.metrics=[]
        code_info=data.metadata(self.positions,'balanced_sources');self.entries=[]
        source_sha='a'*64
        def objective(task,family,positions,context,targets,denominators,bytes_=0):
            return dict(task=task,source_family=family,source_role='train',source_sha256=source_sha,original_positions=positions,
                complete_context_tokens=context,response_EOS_targets=targets,word_UTF8_bytes=bytes_,normalization_denominators=denominators)
        for step,kind in enumerate(['photo','word','native',None],1):
            objects=[]
            if kind is not None:objects.append(objective('text','actual_admitted_code',self.positions,code_info['complete_context_tokens'],code_info['response_EOS_targets'],
                {'original_positive_weighted_targets':code_info['actual_global_weighted_targets']}))
            objects.append(objective('text','synthetic_human_pair_control',[0,1],19,11,{'human_pairs':2,'chosen_response_EOS':5}))
            if kind=='photo':objects.append(objective('image_matching','synthetic_photo',[17,20,23],18,9,{'native_response_EOS':9,'image_pairs':3}))
            elif kind=='word':
                objects.append(objective('native_image','synthetic_native_word',[17,20],12,6,{'native_response_EOS':6}))
                objects.append(objective('word_CTC','synthetic_CTC_word',[17,20],0,0,{'whole_words':2},3))
            elif kind=='native':objects.append(objective('native_image','synthetic_correction',[17,20,23],26,14,{'positive_native_EOS':9,'corrective_pairs':1,'negative_EOS':5}))
            self.entries.append(dict(step=step,objectives=objects))
        self.sources=dict(text_initializer_sha256=source_sha,visual_initializer_sha256='b'*64,tokenizer_sha256=file_hash(data.weak.folder/'tokenizer.json'),
            run_plan_sha256=value_digest(dict(options=self.options,code_identity=data.identity_sha256,entries=self.entries)),
            source_admission_sha256='d'*64,complete_schedule_sha256=value_digest(self.entries),fixed_values_sha256=self.guard.fixed.sha256)

    def step(self):
        number=len(self.metrics);kind=['photo','word','native',None][number]
        probe=[random.random(),float(np.random.random()),float(torch.rand(()))]
        record=update(self.view,self.reference,self.optimizer,self.guard,
            code=dict(data=self.data,positions=self.positions,profile='balanced_sources') if kind else None,
            human=dict(data=self.human,positions=[0,1],DPO_weight=1),visual=visual_packet(kind,self.rank,self.world) if kind else None,
            options=self.options,source_entry_sha256=value_digest(self.entries[number]))
        losses=record['global_normalized_task_losses'];array=([losses['code']] if kind else [])+[losses['human']]
        if kind=='word':
            rows=record['actual_all_rank_task_records']
            array.extend([sum(r['tasks']['visual']['local_native_CE_sum'] for r in rows)/6,
                sum(r['tasks']['visual']['local_byte_normalized_CTC_sum'] for r in rows)/2])
        elif kind:array.append(losses['visual'])
        self.metrics.append(dict(step=number+1,entry_sha256=value_digest(self.entries[number]),active_parameter_names=record['actual_active_parameter_names'],
            normalized_task_losses=array,gradient_norm_before_clip=record['gradient_norm_before_clip'],globally_reduced_task_gradients_before_combining=True,
            gradient_interaction=dict(actual_blend=record['gradient_interaction'],actual_task_transports=record['actual_task_transports'],actual_all_rank_tasks=record['actual_all_rank_task_records'])))
        return probe

    def commit(self,path):
        return durable.commit(self.view,self.optimizer,self.guard.fixed,path,sources=self.sources,entries=self.entries,completed=len(self.metrics),metrics=self.metrics,
            rank=self.rank,world=self.world,maximum_bytes=8*1024**2,minimum_free_bytes=40*1024**3)

    def restore(self,path):
        _,self.metrics,_=durable.restore_latest(self.view,self.optimizer,self.guard.fixed,path,sources=self.sources,entries=self.entries,rank=self.rank,world=self.world)
        self.guard=JointReplicaGuardV1(self.view,self.optimizer)


def live(trial):return tensor_digest(trial.view.named_parameters()),value_digest(trial.optimizer.state_dict()),value_digest(capture_rng('cpu'))


def check_native(trial,folder):
    before=live(trial);graph=signature(trial.view.actor)
    folded,proof=fold_actor(trial.view.actor,sources=trial.sources,working_budget_bytes=256*1024**2)
    folder=Path(folder);folder.mkdir()
    shutil.copyfile(trial.data.weak.folder/'tokenizer.json',folder/'tokenizer.json')
    atomic_checkpoint(folder/'model.pt',export(folded,sources=trial.sources,provenance=proof,completed_updates=len(trial.metrics),
        tokenizer_sha256=trial.sources['tokenizer_sha256']),8*1024**2,40*1024**3)
    cold,metadata=load(folder/'model.pt',expected_checkpoint_sha256=file_hash(folder/'model.pt'),expected_sources=trial.sources)
    assert tensor_digest(cold.named_parameters())==tensor_digest(folded.named_parameters())
    photo,_,_=photo_fixture();words=word_fixture()[1];errors=[]
    probes=[{k:v for k,v in trial.data.batch([trial.positions[0]]).items() if k!='labels'},
        {k:v[:1] for k,v in photo.items() if k!='labels'}]
    with torch.no_grad():
        for batch in probes:
            hot=trial.view.actor(**batch)['logits'];merged=folded(**batch)['logits']
            assert torch.allclose(hot,merged,atol=1e-5,rtol=1e-4)
            assert torch.equal(cold(**batch)['logits'],merged)
            errors.append(float((hot-merged).abs().max()))
        assert torch.equal(trial.view.word_logits(words['word_pixels']),cold.word_logits(words['word_pixels']))
    assert live(trial)==before and signature(trial.view.actor)==graph and metadata['no_old_base_actor_optimizer_or_training_reference_loaded']
    return dict(standalone_sha256=file_hash(folder/'model.pt'),native_values=cold.counts()['total_unique'],
        hot_to_folded_FP32_text_and_photo_maximum_errors=errors,all_native_parameters_and_folded_text_photo_whole_word_outputs_cold_exact=True,
        original_graph_weights_Adam_and_private_RNG_unchanged=True,GPU_BF16_generation_equivalence_not_certified=True)


def worker(rank,folder,precision):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.nice(0)>=19
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo',init_method='file://'+str(Path(folder)/'rendezvous'),rank=rank,world_size=4,timeout=timedelta(minutes=10))
    try:
        trial=Trial(MixedCodingSFTData(ROOT),precision,rank,4)
        random.seed(992100+rank);np.random.seed(992100+rank);torch.manual_seed(992100+rank)
        trial.step();trial.commit(Path(folder)/'state')
        expected_probe=trial.step();expected=live(trial);expected_rows=copy.deepcopy(trial.metrics)
        trial.restore(Path(folder)/'state');assert trial.step()==expected_probe and live(trial)==expected and trial.metrics==expected_rows
        trial.step();trial.step();trial.commit(Path(folder)/'state')
        trial.guard.fixed.check(trial.view,trial.optimizer,full=True);trial.reference.assert_bound(full=True)
        clocks={n:int(trial.optimizer.state[p]['step']) if p in trial.optimizer.state else 0 for n,p in trial.view.named_selected()}
        assert all(v==4 for n,v in clocks.items() if n.startswith('actor.'))
        assert all(v==3 for n,v in clocks.items() if n.startswith('vision.'))
        assert all(v==1 for n,v in clocks.items() if n.startswith(('word.','pointing.')))
        assert 'word.height.bias' not in clocks
        before=live(trial)
        bad_human=pairs();bad_human.batch=WholeHumanData.batch.__get__(bad_human,WholeHumanData)
        if rank==1:bad_human.rows[0]['feedback_source']='machine'
        with pytest.raises(ValueError,match='Coordinated joint checkpoint rejection'):
            update(trial.view,trial.reference,trial.optimizer,trial.guard,human=dict(data=bad_human,positions=[0,1],DPO_weight=1),
                options=trial.options,source_entry_sha256='e'*64)
        assert live(trial)==before and all(p.grad is None for p in trial.view.parameters()) and not torch.cuda.is_initialized()
        native_proof=check_native(trial,Path(folder)/'standalone') if rank==0 else None
        row=dict(rank=rank,precision=precision,actual_small_model_counts=trial.view.actor.counts(),actual_joint_physical_AdamW_calls=5,
            actual_joint_logical_updates=4,exact_next_complete_weights_Adam_private_RNG_and_full_task_record_replay=True,
            actor_steps=4,vision_steps=3,pointing_steps=1,word_steps=1,complete_native_negative_EOS_targets_per_logical_update=5,
            all_original_whole_CTC_bytes_and_native_CE_retained=True,actual_empty_photo_word_native_and_human_task_ranks=True,
            one_bad_human_rank_rejected_before_all_peer_optimizer_mutations=True,
            frozen_language_critic_reward_router_detector_and_redundant_word_bias_unchanged=True,
            actual_source_weighted_code_denominator=trial.entries[0]['objectives'][0]['normalization_denominators']['original_positive_weighted_targets'],
            final_parameters_sha256=tensor_digest(trial.view.named_parameters()),final_Adam_sha256=value_digest(trial.optimizer.state_dict()),
            actual_native_cold_export=native_proof,
            synthetic_image_word_human_controls_and_real_admitted_code_only=True,full_size_GPU_capacity_or_quality_gain=False)
        (Path(folder)/('actual-rank-'+str(rank)+'.json')).write_text(json.dumps(row,indent=2))
        dist.barrier()
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('precision',['fp32','bf16'])
def test_actual_four_replica_joint_objectives_sparse_Adam_and_complete_recovery(tmp_path,precision):
    spawn(worker,args=(str(tmp_path),precision),nprocs=4,join=True)
    rows=[json.loads((tmp_path/('actual-rank-'+str(rank)+'.json')).read_text()) for rank in range(4)]
    assert len({r['final_parameters_sha256'] for r in rows})==len({r['final_Adam_sha256'] for r in rows})==1
    assert sum(r['actual_joint_physical_AdamW_calls'] for r in rows)==20


@pytest.mark.parametrize('value',[0.,-1.,float('nan'),True])
def test_weighted_global_denominator_cannot_be_zero_negative_nonfinite_or_boolean(value):
    parameter=torch.nn.Parameter(torch.ones(3));gradient=torch.tensor([1.,2.,3.])
    with pytest.raises(ValueError):reduce({'value':gradient},named_parameters=[('value',parameter)],task_sha256='a'*64,
        global_denominators={'original_weighted_targets':value},bucket_values=2)
    assert torch.equal(gradient,torch.tensor([1.,2.,3.])) and parameter.grad is None
