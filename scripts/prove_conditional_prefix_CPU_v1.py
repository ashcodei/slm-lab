"""Real small-network conditional CPU updates, exact recovery and failure guards.

This exercises full original program CE plus real conditional suffix sampling.
The explicit2-action CPU mechanics horizon is not full-size capacity or quality.
"""
import copy
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/conditional-prefix-CPU-v1'


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or OUT.exists():raise ValueError('Fresh CPU-hidden conditional mechanics proof required')
    import numpy as np
    import torch
    from slm_lab.admitted_public_prefix_episode_v1 import AdmittedPublicPrefixEpisode
    from slm_lab.anchored_prefix_execution_CPU_v1 import update
    from slm_lab.codenet_checkpoint_v1 import capture_rng
    from slm_lab.conditional_prefix_stage_CPU_v1 import ConditionalPrefixStage,trajectory_digest
    from slm_lab.corpus import file_hash
    from slm_lab.diverse_images import atomic_json
    from slm_lab.expanded_execution_corpus_v1 import ExpandedExecutionCorpus
    from slm_lab.frozen_own_actor_reference_v1 import FrozenOwnActorReference
    from slm_lab.model import ModelConfig
    from slm_lab.own_dense_actor_v1 import OwnDenseActor
    from slm_lab.own_dense_growth_v1 import OwnDenseAssistant,new_backbone
    from slm_lab.whole_human_sft_evidence_v1 import tensor_digest
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);OUT.mkdir()
    corpus=ExpandedExecutionCorpus(ROOT)
    position=corpus.by_id['apps:apps:publisher-train:0001:solution:19:verified-generation-v1']
    episode=AdmittedPublicPrefixEpisode(corpus,position,3,root=ROOT)
    config=ModelConfig(vocab_size=32768,hidden_size=32,intermediate_size=64,num_layers=2,
        num_heads=4,num_kv_heads=2,max_text_length=4096,vision_hidden_size=32,vision_intermediate_size=64,
        vision_layers=1,vision_heads=4,image_size=32,patch_size=16,pool_size=2)
    base=OwnDenseAssistant(new_backbone(config,965117),reward_hidden=8,word_temporal=4,pointing_rank=8,
        router_hidden=8,detector_hidden=4,word_expansion=1).eval()
    model=OwnDenseActor(base,rank=2,seed=965217).eval();parameters=[p for _,p in model.named_selected(('actor',))]
    def optimizer():return torch.optim.AdamW(parameters,lr=1e-4,betas=(.9,.95),weight_decay=.01,foreach=False)
    warmup=optimizer();batch=episode.batch('cpu');model(**batch)['loss'].backward(inputs=parameters)
    warmup.step();warmup.zero_grad(set_to_none=True)
    reference=FrozenOwnActorReference(model,sources=dict(trained_own_starting_actor=tensor_digest(model.parts.named_parameters())))
    with torch.no_grad():
        if torch.equal(reference(batch['input_ids'])['logits'],base(batch['input_ids'])['logits']):
            raise ValueError('A genuinely trained private own reference must differ from the bare base')
    adam=optimizer();sources=dict(kernel_sha256=file_hash(ROOT/'src/slm_lab/anchored_prefix_execution_CPU_v1.py'),
        qualified_source_sha256=episode.binding()['original_native_input_sha256'])
    stage=ConditionalPrefixStage(model,adam,episode,reference,method='anchor_execution',planned_updates=2,
        sources=sources,local_group_size=2,CPU_mechanics_horizon=2,cache_absolute_bound=.0001)
    random.seed(965317);np.random.seed(965317);torch.manual_seed(965317)
    base_digest=tensor_digest(base.named_parameters());critic=model.parts['critic'].weight.detach().clone()
    files=[Path(__file__),ROOT/'src/slm_lab/admitted_public_prefix_episode_v1.py',
        ROOT/'src/slm_lab/anchored_prefix_execution_CPU_v1.py',ROOT/'src/slm_lab/conditional_prefix_stage_CPU_v1.py',
        ROOT/'reports/admitted-public-prefix-episode-v1-final-tests.log',
        ROOT/'reports/public-reference-prefix-curriculum-v1-independent-reader/complete.json']
    bindings={str(p):file_hash(p) for p in files}
    atomic_json(OUT/'plan.json',dict(at=datetime.now(timezone.utc).isoformat(),source_files_sha256=bindings,
        actual_counts=model.counts(),qualified_episode=episode.binding(),logical_updates=2,physical_updates=3,
        CPU_mechanics_action_horizon=2,original_full_reference_EOS_targets=166,
        small_architecture_not_the_current_full_model=True,new_full_GPU_capacity_or_capability_not_proven=True,goal_achieved=False))
    def equal(a,b):
        if isinstance(a,torch.Tensor):return torch.equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return type(a)is type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    first=stage.update();saved=OUT/'after-first.pt';torch.save(stage.state(),saved)
    second=stage.update();weights=model.export(sources);moments=copy.deepcopy(adam.state_dict());rng=capture_rng('cpu')
    raw=torch.load(saved,map_location='cpu',weights_only=True)
    for field in ['episode_binding','own_reference_binding']:
        bad=copy.deepcopy(raw)
        if field=='episode_binding':bad[field]['retained_reference_quarters']=2
        else:bad[field]['frozen_actor_parts_sha256']='f'*64
        before=[(id(p),p._version,p.data_ptr()) for p in model.parameters()]
        try:stage.resume(bad)
        except ValueError:pass
        else:raise ValueError('Changed source hint or own reference must reject before mutation')
        if before!=[(id(p),p._version,p.data_ptr()) for p in model.parameters()] or not equal(rng,capture_rng('cpu')):
            raise ValueError('Rejected prefix/reference recovery changed values or private RNG')
    stage.resume(raw);replayed=stage.update()
    if (not equal(weights,model.export(sources)) or not equal(moments,adam.state_dict())
            or not equal(rng,capture_rng('cpu')) or trajectory_digest(second['trajectories'])!=trajectory_digest(replayed['trajectories'])):
        raise ValueError('Actual next conditional update, Adam, all private RNG and native suffix records must replay exactly')
    prepared=episode.rollout_view()
    for record in [first,second,replayed]:
        if record['source_reference_targets']!=166 or not record['original_reference_used_regardless_of_rewards']:
            raise ValueError('Every update must supervise the complete original reference/EOS targets')
        for item in record['trajectories']:
            if (item['supplied_public_reference_prefix_ids']!=prepared['public_reference_prefix_ids']
                    or item['complete_candidate_response']!=corpus.tokenizer.decode(prepared['public_reference_prefix_ids']+item['response_ids'])):
                raise ValueError('Only the genuine sampled suffix may be attributed to the neural policy')
    original_prefill=model.policy.prefill
    def biased(*args,**kwargs):
        result=original_prefill(*args,**kwargs);logits=result['logits'].clone();logits[:,:,::2]+=.2
        return {**result,'logits':logits}
    model.policy.prefill=biased;failure=None
    try:update(model,adam,episode,own_reference=reference,method='anchor_execution',local_group_size=2,
        CPU_mechanics_horizon=2,cache_absolute_bound=.0001)
    except ValueError as error:failure=str(error)
    finally:model.policy.prefill=original_prefill
    if (failure is None or 'Cached/full actual behavior' not in failure or not equal(weights,model.export(sources))
            or not equal(moments,adam.state_dict()) or any(p.grad is not None for p in model.parameters())):
        raise ValueError('A deliberately wrong sampling probability must fail without any optimizer commit')
    if base_digest!=tensor_digest(base.named_parameters()) or not torch.equal(critic,model.parts['critic'].weight):
        raise ValueError('Own foundation, all vision/auxiliary heads and unused critic must remain exact')
    for name,digest in bindings.items():
        if file_hash(Path(name))!=digest:raise ValueError('Bound CPU proof source changed')
    atomic_json(OUT/'actual-native-steps.json',[first,second,replayed])
    result=dict(at=datetime.now(timezone.utc).isoformat(),complete=True,actual_counts=model.counts(),logical_updates=2,physical_updates=3,
        actual_trained_own_reference_adapter_warmup_updates=1,conditional_own_reference_is_exact_starting_policy=True,
        all_original166_reference_EOS_targets_supervised_on_every_update=True,
        public_prefix_and_actual_sampled_suffix_reconstruction_verified=True,
        next_conditional_update_weights_Adam_Python_NumPy_TorchCPU_RNG_and_actual_trajectory_replay_exact=True,
        wrong_prefix_or_own_reference_restore_rejected_before_mutation=True,
        deliberate_probability_fault_rejected_before_any_optimizer_commit=True,
        base_vision_auxiliary_heads_and_unused_critic_unchanged=True,
        source_files_sha256={**bindings,str(saved):file_hash(saved),str(OUT/'actual-native-steps.json'):file_hash(OUT/'actual-native-steps.json')},
        CPU_short_horizon_small_architecture_only=True,current_full_model_updates=0,
        new_full_size_GPU_capacity_or_held_quality_not_proven=True,candidate_promoted=False,goal_achieved=False)
    atomic_json(OUT/'complete.json',result);print(json.dumps(dict(actual_counts=model.counts(),logical_updates=2,physical_updates=3,
        next_actual_conditional_update_and_all_private_RNG_exact=True,full_original_reference_targets=166,
        probability_fault_optimizer_commits=0,current_full_model_updates=0)),flush=True)


if __name__=='__main__':main()
